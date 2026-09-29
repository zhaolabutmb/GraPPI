#!/usr/bin/env python
"""
random_mut_foldx.py -- stand-alone random interface mutagenesis + FoldX model building.

Given a PDB complex and its receptor/ligand chain assignment, the script:
  1. Finds the binding-interface residues (any heavy-atom pair within --dist-cutoff A).
  2. Randomly picks --num-mut-on-each interface residues on each side (receptor and
     ligand) and assigns each a random non-wild-type amino acid.
  3. Repeats step 2 --n-variants times to build independent mutation sets.
  4. Calls FoldX BuildModel once (all variants in a single individual_list.txt) to
     generate the mutant structures and renames them to <pdb>_<mutations>.pdb.

Mutant structures and FoldX's own output files are written to --out-dir (the current
directory by default); the sampled mutations are reported on stdout as they are made.

Dependencies: numpy, biopython. FoldX is expected to be callable as `foldx` (on PATH);
override with --foldx.

Usage
-----
    # in this project, run inside the `PPI` conda env (has numpy + biopython)
    python random_mut_foldx.py --pdb xxx.pdb --receptor A --ligand B

    # multi-chain sides, 3 mutations per side, 10 variants, custom output dir
    python random_mut_foldx.py --pdb complex.pdb --receptor H,L --ligand A \
        --num-mut-on-each 3 --n-variants 10 --out-dir ./mutants --seed 0

Mutation strings follow the `{chain}_{wt}{position}{mut}` convention used elsewhere in
this project (e.g. `A_Y87R,B_V166L`); they are translated to FoldX's `{wt}{chain}{position}{mut}`
format when the individual_list.txt file is written.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

import numpy as np
from Bio.PDB import PDBParser

# ==============================================================================
# Constants
# ==============================================================================

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

ALL_AA_1 = list(THREE_TO_ONE.values())

# Non-standard residue names mapped onto their standard parent
RESNAME_ALIASES = {'HIE': 'HIS', 'HID': 'HIS', 'HIP': 'HIS', 'MSE': 'MET'}


class NoInterfaceError(Exception):
    """No (or too small an) interface between the two sides of the complex."""


class OversizedError(Exception):
    """The complex exceeds the allowed residue count."""


# ==============================================================================
# PDB parsing
# ==============================================================================

def parse_chain_ids(chain_str):
    """Parse a comma-separated chain specification into a list of chain IDs."""
    return [c.strip() for c in chain_str.split(',') if c.strip()]


def parse_complex(pdb_path, receptor_chains, ligand_chains, model_id=0):
    """
    Read a PDB file and return the list of standard residues belonging to the
    receptor or ligand chains.

    Returns:
        List of dicts with keys: chain, res_seq, icode, res_name, res_letter,
        side ('receptor'/'ligand') and coords (np.ndarray of heavy-atom xyz).
    """
    structure = PDBParser(QUIET=True).get_structure('complex', pdb_path)
    model = list(structure)[model_id]

    side_of_chain = {}
    for cid in receptor_chains:
        side_of_chain[cid] = 'receptor'
    for cid in ligand_chains:
        if cid in side_of_chain:
            raise ValueError(f"Chain '{cid}' listed as both receptor and ligand.")
        side_of_chain[cid] = 'ligand'

    present = {chain.get_id() for chain in model}
    missing = [cid for cid in side_of_chain if cid not in present]
    if missing:
        raise ValueError(
            f"Chain(s) {missing} not found in {pdb_path}. Available: {sorted(present)}"
        )

    residues = []
    for chain in model:
        side = side_of_chain.get(chain.get_id())
        if side is None:
            continue
        for residue in chain.get_residues():
            het_flag, res_seq, icode = residue.get_id()
            if het_flag.strip() and het_flag != 'H_MSE':
                continue
            res_name = RESNAME_ALIASES.get(residue.get_resname(), residue.get_resname())
            if res_name not in THREE_TO_ONE:
                continue
            coords = [a.get_coord() for a in residue.get_atoms()
                      if a.element != 'H' and not a.get_name().startswith('H')]
            if not coords:
                continue
            residues.append({
                'chain': chain.get_id(),
                'res_seq': res_seq,
                'icode': icode.strip(),
                'res_name': res_name,
                'res_letter': THREE_TO_ONE[res_name],
                'side': side,
                'coords': np.asarray(coords, dtype=np.float32),
            })

    return residues


# ==============================================================================
# Interface detection
# ==============================================================================

def _flatten_coords(residues, indices):
    """Stack the heavy-atom coordinates of the given residues and remember the owner."""
    coords = np.concatenate([residues[i]['coords'] for i in indices], axis=0)
    owner = np.repeat(indices, [len(residues[i]['coords']) for i in indices])
    return coords, owner


def find_interface_residues(residues, dist_cutoff=8.0, chunk_size=1024):
    """
    Identify interface residues between the receptor and ligand sides.

    A residue is at the interface when at least one of its heavy atoms lies within
    `dist_cutoff` angstrom of a heavy atom on the opposite side.

    Returns:
        Tuple (receptor_indices, ligand_indices), sorted lists of indices into `residues`.
    """
    rec_idx = np.array([i for i, r in enumerate(residues) if r['side'] == 'receptor'])
    lig_idx = np.array([i for i, r in enumerate(residues) if r['side'] == 'ligand'])
    if len(rec_idx) == 0 or len(lig_idx) == 0:
        raise NoInterfaceError("One side of the complex has no standard residues.")

    rec_coords, rec_owner = _flatten_coords(residues, rec_idx)
    lig_coords, lig_owner = _flatten_coords(residues, lig_idx)

    cutoff_sq = float(dist_cutoff) ** 2
    lig_sq = np.einsum('ij,ij->i', lig_coords, lig_coords)

    rec_hits, lig_hits = set(), set()
    for start in range(0, len(rec_coords), chunk_size):
        block = rec_coords[start:start + chunk_size]
        block_sq = np.einsum('ij,ij->i', block, block)
        dist_sq = block_sq[:, None] + lig_sq[None, :] - 2.0 * (block @ lig_coords.T)
        rows, cols = np.nonzero(dist_sq <= cutoff_sq)
        if len(rows):
            rec_hits.update(rec_owner[start + rows].tolist())
            lig_hits.update(lig_owner[cols].tolist())

    return sorted(rec_hits), sorted(lig_hits)


# ==============================================================================
# Random mutagenesis
# ==============================================================================

def format_mutation(residue, mut_letter):
    """Build a `{chain}_{wt}{position}{icode}{mut}` mutation string."""
    return (f"{residue['chain']}_{residue['res_letter']}"
            f"{residue['res_seq']}{residue['icode']}{mut_letter}")


def sample_mutation_sets(residues, rec_iface, lig_iface, num_mut_on_each,
                         n_variants, rng, max_attempts_factor=20):
    """
    Draw `n_variants` distinct mutation sets, each with `num_mut_on_each` random
    mutations on the receptor interface and the same number on the ligand interface.

    Returns:
        List of lists of mutation strings.
    """
    if len(rec_iface) < num_mut_on_each or len(lig_iface) < num_mut_on_each:
        raise NoInterfaceError(
            f"Insufficient interface residues - receptor: {len(rec_iface)}, "
            f"ligand: {len(lig_iface)}, required: {num_mut_on_each} per side"
        )

    variants, seen = [], set()
    attempts, max_attempts = 0, max(n_variants * max_attempts_factor, 50)

    while len(variants) < n_variants and attempts < max_attempts:
        attempts += 1
        mutations = []
        for side_pool in (rec_iface, lig_iface):
            chosen = rng.choice(len(side_pool), size=num_mut_on_each, replace=False)
            for pos in sorted(chosen):
                res = residues[side_pool[pos]]
                options = [aa for aa in ALL_AA_1 if aa != res['res_letter']]
                mutations.append(format_mutation(res, str(rng.choice(options))))
        key = frozenset(mutations)
        if key in seen:
            continue
        seen.add(key)
        variants.append(mutations)

    if len(variants) < n_variants:
        print(f"Warning: only {len(variants)}/{n_variants} distinct mutation sets could "
              f"be sampled from the available interface residues.", file=sys.stderr)
    return variants


# ==============================================================================
# FoldX
# ==============================================================================

_MUT_RE = re.compile(r'^([A-Za-z0-9])_([A-Z])(-?\d+[A-Za-z]?)([A-Z])$')


def to_foldx_mutation(mutation):
    """Convert `A_Y87R` into FoldX's `YA87R` notation."""
    m = _MUT_RE.match(mutation.strip())
    if not m:
        raise ValueError(f"Cannot parse mutation string: '{mutation}'")
    chain, wt, position, mut = m.groups()
    return f"{wt}{chain}{position}{mut}"


def write_individual_list(variants, output_file):
    """Write the FoldX individual_list.txt: one ';'-terminated line per variant."""
    with open(output_file, 'w') as fh:
        for mutations in variants:
            fh.write(','.join(to_foldx_mutation(m) for m in mutations) + ';\n')


def run_foldx(foldx_executable, pdb_name, individual_list, working_dir):
    """Run `FoldX --command=BuildModel` inside `working_dir`."""
    cmd = [
        foldx_executable,
        '--command=BuildModel',
        f'--pdb={pdb_name}',
        f'--mutant-file={os.path.abspath(individual_list)}',
        '--numberOfRuns=1',
    ]
    result = subprocess.run(cmd, cwd=working_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FoldX failed (exit {result.returncode}).\n"
            f"command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def collect_mutant_pdbs(variants, pdb_stem, working_dir, max_name_len=200):
    """
    Rename FoldX's `<stem>_<i>.pdb` outputs to `<stem>_<mutations>.pdb` and drop the
    `WT_<stem>_<i>.pdb` companions.

    Returns:
        List of output paths (None where FoldX produced no file).
    """
    produced = []
    for idx, mutations in enumerate(variants, start=1):
        foldx_out = os.path.join(working_dir, f"{pdb_stem}_{idx}.pdb")
        wt_out = os.path.join(working_dir, f"WT_{pdb_stem}_{idx}.pdb")
        if os.path.exists(wt_out):
            os.remove(wt_out)

        if not os.path.exists(foldx_out):
            print(f"Warning: expected FoldX output {foldx_out} not found.", file=sys.stderr)
            produced.append(None)
            continue

        new_name = f"{pdb_stem}_{'-'.join(mutations)}.pdb"
        if len(new_name) > max_name_len:
            new_name = f"{pdb_stem}_mut{idx}.pdb"
        new_path = os.path.join(working_dir, new_name)
        os.replace(foldx_out, new_path)
        produced.append(new_path)
    return produced


# ==============================================================================
# Driver
# ==============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Randomly mutate binding-interface residues of a complex and '
                    'build the mutant structures with FoldX.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--pdb', required=True, help='Path to the wild-type PDB file.')
    parser.add_argument('--receptor', required=True,
                        help="Receptor chain ID(s), comma separated, e.g. 'H,L'.")
    parser.add_argument('--ligand', required=True,
                        help="Ligand chain ID(s), comma separated, e.g. 'A'.")
    parser.add_argument('--num-mut-on-each', type=int, default=5,
                        help='Number of mutations placed on each side of the interface.')
    parser.add_argument('--n-variants', type=int, default=1,
                        help='Number of independent mutant structures to generate.')
    parser.add_argument('--dist-cutoff', type=float, default=8.0,
                        help='Heavy-atom distance cutoff (A) defining the interface.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--out-dir', default='.',
                        help='Directory where FoldX runs and the mutant PDBs are written.')
    parser.add_argument('--foldx', default='foldx',
                        help='FoldX executable (name on PATH or explicit path).')
    parser.add_argument('--max-residues', type=int, default=2000,
                        help='Reject complexes larger than this many residues (0 disables).')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.num_mut_on_each < 1:
        raise ValueError('--num-mut-on-each must be >= 1')
    if args.n_variants < 1:
        raise ValueError('--n-variants must be >= 1')

    pdb_path = os.path.abspath(args.pdb)
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    foldx_executable = shutil.which(args.foldx) or (
        os.path.abspath(args.foldx) if os.path.isfile(args.foldx) else None
    )
    if foldx_executable is None:
        raise FileNotFoundError(
            f"FoldX executable '{args.foldx}' not found on PATH or as a file. "
            f"Pass an explicit path with --foldx."
        )

    receptor_chains = parse_chain_ids(args.receptor)
    ligand_chains = parse_chain_ids(args.ligand)

    residues = parse_complex(pdb_path, receptor_chains, ligand_chains)
    if args.max_residues and len(residues) > args.max_residues:
        raise OversizedError(
            f"{os.path.basename(pdb_path)} has {len(residues)} residues "
            f"(max allowed: {args.max_residues})"
        )
    print(f"Parsed {len(residues)} residues from {os.path.basename(pdb_path)}")

    rec_iface, lig_iface = find_interface_residues(residues, args.dist_cutoff)
    if not rec_iface or not lig_iface:
        raise NoInterfaceError(
            f"No interface residues found for {os.path.basename(pdb_path)} - receptor and "
            f"ligand are not interacting within {args.dist_cutoff}A threshold"
        )
    print(f"Interface residues - receptor: {len(rec_iface)}, ligand: {len(lig_iface)}")

    rng = np.random.default_rng(args.seed)
    variants = sample_mutation_sets(residues, rec_iface, lig_iface,
                                    args.num_mut_on_each, args.n_variants, rng)
    print(f"Sampled {len(variants)} mutation set(s):")
    for i, mutations in enumerate(variants, start=1):
        print(f"  [{i}] {','.join(mutations)}")

    pdb_stem = os.path.splitext(os.path.basename(pdb_path))[0].lower()
    working_dir = os.path.abspath(args.out_dir)
    os.makedirs(working_dir, exist_ok=True)

    local_pdb = f"{pdb_stem}.pdb"
    local_pdb_path = os.path.join(working_dir, local_pdb)
    if not os.path.exists(local_pdb_path) or not os.path.samefile(pdb_path, local_pdb_path):
        shutil.copy(pdb_path, local_pdb_path)

    individual_list = os.path.join(working_dir, 'individual_list.txt')
    write_individual_list(variants, individual_list)

    print(f"Running FoldX BuildModel in {working_dir} ...")
    run_foldx(foldx_executable, local_pdb, individual_list, working_dir)

    mutant_paths = collect_mutant_pdbs(variants, pdb_stem, working_dir)

    n_ok = 0
    for idx, (mutations, path) in enumerate(zip(variants, mutant_paths), start=1):
        if path:
            n_ok += 1
            print(f"  [{idx}] {','.join(mutations)} -> {path}")
    print(f"Generated {n_ok}/{len(variants)} mutant structure(s) in {working_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
