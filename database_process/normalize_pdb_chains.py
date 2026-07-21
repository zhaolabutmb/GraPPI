"""
normalize_pdb_chains.py

Reads a CSV with columns:
    PDB, Receptor Chains, Ligand Chains, paths, affinity

For each row, normalizes chain IDs in the PDB file at `paths` so that:
    Receptor Chains  →  single chain A
    Ligand Chains    →  single chain B

Behavior:
  - Receptor=['A'] and Ligand=['B']  →  copy source file to DEST_DIR unchanged.
  - Anything else                    →  load PDB, extract/merge the listed
                                        chains into new chains A and B, write
                                        to DEST_DIR.  Extra chains in the file
                                        are silently ignored.
                                        Residues are renumbered only when
                                        multiple source chains are merged into
                                        one output chain (to avoid ID clashes).

Skipped with a warning if:
  - `paths` entry is missing / NaN
  - Source file does not exist
  - Receptor/Ligand chain info is missing in the row
"""

import copy
import os
import shutil
import warnings

import pandas as pd
from Bio import BiopythonWarning
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure

warnings.filterwarnings("ignore", category=BiopythonWarning)

# ── Configuration ───────────────────────────────────────────────────────────
CSV_PATH = './saved_tables/test_rand_mut_table.csv' #'test_preppi_df.csv' 'test_posi_df.csv' 'test_rand_mut_table.csv' AbEpiTope_table.csv
DEST_DIR = '/data/zhisong/GNN_Dove_data/rand_mut_pdb' #'preppi_pdb' 'posi_pdb' 'rand_mut_pdb' Abag_pdb
if_mut = True
# ────────────────────────────────────────────────────────────────────────────


def parse_chain_ids(chain_str) -> list[str]:
    """Parse 'H, L' or 'A' or 'A, G, H' into a list of stripped IDs."""
    if pd.isna(chain_str):
        return []
    return [c.strip() for c in str(chain_str).split(',') if c.strip()]


def build_ab_structure(
    orig_structure: Structure,
    receptor_ids: list[str],
    ligand_ids: list[str],
    renumber: bool,
) -> Structure:
    """
    Build a new Structure with exactly two chains (A and B) by extracting
    residues from the specified source chains.

    renumber=True  – residues in each output chain are numbered 1, 2, 3, …
                     (required when merging multiple chains to avoid ID clashes)
    renumber=False – original residue IDs are preserved
    """
    orig_model = orig_structure[0]

    new_struct = Structure(orig_structure.id)
    new_model  = Model(0)
    new_struct.add(new_model)

    for new_chain_id, source_ids in (('A', receptor_ids), ('B', ligand_ids)):
        new_chain = Chain(new_chain_id)
        new_model.add(new_chain)
        res_num = 1
        for cid in source_ids:
            if cid not in orig_model:
                print(f"  [WARN] chain {cid!r} not found in {orig_structure.id}, skipping")
                continue
            for residue in orig_model[cid].get_residues():
                if residue.id[0] != ' ':   # skip HETATM (waters, ions, ligands)
                    continue
                new_res = copy.deepcopy(residue)
                new_res.detach_parent()
                if renumber:
                    het, _, _ = new_res.id
                    new_res.id = (het, res_num, ' ')
                new_chain.add(new_res)
                res_num += 1

    return new_struct


def process_pdb(
    pdb_path: str,
    dest_path: str,
    receptor_ids: list[str],
    ligand_ids: list[str],
) -> str:
    """
    Normalize chain IDs for one PDB file and write to dest_path.

    Decision is made from the TABLE values only (receptor_ids / ligand_ids):
      - Receptor=['A'], Ligand=['B']  → copy as-is (already correct)
      - Anything else                 → load PDB, extract the listed chains
                                        into new chains A and B, write output.
                                        Extra chains in the PDB are ignored.
                                        Residues are renumbered only when
                                        multiple source chains are merged into
                                        one output chain (to avoid ID clashes).

    Returns one of: 'copied', 'renamed', 'merged'.
    """
    # ── Fast path: table already says A / B ──────────────────────────────
    if receptor_ids == ['A'] and ligand_ids == ['B']:
        shutil.copy2(pdb_path, dest_path)
        return 'copied'

    # ── Need to transform: load PDB and build a fresh two-chain structure ─
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('s', pdb_path)

    renumber = len(receptor_ids) > 1 or len(ligand_ids) > 1
    new_struct = build_ab_structure(structure, receptor_ids, ligand_ids, renumber)

    io = PDBIO()
    io.set_structure(new_struct)
    io.save(dest_path)

    return 'merged' if renumber else 'renamed'


def main() -> None:
    table = pd.read_csv(CSV_PATH)
    os.makedirs(DEST_DIR, exist_ok=True)

    # Resolve paths in the CSV relative to the script's directory so the script
    # works regardless of the current working directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    stats = {'copied': 0, 'renamed': 0, 'merged': 0, 'skipped': 0}

    for _, row in table.iterrows():
        pdb_id   = str(row['PDB']).lower()
        src_raw  = row.get('paths', None)

        if pd.isna(src_raw) or not src_raw:
            print(f"[SKIP]  {pdb_id}: missing 'paths' value")
            stats['skipped'] += 1
            continue

        # Resolve source path (handles both absolute and relative paths)
        if if_mut:
            src_path = '../../GraPPI_data/'+ src_raw
        else:
            src_path = (
                src_raw if os.path.isabs(src_raw)
                else os.path.normpath(os.path.join(script_dir, src_raw))
            )

        if not os.path.exists(src_path):
            print(f"[SKIP]  {pdb_id}: source file not found — {src_path}")
            stats['skipped'] += 1
            continue

        receptor_ids = parse_chain_ids(row['Receptor Chains'])
        ligand_ids   = parse_chain_ids(row['Ligand Chains'])

        if not receptor_ids or not ligand_ids:
            print(f"[SKIP]  {pdb_id}: missing Receptor/Ligand chain info")
            stats['skipped'] += 1
            continue
        
        fname     = os.path.basename(src_path)
        dest_path = os.path.join(DEST_DIR, fname)

        try:
            result = process_pdb(src_path, dest_path, receptor_ids, ligand_ids)
            stats[result] += 1
            print(f"[{result.upper():<7}] {pdb_id}  ({fname})")
        except Exception as exc:
            print(f"[ERROR] {pdb_id}: {exc}")
            stats['skipped'] += 1

    total = sum(stats.values())
    print(
        f"\nDone — {total} rows processed.\n"
        f"  Copied : {stats['copied']}\n"
        f"  Renamed: {stats['renamed']}\n"
        f"  Merged : {stats['merged']}\n"
        f"  Skipped: {stats['skipped']}"
    )


if __name__ == '__main__':
    main()
