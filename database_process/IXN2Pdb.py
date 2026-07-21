import copy
import io
import os
import requests
import numpy as np
from scipy.spatial import cKDTree
from Bio import PDB
from Bio.PDB import Structure, Model
import subprocess
import argparse
import pandas as pd
from tqdm import tqdm

# Path to USALIGN binary (adjust if needed)
'''
Usage:
python IXN2Pdb.py --IXN_csv_file ../../GraPPI_data/ppi_axin_negative_df.csv --base_dir path_to_PrePPI_human_db --aligned_dir ../../GraPPI_data/PDB/preppi_aligned/
'''
USALIGN_BIN = "path_to_USALIGN"

# Standard 3-letter to 1-letter amino acid mapping (used to filter non-standard residues)
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O',
}


def download_pdb(pdb_id):
    """Download PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to download {pdb_id} from PDB")


def save_selected_chains(pdb_text, pdb_id, chains, output_dir="../../GraPPI_data/preppi_ref_pdbs"):
    """Save only specified chains from a PDB file."""
    os.makedirs(output_dir, exist_ok=True)
    parser = PDB.PDBParser(QUIET=True, PERMISSIVE=True)  # allow duplicate chains
    handle = io.StringIO(pdb_text)
    structure = parser.get_structure(pdb_id, handle)

    io_obj = PDB.PDBIO()
    io_obj.set_structure(structure)

    class ChainSelect(PDB.Select):
        def accept_chain(self, chain):
            return chain.id in chains

    filename = os.path.join(output_dir, f"{pdb_id}_{''.join(chains)}.pdb")
    io_obj.save(filename, ChainSelect())
    #print(f"Saved {filename}")
    return filename


def count_residue_contacts(chain1, chain2, distance_cutoff=6.5):
    """
    Count the number of inter-chain residue contacts between two chains.

    A residue contact is defined as: at least one pair of heavy atoms
    (non-hydrogen) from two residues (one from each chain) within
    distance_cutoff Angstroms.

    Uses KD-tree for efficient spatial querying.

    Args:
        chain1: Bio.PDB Chain object
        chain2: Bio.PDB Chain object
        distance_cutoff: distance threshold in Angstroms (default 6.5)

    Returns:
        int: number of residue-residue contacts
    """
    chain1_residues = []
    for res_idx, residue in enumerate(chain1.get_residues()):
        hetfield = residue.get_id()[0]
        resname = residue.get_resname().strip()
        if hetfield == " " or resname in THREE_TO_ONE:
            atoms_coords = []
            for atom in residue.get_atoms():
                if atom.element == "H" or atom.get_name().startswith("H"):
                    continue
                atoms_coords.append(atom.get_vector().get_array())
            if atoms_coords:
                chain1_residues.append(atoms_coords)

    chain2_residues = []
    for res_idx, residue in enumerate(chain2.get_residues()):
        hetfield = residue.get_id()[0]
        resname = residue.get_resname().strip()
        if hetfield == " " or resname in THREE_TO_ONE:
            atoms_coords = []
            for atom in residue.get_atoms():
                if atom.element == "H" or atom.get_name().startswith("H"):
                    continue
                atoms_coords.append(atom.get_vector().get_array())
            if atoms_coords:
                chain2_residues.append(atoms_coords)

    if not chain1_residues or not chain2_residues:
        return 0

    all_chain2_coords = []
    chain2_atom_to_residue = []
    for ridx, atom_coords in enumerate(chain2_residues):
        for coord in atom_coords:
            all_chain2_coords.append(coord)
            chain2_atom_to_residue.append(ridx)

    all_chain2_coords = np.array(all_chain2_coords)
    chain2_atom_to_residue = np.array(chain2_atom_to_residue)

    tree = cKDTree(all_chain2_coords)

    contacted_residue_pairs = set()
    for ridx1, atom_coords in enumerate(chain1_residues):
        for coord in atom_coords:
            neighbors = tree.query_ball_point(coord, distance_cutoff)
            if neighbors:
                for neighbor_idx in neighbors:
                    ridx2 = chain2_atom_to_residue[neighbor_idx]
                    contacted_residue_pairs.add((ridx1, ridx2))

    return len(contacted_residue_pairs)


def load_first_chain(pdb_path):
    """Load a PDB file and return the first chain as a Bio.PDB Chain object."""
    parser = PDB.PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure("tmp", pdb_path)
    for model in structure:
        for chain in model:
            return chain
    return None


def count_standard_residues(chain):
    """Count the number of standard amino acid residues in a chain."""
    count = 0
    for residue in chain.get_residues():
        hetfield = residue.get_id()[0]
        resname = residue.get_resname().strip()
        if hetfield == " " or resname in THREE_TO_ONE:
            count += 1
    return count


def save_combined_structure(chain1, chain2, out_path, chain_id1="A", chain_id2="B"):
    """
    Combine two Bio.PDB Chain objects into a single PDB file with distinct chain IDs.

    Args:
        chain1: Bio.PDB Chain object (saved as chain_id1)
        chain2: Bio.PDB Chain object (saved as chain_id2)
        out_path: output file path
        chain_id1: chain ID to assign to chain1 (default 'A')
        chain_id2: chain ID to assign to chain2 (default 'B')
    """
    new_struct = Structure.Structure("combined")
    new_model = Model.Model(0)
    new_struct.add(new_model)

    c1 = copy.deepcopy(chain1)
    c1.id = chain_id1
    c2 = copy.deepcopy(chain2)
    c2.id = chain_id2

    new_model.add(c1)
    new_model.add(c2)

    io_obj = PDB.PDBIO()
    io_obj.set_structure(new_struct)
    io_obj.save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--IXN_csv_file", required=True, help="Path to ixn csv file")
    parser.add_argument("--base_dir", required=True, help="Base directory for domain location")
    parser.add_argument("--aligned_dir", required=True, help="Aligned dir for aligned structure")
    parser.add_argument("--min_contacts", type=int, default=5,
                        help="Minimum residue contacts to keep a pair (default: 5)")
    parser.add_argument("--max_clashes", type=int, default=5,
                        help="Maximum allowed clashing residue pairs (heavy atom dist < 2A) before excluding (default: 5)")
    parser.add_argument("--max_complex_size", type=int, default=1600,
                        help="Maximum total residues in the complex (default: 1600)")
    parser.add_argument("--max_chain_size", type=int, default=1300,
                        help="Maximum residues in a single chain (default: 1300)")
    parser.add_argument("--target_count", type=int, default=15000,
                        help="Stop after collecting this many passed PDBs (default: 15000)")
    parser.add_argument("--n_splits", type=int, default=50,
                        help="Number of splits to divide the shuffled dataframe into (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    args = parser.parse_args()

    print("IXN_csv_file:", args.IXN_csv_file)
    print("base_dir:", args.base_dir)
    print("aligned_dir:", args.aligned_dir)
    print("min_contacts:", args.min_contacts)
    print("max_clashes:", args.max_clashes)
    print("max_complex_size:", args.max_complex_size)
    print("max_chain_size:", args.max_chain_size)
    print("target_count:", args.target_count)
    print("n_splits:", args.n_splits)
    print("seed:", args.seed)

    IXN_csv_file = args.IXN_csv_file
    base_dir = args.base_dir
    aligned_dir = args.aligned_dir
    min_contacts = args.min_contacts
    max_clashes = args.max_clashes
    max_complex_size = args.max_complex_size
    max_chain_size = args.max_chain_size
    target_count = args.target_count
    n_splits = args.n_splits

    os.makedirs(aligned_dir, exist_ok=True)

    # Subdirectories for USALIGN intermediate outputs and logs
    aligned_structures_dir = os.path.join(aligned_dir, "aligned_structures")
    logs_dir = os.path.join(aligned_dir, "log_files")
    os.makedirs(aligned_structures_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    df = pd.read_csv(IXN_csv_file, keep_default_na=False, na_values=[])
    print(f"Loaded {len(df)} rows from CSV")

    # Shuffle the dataframe (originally ranked by score)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print(f"Shuffled dataframe with seed={args.seed}")

    # Split into n_splits chunks
    splits = np.array_split(df, n_splits)
    print(f"Split into {len(splits)} chunks of ~{len(splits[0])} rows each")

    n_total = 0
    n_passed = 0
    n_skipped_missing = 0
    n_skipped_contacts = 0
    n_skipped_clashes = 0
    n_skipped_size = 0

    skip_log_path = os.path.join(aligned_dir, "skipped.log")
    skip_log = open(skip_log_path, "a")

    # Load previously processed pairs for resume support
    processed_log_path = os.path.join(aligned_dir, "processed.log")
    processed_pairs = set()
    if os.path.exists(processed_log_path):
        with open(processed_log_path, "r") as f:
            for line in f:
                pair_id = line.strip()
                if pair_id:
                    processed_pairs.add(pair_id)
        # Count how many of those were passed (have output PDBs)
        for pair_id in processed_pairs:
            if os.path.exists(os.path.join(aligned_dir, f"{pair_id}.pdb")):
                n_passed += 1
        print(f"Resuming: {len(processed_pairs)} pairs already processed, {n_passed} passed")
    processed_log = open(processed_log_path, "a")

    reached_target = False
    for split_idx, split_df in enumerate(splits):
        print(f"\n=== Processing split {split_idx + 1}/{n_splits} ({len(split_df)} rows) | "
              f"Passed so far: {n_passed}/{target_count} ===")

        for index, row in tqdm(split_df.iterrows(), total=len(split_df),
                               desc=f"Split {split_idx + 1}/{n_splits}"):
            pdb_id = row['pdb_id']
            chain_order = row['chain_order']
            dom1 = 'M'+row['dom1']
            dom2 = 'M'+row['dom2']
            pair_id = f"{dom1}_{dom2}"

            if pair_id in processed_pairs:
                continue

            def mark_processed():
                processed_pairs.add(pair_id)
                processed_log.write(f"{pair_id}\n")
                processed_log.flush()

            try:
                pdb_text = download_pdb(pdb_id)
            except Exception as e:
                skip_log.write(f"SKIP_DOWNLOAD {dom1}_{dom2}: failed to download PDB {pdb_id} - {e}\n")
                n_skipped_missing += 1
                mark_processed()
                continue
            ref_file = save_selected_chains(pdb_text, pdb_id, list(chain_order),
                                            output_dir="../../GraPPI_data/preppi_ref_pdbs")

            # Domain files are inside subdirectories named after domain IDs
            dom1_file = os.path.join(base_dir, dom1, "Models", f"{dom1}_m.pdb")
            dom2_file = os.path.join(base_dir, dom2, "Models", f"{dom2}_m.pdb")

            # USALIGN output prefixes inside aligned_structures/
            out1_prefix = os.path.join(aligned_structures_dir, f"{dom1}_aligned")
            out2_prefix = os.path.join(aligned_structures_dir, f"{dom2}_aligned")

            if not (os.path.exists(dom1_file) and os.path.exists(dom2_file) and len(chain_order) >= 2):
                if not os.path.exists(dom1_file):
                    skip_log.write(f"SKIP_MISSING {dom1}_{dom2}: domain file {dom1_file} not found\n")
                elif not os.path.exists(dom2_file):
                    skip_log.write(f"SKIP_MISSING {dom1}_{dom2}: domain file {dom2_file} not found\n")
                else:
                    skip_log.write(f"SKIP_MISSING {dom1}_{dom2}: invalid chain order '{chain_order}'\n")
                n_skipped_missing += 1
                mark_processed()
                continue

            n_total += 1

            # Run USALIGN
            cmd1 = [USALIGN_BIN, dom1_file, ref_file, "-chain2", chain_order[0],
                    "-o", out1_prefix, "-outfmt", "2"]
            cmd2 = [USALIGN_BIN, dom2_file, ref_file, "-chain2", chain_order[1],
                    "-o", out2_prefix, "-outfmt", "2"]

            log1 = os.path.join(logs_dir, f"{dom1}_usalign.log")
            log2 = os.path.join(logs_dir, f"{dom2}_usalign.log")
            try:
                with open(log1, "w") as L1, open(log2, "w") as L2:
                    subprocess.run(cmd1, check=True, stdout=L1, stderr=L1)
                    subprocess.run(cmd2, check=True, stdout=L2, stderr=L2)
            except subprocess.CalledProcessError as e:
                skip_log.write(f"SKIP_USALIGN {dom1}_{dom2}: USALIGN failed - {e}\n")
                n_skipped_missing += 1
                mark_processed()
                continue

            # Load both aligned structures
            aligned1_pdb = out1_prefix + ".pdb"
            aligned2_pdb = out2_prefix + ".pdb"

            if not os.path.exists(aligned1_pdb) or not os.path.exists(aligned2_pdb):
                skip_log.write(f"SKIP_USALIGN {dom1}_{dom2}: aligned PDB(s) not produced by USALIGN\n")
                n_skipped_missing += 1
                mark_processed()
                continue

            chain1 = load_first_chain(aligned1_pdb)
            chain2 = load_first_chain(aligned2_pdb)

            if chain1 is None or chain2 is None:
                skip_log.write(f"SKIP_MISSING {dom1}_{dom2}: could not load aligned PDB(s)\n")
                n_skipped_missing += 1
                mark_processed()
                continue

            # Size filter: check single chain and complex size
            n_res1 = count_standard_residues(chain1)
            n_res2 = count_standard_residues(chain2)
            n_complex = n_res1 + n_res2
            if max(n_res1, n_res2) > max_chain_size or n_complex > max_complex_size:
                skip_log.write(f"SKIP_SIZE {dom1}_{dom2}: chain1={n_res1}, chain2={n_res2}, "
                               f"complex={n_complex} (max_chain={max_chain_size}, max_complex={max_complex_size})\n")
                n_skipped_size += 1
                mark_processed()
                continue

            # Contact filter
            n_contacts = count_residue_contacts(chain1, chain2)
            if n_contacts < min_contacts:
                skip_log.write(f"SKIP_CONTACTS {dom1}_{dom2}: only {n_contacts} contacts (< {min_contacts})\n")
                n_skipped_contacts += 1
                mark_processed()
                continue

            # Clash filter
            n_clashes = count_residue_contacts(chain1, chain2, distance_cutoff=2.0)
            if n_clashes > max_clashes:
                skip_log.write(f"SKIP_CLASHES {dom1}_{dom2}: {n_clashes} clashing residue pairs (> {max_clashes})\n")
                n_skipped_clashes += 1
                mark_processed()
                continue

            # All filters passed — save
            combined_out = os.path.join(aligned_dir, f"{dom1}_{dom2}.pdb")
            save_combined_structure(chain1, chain2, combined_out, chain_id1="A", chain_id2="B")
            n_passed += 1
            mark_processed()

            if n_passed >= target_count:
                print(f"\nReached target count of {target_count} passed PDBs. Stopping.")
                reached_target = True
                break

        if reached_target:
            break

        print(f"Split {split_idx + 1} done. Passed so far: {n_passed}/{target_count}")

    skip_log.close()
    processed_log.close()

    print(f"\nDone. Total processed: {n_total} | Passed: {n_passed} | "
          f"Skipped (size): {n_skipped_size} | Skipped (contacts): {n_skipped_contacts} | "
          f"Skipped (clashes): {n_skipped_clashes} | Skipped (missing): {n_skipped_missing}")
    print(f"Skip/error details written to: {skip_log_path}")

    if not reached_target:
        if n_passed >= 10000:
            print(f"\nWARNING: Did not reach target ({target_count}), but collected {n_passed} PDBs (>= 10000). "
                  f"Proceeding with available data.")
        else:
            print(f"\nERROR: Only collected {n_passed} PDBs (< 10000) after processing all {n_splits} splits. "
                  f"Consider relaxing filter cutoffs or using more data.")