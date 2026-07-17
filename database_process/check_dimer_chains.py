#!/usr/bin/env python3
"""
Check and correct dimer chain assignments in new_dimer_db_table.csv.

For each PDB entry:
1. Check if the two selected chains have sequence identity < 0.95 (hetero-dimer check)
2. Check if they have >= 5 inter-chain residue contacts (binding check)
   A residue contact = at least one heavy atom pair from two residues < 6.5 Angstrom

For PDBs with exactly 2 chains:
  - If both checks pass, keep as-is
  - Otherwise, add to questioning list

For PDBs with >2 chains:
  - Try all chain combinations, pick the pair with different sequences AND max contacts
  - If no valid pair found, add to questioning list

Outputs:
  - corrected_new_dimer_db_table.csv
  - questioning_pdbs.csv
"""

import os
import sys
import csv
import warnings
import logging
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from scipy.spatial import cKDTree

# Suppress Biopython warnings about discontinuous chains, etc.
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("check_dimer_chains.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# --- Three-letter to one-letter amino acid code mapping ---
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # common modified residues mapped to standard
    "MSE": "M", "HYP": "P", "SEP": "S", "TPO": "T", "CSO": "C",
    "PTR": "Y", "MLY": "K", "CME": "C", "CSD": "C",
}


def extract_chain_sequence(chain):
    """
    Extract the amino acid sequence (one-letter code) from a Bio.PDB Chain object.
    Only considers standard amino acid residues (excludes HETATM / water).
    """
    seq = []
    for residue in chain.get_residues():
        # Skip hetero atoms and water
        hetfield = residue.get_id()[0]
        if hetfield != " ":
            # Allow MSE (selenomethionine) which is flagged as HETATM
            resname = residue.get_resname().strip()
            if resname in THREE_TO_ONE:
                seq.append(THREE_TO_ONE[resname])
            continue
        resname = residue.get_resname().strip()
        if resname in THREE_TO_ONE:
            seq.append(THREE_TO_ONE[resname])
    return "".join(seq)


def compute_sequence_identity(seq1, seq2):
    """
    Compute pairwise sequence identity between two sequences using
    a simple global alignment approach (Needleman-Wunsch style via
    dynamic programming with identity scoring).

    Returns the fraction of identical positions relative to the
    length of the shorter alignment.

    For efficiency on this large dataset, we use a simpler approach:
    if sequences are very different in length (ratio < 0.5), return 0.0 early.
    Otherwise, do a banded alignment.
    """
    if not seq1 or not seq2:
        return 0.0

    len1, len2 = len(seq1), len(seq2)

    # Quick length ratio check - if very different lengths, not identical
    if min(len1, len2) / max(len1, len2) < 0.5:
        return 0.0

    # For very short sequences, use simple alignment
    # For efficiency, use a simplified approach:
    # Use the shorter sequence and align it against the longer one
    # with a simple DP matching

    # Simple Needleman-Wunsch with match=1, mismatch=0, gap=-0.5
    # We only need identity, so keep it simple
    match_score = 1
    mismatch_score = 0
    gap_penalty = -0.5

    # Create DP matrix
    dp = np.zeros((len1 + 1, len2 + 1), dtype=np.float32)
    for i in range(1, len1 + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
    for j in range(1, len2 + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                diag = dp[i - 1][j - 1] + match_score
            else:
                diag = dp[i - 1][j - 1] + mismatch_score
            up = dp[i - 1][j] + gap_penalty
            left = dp[i][j - 1] + gap_penalty
            dp[i][j] = max(diag, up, left)

    # Traceback to count matches
    i, j = len1, len2
    matches = 0
    aligned_len = 0
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            score_diag = dp[i - 1][j - 1] + match_score
        else:
            score_diag = dp[i - 1][j - 1] + mismatch_score

        if dp[i][j] == score_diag:
            if seq1[i - 1] == seq2[j - 1]:
                matches += 1
            aligned_len += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + gap_penalty:
            aligned_len += 1
            i -= 1
        else:
            aligned_len += 1
            j -= 1

    # Count remaining gaps
    aligned_len += i + j

    if aligned_len == 0:
        return 0.0

    return matches / aligned_len


def check_sequence_identity(chain1, chain2, threshold=0.95):
    """
    Check if two chains are hetero-dimeric (sequence identity < threshold).

    Args:
        chain1: Bio.PDB Chain object
        chain2: Bio.PDB Chain object
        threshold: sequence identity threshold (default 0.95)

    Returns:
        (is_hetero, identity): tuple of (bool, float)
            is_hetero = True if identity < threshold (chains are different)
            identity = computed sequence identity
    """
    seq1 = extract_chain_sequence(chain1)
    seq2 = extract_chain_sequence(chain2)

    if not seq1 or not seq2:
        # If we can't extract sequence, flag it
        return False, -1.0

    identity = compute_sequence_identity(seq1, seq2)
    is_hetero = identity < threshold
    return is_hetero, identity


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
    # Collect heavy atom coordinates for each residue in chain1
    chain1_residues = []
    chain1_coords = []
    chain1_residue_ids = []

    for res_idx, residue in enumerate(chain1.get_residues()):
        hetfield = residue.get_id()[0]
        resname = residue.get_resname().strip()
        # Only consider amino acid residues
        if hetfield == " " or resname in THREE_TO_ONE:
            atoms_coords = []
            for atom in residue.get_atoms():
                # Skip hydrogens
                if atom.element == "H" or atom.get_name().startswith("H"):
                    continue
                atoms_coords.append(atom.get_vector().get_array())
            if atoms_coords:
                chain1_residues.append(atoms_coords)
                chain1_residue_ids.append(res_idx)

    # Collect heavy atom coordinates for each residue in chain2
    chain2_residues = []
    chain2_coords = []
    chain2_residue_ids = []

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
                chain2_residue_ids.append(res_idx)

    if not chain1_residues or not chain2_residues:
        return 0

    # Build a single coordinate array for chain2 with residue index mapping
    all_chain2_coords = []
    chain2_atom_to_residue = []
    for ridx, atom_coords in enumerate(chain2_residues):
        for coord in atom_coords:
            all_chain2_coords.append(coord)
            chain2_atom_to_residue.append(ridx)

    all_chain2_coords = np.array(all_chain2_coords)
    chain2_atom_to_residue = np.array(chain2_atom_to_residue)

    # Build KD-tree for chain2 atoms
    tree = cKDTree(all_chain2_coords)

    # For each residue in chain1, query if any of its atoms are close to chain2 atoms
    contacted_residue_pairs = set()

    for ridx1, atom_coords in enumerate(chain1_residues):
        for coord in atom_coords:
            # Query all chain2 atoms within cutoff
            neighbors = tree.query_ball_point(coord, distance_cutoff)
            if neighbors:
                # Get unique chain2 residue indices that are in contact
                for neighbor_idx in neighbors:
                    ridx2 = chain2_atom_to_residue[neighbor_idx]
                    contacted_residue_pairs.add((ridx1, ridx2))

    # Count unique residue contacts (counting each pair once)
    # But the question asks for "at least 5 residue contact"
    # which means 5 residue pairs in contact
    return len(contacted_residue_pairs)


def check_binding(chain1, chain2, min_contacts=5, distance_cutoff=6.5):
    """
    Check if two chains bind to each other by counting inter-chain
    residue contacts.

    Args:
        chain1: Bio.PDB Chain object
        chain2: Bio.PDB Chain object
        min_contacts: minimum number of residue contacts required (default 5)
        distance_cutoff: distance threshold in Angstroms (default 6.5)

    Returns:
        (is_binding, n_contacts): tuple of (bool, int)
            is_binding = True if n_contacts >= min_contacts
            n_contacts = number of residue-residue contacts
    """
    n_contacts = count_residue_contacts(chain1, chain2, distance_cutoff)
    is_binding = n_contacts >= min_contacts
    return is_binding, n_contacts


def process_single_pdb(pdb_id, chain_1_id, chain_2_id, all_chains_str, pdb_path,
                       parser, seq_threshold=0.95, contact_min=5, contact_cutoff=6.5):
    """
    Process a single PDB entry.

    Returns:
        dict with keys:
            'status': 'ok' | 'corrected' | 'questioning' | 'error'
            'new_chain_1': corrected chain 1 (or original)
            'new_chain_2': corrected chain 2 (or original)
            'reason': explanation string
            'seq_identity': sequence identity of final pair
            'n_contacts': number of contacts of final pair
    """
    result = {
        "status": "ok",
        "new_chain_1": chain_1_id,
        "new_chain_2": chain_2_id,
        "reason": "",
        "seq_identity": -1.0,
        "n_contacts": -1,
    }

    # Resolve path
    if not os.path.isabs(pdb_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pdb_path = os.path.normpath(os.path.join(base_dir, pdb_path))

    if not os.path.exists(pdb_path):
        result["status"] = "error"
        result["reason"] = f"PDB file not found: {pdb_path}"
        return result

    # Parse PDB
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
    except Exception as e:
        result["status"] = "error"
        result["reason"] = f"Failed to parse PDB: {e}"
        return result

    model = structure[0]  # Use first model

    # Get available chain IDs
    available_chains = [ch.get_id() for ch in model.get_chains()]

    # Parse all_chains from CSV
    all_chains_csv = [c.strip() for c in all_chains_str.split(",")]
    num_chains = len(all_chains_csv)

    # Check that requested chains exist
    if chain_1_id not in available_chains or chain_2_id not in available_chains:
        result["status"] = "error"
        result["reason"] = (
            f"Chain(s) not found in structure. "
            f"Requested: {chain_1_id}, {chain_2_id}. "
            f"Available: {available_chains}"
        )
        return result

    if num_chains == 2:
        # --- Exactly 2 chains: validate current assignment ---
        ch1 = model[chain_1_id]
        ch2 = model[chain_2_id]

        is_hetero, identity = check_sequence_identity(ch1, ch2, seq_threshold)
        is_binding, n_contacts = check_binding(ch1, ch2, contact_min, contact_cutoff)

        result["seq_identity"] = identity
        result["n_contacts"] = n_contacts

        if is_hetero and is_binding:
            result["status"] = "ok"
            result["reason"] = f"2-chain OK: identity={identity:.3f}, contacts={n_contacts}"
        else:
            result["status"] = "questioning"
            reasons = []
            if not is_hetero:
                reasons.append(f"high_seq_identity={identity:.3f}")
            if not is_binding:
                reasons.append(f"low_contacts={n_contacts}")
            result["reason"] = f"2-chain FAIL: {'; '.join(reasons)}"

    else:
        # --- More than 2 chains: try all combinations ---
        # First check if the original pair is already valid
        ch1_orig = model[chain_1_id]
        ch2_orig = model[chain_2_id]

        is_hetero_orig, identity_orig = check_sequence_identity(
            ch1_orig, ch2_orig, seq_threshold
        )
        is_binding_orig, n_contacts_orig = check_binding(
            ch1_orig, ch2_orig, contact_min, contact_cutoff
        )

        if is_hetero_orig and is_binding_orig:
            # Original assignment is fine
            result["status"] = "ok"
            result["seq_identity"] = identity_orig
            result["n_contacts"] = n_contacts_orig
            result["reason"] = (
                f"multi-chain OK (original): identity={identity_orig:.3f}, "
                f"contacts={n_contacts_orig}"
            )
            return result

        # Try all chain combinations from available chains in the structure
        # Use the chains listed in all_chains that actually exist in structure
        valid_chains = [c for c in all_chains_csv if c in available_chains]

        best_pair = None
        best_contacts = -1
        best_identity = -1.0

        # Precompute chain sequences for efficiency
        chain_seqs = {}
        for cid in valid_chains:
            chain_seqs[cid] = extract_chain_sequence(model[cid])

        for c1, c2 in combinations(valid_chains, 2):
            seq1 = chain_seqs[c1]
            seq2 = chain_seqs[c2]

            if not seq1 or not seq2:
                continue

            identity = compute_sequence_identity(seq1, seq2)
            if identity >= seq_threshold:
                # Same sequence, skip (we want hetero-dimer)
                continue

            # Check contacts
            n_contacts = count_residue_contacts(model[c1], model[c2], contact_cutoff)

            if n_contacts >= contact_min and n_contacts > best_contacts:
                best_pair = (c1, c2)
                best_contacts = n_contacts
                best_identity = identity

        if best_pair is not None:
            if best_pair == (chain_1_id, chain_2_id):
                result["status"] = "ok"
                result["reason"] = (
                    f"multi-chain OK (confirmed by search): "
                    f"identity={best_identity:.3f}, contacts={best_contacts}"
                )
            else:
                result["status"] = "corrected"
                result["new_chain_1"] = best_pair[0]
                result["new_chain_2"] = best_pair[1]
                result["reason"] = (
                    f"multi-chain CORRECTED: {chain_1_id},{chain_2_id} -> "
                    f"{best_pair[0]},{best_pair[1]} "
                    f"(identity={best_identity:.3f}, contacts={best_contacts})"
                )
                # Also log original pair info
                if is_hetero_orig:
                    result["reason"] += (
                        f" [original: identity={identity_orig:.3f}, "
                        f"contacts={n_contacts_orig}, binding_fail]"
                    )
                else:
                    result["reason"] += (
                        f" [original: identity={identity_orig:.3f}, seq_too_similar]"
                    )
            result["seq_identity"] = best_identity
            result["n_contacts"] = best_contacts
        else:
            result["status"] = "questioning"
            result["seq_identity"] = identity_orig
            result["n_contacts"] = n_contacts_orig
            reasons = []
            if not is_hetero_orig:
                reasons.append(f"original_high_seq_identity={identity_orig:.3f}")
            if not is_binding_orig:
                reasons.append(f"original_low_contacts={n_contacts_orig}")
            reasons.append("no_valid_alternative_pair_found")
            result["reason"] = f"multi-chain FAIL: {'; '.join(reasons)}"

    return result


def main():
    argparser = argparse.ArgumentParser(
        description="Check and correct dimer chain assignments"
    )
    argparser.add_argument(
        "--input",
        default="saved_tables/new_dimer_db_table.csv",
        help="Input CSV file",
    )
    argparser.add_argument(
        "--output",
        default="saved_tables/corrected_new_dimer_db_table.csv",
        help="Output corrected CSV file",
    )
    argparser.add_argument(
        "--questioning",
        default="saved_tables/questioning_pdbs.csv",
        help="Output questioning PDBs CSV file",
    )
    argparser.add_argument(
        "--seq-threshold",
        type=float,
        default=0.95,
        help="Sequence identity threshold for hetero-dimer check (default: 0.95)",
    )
    argparser.add_argument(
        "--min-contacts",
        type=int,
        default=5,
        help="Minimum residue contacts for binding check (default: 5)",
    )
    argparser.add_argument(
        "--contact-cutoff",
        type=float,
        default=6.5,
        help="Distance cutoff in Angstroms for contact (default: 6.5)",
    )
    args = argparser.parse_args()

    # Read input CSV
    logger.info(f"Reading input file: {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Total entries: {len(df)}")

    parser = PDBParser(QUIET=True)

    # Results storage
    results = []
    questioning_list = []
    stats = {"ok": 0, "corrected": 0, "questioning": 0, "error": 0}

    for idx, row in df.iterrows():
        pdb_id = str(row["PDB"])
        chain_1 = str(row["chain_1"]).strip()
        chain_2 = str(row["chain_2"]).strip()
        all_chains = str(row["all_chains"]).strip()
        pdb_path = str(row["path"]).strip()

        if idx % 500 == 0:
            logger.info(
                f"Processing {idx + 1}/{len(df)} (PDB: {pdb_id}) | "
                f"Stats: ok={stats['ok']}, corrected={stats['corrected']}, "
                f"questioning={stats['questioning']}, error={stats['error']}"
            )

        result = process_single_pdb(
            pdb_id, chain_1, chain_2, all_chains, pdb_path, parser,
            args.seq_threshold, args.min_contacts, args.contact_cutoff,
        )

        stats[result["status"]] += 1

        # Store result
        results.append(result)

        # Build questioning entry if needed
        if result["status"] in ("questioning", "error"):
            questioning_list.append({
                "PDB": pdb_id,
                "original_chain_1": chain_1,
                "original_chain_2": chain_2,
                "all_chains": all_chains,
                "num_chains": row["num_chains"],
                "status": result["status"],
                "reason": result["reason"],
                "seq_identity": result["seq_identity"],
                "n_contacts": result["n_contacts"],
                "path": pdb_path,
            })

    # Build corrected DataFrame
    logger.info("Building corrected table...")
    corrected_df = df.copy()
    for idx, result in enumerate(results):
        if result["status"] == "corrected":
            corrected_df.at[idx, "chain_1"] = result["new_chain_1"]
            corrected_df.at[idx, "chain_2"] = result["new_chain_2"]
            corrected_df.at[idx, "chain_determination_method"] = "binding_check_corrected"

    # Remove questioning/error entries from corrected table
    remove_indices = [
        idx for idx, r in enumerate(results) if r["status"] in ("questioning", "error")
    ]
    corrected_df = corrected_df.drop(index=remove_indices).reset_index(drop=True)

    # Save outputs
    corrected_df.to_csv(args.output, index=False)
    logger.info(f"Saved corrected table: {args.output} ({len(corrected_df)} entries)")

    questioning_df = pd.DataFrame(questioning_list)
    questioning_df.to_csv(args.questioning, index=False)
    logger.info(
        f"Saved questioning PDBs: {args.questioning} ({len(questioning_df)} entries)"
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total processed : {len(df)}")
    logger.info(f"  OK (unchanged)  : {stats['ok']}")
    logger.info(f"  Corrected       : {stats['corrected']}")
    logger.info(f"  Questioning     : {stats['questioning']}")
    logger.info(f"  Error           : {stats['error']}")
    logger.info(f"  Final table size: {len(corrected_df)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
