"""
Download Dimer PDB Structures from the Protein Data Bank (RCSB PDB)

This script:
1. Queries the RCSB PDB Search API for dimer structures (hetero-dimers).
2. Downloads PDB files for each hit.
3. Parses each PDB to identify chain IDs.
4. For structures with >2 chains (crystal packing), determines the actual
   biological dimer chains using RCSB assembly info or BIOMT records.
5. Records all metadata (PDB ID, all chains, dimer chain IDs, homo/hetero)
   in a CSV table.

Usage:
    python download_dimer_pdbs.py [--max_results N] [--resume]

Output:
    - PDB files saved to: ../../PDB/new_dimer_db/
    - Metadata CSV saved to: ./saved_tables/new_dimer_db_table.csv
"""

import os
import sys
import time
import json
import gzip
import argparse
import logging
from io import StringIO
from collections import Counter

import requests
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.MMCIFParser import MMCIFParser

# =====================================================================
# PDB Select filter — keep only standard protein residues + water
# (avoids HETATM ligands with >3-char residue names that corrupt
# PDB column alignment during mmCIF → PDB conversion)
# =====================================================================

class ProteinSelect(Select):
    """Select only standard amino acid residues (and water).
    Skips HETATM ligands whose residue names may exceed 3 characters,
    which corrupts the fixed-width PDB format columns."""
    def accept_residue(self, residue):
        # residue.id[0]: ' ' for standard, 'W' for water, 'H_xxx' for HETATM
        hetflag = residue.id[0]
        return hetflag in (' ', 'W')


# =====================================================================
# Configuration
# =====================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PDB_SAVE_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'PDB', 'new_dimer_db')
TABLE_SAVE_DIR = os.path.join(SCRIPT_DIR, 'saved_tables')
TABLE_SAVE_PATH = os.path.join(TABLE_SAVE_DIR, 'new_dimer_db_table.csv')

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{}.pdb"
RCSB_DOWNLOAD_CIF_URL = "https://files.rcsb.org/download/{}.cif"
RCSB_ASSEMBLY_URL = "https://data.rcsb.org/rest/v1/core/assembly/{}/{}"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{}"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SCRIPT_DIR, 'download_dimer.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)


# =====================================================================
# RCSB PDB Search Query (Hetero-dimers only, with pagination)
# =====================================================================

PAGE_SIZE = 10000  # RCSB API max rows per request


def build_hetero_dimer_search_query(start=0, rows=10000):
    """
    Build RCSB PDB search query for hetero-dimer structures.
    Filters:
      - Oligomeric state: Hetero 2-mer
      - Experiment method: X-ray, cryo-EM, or NMR
      - Resolution: <= 4.0 Å (for X-ray/cryo-EM) OR NMR (no resolution)
      - Polymer entity count: >= 2 (at least 2 protein chains)
      - Protein polymer type
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_struct_symmetry.oligomeric_state",
                        "operator": "exact_match",
                        "negation": False,
                        "value": "Hetero 2-mer"
                    }
                },
                {
                    "type": "group",
                    "logical_operator": "or",
                    "nodes": [
                        {
                            "type": "group",
                            "logical_operator": "and",
                            "nodes": [
                                {
                                    "type": "terminal",
                                    "service": "text",
                                    "parameters": {
                                        "attribute": "exptl.method",
                                        "operator": "in",
                                        "negation": False,
                                        "value": [
                                            "X-RAY DIFFRACTION",
                                            "ELECTRON MICROSCOPY"
                                        ]
                                    }
                                },
                                {
                                    "type": "terminal",
                                    "service": "text",
                                    "parameters": {
                                        "attribute": "rcsb_entry_info.resolution_combined",
                                        "operator": "less_or_equal",
                                        "negation": False,
                                        "value": 4.0
                                    }
                                }
                            ]
                        },
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": "exptl.method",
                                "operator": "exact_match",
                                "negation": False,
                                "value": "SOLUTION NMR"
                            }
                        }
                    ]
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "greater_or_equal",
                        "negation": False,
                        "value": 2
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "negation": False,
                        "value": "Protein"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": start,
                "rows": rows
            },
            "results_content_type": ["experimental"],
            "sort": [
                {
                    "sort_by": "score",
                    "direction": "desc"
                }
            ]
        }
    }
    return query


def search_rcsb_page(query):
    """Execute a search query against the RCSB PDB Search API.
    Returns (results_list, total_count).
    """
    response = requests.post(RCSB_SEARCH_URL, json=query, timeout=120)
    if response.status_code == 200:
        data = response.json()
        total = data.get("total_count", 0)
        results = [r["identifier"] for r in data.get("result_set", [])]
        logger.info(f"Search returned {total} total results, fetched {len(results)}")
        return results, total
    else:
        logger.error(f"Search API error: {response.status_code} - {response.text[:500]}")
        return [], 0


def search_hetero_dimers():
    """
    Search for all hetero-dimers using pagination.
    Fetches PAGE_SIZE results per request until all are collected.
    """
    logger.info("Searching for hetero-dimers (with pagination)...")
    all_ids = []
    start = 0

    while True:
        query = build_hetero_dimer_search_query(start=start, rows=PAGE_SIZE)
        page_ids, total = search_rcsb_page(query)

        if not page_ids:
            break

        all_ids.extend(page_ids)
        logger.info(f"Pagination: fetched {len(all_ids)}/{total} hetero-dimer IDs")

        if len(all_ids) >= total or len(page_ids) < PAGE_SIZE:
            break

        start += PAGE_SIZE
        time.sleep(0.5)  # Rate limit between pages

    # Deduplicate (in case of overlap)
    all_ids = list(dict.fromkeys(all_ids))
    logger.info(f"Total unique hetero-dimer PDB IDs: {len(all_ids)}")

    pdb_type = {pid: "hetero" for pid in all_ids}
    return all_ids, pdb_type


# =====================================================================
# PDB Download
# =====================================================================

def download_pdb(pdb_id, save_dir):
    """
    Download a PDB file from RCSB. If PDB format is unavailable (404),
    fall back to downloading mmCIF and converting to PDB format.
    Returns (saved_file_path, converted_from_cif) or (None, False).
    """
    save_path = os.path.join(save_dir, f"{pdb_id}.pdb")
    if os.path.exists(save_path):
        return save_path, False  # Already downloaded

    # --- Try legacy PDB format first ---
    url = RCSB_DOWNLOAD_URL.format(pdb_id)
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            with open(save_path, 'w') as f:
                f.write(response.text)
            return save_path, False
        elif response.status_code != 404:
            # Non-404 error (server issue, etc.) — don't try CIF fallback
            logger.warning(f"Failed to download {pdb_id}.pdb: HTTP {response.status_code}")
            return None, False
    except requests.RequestException as e:
        logger.warning(f"Network error downloading {pdb_id}.pdb: {e}")
        return None, False

    # --- PDB format not available (404), try mmCIF fallback ---
    logger.info(f"{pdb_id}: PDB format unavailable, trying mmCIF...")
    cif_url = RCSB_DOWNLOAD_CIF_URL.format(pdb_id)
    cif_tmp_path = os.path.join(save_dir, f"{pdb_id}.cif")
    try:
        response = requests.get(cif_url, timeout=60)
        if response.status_code != 200:
            logger.warning(f"Failed to download {pdb_id}.cif: HTTP {response.status_code}")
            return None, False

        # Save CIF temporarily
        with open(cif_tmp_path, 'w') as f:
            f.write(response.text)

        # Parse mmCIF and convert to PDB format
        cif_parser = MMCIFParser(QUIET=True)
        structure = cif_parser.get_structure(pdb_id, cif_tmp_path)

        # Remap multi-character chain IDs to single characters for PDB format
        chain_id_map = {}
        available_ids = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
        used_ids = set()
        needs_remap = False
        for model in structure:
            for chain in model:
                old_id = chain.id
                if len(old_id) > 1:
                    needs_remap = True
                if old_id not in chain_id_map:
                    if len(old_id) == 1 and old_id not in used_ids:
                        chain_id_map[old_id] = old_id
                        used_ids.add(old_id)
                    else:
                        # Assign next available single-character ID
                        for new_id in available_ids:
                            if new_id not in used_ids:
                                chain_id_map[old_id] = new_id
                                used_ids.add(new_id)
                                break
                        else:
                            logger.warning(f"{pdb_id}: Too many chains for PDB format remap")
                            os.remove(cif_tmp_path)
                            return None, False

        if needs_remap:
            for model in structure:
                for chain in model:
                    chain.id = chain_id_map[chain.id]
            remap_str = ", ".join(f"{k}->{v}" for k, v in chain_id_map.items())
            logger.info(f"{pdb_id}: Remapped chain IDs: {remap_str}")

        io = PDBIO()
        io.set_structure(structure)
        io.save(save_path, select=ProteinSelect())

        # Clean up temporary CIF file
        os.remove(cif_tmp_path)

        # Verify the converted file is valid
        if os.path.getsize(save_path) < 100:
            logger.warning(f"{pdb_id}: Converted PDB file is suspiciously small, "
                           f"conversion may have failed")
            os.remove(save_path)
            return None, False

        logger.info(f"{pdb_id}: Successfully converted from mmCIF to PDB format")
        return save_path, True

    except Exception as e:
        logger.warning(f"{pdb_id}: mmCIF download/conversion FAILED — {type(e).__name__}: {e}")
        # Clean up partial files
        if os.path.exists(cif_tmp_path):
            os.remove(cif_tmp_path)
        if os.path.exists(save_path):
            os.remove(save_path)
        return None, False


# =====================================================================
# Chain Analysis
# =====================================================================

def get_protein_chains_from_pdb(pdb_path, pdb_id):
    """
    Parse a PDB file and return a list of protein chain IDs.
    Protein chains are identified by having CA atoms (alpha carbons).
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
    except Exception as e:
        logger.warning(f"Failed to parse {pdb_id}: {e}")
        return []

    model = structure[0]  # First model only
    protein_chains = []
    for chain in model:
        # Check if chain has amino acid residues (CA atoms)
        has_ca = False
        residue_count = 0
        for residue in chain:
            if residue.id[0] == ' ':  # Standard residue (not HETATM)
                residue_count += 1
                if 'CA' in residue:
                    has_ca = True
        # Consider it a protein chain if it has CA atoms and enough residues
        if has_ca and residue_count >= 10:
            protein_chains.append(chain.id)

    return protein_chains


def get_assembly_chain_info(pdb_id):
    """
    Query RCSB API for biological assembly information to determine
    which chains form the dimer in the biological assembly.
    Returns a list of chain IDs for the first biological assembly (the dimer).
    """
    try:
        # Get assembly info for assembly 1 (the most likely biological unit)
        url = RCSB_ASSEMBLY_URL.format(pdb_id, 1)
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None

        data = response.json()

        # Extract chain IDs from the assembly
        # The assembly describes which chains are in the biological unit
        assembly_chains = []
        if 'rcsb_struct_symmetry' in data:
            pass  # This info is at entry level, not assembly level

        # Try to get polymer entity instances in the assembly
        # Use the pdbx_struct_assembly_gen for operator and chain info
        if 'pdbx_struct_assembly_gen' in data:
            for gen in data['pdbx_struct_assembly_gen']:
                asym_ids = gen.get('asym_id_list', [])
                if isinstance(asym_ids, list):
                    assembly_chains.extend(asym_ids)
                elif isinstance(asym_ids, str):
                    assembly_chains.extend(asym_ids.split(','))

        if assembly_chains:
            return assembly_chains

    except Exception as e:
        logger.debug(f"Assembly API call failed for {pdb_id}: {e}")

    return None


def get_entity_chain_mapping(pdb_id):
    """
    Query RCSB API for entity-to-chain mapping.
    Returns dict: {entity_id: [chain_ids]} for polymer entities only.
    """
    try:
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None

        data = response.json()
        # Get polymer entity count
        entity_info = data.get('rcsb_entry_info', {})
        polymer_count = entity_info.get('deposited_polymer_entity_instance_count', 0)

        return {'polymer_instance_count': polymer_count}
    except Exception:
        return None


def determine_dimer_chains(pdb_id, all_protein_chains, pdb_path):
    """
    For PDBs with more than 2 protein chains, determine which 2 chains
    form the biological dimer.

    Strategy:
    1. Query RCSB assembly API for biological assembly chain info.
    2. If assembly info indicates 2 chains, use those.
    3. Otherwise, default to first two chains (A, B) as most PDBs
       list the biological unit chains first.

    Returns:
        (chain1, chain2): The two chain IDs forming the dimer.
        method: How the chains were determined.
    """
    if len(all_protein_chains) == 2:
        return all_protein_chains[0], all_protein_chains[1], "exact_2_chains"

    if len(all_protein_chains) < 2:
        return None, None, "insufficient_chains"

    # For >2 chains, try to get assembly info
    assembly_chains = get_assembly_chain_info(pdb_id)

    if assembly_chains and len(assembly_chains) == 2:
        # Assembly says exactly 2 chains
        # Map auth chain IDs — assembly may use asym_id (label) not auth_id
        # For simplicity, check if assembly chains match our protein chains
        matched = [c for c in assembly_chains if c in all_protein_chains]
        if len(matched) == 2:
            return matched[0], matched[1], "assembly_api"

    # Try reading REMARK 350 (BIOMT) from the PDB file for biological assembly
    bio_chains = parse_remark350_chains(pdb_path)
    if bio_chains and len(bio_chains) == 2:
        matched = [c for c in bio_chains if c in all_protein_chains]
        if len(matched) == 2:
            return matched[0], matched[1], "remark350"

    # Fallback: just take the first two protein chains
    return all_protein_chains[0], all_protein_chains[1], "first_two_fallback"


def parse_remark350_chains(pdb_path):
    """
    Parse REMARK 350 from a PDB file to extract chains in the
    first biological assembly (BIOMOLECULE: 1).
    """
    chains = set()
    in_biomol_1 = False
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("REMARK 350"):
                    if "BIOMOLECULE: 1" in line:
                        in_biomol_1 = True
                    elif "BIOMOLECULE:" in line and "BIOMOLECULE: 1" not in line:
                        # We've reached a different biomolecule
                        if in_biomol_1:
                            break
                    if in_biomol_1 and "APPLY THE FOLLOWING TO CHAINS:" in line:
                        # Extract chain IDs
                        parts = line.split("CHAINS:")[1]
                        chain_ids = [c.strip().rstrip(',') for c in parts.split(',')]
                        chains.update([c for c in chain_ids if c])
                    if in_biomol_1 and "AND CHAINS:" in line:
                        parts = line.split("AND CHAINS:")[1]
                        chain_ids = [c.strip().rstrip(',') for c in parts.split(',')]
                        chains.update([c for c in chain_ids if c])
                elif not line.startswith("REMARK") and in_biomol_1:
                    break
    except Exception:
        pass
    return list(chains)


# =====================================================================
# Main Pipeline
# =====================================================================

def process_single_pdb(pdb_id, pdb_type_label, save_dir):
    """
    Download and analyze a single PDB.
    Returns a dict with metadata or None on failure.
    """
    # Download
    pdb_path, converted_from_cif = download_pdb(pdb_id, save_dir)
    if pdb_path is None:
        return None

    # Get all protein chains
    all_protein_chains = get_protein_chains_from_pdb(pdb_path, pdb_id)
    if len(all_protein_chains) < 2:
        logger.warning(f"{pdb_id}: Only {len(all_protein_chains)} protein chain(s) found, skipping")
        # Remove the downloaded file since it's not a usable dimer
        os.remove(pdb_path)
        return None

    # Determine the dimer chains
    chain1, chain2, method = determine_dimer_chains(pdb_id, all_protein_chains, pdb_path)
    if chain1 is None:
        logger.warning(f"{pdb_id}: Could not determine dimer chains, skipping")
        os.remove(pdb_path)
        return None

    record = {
        'PDB': pdb_id,
        'chain_1': chain1,
        'chain_2': chain2,
        'all_chains': ','.join(all_protein_chains),
        'num_chains': len(all_protein_chains),
        'dimer_type': pdb_type_label,
        'chain_determination_method': method,
        'converted_from_cif': converted_from_cif,
        'path': os.path.relpath(pdb_path, SCRIPT_DIR),
    }

    return record


def main():
    parser = argparse.ArgumentParser(description="Download dimer PDBs from RCSB")
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing CSV, skip already processed PDBs')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Save progress every N PDBs')
    args = parser.parse_args()

    # Create directories
    os.makedirs(PDB_SAVE_DIR, exist_ok=True)
    os.makedirs(TABLE_SAVE_DIR, exist_ok=True)

    # Search for hetero-dimers (with automatic pagination)
    pdb_ids, pdb_type_map = search_hetero_dimers()

    if not pdb_ids:
        logger.error("No PDB IDs found from search. Exiting.")
        return

    # Resume support: collect PDB IDs we already have from multiple sources
    existing_records = []
    processed_ids = set()

    # 1. From new_dimer_db_table.csv (our own previous downloads)
    if args.resume and os.path.exists(TABLE_SAVE_PATH):
        existing_df = pd.read_csv(TABLE_SAVE_PATH)
        existing_records = existing_df.to_dict('records')
        processed_ids.update(existing_df['PDB'].tolist())
        logger.info(f"Resuming: {len(existing_df)} PDBs from new_dimer_db_table")

    # 2. From old_dimer_table.csv (old dimer database)
    old_dimer_path = os.path.join(TABLE_SAVE_DIR, 'old_dimer_table.csv')
    if os.path.exists(old_dimer_path):
        old_df = pd.read_csv(old_dimer_path)
        processed_ids.update(old_df['PDB'].tolist())
        logger.info(f"Skipping {len(old_df)} PDBs from dimer_table.csv")

    # 3. From wt_table.csv (wild-type complexes)
    wt_path = os.path.join(TABLE_SAVE_DIR, 'wt_table.csv')
    if os.path.exists(wt_path):
        wt_df = pd.read_csv(wt_path)
        processed_ids.update(wt_df['PDB'].tolist())
        logger.info(f"Skipping {len(wt_df)} PDBs from wt_table.csv")

    logger.info(f"Total PDB IDs to skip: {len(processed_ids)}")

    remaining_ids = [pid for pid in pdb_ids if pid not in processed_ids]
    logger.info(f"PDBs to process: {len(remaining_ids)}")

    # Process PDBs
    records = list(existing_records)
    success_count = len(existing_records)
    fail_count = 0

    for i, pdb_id in enumerate(remaining_ids):
        pdb_type_label = pdb_type_map.get(pdb_id, "unknown")
        record = process_single_pdb(pdb_id, pdb_type_label, PDB_SAVE_DIR)

        if record is not None:
            records.append(record)
            success_count += 1
        else:
            fail_count += 1

        # Progress logging
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{len(remaining_ids)} | "
                        f"Success: {success_count} | Failed: {fail_count}")

        # Periodic save
        if (i + 1) % args.batch_size == 0:
            df = pd.DataFrame(records)
            df.to_csv(TABLE_SAVE_PATH, index=False)
            logger.info(f"Checkpoint saved at {i+1} PDBs")

        # Rate limiting — be kind to RCSB servers
        time.sleep(0.2)

    # Final save
    df = pd.DataFrame(records)
    df.to_csv(TABLE_SAVE_PATH, index=False)

    # Summary
    logger.info("=" * 60)
    logger.info("Download Complete!")
    logger.info(f"Total PDBs downloaded: {success_count}")
    logger.info(f"Failed/skipped: {fail_count}")
    logger.info(f"PDB files saved to: {os.path.abspath(PDB_SAVE_DIR)}")
    logger.info(f"Table saved to: {os.path.abspath(TABLE_SAVE_PATH)}")

    if len(df) > 0:
        logger.info(f"\nDimer type distribution:")
        logger.info(f"\n{df['dimer_type'].value_counts().to_string()}")
        logger.info(f"\nChain determination method distribution:")
        logger.info(f"\n{df['chain_determination_method'].value_counts().to_string()}")
        logger.info(f"\nPDBs with >2 protein chains: "
                     f"{(df['num_chains'] > 2).sum()}")


if __name__ == '__main__':
    main()
