"""
Cluster Dimer Database by Structural Similarity using Foldseek

This script:
1. Reads the combined_dimer_table.csv (produced by cluster_dimers_mmseqs.py Step 1).
2. Extracts per-dimer PDB files containing only chain_1 and chain_2 (first model only)
   into a working directory, so foldseek clusters by the dimer complex structure.
3. Runs foldseek easy-multimercluster at the specified coverage thresholds.
   This clusters at the COMPLEX level (both chains together), not per-chain.
4. Outputs:
   - A cluster info table: saved_tables/foldseek_cluster_info_tmsXX.csv
   - A representative (non-redundant) table: saved_tables/nr_dimer_table_foldseek_tmsXX.csv

Usage:
    python cluster_dimers_foldseek.py [--coverage 0.8] [--threads 32]

Requires:
    - foldseek binary (set via --foldseek-bin or auto-detected)
    - Biopython
"""

import os
import sys
import re
import argparse
import subprocess
import shutil
import logging
from collections import OrderedDict

import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select

# =====================================================================
# Configuration
# =====================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLE_DIR = os.path.join(SCRIPT_DIR, 'saved_tables')
WORK_DIR = '../../GraPPI_data/foldseek_workdir'

# Default foldseek binary location
DEFAULT_FOLDSEEK_BIN = os.path.expanduser('~/foldseek/foldseek/bin/foldseek') # need to asign path for foldseek

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SCRIPT_DIR, 'cluster_foldseek.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)


# =====================================================================
# Step 1: Extract Dimer-Only PDB Files
# =====================================================================

class ChainSelect(Select):
    """Select only specific chains from a structure, keeping only protein residues."""

    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)

    def accept_model(self, model):
        # Only keep the first model (avoid NMR model duplicates)
        return model.id == 0

    def accept_chain(self, chain):
        return chain.id in self.chain_ids

    def accept_residue(self, residue):
        # Keep only standard amino acid residues (hetflag ' ')
        # Exclude water ('W'), ions, ligands, and other HETATM ('H_xxx')
        hetflag = residue.id[0]
        return hetflag == ' '


def extract_dimer_pdbs(combined_df, output_dir):
    """
    For each dimer, extract only chain_1 and chain_2 into a new PDB file
    in output_dir. This ensures foldseek clusters by the dimer structure,
    not by extra crystal packing chains.

    Returns:
        extracted: dict {pdb_id: extracted_pdb_path}
        failed: list of pdb_ids that failed
    """
    os.makedirs(output_dir, exist_ok=True)
    parser = PDBParser(QUIET=True)
    io = PDBIO()

    extracted = OrderedDict()
    failed = []

    for i, row in combined_df.iterrows():
        pdb_id = row['PDB']
        chain1 = str(row['chain_1']).strip()
        chain2 = str(row['chain_2']).strip()
        pdb_path = row['path']

        # Resolve relative paths (relative to SCRIPT_DIR)
        if not os.path.isabs(pdb_path):
            pdb_path = os.path.join(SCRIPT_DIR, pdb_path)

        out_path = os.path.join(output_dir, f"{pdb_id}.pdb")

        # Skip if already extracted
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
            extracted[pdb_id] = out_path
            if (i + 1) % 5000 == 0:
                logger.info(f"Extraction progress: {i+1}/{len(combined_df)} "
                            f"(reusing existing)")
            continue

        if not os.path.exists(pdb_path):
            logger.warning(f"{pdb_id}: PDB file not found at {pdb_path}")
            failed.append(pdb_id)
            continue

        try:
            structure = parser.get_structure(pdb_id, pdb_path)
            model = structure[0]

            # Verify both chains exist
            chain_ids_in_file = {c.id for c in model}
            if chain1 not in chain_ids_in_file or chain2 not in chain_ids_in_file:
                logger.warning(f"{pdb_id}: Chain(s) {chain1}/{chain2} not found "
                               f"in file (available: {chain_ids_in_file})")
                failed.append(pdb_id)
                continue

            # Check chains have enough residues
            for cid in [chain1, chain2]:
                chain_obj = model[cid]
                res_count = sum(1 for r in chain_obj if r.id[0] == ' ')
                if res_count < 10:
                    logger.warning(f"{pdb_id}: Chain {cid} too short "
                                   f"({res_count} residues)")
                    failed.append(pdb_id)
                    break
            else:
                # Both chains are OK — extract
                io.set_structure(structure)
                io.save(out_path, select=ChainSelect([chain1, chain2]))

                if os.path.getsize(out_path) < 100:
                    logger.warning(f"{pdb_id}: Extracted PDB is suspiciously small")
                    os.remove(out_path)
                    failed.append(pdb_id)
                    continue

                extracted[pdb_id] = out_path

        except Exception as e:
            logger.warning(f"{pdb_id}: Extraction failed — {type(e).__name__}: {e}")
            failed.append(pdb_id)
            if os.path.exists(out_path):
                os.remove(out_path)

        if (i + 1) % 2000 == 0:
            logger.info(f"Extraction progress: {i+1}/{len(combined_df)} "
                        f"({len(extracted)} OK, {len(failed)} failed)")

    logger.info(f"Extracted {len(extracted)} dimer PDBs, {len(failed)} failed")
    return extracted, failed


# =====================================================================
# Step 2: Run Foldseek Multimer Clustering
# =====================================================================

def run_foldseek_cluster(foldseek_bin, pdb_dir, output_prefix, tmp_dir,
                         coverage=0.8, tmscore_threshold=None,
                         exhaustive_search=False,
                         threads=4, extra_args=None):
    """
    Run foldseek easy-multimercluster on a directory of PDB files.
    This clusters at the COMPLEX level, comparing multi-chain structures
    as whole units rather than individual chains.

    Args:
        foldseek_bin: Path to foldseek binary
        pdb_dir: Directory containing PDB files to cluster
        output_prefix: Prefix for output files
        tmp_dir: Temporary directory for foldseek
        coverage: Min coverage threshold (0-1)
        tmscore_threshold: Optional TM-score threshold
        exhaustive_search: If True, bypass k-mer prefilter (slower but catches small proteins)
        threads: Number of CPU threads
        extra_args: Additional foldseek arguments as list of strings
    """
    cmd = [
        foldseek_bin, 'easy-multimercluster',
        pdb_dir,
        output_prefix,
        tmp_dir,
        '-c', str(coverage),
        '--cov-mode', '0',          # coverage of query AND target
        '--threads', str(threads),
        '-v', '3',                  # verbosity
    ]

    if exhaustive_search:
        cmd.extend(['--exhaustive-search', '1'])

    if tmscore_threshold is not None:
        cmd.extend(['--tmscore-threshold', str(tmscore_threshold)])

    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Running foldseek: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Foldseek failed!\nstdout: {result.stdout[-1000:]}\n"
                     f"stderr: {result.stderr[-1000:]}")
        sys.exit(1)

    logger.info("Foldseek finished successfully")
    # Print last part of output
    if result.stdout:
        logger.info(result.stdout[-500:])


# =====================================================================
# Step 3: Parse Foldseek Multimercluster Results
# =====================================================================

def strip_model_suffix(foldseek_id):
    """
    Strip _MODEL_N suffix from foldseek multimercluster IDs.
    Examples:
        '10LG_MODEL_1'  -> '10LG'
        '1A0N_MODEL_25' -> '1A0N'
        '12E8_MODEL_1'  -> '12E8'
    """
    match = re.match(r'^(.+)_MODEL_\d+$', foldseek_id)
    if match:
        return match.group(1)
    return foldseek_id


def parse_foldseek_clusters(cluster_tsv_path, extracted_pdb_ids):
    """
    Parse foldseek multimercluster TSV output.
    Format: representative_id\tmember_id (one pair per line)

    Foldseek appends '_MODEL_N' to all IDs (e.g., '10LG_MODEL_1').
    For NMR structures with multiple models, there will be multiple entries
    (e.g., '1A0N_MODEL_1' through '1A0N_MODEL_25').
    We strip these suffixes and collapse to PDB-level clustering.

    Args:
        cluster_tsv_path: Path to foldseek cluster TSV output
        extracted_pdb_ids: Set of PDB IDs we extracted (for validation)

    Returns:
        rep_to_members: dict {representative_pdb: [member_pdbs]}
        member_to_rep: dict {member_pdb: representative_pdb}
    """
    # First pass: read raw foldseek IDs and map to PDB IDs
    raw_rep_to_members = OrderedDict()
    with open(cluster_tsv_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            rep_raw, member_raw = parts

            # Strip _MODEL_N suffix to get PDB IDs
            rep_id = strip_model_suffix(rep_raw)
            member_id = strip_model_suffix(member_raw)

            # Skip entries not in our extracted set (e.g., extra NMR models)
            if member_id not in extracted_pdb_ids:
                continue

            if rep_id not in raw_rep_to_members:
                raw_rep_to_members[rep_id] = set()
            raw_rep_to_members[rep_id].add(member_id)

    # Build clean cluster dictionaries at PDB level
    rep_to_members = OrderedDict()
    member_to_rep = {}

    for rep_id, members in raw_rep_to_members.items():
        # If the representative itself is not in our extracted set,
        # pick the first member as the representative instead
        if rep_id not in extracted_pdb_ids:
            members_list = sorted(members)
            rep_id = members_list[0]

        if rep_id not in rep_to_members:
            rep_to_members[rep_id] = []

        for member_id in sorted(members):
            if member_id not in member_to_rep:
                rep_to_members[rep_id].append(member_id)
                member_to_rep[member_id] = rep_id

    # Log any IDs we extracted but didn't appear in clustering
    clustered_ids = set(member_to_rep.keys())
    missing = extracted_pdb_ids - clustered_ids
    if missing:
        logger.warning(f"{len(missing)} extracted PDBs not found in cluster output "
                       f"(first 5: {list(missing)[:5]})")

    return rep_to_members, member_to_rep


# =====================================================================
# Main Pipeline
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cluster dimer database by structural similarity using Foldseek")
    parser.add_argument('--coverage', type=float, default=0.8,
                        help='Coverage threshold (0-1, default: 0.8)')
    parser.add_argument('--tmscore-threshold', type=float, default=None,
                        help='Optional TM-score threshold for filtering alignments (0-1)')
    parser.add_argument('--exhaustive-search', action='store_true',
                        help='Use exhaustive search (bypasses k-mer prefilter, slower but catches small proteins)')
    parser.add_argument('--threads', type=int, default=32,
                        help='Number of CPU threads for foldseek (default: 32)')
    parser.add_argument('--foldseek-bin', type=str, default=DEFAULT_FOLDSEEK_BIN,
                        help=f'Path to foldseek binary (default: {DEFAULT_FOLDSEEK_BIN})')
    parser.add_argument('--input-table', type=str, default=None,
                        help='Input CSV table (default: saved_tables/combined_dimer_table.csv)')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Skip PDB extraction step (reuse existing dimer PDBs)')
    parser.add_argument('--extra-args', nargs='*', default=None,
                        help='Additional arguments to pass to foldseek easy-multimercluster')
    args = parser.parse_args()

    # Validate foldseek binary
    if not os.path.exists(args.foldseek_bin):
        # Try to find it in PATH
        foldseek_path = shutil.which('foldseek')
        if foldseek_path:
            args.foldseek_bin = foldseek_path
        else:
            logger.error(f"Foldseek binary not found at {args.foldseek_bin} "
                         f"or in PATH. Please install foldseek or specify "
                         f"--foldseek-bin.")
            sys.exit(1)

    logger.info(f"Using foldseek binary: {args.foldseek_bin}")

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    # Build tag for output filenames based on parameters
    cov_str = str(args.coverage).replace('.', '')
    tag = f'cov{cov_str}'
    if args.tmscore_threshold is not None:
        tms_str = str(args.tmscore_threshold).replace('.', '')
        tag += f'_tms{tms_str}'
    if args.exhaustive_search:
        tag += '_exh'

    # Directories — dimer_pdbs is shared (same PDBs), but output/tmp are per-setting
    dimer_pdb_dir = os.path.join(WORK_DIR, 'dimer_pdbs')
    tmp_dir = os.path.join(WORK_DIR, f'tmp_{tag}')
    output_prefix = os.path.join(WORK_DIR, f'dimer_cluster_{tag}')

    # ------------------------------------------------------------------
    # Step 1: Load combined dimer table
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1: Loading dimer table...")

    input_table = args.input_table or os.path.join(TABLE_DIR, 'combined_dimer_table.csv')
    if not os.path.exists(input_table):
        logger.error(f"Input table not found: {input_table}")
        logger.error("Run cluster_dimers_mmseqs.py first to generate "
                     "combined_dimer_table.csv, or specify --input-table.")
        sys.exit(1)

    combined_df = pd.read_csv(input_table)
    logger.info(f"Loaded {len(combined_df)} dimers from {input_table}")

    # ------------------------------------------------------------------
    # Step 2: Extract dimer-only PDB files
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2: Extracting dimer PDB files (chain_1 + chain_2 only)...")

    if args.skip_extraction and os.path.exists(dimer_pdb_dir):
        # Count existing PDBs
        existing = [f for f in os.listdir(dimer_pdb_dir) if f.endswith('.pdb')]
        logger.info(f"Skipping extraction — reusing {len(existing)} existing "
                    f"dimer PDBs in {dimer_pdb_dir}")
        extracted = {f[:-4]: os.path.join(dimer_pdb_dir, f) for f in existing}
        failed = []
    else:
        extracted, failed = extract_dimer_pdbs(combined_df, dimer_pdb_dir)

    if not extracted:
        logger.error("No PDB files extracted! Check that paths in the input "
                     "table are correct.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Run foldseek clustering
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    tms_msg = f", tmscore={args.tmscore_threshold}" if args.tmscore_threshold else ""
    exh_msg = ", exhaustive" if args.exhaustive_search else ""
    logger.info(f"Step 3: Running foldseek multimer structural clustering "
                f"(coverage={args.coverage}{tms_msg}{exh_msg})...")

    # Clean up previous run's tmp dir to avoid stale data
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    run_foldseek_cluster(
        foldseek_bin=args.foldseek_bin,
        pdb_dir=dimer_pdb_dir,
        output_prefix=output_prefix,
        tmp_dir=tmp_dir,
        coverage=args.coverage,
        tmscore_threshold=args.tmscore_threshold,
        exhaustive_search=args.exhaustive_search,
        threads=args.threads,
        extra_args=args.extra_args,
    )

    # ------------------------------------------------------------------
    # Step 4: Parse results and build output tables
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 4: Parsing clustering results...")

    cluster_tsv = output_prefix + '_cluster.tsv'
    if not os.path.exists(cluster_tsv):
        logger.error(f"Cluster TSV not found at {cluster_tsv}")
        sys.exit(1)

    rep_to_members, member_to_rep = parse_foldseek_clusters(
        cluster_tsv, set(extracted.keys())
    )
    representatives = set(rep_to_members.keys())

    logger.info(f"Total clusters: {len(rep_to_members)}")
    logger.info(f"Total clustered structures: {len(member_to_rep)}")

    # Cluster size distribution
    sizes = [len(members) for members in rep_to_members.values()]
    if sizes:
        logger.info(f"Cluster size stats: "
                    f"min={min(sizes)}, max={max(sizes)}, "
                    f"mean={sum(sizes)/len(sizes):.1f}, "
                    f"median={sorted(sizes)[len(sizes)//2]}")

    # Build cluster info table
    cluster_records = []
    for rep_id, members in rep_to_members.items():
        for member in members:
            cluster_records.append({
                'PDB': member,
                'cluster_rep': rep_id,
                'cluster_size': len(members),
                'is_representative': member == rep_id
            })
    cluster_df = pd.DataFrame(cluster_records)

    # Merge cluster info with combined table
    combined_with_cluster = combined_df.merge(cluster_df, on='PDB', how='left')

    # Mark failed/unclustered PDBs
    combined_with_cluster.loc[
        combined_with_cluster['cluster_rep'].isna(), 'cluster_rep'
    ] = 'unclustered'

    # Save cluster info
    cluster_info_path = os.path.join(TABLE_DIR, f'foldseek_cluster_info_{tag}.csv')
    combined_with_cluster.to_csv(cluster_info_path, index=False)
    logger.info(f"Saved cluster info to {cluster_info_path}")

    # Build non-redundant (representative) table
    nr_df = combined_with_cluster[
        combined_with_cluster['is_representative'] == True
    ].copy()
    nr_df = nr_df.reset_index(drop=True)

    nr_path = os.path.join(TABLE_DIR, f'nr_dimer_table_foldseek_{tag}.csv')
    nr_df.to_csv(nr_path, index=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Foldseek Multimer Structural Clustering Complete!")
    logger.info(f"  Coverage threshold: {args.coverage}")
    logger.info(f"  Input dimers: {len(combined_df)}")
    logger.info(f"  Structures extracted: {len(extracted)}")
    logger.info(f"  Failed extractions: {len(failed)}")
    logger.info(f"  Clusters: {len(rep_to_members)}")
    logger.info(f"  Representative (non-redundant) dimers: {len(nr_df)}")

    if len(extracted) > 0 and len(nr_df) > 0:
        logger.info(f"  Redundancy reduction: "
                    f"{len(extracted)} -> {len(nr_df)} "
                    f"({100*(1 - len(nr_df)/len(extracted)):.1f}% removed)")

    logger.info(f"\nOutput files:")
    logger.info(f"  Cluster info: {os.path.abspath(cluster_info_path)}")
    logger.info(f"  Non-redundant table: {os.path.abspath(nr_path)}")

    # Source distribution in NR set
    if 'source' in nr_df.columns:
        logger.info(f"\nNon-redundant set by source:")
        logger.info(f"\n{nr_df['source'].value_counts().to_string()}")


if __name__ == '__main__':
    main()
