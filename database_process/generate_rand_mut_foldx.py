"""
generate_rand_mut_foldx.py

Generates mutant PDB structures using FoldX for random mutations stored in pkl files.

Each pkl file (e.g. 10lg_mut.pkl) corresponds to one wild-type PDB and contains
a StructureToHeteroGraph object with:
  - mut_info: mutation string in format 'A_Y87R,A_V166L,...' (chain_AA-pos-newAA)
  - AA_df:    dataframe with chain_id, chain_letter, chain_type (receptor/ligand) columns

The script:
  1. Scans all *_mut.pkl files in PKL_DIR
  2. Builds a mutation dataframe (optionally saved as CSV)
  3. Runs FoldX BuildModel for each PDB using multiprocessing
"""

import os
import sys
import glob
import pickle
import multiprocessing as mp
from functools import partial

import pandas as pd
from tqdm import tqdm

# Ensure project root is on sys.path so utils can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.database_process import create_individual_list, run_foldx, rename_mutant_pdbs

# ---------------------------------------------------------------------------
# Configuration — adjust these paths as needed
# ---------------------------------------------------------------------------
BASE_PDB_DIR    = '../../GraPPI_data'
FOLDX_EXEC      = '/mnt/poisson/home/zhisong/projects/foldx5/foldx_20270131'
WORKING_DIR_ROOT = os.path.join(BASE_PDB_DIR, 'PDB/test_rand_mut_pdbs')
OUTPUT_CSV      = './saved_tables/test_rand_mut_table.csv'
N_PROCESSES     = 16
# ---------------------------------------------------------------------------


def build_rand_mut_dataframe(pkl_dir: str, base_pdb_dir: str) -> pd.DataFrame:
    """
    Scan all *_mut.pkl files in pkl_dir and build a mutation dataframe
    compatible with process_pdb_mutations().

    Columns produced:
        Mutations, Source Data Set, PDB_ori, KD(M),
        Ligand Chains, Receptor Chains, wt_paths,
        affinity, PDB, new_pdb_name, mut_paths
    """
    # Load PDB path lookup tables
    dimer_table = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'saved_tables/dimer_table.csv')
    )
    wt_table = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'saved_tables/wt_table.csv')
    )

    # Upper-case PDB column for case-insensitive matching
    dimer_table['_PDB_upper'] = dimer_table['PDB'].str.upper()
    wt_table['_PDB_upper']    = wt_table['PDB'].str.upper()

    # Paths in the lookup tables are relative to database_process/ (the script's dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_base   = os.path.abspath(os.path.join(script_dir, base_pdb_dir))

    def lookup_wt_path(pdb_upper: str):
        """Return wt_paths relative to base_pdb_dir, or None if not found."""
        row = dimer_table[dimer_table['_PDB_upper'] == pdb_upper]
        if row.empty:
            row = wt_table[wt_table['_PDB_upper'] == pdb_upper]
        if row.empty:
            return None
        full_path = row.iloc[0]['paths']
        # Resolve the table path (which is relative to database_process/)
        if os.path.isabs(full_path):
            abs_full = full_path
        else:
            abs_full = os.path.abspath(os.path.join(script_dir, full_path))
        return os.path.relpath(abs_full, abs_base)

    pkl_files = sorted(glob.glob(os.path.join(pkl_dir, '*_mut.pkl')))
    if not pkl_files:
        raise FileNotFoundError(f"No *_mut.pkl files found in {pkl_dir}")

    rows = []
    skipped = []

    for pkl_path in tqdm(pkl_files, desc="Loading pkl files"):
        filename   = os.path.basename(pkl_path)
        pdb_id     = filename.replace('_mut.pkl', '')   # e.g. '10lg'
        pdb_upper  = pdb_id.upper()                     # e.g. '10LG'
        pdb_lower  = pdb_id.lower()

        try:
            with open(pkl_path, 'rb') as f:
                mut_obj = pickle.load(f)
        except Exception as exc:
            print(f"Warning: could not load {filename}: {exc}")
            skipped.append(pdb_upper)
            continue

        mut_info = getattr(mut_obj, 'mut_info', None)
        if not mut_info:
            print(f"Warning: empty mut_info for {pdb_upper}, skipping.")
            skipped.append(pdb_upper)
            continue

        # Chain info from AA_df
        aa_df = mut_obj.AA_df
        receptor_chains = ','.join(
            sorted(aa_df.loc[aa_df['chain_type'] == 'receptor', 'chain_letter'].unique())
        )
        ligand_chains = ','.join(
            sorted(aa_df.loc[aa_df['chain_type'] == 'ligand', 'chain_letter'].unique())
        )

        # Wild-type PDB path (relative to base_pdb_dir)
        wt_paths = lookup_wt_path(pdb_upper)
        if wt_paths is None:
            print(f"Warning: PDB {pdb_upper} not found in lookup tables, skipping.")
            skipped.append(pdb_upper)
            continue

        new_pdb_name = f"{pdb_lower}_mut"
        mut_paths    = f"PDB/rand_mut_pdbs/{new_pdb_name}.pdb"

        rows.append({
            'Mutations':       mut_info,
            'Source Data Set': 'random_mut',
            'PDB_ori':         pdb_upper,
            'KD(M)':           float('nan'),
            'Ligand Chains':   ligand_chains,
            'Receptor Chains': receptor_chains,
            'wt_paths':        wt_paths,
            'affinity':        float('nan'),
            'PDB':             pdb_upper,
            'new_pdb_name':    new_pdb_name,
            'mut_paths':       mut_paths,
        })

    if skipped:
        print(f"\nSkipped {len(skipped)} entries: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    return pd.DataFrame(rows)


def process_pdb_mutations(pdb_subset_df, foldx_executable, working_dir_root,
                          base_pdb_dir='../../GraPPI_data'):
    """Run FoldX BuildModel for all mutations belonging to one PDB."""
    pdb_code    = pdb_subset_df['PDB'].iloc[0]
    pdb_lower   = pdb_code.lower()
    wt_pdb_path = os.path.join(base_pdb_dir, pdb_subset_df['wt_paths'].iloc[0])
    working_dir = os.path.join(working_dir_root, pdb_lower)

    if not os.path.exists(wt_pdb_path):
        raise FileNotFoundError(f"Wild-type PDB not found: {wt_pdb_path}")

    os.makedirs(working_dir, exist_ok=True)

    mutations_list        = pdb_subset_df['Mutations'].tolist()
    individual_list_path  = os.path.join(working_dir, 'individual_list.txt')

    create_individual_list(mutations_list, individual_list_path)
    run_foldx(wt_pdb_path, individual_list_path, foldx_executable, working_dir)
    rename_mutant_pdbs(pdb_subset_df, working_dir, only_remove_wt=True)


def process_single_pdb(pdb, mu_df, foldx_executable, working_dir_root):
    """Worker function called by each process in the pool."""
    try:
        pdb_subset = mu_df[mu_df['PDB'] == pdb]
        if pdb_subset.empty:
            print(f"Error: no data for PDB {pdb}")
            return pdb, False
        process_pdb_mutations(pdb_subset, foldx_executable, working_dir_root)
        return pdb, True
    except Exception as exc:
        print(f"Error processing {pdb}: {exc}")
        return pdb, False


if __name__ == '__main__':
    # ---- Step 1: build mutation dataframe from pkl files ----
    #print("Building mutation dataframe from pkl files...")
    #mu_df = build_rand_mut_dataframe(PKL_DIR, BASE_PDB_DIR)
    #print(f"Found {len(mu_df)} mutation entries across {mu_df['PDB'].nunique()} PDBs")

    # Save the intermediate table for reference
    #os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    #mu_df.to_csv(OUTPUT_CSV, index=False)
    #print(f"Saved mutation table to {OUTPUT_CSV}")

    # ---- Step 2: run FoldX with multiprocessing ----
    mu_df = pd.read_csv('./saved_tables/test_rand_mut_table.csv')
    pdb_list = mu_df['PDB'].unique().tolist()
    print(f"\nProcessing {len(pdb_list)} PDBs using {N_PROCESSES} processes...")

    process_func = partial(
        process_single_pdb,
        mu_df=mu_df,
        foldx_executable=FOLDX_EXEC,
        working_dir_root=WORKING_DIR_ROOT,
    )

    with mp.Pool(processes=N_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap(process_func, pdb_list),
            total=len(pdb_list),
            desc="Processing PDBs",
        ))

    successful = sum(1 for _, ok in results if ok)
    failed_pdbs = [pdb for pdb, ok in results if not ok]

    print(f"\nCompleted: {successful}/{len(pdb_list)} successful")
    if failed_pdbs:
        print(f"Failed PDBs ({len(failed_pdbs)}): {failed_pdbs}")
