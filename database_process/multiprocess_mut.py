import multiprocessing as mp
from functools import partial
import pandas as pd
import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.database_process import create_individual_list, run_foldx, rename_mutant_pdbs
from tqdm import tqdm
tqdm.pandas()

mu_df = pd.read_csv('./saved_tables/mutant_table.csv')

def process_pdb_mutations(pdb_subset_df, foldx_executable, working_dir_root, rotabase_location=None, base_pdb_dir='../../GraPPI_data'):
    pdb_code = pdb_subset_df['PDB'].iloc[0]
    pdb_lower = pdb_code.lower()
    wt_pdb_path = os.path.join(base_pdb_dir, f"{pdb_subset_df['wt_paths'].iloc[0]}")
    working_dir = os.path.join(working_dir_root, pdb_lower)
    if not os.path.exists(wt_pdb_path):
        print(f"Error: Wild type PDB not found at {wt_pdb_path}")
        return

    os.makedirs(working_dir, exist_ok=True)
    mutations_list = pdb_subset_df['Mutations'].tolist()
    individual_list_path = os.path.join(working_dir, 'individual_list.txt')
    create_individual_list(mutations_list, individual_list_path)
    run_foldx(wt_pdb_path, individual_list_path, foldx_executable, working_dir, rotabase_location=rotabase_location)
    rename_mutant_pdbs(pdb_subset_df, working_dir, True)

def process_single_pdb(pdb, mu_df, foldx_executable, working_dir_root):
    """
    Wrapper function to process a single PDB
    This function will be called by each worker process
    """
    try:
        pdb_subset = mu_df[mu_df['PDB'] == pdb]
        # Check wild-type PDB path
        if pdb_subset.empty:
            print(f"Error: No data found for PDB {pdb}")
            return pdb, False
        # Process mutations
        process_pdb_mutations(pdb_subset, foldx_executable, working_dir_root)
        return pdb, True
    except Exception as e:
        print(f"Error processing {pdb}: {str(e)}")
        return pdb, False


# Main multiprocessing code
if __name__ == '__main__':  # Important for multiprocessing
    foldx_executable = '../../GraPPI_data/foldx' # need to assign foldx path
    working_dir_root = '../../GraPPI_data/PDB/mutant_pdbs'
    
    # Get list of unique PDBs
    pdb_list = mu_df['PDB'].unique().tolist()
    
    # Number of processes (adjust based on your CPU cores)
    n_processes = 16 #mp.cpu_count() - 1  # Leave one core free
    # Or set manually: n_processes = 8
    
    print(f"Processing {len(pdb_list)} PDBs using {n_processes} processes...")
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_pdb,
        mu_df=mu_df,
        foldx_executable=foldx_executable,
        working_dir_root=working_dir_root
    )
    
    # Use multiprocessing Pool
    with mp.Pool(processes=n_processes) as pool:
        # Use imap for progress tracking with tqdm
        results = list(tqdm(
            pool.imap(process_func, pdb_list),
            total=len(pdb_list),
            desc="Processing PDBs"
        ))
    
    # Check results
    successful = sum(1 for _, success in results if success)
    failed = [(pdb, success) for pdb, success in results if not success]
    
    print(f"\nCompleted: {successful}/{len(pdb_list)} successful")
    if failed:
        print(f"Failed PDBs: {[pdb for pdb, _ in failed]}")