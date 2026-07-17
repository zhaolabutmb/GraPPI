import multiprocessing as mp
from functools import partial
import os
import pandas as pd
import subprocess
import re
from tqdm import tqdm
import numpy as np

def run_usalign_comparison(args):
    """
    Run USalign for a single pair of structures.
    Returns: (idx, jdx, tm_score, rmsd)
    """
    idx, jdx, pdb_path1, pdb_path2, chain_ids1, chain_ids2 = args
    
    cmd_line = f'USalign -mol prot -mm 1 -chain1 {chain_ids1} {pdb_path1} -chain2 {chain_ids2} {pdb_path2} -outfmt 1 -a T'
    process = subprocess.Popen(cmd_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    tm_pattern = r'# TM-score=([\d.]+)\s+\(normalized by average length.*?L=([\d.]+)\s+d0=([\d.]+)\)'
    rmsd_pattern = r'# Lali=(\d+)\s+RMSD=([\d.]+)\s+seqID_ali=([\d.]+)'
    output_text = stdout.decode('utf-8')
    
    tm_match = re.search(tm_pattern, output_text)
    rmsd_match = re.search(rmsd_pattern, output_text)
    
    if tm_match and rmsd_match:
        tm_score = float(tm_match.group(1))
        rmsd_val = float(rmsd_match.group(2))
    else:
        tm_score = np.nan
        rmsd_val = np.nan
    
    return (idx, jdx, tm_score, rmsd_val)

if __name__ == '__main__':
    # Load dataframe
    print("Loading dataframe...")
    wt_df = pd.read_csv('./saved_tables/wt_table.csv').reset_index(drop=True)
    
    # Prepare all comparison tasks
    print("Preparing comparison tasks...")
    tasks = []
    for idx in range(len(wt_df)): # 0 to len(wt_df)-1
        for jdx in range(idx + 1, len(wt_df)):
            pdb_path1 = wt_df.loc[idx, 'paths']
            pdb_path2 = wt_df.loc[jdx, 'paths']
            
            chain_ids1 = ','.join(list(set([cid.strip() for cid in (wt_df.loc[idx, 'Ligand Chains']+','+wt_df.loc[idx, 'Receptor Chains']).split(',')])))
            chain_ids2 = ','.join(list(set([cid.strip() for cid in (wt_df.loc[jdx, 'Ligand Chains']+','+wt_df.loc[jdx, 'Receptor Chains']).split(',')])))
            
            tasks.append((idx, jdx, pdb_path1, pdb_path2, chain_ids1, chain_ids2))
    
    print(f"Total tasks: {len(tasks)}")
    
    # Initialize result matrices
    rmsd_matrix = np.zeros((len(wt_df),len(wt_df)))
    tmscore_matrix = np.ones((len(wt_df), len(wt_df)))
    
    # Run in parallel with progress bar
    num_processes = mp.cpu_count()-2  # Use all available CPUs
    print(f"Using {num_processes} CPU cores")
    
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(run_usalign_comparison, tasks),
            total=len(tasks),
            desc="Computing alignments"
        ))
    
    # Fill matrices with results
    print("Filling result matrices...")
    for idx, jdx, tm_score, rmsd_val in results:
        tmscore_matrix[idx, jdx] = tm_score
        rmsd_matrix[idx, jdx] = rmsd_val 

    print("Done!")
    print(f"TM-score matrix shape: {tmscore_matrix.shape}")
    print(f"RMSD matrix shape: {rmsd_matrix.shape}")
    
    # Save results
    np.save('./saved_tables/updated_wt_rmsd_matrix.npy', rmsd_matrix)
    np.save('./saved_tables/updated_wt_tmscore_matrix.npy', tmscore_matrix)
    print("Matrices saved!")