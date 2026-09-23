import os
import sys
import pickle
import json
from datetime import datetime
from tqdm.contrib.concurrent import process_map
import pandas as pd

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.gen_graphs_unified import (
    StructureToHeteroGraph,
    NoInterfaceError,
    OversizedError,
)

# =====================================================================
# Configuration
# =====================================================================

# --- Processing mode (mutually exclusive flags) ---
# if_mut        : apply feature-space mutations from a pre-built mutant table
# if_random_mut : generate random interface mutations on-the-fly from a WT table
if_mut = False
if_random_mut = True
num_mut_on_each = 5   # mutations per side when if_random_mut=True

assert not (if_mut and if_random_mut), "if_mut and if_random_mut are mutually exclusive"
max_residues=2000
# Load data tables
if if_mut:
    process_df = pd.read_csv('./saved_tables/mutant_table.csv')
    save_path_hg = '../../GraPPI_data/curated_db/mutant_sthg_8A'
    pdb_root_path = '../../GraPPI_data/'
elif if_random_mut:
    random_seed = 42
    process_df = pd.read_csv('./saved_tables/dimer_table.csv')
    save_path_hg = '../../GraPPI_data/curated_db/random_mut_sthg_8A'
    pdb_root_path = ''
    max_residues=1200
else:
    #wt_df = pd.read_csv('./saved_tables/wt_table.csv') # ~ 3k
    #abet_df = pd.read_csv('./saved_tables/AbEpiTope_table.csv') # ~ 0.3k
    #golden_df = pd.read_csv('./saved_tables/dimer_table.csv') # ~ 8.69k
    #swap_df = pd.read_csv('./saved_tables/swapped_table.csv') # ~ 3.0k
    process_df = pd.read_csv('./saved_tables/preppi_db.csv') # ~ 15k
    #df = pd.concat([wt_df, golden_df, abet_df, swap_df], ignore_index=True)
    save_path_hg = '../../GraPPI_data/curated_db/preppi_sthg_8A'
    pdb_root_path = ''
#save_path_3g = '../../curated_db/all_sttgs_8A'

# Log files for tracking
log_dir = './processing_logs'
skipped_log_path = os.path.join(log_dir, 'skipped_no_interface.jsonl')
error_log_path = os.path.join(log_dir, 'processing_errors.jsonl')

# =====================================================================
# Processing Functions
# =====================================================================

def make_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def log_skipped(pdb_name, reason, interface_info=None):
    """Log a skipped PDB to the skip log file."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'pdb_name': pdb_name,
        'reason': reason,
        'interface_info': make_json_serializable(interface_info)
    }
    with open(skipped_log_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def log_error(pdb_name, error_type, error_message):
    """Log an error to the error log file."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'pdb_name': pdb_name,
        'error_type': error_type,
        'error_message': error_message
    }
    with open(error_log_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def load_process_pdb_save_indiv(ind_row_of_df):
    """
    Process a single PDB and save the graphs.
    
    Handles non-interacting complexes gracefully by skipping them
    and logging the skip.
    
    Args:
        ind_row_of_df: Tuple of (index, row_of_df) from DataFrame.iterrows()
        
    Returns:
        Dict with processing status information
    """
    global save_path_hg, if_mut, if_random_mut, num_mut_on_each, pdb_root_path
    
    idx, row_of_df = ind_row_of_df
    
    # Determine filename and mutation mode
    if if_mut:
        filename = row_of_df['new_pdb_name'].lower()
        mutations = row_of_df['Mutations']
    elif if_random_mut:
        filename = row_of_df["paths"].split("/")[-1].replace(".pdb", "").lower() + '_mut'
        mutations = None
    else:
        filename = row_of_df["paths"].split("/")[-1].replace(".pdb", "").lower()
        mutations = None
    
    # Output paths
    hg_output_path = f'{save_path_hg}/{filename}.pkl'
    
    result = {
        'filename': filename,
        'triple_graphs': 'skipped',
        'hetero_graph': 'skipped',
        'has_interface': True,
        'error': None
    }
    
    # ===== Process HeteroGraph =====
    if not os.path.exists(hg_output_path):
        try:
            sthg = StructureToHeteroGraph(
                aa_feature_path='../stored_data/AA_At_dict.json'
            )
            sthg.process_pdb(row_of_df, root_path=pdb_root_path, mutations=mutations,
                             if_random_mut=if_random_mut, num_mut_on_each=num_mut_on_each,
                             random_seed=random_seed, max_residues=max_residues)

            sthg.get_hetero_graph(distance_threshold=8, validate_interface=True)
            
            with open(hg_output_path, 'wb') as f:
                pickle.dump(sthg, f)
            
            result['hetero_graph'] = 'success'
            
        except NoInterfaceError as e:
            result['hetero_graph'] = 'no_interface'
            result['has_interface'] = False
            result['error'] = str(e)
            log_skipped(filename, 'NoInterfaceError (HeteroGraph)',
                       sthg.interface_info if hasattr(sthg, 'interface_info') else None)

        except OversizedError as e:
            result['hetero_graph'] = 'oversized'
            result['error'] = str(e)

        except Exception as e:
            result['hetero_graph'] = 'error'
            result['error'] = str(e)
            log_error(filename, type(e).__name__, str(e))
    else:
        result['hetero_graph'] = 'exists'
    
    return result


def mp_process(df, n_processor=8):
    """
    Main multiprocess function to process all PDBs in the dataframe.
    
    Args:
        n_processor: Number of parallel processes to use
    """
    # Create output directories
    os.makedirs(save_path_hg, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Processing {len(df)} PDB structures...")
    print(f"Output paths:")
    print(f"  HeteroGraphs: {save_path_hg}")
    print(f"  Logs: {log_dir}")
    print()
    
    items = list(df.iterrows())
    
    # Process in parallel
    results = process_map(
        load_process_pdb_save_indiv, 
        items, 
        max_workers=n_processor, 
        chunksize=4
    )
    
    # Summarize results
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    
    summary = {
        'hetero_graph': {'success': 0, 'exists': 0, 'no_interface': 0,
                         'oversized': 0, 'error': 0, 'skipped': 0}
    }
    
    no_interface_list = []
    
    for r in results:
        summary['hetero_graph'][r['hetero_graph']] += 1
        
        if not r['has_interface']:
            no_interface_list.append(r['filename'])
    
    
    print("\nHetero Graphs:")
    for status, count in summary['hetero_graph'].items():
        if count > 0:
            print(f"  {status}: {count}")
    
    if no_interface_list:
        print(f"\nStructures with no interface ({len(no_interface_list)} total):")
        for name in no_interface_list[:10]:  # Show first 10
            print(f"  - {name}")
        if len(no_interface_list) > 10:
            print(f"  ... and {len(no_interface_list) - 10} more (see {skipped_log_path})")
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)


if __name__ == '__main__':
    mp_process(n_processor=20, df=process_df)
