import os
import subprocess

def run_foldx(pdb_path, individual_list, 
              foldx_executable, working_dir):
    """
    Run FoldX BuildModel command
    
    Args:
        pdb_path: path to the wild type PDB file
        individual_list: path to the mutation list file
        foldx_executable: path to foldx executable
        working_dir: directory to run foldx in
    """
    pdb_path = os.path.abspath(pdb_path)
    individual_list = os.path.abspath(individual_list)
    foldx_executable = os.path.abspath(foldx_executable)
    if 'ATLAS' in pdb_path:
        pdb_id = os.path.basename(pdb_path).lower().split()('.')[0]
        pdb_name = pdb_id + '_at' + '.pdb'
    else:
        pdb_name = os.path.basename(pdb_path).lower()
    
    # Copy PDB to working directory
    dest_pdb = os.path.join(working_dir, pdb_name)
    if not os.path.exists(dest_pdb):
        import shutil
        shutil.copy(pdb_path, dest_pdb)
    
    # Run FoldX with cwd set to working_dir
    cmd = [
        foldx_executable,
        '--command=BuildModel',
        f'--pdb={pdb_name}',
        f'--mutant-file={individual_list}',
        '--numberOfRuns=1'
    ]
    
    #print(f"Running FoldX command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=working_dir, capture_output=True, text=True)
    
    if result.returncode != 0:
        #print(f"FoldX error: {result.stderr}")
        raise RuntimeError("FoldX execution failed")
    #else:
    #    print("FoldX completed successfully")
    #    print(result.stdout)
        
    return result