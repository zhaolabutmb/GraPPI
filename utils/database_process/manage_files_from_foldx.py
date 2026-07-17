import os
def rename_mutant_pdbs(mutations_df, working_dir, only_remove_wt=True):
    """
    Rename FoldX output files to the format: pdb_chainid_mutinfo.pdb
    
    Args:
        pdb_code: PDB code (e.g., '1CSE')
        mutations_df: dataframe subset containing mutation info
    """
    pdb_code = mutations_df['PDB'].tolist()[0]
    pdb_lower = pdb_code.lower() # get pdb code in lower case
    # Change to working directory
    original_dir = os.getcwd()
    os.chdir(working_dir)

    # FoldX output format is typically: PDBName_1.pdb, PDBName_2.pdb, etc.
    idx = 1
    for _, row in mutations_df.iterrows():
        if not only_remove_wt:
            mutation_str = row['Mutations']
            # Parse chain info and mutation info from mutation string
            # Example: 'I_L38G' -> chain='I', mut_info='L38G'
            mutations = [m.strip() for m in mutation_str.split(',')]
            mut_description = '-'.join(mutations)
            foldx_output = f"{pdb_code}_{idx}.pdb"  # Adjust based on actual FoldX output
            # New filename
            new_filename = f"{pdb_lower}_{mut_description}.pdb"
            
            # Rename if file exists
            if os.path.exists(foldx_output):
                os.rename(foldx_output, new_filename)
                #print(f"Renamed: {foldx_output} -> {new_filename}")
            else:
                #print(f"Error: Expected file {foldx_output} not found")
                raise FileNotFoundError(f"Expected file {foldx_output} not found")
        # Check and remove WT file
        if os.path.exists(f'WT_{pdb_code}_{idx}.pdb'):
            os.remove(f'WT_{pdb_code}_{idx}.pdb')
        
        idx += 1
    # Return to original directory
    os.chdir(original_dir)