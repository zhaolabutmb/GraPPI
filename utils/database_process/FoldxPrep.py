def convert_mutation_to_foldx_format(mutation_str:str) -> list[str]:
    """
    Convert mutation format from 'I_L38G' to 'LI38G' (FoldX format)
    Handles multiple mutations like 'A_A27K, D_A43T' or 'A_A27K,D_A43T'
    
    Returns: list of mutations in FoldX format
    """
    # Remove spaces and split by comma
    mutations = [m.strip() for m in mutation_str.split(',')]
    
    foldx_mutations = []
    for mut in mutations:
        # Format: Chain_OldAAPositionNewAA -> ChainOldAAPositionNewAA
        parts = mut.split('_')
        if len(parts) == 2:
            chain = parts[0]
            aa_mutation = parts[1]  # e.g., L38G
            # FoldX format: ChainOldAAPositionNewAA
            foldx_mut = aa_mutation[0]+ chain + aa_mutation[1:]
            foldx_mutations.append(foldx_mut)
    
    return foldx_mutations
def create_individual_list(mutations_list, output_file):
    """
    Create the individual_list file required by FoldX.
    Each mutation case should end with ';'
    Multiple simultaneous mutations are separated by ','
    
    Args:
        mutations_list: list of mutation strings (each can contain multiple mutations)
        output_file: path to save the individual_list file
    """
    with open(output_file, 'w') as f:
        for mutation_str in mutations_list:
            foldx_muts = convert_mutation_to_foldx_format(mutation_str)
            # Join simultaneous mutations with comma
            line = ','.join(foldx_muts) + ';\n'
            f.write(line)