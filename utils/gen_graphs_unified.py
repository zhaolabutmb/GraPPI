import pandas as pd
import torch
from Bio.PDB import PDBParser
import json
import numpy as np
from torch_geometric.data import HeteroData, Data
from itertools import combinations

# ==============================================================================
# Constants
# ==============================================================================

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}
_ALL_AA_1 = list(THREE_TO_ONE.values())  # All 20 standard one-letter amino acid codes

# ==============================================================================
# Exception Classes
# ==============================================================================

class NoInterfaceError(Exception):
    """Raised when no interface residues are found between receptor and ligand."""
    pass

class InsufficientInterfaceError(Exception):
    """Raised when interface has too few residues to form meaningful edges."""
    pass

class OversizedError(Exception):
    """Raised when a complex exceeds the maximum allowed residue count."""
    pass

# ==============================================================================
# Common Utility Functions
# ==============================================================================

def load_aa_dict(aa_feature_path):
    """Load amino acid feature dictionary from JSON file."""
    with open(aa_feature_path, 'r') as file:
        return json.load(file)


def parse_chain_ids(chain_str):
    """Parse comma-separated chain IDs into a list."""
    return [x.strip() for x in chain_str.split(',')]


def parse_mutations(mutation_str):
    """
    Parse mutation string into a lookup dict.
    
    Args:
        mutation_str: Comma-separated mutations, e.g. 'A_D24E, B_N31A'
            Format per mutation: {Chain}_{WtAA}{Position}{MutAA}
            
    Returns:
        Dict mapping (chain_letter, res_seq) -> (wt_letter, mut_3letter)
    """
    import re
    mutation_lookup = {}
    for mut in mutation_str.split(','):
        mut = mut.strip()
        m = re.match(r'^([A-Z])_([A-Z])(\d+)([A-Z])$', mut)
        if m:
            chain, wt_letter, pos, mut_letter = m.groups()
            mutation_lookup[(chain, int(pos))] = (wt_letter, ONE_TO_THREE[mut_letter])
        else:
            raise ValueError(f"Cannot parse mutation string: '{mut}'")
    return mutation_lookup


def compute_residue_pair_distance(coords1, coords2):
    """
    Compute pairwise distance matrix between two sets of atomic coordinates.
    
    Args:
        coords1: List of [x, y, z] coordinates for residue 1
        coords2: List of [x, y, z] coordinates for residue 2
        
    Returns:
        torch.Tensor: Distance matrix between all atom pairs
    """
    return torch.cdist(
        torch.tensor(coords1, dtype=torch.float32),
        torch.tensor(coords2, dtype=torch.float32)
    )


def compute_edge_features(coords_i, coords_j, dist_matrix, hist_min=0, hist_max=15, hist_bins=5):
    """
    Compute edge features including distance histogram and unit direction vector.
    
    Args:
        coords_i: Atomic coordinates of residue i
        coords_j: Atomic coordinates of residue j
        dist_matrix: Pre-computed distance matrix between residues
        hist_min: Minimum value for histogram
        hist_max: Maximum value for histogram
        hist_bins: Number of histogram bins
        
    Returns:
        Tuple of (forward_edge_feature, reverse_edge_feature)
    """
    # Compute distance histogram
    dist_hist = torch.histc(
        dist_matrix.ravel(), 
        min=hist_min, 
        max=hist_max, 
        bins=hist_bins
    ) / len(dist_matrix.ravel())
    
    # Compute unit direction vector from centroid i to centroid j
    centroid_i = torch.tensor(coords_i, dtype=torch.float32).mean(dim=0)
    centroid_j = torch.tensor(coords_j, dtype=torch.float32).mean(dim=0)
    vec = centroid_j - centroid_i
    unit_vec = vec / torch.linalg.norm(vec)
    
    # Forward edge: i -> j
    forward_edge = torch.cat([dist_hist, unit_vec])
    # Reverse edge: j -> i
    reverse_edge = torch.cat([dist_hist, -unit_vec])
    
    return forward_edge, reverse_edge


def compute_node_features(res_name, chain_type, aa_dict):
    """
    Compute node features for a residue.
    
    Args:
        res_name: Three-letter amino acid name
        chain_type: 'receptor' or 'ligand' (or 'R'/'L')
        aa_dict: Amino acid feature dictionary
        
    Returns:
        torch.Tensor: Node feature vector
    """
    res_map = {res: i for i, res in enumerate(aa_dict.keys())}
    
    # One-hot encoding (20 amino acids)
    one_hot = torch.zeros(20)
    if res_name in res_map:
        one_hot[res_map[res_name]] = 1.0
    
    # Receptor/Ligand indicator
    is_receptor = chain_type in ['receptor', 'R', 'Receptor']
    one_hot_rl = torch.tensor([1.0, 0.0]) if is_receptor else torch.tensor([0.0, 1.0])
    
    # Dictionary features
    dict_feats = torch.tensor(aa_dict[res_name]['features'], dtype=torch.float32)
    
    return torch.cat([one_hot, one_hot_rl, dict_feats])


def check_interface_exists(receptor_df, ligand_df, distance_threshold=8):
    """
    Check if any interface residues exist between receptor and ligand.
    
    Args:
        receptor_df: DataFrame with receptor residue information
        ligand_df: DataFrame with ligand residue information
        distance_threshold: Distance cutoff for interface definition
        
    Returns:
        Tuple of (has_interface, receptor_interface_count, ligand_interface_count)
    """
    receptor_interface_indices = set()
    ligand_interface_indices = set()
    
    for idx, row_r in receptor_df.iterrows():
        for jdx, row_l in ligand_df.iterrows():
            dist_matrix = compute_residue_pair_distance(
                row_r['atom_coords'], 
                row_l['atom_coords']
            )
            if (dist_matrix <= distance_threshold).any():
                receptor_interface_indices.add(idx)
                ligand_interface_indices.add(jdx)
    
    has_interface = len(receptor_interface_indices) > 0 and len(ligand_interface_indices) > 0
    return has_interface, len(receptor_interface_indices), len(ligand_interface_indices)


# ==============================================================================
# Common PDB Processing Mixin
# ==============================================================================

class PDBProcessorMixin:
    """Mixin class providing common PDB processing functionality."""
    
    def _parse_pdb_to_residue_df(self, pdb_path, receptor_chains, ligand_chains, aa_dict, 
                                  separate_dfs=False, mutations=None):
        """
        Parse PDB file and extract residue information into DataFrame(s).
        
        Args:
            pdb_path: Path to PDB file
            receptor_chains: List of receptor chain IDs
            ligand_chains: List of ligand chain IDs
            aa_dict: Amino acid feature dictionary
            separate_dfs: If True, return separate DataFrames for receptor and ligand
            mutations: Optional mutation lookup dict from parse_mutations().
                       If provided, node features at mutation sites are swapped
                       to the mutant residue type while keeping WT coordinates.
            
        Returns:
            If separate_dfs=False: Single DataFrame with 'chain_type' column
            If separate_dfs=True: Dict with 'Receptor' and 'Ligand' DataFrames
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)[0]
        aa_dict_list = list(aa_dict.keys())
        
        # Track which mutations were applied (for validation)
        if mutations:
            mutations_applied = set()
        
        if separate_dfs:
            AA_dfs = {
                x: {
                    'res_name': [], 'res_letter': [], 'chain_id': [], 'res_seq': [],
                    'atom_names': [], 'atom_coords': []
                } 
                for x in ['Receptor', 'Ligand']
            }
        else:
            AA_df = {
                'res_name': [], 'res_letter': [], 'chain_id': [], 'chain_letter': [], 'res_seq': [],
                'atom_names': [], 'atom_coords': [], 'chain_type': []
            }
        
        chain_count = 0
        for chain in structure:
            chain_id = chain.get_id()
            if chain_id in receptor_chains:
                chain_type = 'Receptor' if separate_dfs else 'receptor'
            elif chain_id in ligand_chains:
                chain_type = 'Ligand' if separate_dfs else 'ligand'
            else:
                chain_count += 1
                continue
            
            for residue in chain.get_residues():
                if residue.get_id()[0] == ' ':  # Standard residue
                    resname = residue.get_resname()
                    if resname == 'HIE':
                        resname = 'HIS'
                    
                    if resname in aa_dict_list:
                        # Apply mutation if this site is in the mutation lookup
                        if mutations:
                            res_seq = residue.get_id()[1]
                            mut_key = (chain_id, res_seq)
                            if mut_key in mutations:
                                wt_expected, mut_3letter = mutations[mut_key]
                                if THREE_TO_ONE[resname] != wt_expected:
                                    import warnings
                                    warnings.warn(
                                        f"Mutation WT mismatch at chain {chain_id} pos {res_seq}: "
                                        f"expected {wt_expected}, found {THREE_TO_ONE[resname]}. "
                                        f"Skipping this mutation."
                                    )
                                else:
                                    resname = mut_3letter
                                    mutations_applied.add(mut_key)
                        heavy_atoms, coords = [], []
                        for atom in residue.get_atoms():
                            if atom.get_name()[0] != 'H':  # Heavy atoms only
                                atom_name = 'O' if atom.get_name() == 'OXT' else atom.get_name()
                                heavy_atoms.append(atom_name)
                                coords.append(atom.get_coord().tolist())
                        
                        if separate_dfs:
                            AA_dfs[chain_type]['res_name'].append(resname)
                            AA_dfs[chain_type]['res_letter'].append(THREE_TO_ONE[resname])
                            AA_dfs[chain_type]['chain_id'].append(chain_id)
                            AA_dfs[chain_type]['res_seq'].append(residue.get_id()[1])
                            AA_dfs[chain_type]['atom_names'].append(heavy_atoms)
                            AA_dfs[chain_type]['atom_coords'].append(coords)
                        else:
                            AA_df['res_name'].append(resname)
                            AA_df['res_letter'].append(THREE_TO_ONE[resname])
                            AA_df['chain_id'].append(chain_count)
                            AA_df['chain_letter'].append(chain_id)
                            AA_df['res_seq'].append(residue.get_id()[1])
                            AA_df['atom_names'].append(heavy_atoms)
                            AA_df['atom_coords'].append(coords)
                            AA_df['chain_type'].append(chain_type)
            
            chain_count += 1
        
        # Warn if any mutations were not found in the PDB
        if mutations:
            mutations_not_found = set(mutations.keys()) - mutations_applied
            if mutations_not_found:
                import warnings
                warnings.warn(
                    f"Mutations not found in PDB {pdb_path}: "
                    f"{[(k[0], k[1], mutations[k]) for k in mutations_not_found]}"
                )
        
        if separate_dfs:
            AA_dfs['Receptor'] = pd.DataFrame(AA_dfs['Receptor'])
            AA_dfs['Ligand'] = pd.DataFrame(AA_dfs['Ligand'])
            AA_dfs['Receptor']['part'] = 'R'
            AA_dfs['Ligand']['part'] = 'L'
            AA_dfs['Receptor']['if_interface'] = False
            AA_dfs['Ligand']['if_interface'] = False
            return AA_dfs
        else:
            return pd.DataFrame(AA_df)


# ==============================================================================
# StructureToHeteroGraph Class (Unified)
# ==============================================================================

class StructureToHeteroGraph(PDBProcessorMixin):
    """
    Convert PDB to heterogeneous graph with receptor and ligand as separate node types.
    
    Includes interface validation to detect non-interacting complexes.
    """
    
    def __init__(self, aa_feature_path='../stored_data/AA_At_dict.json'):
        self.AA_df = None
        self.protein_graph = None
        self.row_of_df = None
        self.has_interface = True  # Flag for interface existence
        self.interface_info = None  # Store interface statistics
        self.mut_info = None  # Mutation string when if_random_mut=True
        self.if_random_mut = False
        self.AA_dict = load_aa_dict(aa_feature_path)

    def process_pdb(self, row_of_df, root_path='../../GraPPI_data/',
                     mutations=None, if_random_mut=False, num_mut_on_each=5,
                     random_seed=42, dist_cutoff=8, max_residues=2000):
        """
        Process PDB file and extract residue information.

        Args:
            row_of_df: DataFrame row with PDB info (paths, Receptor Chains, Ligand Chains)
            root_path: Root path prefix for PDB files
            mutations: Optional mutation string (e.g. 'A_D24E, B_N31A').
                       Applies mutations in feature space during parsing.
                       Mutually exclusive with if_random_mut.
            if_random_mut: If True, randomly select interface residues and assign
                           random mutations on-the-fly. Mutually exclusive with mutations.
                           Stores the generated mutation string in self.mut_info.
            num_mut_on_each: Number of mutations per side (receptor and ligand)
                             when if_random_mut=True. Raises NoInterfaceError if
                             either side has fewer interface residues than this.
            random_seed: Random seed for reproducibility when if_random_mut=True.
            dist_cutoff: Distance cutoff (Å) for interface detection when if_random_mut=True.
        """
        if mutations and if_random_mut:
            raise ValueError("mutations and if_random_mut are mutually exclusive.")

        pdb_path = root_path + row_of_df['paths']
        receptor_chains = parse_chain_ids(row_of_df['Receptor Chains'])
        ligand_chains = parse_chain_ids(row_of_df['Ligand Chains'])

        mutation_lookup = parse_mutations(mutations) if mutations else None

        self.AA_df = self._parse_pdb_to_residue_df(
            pdb_path, receptor_chains, ligand_chains, self.AA_dict,
            separate_dfs=False, mutations=mutation_lookup
        )
        self.row_of_df = row_of_df
        self.if_random_mut = if_random_mut
        self.n_residues = len(self.AA_df)

        if max_residues is not None and self.n_residues > max_residues:
            raise OversizedError(
                f"{self.row_of_df.get('PDB', 'unknown')} has {self.n_residues} residues "
                f"(max allowed: {max_residues})"
            )

        if if_random_mut:
            self._apply_random_mutations(num_mut_on_each, random_seed, dist_cutoff)

    def _apply_random_mutations(self, num_mut_on_each, random_seed, dist_cutoff):
        """
        Identify interface residues, randomly select num_mut_on_each per side,
        apply random mutations in-place to self.AA_df, and store the mutation
        string in self.mut_info.

        Raises:
            NoInterfaceError: If no interface exists or either side has fewer
                              than num_mut_on_each interface residues.
        """
        df = self.AA_df
        receptor_df = df[df['chain_type'] == 'receptor']
        ligand_df = df[df['chain_type'] == 'ligand']

        # Identify interface residues
        receptor_iface_indices = set()
        ligand_iface_indices = set()

        for ridx, rrow in receptor_df.iterrows():
            for lidx, lrow in ligand_df.iterrows():
                dist_matrix = compute_residue_pair_distance(
                    rrow['atom_coords'], lrow['atom_coords']
                )
                if (dist_matrix <= dist_cutoff).any():
                    receptor_iface_indices.add(ridx)
                    ligand_iface_indices.add(lidx)

        if not receptor_iface_indices or not ligand_iface_indices:
            self.has_interface = False
            raise NoInterfaceError(
                f"No interface residues found for {self.row_of_df.get('PDB', 'unknown')} - "
                f"receptor and ligand are not interacting within {dist_cutoff}\u00c5 threshold"
            )

        receptor_iface = df.loc[sorted(receptor_iface_indices)]
        ligand_iface = df.loc[sorted(ligand_iface_indices)]

        if len(receptor_iface) < num_mut_on_each or len(ligand_iface) < num_mut_on_each:
            raise NoInterfaceError(
                f"Insufficient interface residues for {self.row_of_df.get('PDB', 'unknown')} - "
                f"receptor: {len(receptor_iface)}, ligand: {len(ligand_iface)}, "
                f"required: {num_mut_on_each} per side"
            )

        rng = np.random.default_rng(random_seed)
        mut_parts = []

        for side_df in [receptor_iface, ligand_iface]:
            chosen = rng.choice(len(side_df), size=num_mut_on_each, replace=False)
            selected = side_df.iloc[sorted(chosen)]

            for idx, res in selected.iterrows():
                chain = res['chain_letter']
                pos = res['res_seq']
                wt = res['res_letter']
                mut_options = [aa for aa in _ALL_AA_1 if aa != wt]
                mut = rng.choice(mut_options)
                # Apply mutation in-place
                self.AA_df.at[idx, 'res_name'] = ONE_TO_THREE[mut]
                self.AA_df.at[idx, 'res_letter'] = mut
                mut_parts.append(f"{chain}_{wt}{pos}{mut}")

        self.mut_info = ','.join(mut_parts)

    def validate_interface(self, distance_threshold=8):
        """
        Validate that interface residues exist between receptor and ligand.
        
        Args:
            distance_threshold: Distance cutoff for interface definition
            
        Returns:
            bool: True if interface exists, False otherwise
        """
        df = self.AA_df
        receptor_df = df[df['chain_type'] == 'receptor'].reset_index(drop=True)
        ligand_df = df[df['chain_type'] == 'ligand'].reset_index(drop=True)
        
        has_interface, r_count, l_count = check_interface_exists(
            receptor_df, ligand_df, distance_threshold
        )
        
        self.has_interface = has_interface
        self.interface_info = {
            'receptor_interface_count': r_count,
            'ligand_interface_count': l_count,
            'has_interface': has_interface
        }
        
        return has_interface

    def get_hetero_graph(self, distance_threshold=8, validate_interface=True):
        """
        Creates a heterogeneous graph from the protein complex DataFrame.
        
        Args:
            distance_threshold: Distance cutoff for edge creation
            validate_interface: If True, validate interface before processing
            
        Returns:
            HeteroData object or None if no interface exists
            
        Raises:
            NoInterfaceError: If validate_interface=True and no interface exists
        """
        df = self.AA_df
        aa_dict = self.AA_dict
        res_map = {res: i for i, res in enumerate(aa_dict.keys())}

        # === 1. SPLIT NODES BY TYPE ===
        receptor_features, ligand_features = [], []
        receptor_idx_map, ligand_idx_map = {}, {}

        for idx, row in df.iterrows():
            feat = compute_node_features(row['res_name'], row['chain_type'], aa_dict)

            if row['chain_type'] == 'receptor':
                local_idx = len(receptor_features)
                receptor_idx_map[idx] = local_idx
                receptor_features.append(feat)
            else:  # ligand
                local_idx = len(ligand_features)
                ligand_idx_map[idx] = local_idx
                ligand_features.append(feat)

        receptor_features = torch.stack(receptor_features)
        ligand_features = torch.stack(ligand_features)

        # === 2. EDGE LISTS ===
        receptor_receptor_edges, receptor_receptor_attrs = [], []
        ligand_ligand_edges, ligand_ligand_attrs = [], []
        receptor_ligand_edges, receptor_ligand_attrs = [], []
        ligand_receptor_edges, ligand_receptor_attrs = [], []

        # Precompute centroids
        centroids_all = np.array([np.mean(coords, axis=0) for coords in df['atom_coords']])

        for i, j in combinations(range(len(df)), 2):
            residue_pair_dist_mtx = compute_residue_pair_distance(
                df.iloc[i]['atom_coords'],
                df.iloc[j]['atom_coords']
            )
            
            if (residue_pair_dist_mtx <= distance_threshold).any():
                forward_edge, reverse_edge = compute_edge_features(
                    df.iloc[i]['atom_coords'],
                    df.iloc[j]['atom_coords'],
                    residue_pair_dist_mtx
                )

                if df['chain_type'].iloc[i] == 'receptor' and df['chain_type'].iloc[j] == 'receptor':
                    src = receptor_idx_map[i]
                    dst = receptor_idx_map[j]
                    receptor_receptor_edges.append([src, dst])
                    receptor_receptor_edges.append([dst, src])
                    receptor_receptor_attrs.append(forward_edge)
                    receptor_receptor_attrs.append(reverse_edge)

                elif df['chain_type'].iloc[i] == 'ligand' and df['chain_type'].iloc[j] == 'ligand':
                    src = ligand_idx_map[i]
                    dst = ligand_idx_map[j]
                    ligand_ligand_edges.append([src, dst])
                    ligand_ligand_edges.append([dst, src])
                    ligand_ligand_attrs.append(forward_edge)
                    ligand_ligand_attrs.append(reverse_edge)

                elif df['chain_type'].iloc[i] == 'receptor' and df['chain_type'].iloc[j] == 'ligand':
                    src = receptor_idx_map[i]
                    dst = ligand_idx_map[j]
                    receptor_ligand_edges.append([src, dst])
                    receptor_ligand_attrs.append(forward_edge)
                    ligand_receptor_edges.append([dst, src])
                    ligand_receptor_attrs.append(reverse_edge)
                        
                else:  # i is ligand, j is receptor
                    src = ligand_idx_map[i]
                    dst = receptor_idx_map[j]
                    ligand_receptor_edges.append([src, dst])
                    ligand_receptor_attrs.append(forward_edge)
                    receptor_ligand_edges.append([dst, src])
                    receptor_ligand_attrs.append(reverse_edge)

        # === 3. VALIDATE INTERFACE ===
        if validate_interface and not receptor_ligand_edges:
            self.has_interface = False
            self.interface_info = {
                'receptor_interface_count': 0,
                'ligand_interface_count': 0,
                'has_interface': False
            }
            raise NoInterfaceError(
                f"No interface edges found for {self.row_of_df.get('PDB', 'unknown')} - "
                f"receptor and ligand are not interacting within {distance_threshold}Å threshold"
            )

        # === 4. BUILD HETERODATA ===
        graph = HeteroData()
        graph['ligand'].x = ligand_features
        graph['receptor'].x = receptor_features
        graph.y = torch.tensor(0.0) if self.if_random_mut else torch.tensor(self.row_of_df['affinity'])
        if self.if_random_mut and self.mut_info is not None:
            graph.mut_info = self.mut_info

        if 'new_pdb_name' in self.row_of_df and pd.notna(self.row_of_df['new_pdb_name']):
            graph.pdb_name = self.row_of_df['new_pdb_name'].lower()
        else:
            graph.pdb_name = self.row_of_df['PDB'].lower()

        if receptor_receptor_edges:
            graph['receptor', 'receptor_receptor', 'receptor'].edge_index = torch.tensor(
                receptor_receptor_edges, dtype=torch.long).t().contiguous()
            graph['receptor', 'receptor_receptor', 'receptor'].edge_attr = torch.stack(receptor_receptor_attrs)
        
        if ligand_ligand_edges:
            graph['ligand', 'ligand_ligand', 'ligand'].edge_index = torch.tensor(
                ligand_ligand_edges, dtype=torch.long).t().contiguous()
            graph['ligand', 'ligand_ligand', 'ligand'].edge_attr = torch.stack(ligand_ligand_attrs)

        if receptor_ligand_edges:
            graph['receptor', 'receptor_ligand', 'ligand'].edge_index = torch.tensor(
                receptor_ligand_edges, dtype=torch.long).t().contiguous()
            graph['receptor', 'receptor_ligand', 'ligand'].edge_attr = torch.stack(receptor_ligand_attrs)

        if ligand_receptor_edges:
            graph['ligand', 'ligand_receptor', 'receptor'].edge_index = torch.tensor(
                ligand_receptor_edges, dtype=torch.long).t().contiguous()
            graph['ligand', 'ligand_receptor', 'receptor'].edge_attr = torch.stack(ligand_receptor_attrs)

        self.protein_graph = graph
        self.has_interface = True
        self.interface_info = {
            'receptor_interface_count': len(set(e[0] for e in receptor_ligand_edges)),
            'ligand_interface_count': len(set(e[1] for e in receptor_ligand_edges)),
            'has_interface': True
        }
        
        return graph

# ==============================================================================
# Wrapper functions for backwards compatibility
# ==============================================================================

def create_hetero_graph_safe(row_of_df, root_path='', distance_threshold=8, 
                             aa_feature_path='../stored_data/AA_At_dict.json'):
    """
    Safely create heterogeneous graph with interface validation.
    
    Returns:
        Tuple of (graph, has_interface, interface_info)
        - graph: HeteroData object or None if no interface
        - has_interface: Boolean indicating if interface exists
        - interface_info: Dict with interface statistics
    """
    sthg = StructureToHeteroGraph(aa_feature_path=aa_feature_path)
    sthg.process_pdb(row_of_df, root_path=root_path)
    
    try:
        graph = sthg.get_hetero_graph(distance_threshold=distance_threshold, validate_interface=True)
        return graph, True, sthg.interface_info
    except NoInterfaceError:
        return None, False, sthg.interface_info