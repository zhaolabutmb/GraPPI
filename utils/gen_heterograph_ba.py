import pandas as pd
import torch
from Bio.PDB import PDBParser
import json
import numpy as np
from torch_geometric.data import HeteroData
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
def rbf_expansion(distances, k, mu=0, delta=1):
    rbf = torch.exp(-((distances.unsqueeze(-1) - torch.arange(mu, k*delta, delta))**2) / (2*delta**2))
    return rbf

class StructureToHeteroGraph:
    """
    Convert PDB to hetero-graph with receptor and ligand as separate node types.
    """
    def __init__(self, aa_feature_path='../stored_data/AA_At_dict.json'):
        self.AA_df = None
        self.protein_graph = None
        self.row_of_df = None
        with open(aa_feature_path, 'r') as file:
            self.AA_dict = json.load(file)

    def process_pdb(self, row_of_df, root_path='/home/zhisong/projects/EntroPPI_files/'):
        parser = PDBParser(QUIET=True)
        if 'mut_paths' in row_of_df and pd.notna(row_of_df['mut_paths']):
            structure = parser.get_structure('protein', root_path+row_of_df['mut_paths'])[0]
        else:
            structure = parser.get_structure('protein', root_path+row_of_df['paths'])[0]
        AA_dict = self.AA_dict
        AA_dict_list = list(AA_dict.keys())
        AA_df = {'res_name': [], 'res_letter': [], 'chain_id': [], 'atom_names': [], 'atom_coords': [], 'chain_type': []}
        atom_coords = []
        chain_count = 0
        receptor_chains = [x.strip() for x in row_of_df['Receptor Chains'].split(',')]
        ligand_chains = [x.strip() for x in row_of_df['Ligand Chains'].split(',')]
        # === Build AA_df and collect atomic coordinates ===
        for chain in structure:
            if chain.get_id() not in (receptor_chains + ligand_chains):
                continue
            for residue in chain.get_residues():
                if residue.get_id()[0] == ' ':
                    resname = residue.get_resname()
                    if resname == 'HIE':
                        resname = 'HIS'  # remap HIE to HIS
                    if resname in AA_dict_list:
                        heavy_atoms, bfactor, coords = [], [], []
                        for atom in residue.get_atoms():
                            if atom.get_name()[0] != 'H':
                                heavy_atoms.append('O' if atom.get_name() == 'OXT' else atom.get_name())
                                bfactor.append(atom.get_bfactor())
                                coords.append(atom.get_coord().tolist())
                        AA_df['res_name'].append(resname)
                        AA_df['res_letter'].append(three_to_one[resname])
                        AA_df['chain_id'].append(chain_count)
                        AA_df['atom_names'].append(heavy_atoms)
                        AA_df['atom_coords'].append(coords)
                        AA_df['chain_type'].append('receptor' if chain.get_id() in receptor_chains else 'ligand')
                        atom_coords.extend([atom.get_coord().tolist() for atom in residue.get_atoms() if atom.get_name()[0] != 'H'])
            chain_count += 1
        self.AA_df = pd.DataFrame(AA_df)
        self.row_of_df = row_of_df

    def get_hetero_graph(self, distance_threshold=8):
        """
        Creates a heterogeneous graph from the protein complex DataFrame.
        """
        df = self.AA_df
        aa_dict = self.AA_dict
        res_map = {res: i for i, res in enumerate(aa_dict.keys())}

        # === 1. SPLIT NODES BY TYPE ===
        receptor_features, ligand_features = [], []
        receptor_idx_map, ligand_idx_map = {}, {}
        #centroids_receptor, centroids_ligand = [], []

        for idx, row in df.iterrows():
            # one-hot + dict features
            one_hot = torch.zeros(20)
            one_hot[res_map[row['res_name']]] = 1.0
            one_hot_R_or_L = torch.tensor([1.0, 0.0]) if df.iloc[idx]['chain_type'] == 'receptor' else torch.tensor([0.0, 1.0])
            dict_feats = torch.tensor(aa_dict[row['res_name']]['features'], dtype=torch.float32)
            feat = torch.cat([one_hot, one_hot_R_or_L, dict_feats])

            #coords = np.mean(row['atom_coords'], axis=0)

            if row['chain_type'] == 'receptor':
                local_idx = len(receptor_features)
                receptor_idx_map[idx] = local_idx
                receptor_features.append(feat)
                #centroids_receptor.append(coords)
            else:  # ligand
                local_idx = len(ligand_features)
                ligand_idx_map[idx] = local_idx
                ligand_features.append(feat)
                #centroids_ligand.append(coords)

        receptor_features = torch.stack(receptor_features)
        ligand_features = torch.stack(ligand_features)

        #centroids_receptor = np.array(centroids_receptor)
        #centroids_ligand = np.array(centroids_ligand)

        # === 2. EDGE LISTS ===
        receptor_receptor_edges, receptor_receptor_attrs = [], []
        ligand_ligand_edges, ligand_ligand_attrs = [], []
        receptor_ligand_edges, receptor_ligand_attrs = [], []
        ligand_receptor_edges, ligand_receptor_attrs = [], []

        # Precompute distances between all nodes globally for convenience
        centroids_all = np.array([np.mean(coords, axis=0) for coords in df['atom_coords']])

        for i, j in combinations(range(len(df)), 2):
            residue_pair_dist_mtx = torch.cdist(
                        torch.tensor(df.iloc[i]['atom_coords'], dtype=torch.float32),
                        torch.tensor(df.iloc[j]['atom_coords'], dtype=torch.float32)
            )
            if (residue_pair_dist_mtx <= distance_threshold).any():
                vec = torch.tensor(centroids_all[j] - centroids_all[i], dtype=torch.float32)
                unit_vec = vec / torch.linalg.norm(vec)
                #dist_hist = torch.histc(residue_pair_dist_mtx.ravel(),min=0,max=2*distance_threshold,bins=int(2*distance_threshold+1))/len(residue_pair_dist_mtx.ravel())
                dist_hist = torch.histc(residue_pair_dist_mtx.ravel(),min=0, max=15, bins=5)/len(residue_pair_dist_mtx.ravel())
                edge_feature = torch.cat([dist_hist, unit_vec])

                if df['chain_type'].iloc[i] == 'receptor' and df['chain_type'].iloc[j] == 'receptor':
                    src = receptor_idx_map[i]; dst = receptor_idx_map[j]
                    receptor_receptor_edges.append([src, dst])
                    receptor_receptor_edges.append([dst, src])
                    receptor_receptor_attrs.append(edge_feature)
                    receptor_receptor_attrs.append(torch.cat([dist_hist, -unit_vec]))

                elif df['chain_type'].iloc[i] == 'ligand' and df['chain_type'].iloc[j] == 'ligand':
                    src = ligand_idx_map[i]; dst = ligand_idx_map[j]
                    ligand_ligand_edges.append([src, dst])
                    ligand_ligand_edges.append([dst, src])
                    ligand_ligand_attrs.append(edge_feature)
                    ligand_ligand_attrs.append(torch.cat([dist_hist, -unit_vec]))

                elif df['chain_type'].iloc[i] == 'receptor' and df['chain_type'].iloc[j] == 'ligand':
                    # i is receptor, j is ligand
                    src = receptor_idx_map[i]; dst = ligand_idx_map[j]
                    receptor_ligand_edges.append([src, dst])
                    receptor_ligand_attrs.append(edge_feature)
                    ligand_receptor_edges.append([dst, src])
                    ligand_receptor_attrs.append(torch.cat([dist_hist, -unit_vec]))
                        
                else:  # i is ligand, j is receptor
                    src = ligand_idx_map[i]; dst = receptor_idx_map[j]
                    ligand_receptor_edges.append([src, dst])
                    ligand_receptor_attrs.append(edge_feature)
                    receptor_ligand_edges.append([dst, src])
                    receptor_ligand_attrs.append(torch.cat([dist_hist, -unit_vec]))

        # === 3. BUILD HETERODATA ===
        graph = HeteroData()
        graph['ligand'].x = ligand_features
        graph['receptor'].x = receptor_features
        graph.y = torch.tensor(self.row_of_df['affinity'])
        if 'new_pdb_name' in self.row_of_df and pd.notna(self.row_of_df['new_pdb_name']):
            graph.pdb_name = self.row_of_df['new_pdb_name']
        else:
            graph.pdb_name = self.row_of_df['PDB']

        if receptor_receptor_edges:
            graph['receptor','receptor_receptor','receptor'].edge_index = torch.tensor(
                receptor_receptor_edges, dtype=torch.long).t().contiguous()
            graph['receptor','receptor_receptor','receptor'].edge_attr = torch.stack(receptor_receptor_attrs)
        
        if ligand_ligand_edges:
            graph['ligand','ligand_ligand','ligand'].edge_index = torch.tensor(
                ligand_ligand_edges, dtype=torch.long).t().contiguous()
            graph['ligand','ligand_ligand','ligand'].edge_attr = torch.stack(ligand_ligand_attrs)

        if receptor_ligand_edges:
            graph['receptor','receptor_ligand','ligand'].edge_index = torch.tensor(
                receptor_ligand_edges, dtype=torch.long).t().contiguous()
            graph['receptor','receptor_ligand','ligand'].edge_attr = torch.stack(receptor_ligand_attrs)
        else:
            print("No receptor-ligand edges found.")

        if ligand_receptor_edges:
            graph['ligand','ligand_receptor','receptor'].edge_index = torch.tensor(
                ligand_receptor_edges, dtype=torch.long).t().contiguous()
            graph['ligand','ligand_receptor','receptor'].edge_attr = torch.stack(ligand_receptor_attrs)
        else:
            print("No ligand-receptor edges found.")

        self.protein_graph = graph
