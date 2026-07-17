"""
SSL Transforms for Heterogeneous Protein Complex Graphs.

Implements masking strategies for self-supervised pre-training:
1. MaskInterfaceNodes: Mask amino acid identity at interface residues
2. MaskInterfaceEdges: Mask inter-molecular edges (receptor-ligand connections)
3. MaskInterfaceNodesAndEdges: Combined masking strategy

The node features are structured as:
- [0:20]: One-hot encoded amino acid type (20 classes)
- [20:22]: One-hot receptor/ligand indicator
- [22:25]: Amino acid properties (from AA_At_dict: hydrophobicity, polarity, charge)

For node masking, we replace the 20-dim one-hot with a learnable [MASK] token or zeros,
and the task is to predict the original amino acid type.

For edge masking, we remove some inter-molecular edges and the task is to predict
whether an edge should exist between pairs of nodes.
"""

import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import Tuple, Optional, Dict, List
import copy


class MaskInterfaceNodes:
    """
    Mask amino acid identity at interface residues.
    
    Interface nodes are those connected by receptor_ligand or ligand_receptor edges.
    
    Args:
        mask_ratio: Fraction of interface nodes to mask (default: 0.15)
        num_aa_classes: Number of amino acid classes (default: 20)
        mask_token_value: Value to use for masked positions (default: 0.0)
        preserve_aa_properties: Whether to keep AA property features (indices 22:25)
    """
    def __init__(
        self,
        mask_ratio: float = 0.15,
        num_aa_classes: int = 20,
        mask_token_value: float = 0.0,
        preserve_aa_properties: bool = True,
    ):
        self.mask_ratio = mask_ratio
        self.num_aa_classes = num_aa_classes
        self.mask_token_value = mask_token_value
        self.preserve_aa_properties = preserve_aa_properties
    
    def __call__(self, data: HeteroData) -> Tuple[HeteroData, Dict]:
        """
        Apply node masking to interface residues.
        
        Returns:
            Tuple of (masked_data, mask_info)
            - masked_data: HeteroData with masked node features
            - mask_info: Dict containing:
                - 'receptor_mask_indices': indices of masked receptor nodes
                - 'ligand_mask_indices': indices of masked ligand nodes
                - 'receptor_labels': original AA types for masked receptor nodes
                - 'ligand_labels': original AA types for masked ligand nodes
        """
        data = copy.deepcopy(data)
        
        # Find interface nodes (nodes with inter-molecular edges)
        receptor_interface_nodes = set()
        ligand_interface_nodes = set()
        
        # Get nodes from receptor_ligand edges
        if ('receptor', 'receptor_ligand', 'ligand') in data.edge_index_dict:
            edge_index = data[('receptor', 'receptor_ligand', 'ligand')].edge_index
            receptor_interface_nodes.update(edge_index[0].tolist())
            ligand_interface_nodes.update(edge_index[1].tolist())
        
        # Get nodes from ligand_receptor edges (reverse direction)
        if ('ligand', 'ligand_receptor', 'receptor') in data.edge_index_dict:
            edge_index = data[('ligand', 'ligand_receptor', 'receptor')].edge_index
            ligand_interface_nodes.update(edge_index[0].tolist())
            receptor_interface_nodes.update(edge_index[1].tolist())
        
        receptor_interface_nodes = sorted(list(receptor_interface_nodes))
        ligand_interface_nodes = sorted(list(ligand_interface_nodes))
        
        mask_info = {
            'receptor_mask_indices': [],
            'ligand_mask_indices': [],
            'receptor_labels': [],
            'ligand_labels': [],
            'all_receptor_interface': receptor_interface_nodes,
            'all_ligand_interface': ligand_interface_nodes,
        }
        
        # Mask receptor interface nodes
        if receptor_interface_nodes:
            n_mask = max(1, int(len(receptor_interface_nodes) * self.mask_ratio))
            mask_indices = np.random.choice(receptor_interface_nodes, n_mask, replace=False).tolist()
            
            for idx in mask_indices:
                # Get original AA label (argmax of one-hot)
                original_onehot = data['receptor'].x[idx, :self.num_aa_classes]
                aa_label = original_onehot.argmax().item()
                mask_info['receptor_labels'].append(aa_label)
                mask_info['receptor_mask_indices'].append(idx)
                
                # Mask the one-hot encoding
                data['receptor'].x[idx, :self.num_aa_classes] = self.mask_token_value
                
                if not self.preserve_aa_properties:
                    # Also mask AA properties (indices 22:25 if they exist)
                    if data['receptor'].x.size(1) > 22:
                        data['receptor'].x[idx, 22:25] = self.mask_token_value
        
        # Mask ligand interface nodes
        if ligand_interface_nodes:
            n_mask = max(1, int(len(ligand_interface_nodes) * self.mask_ratio))
            mask_indices = np.random.choice(ligand_interface_nodes, n_mask, replace=False).tolist()
            
            for idx in mask_indices:
                original_onehot = data['ligand'].x[idx, :self.num_aa_classes]
                aa_label = original_onehot.argmax().item()
                mask_info['ligand_labels'].append(aa_label)
                mask_info['ligand_mask_indices'].append(idx)
                
                data['ligand'].x[idx, :self.num_aa_classes] = self.mask_token_value
                
                if not self.preserve_aa_properties:
                    if data['ligand'].x.size(1) > 22:
                        data['ligand'].x[idx, 22:25] = self.mask_token_value
        
        # Convert to tensors
        mask_info['receptor_mask_indices'] = torch.tensor(mask_info['receptor_mask_indices'], dtype=torch.long)
        mask_info['ligand_mask_indices'] = torch.tensor(mask_info['ligand_mask_indices'], dtype=torch.long)
        mask_info['receptor_labels'] = torch.tensor(mask_info['receptor_labels'], dtype=torch.long)
        mask_info['ligand_labels'] = torch.tensor(mask_info['ligand_labels'], dtype=torch.long)
        
        return data, mask_info


class MaskInterfaceEdges:
    """
    Mask inter-molecular edges for link prediction task.
    
    Masks a fraction of receptor_ligand and ligand_receptor edges.
    The task is to predict whether an edge should exist between pairs of interface nodes.
    
    Args:
        mask_ratio: Fraction of inter-molecular edges to mask (default: 0.15)
        negative_ratio: Ratio of negative samples to positive samples (default: 1.0)
    """
    def __init__(
        self,
        mask_ratio: float = 0.15,
        negative_ratio: float = 1.0,
    ):
        self.mask_ratio = mask_ratio
        self.negative_ratio = negative_ratio
    
    def __call__(self, data: HeteroData) -> Tuple[HeteroData, Dict]:
        """
        Apply edge masking to inter-molecular edges.
        
        Returns:
            Tuple of (masked_data, mask_info)
            - masked_data: HeteroData with some edges removed
            - mask_info: Dict containing:
                - 'positive_edges': (receptor_idx, ligand_idx) pairs that were masked (should exist)
                - 'negative_edges': (receptor_idx, ligand_idx) pairs that don't exist (negative samples)
                - 'edge_labels': 1 for positive, 0 for negative
        """
        data = copy.deepcopy(data)
        
        # Get receptor_ligand edges
        if ('receptor', 'receptor_ligand', 'ligand') not in data.edge_index_dict:
            return data, {'positive_edges': torch.empty(2, 0), 'negative_edges': torch.empty(2, 0), 'edge_labels': torch.empty(0)}
        
        edge_index = data[('receptor', 'receptor_ligand', 'ligand')].edge_index
        edge_attr = data[('receptor', 'receptor_ligand', 'ligand')].edge_attr
        
        num_edges = edge_index.size(1)
        n_receptor = data['receptor'].x.size(0)
        n_ligand = data['ligand'].x.size(0)
        
        # Sample edges to mask
        n_mask = max(1, int(num_edges * self.mask_ratio))
        mask_edge_indices = np.random.choice(num_edges, n_mask, replace=False)
        
        # Get masked edge pairs
        positive_receptor_idx = edge_index[0, mask_edge_indices]
        positive_ligand_idx = edge_index[1, mask_edge_indices]
        
        # Create set of existing edges for fast lookup
        existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        
        # Generate negative samples (non-existing edges between interface nodes)
        n_negative = int(n_mask * self.negative_ratio)
        negative_pairs = []
        
        # Get all interface nodes
        receptor_interface = set(edge_index[0].tolist())
        ligand_interface = set(edge_index[1].tolist())
        receptor_interface = list(receptor_interface)
        ligand_interface = list(ligand_interface)
        
        attempts = 0
        max_attempts = n_negative * 10
        while len(negative_pairs) < n_negative and attempts < max_attempts:
            r_idx = np.random.choice(receptor_interface)
            l_idx = np.random.choice(ligand_interface)
            if (r_idx, l_idx) not in existing_edges:
                negative_pairs.append((r_idx, l_idx))
                existing_edges.add((r_idx, l_idx))  # Avoid duplicates
            attempts += 1
        
        # Remove masked edges from graph
        keep_mask = torch.ones(num_edges, dtype=torch.bool)
        keep_mask[mask_edge_indices] = False
        
        data[('receptor', 'receptor_ligand', 'ligand')].edge_index = edge_index[:, keep_mask]
        data[('receptor', 'receptor_ligand', 'ligand')].edge_attr = edge_attr[keep_mask]
        
        # Also remove corresponding reverse edges
        if ('ligand', 'ligand_receptor', 'receptor') in data.edge_index_dict:
            rev_edge_index = data[('ligand', 'ligand_receptor', 'receptor')].edge_index
            rev_edge_attr = data[('ligand', 'ligand_receptor', 'receptor')].edge_attr
            
            # Find reverse edges that correspond to masked edges
            masked_pairs = set(zip(positive_receptor_idx.tolist(), positive_ligand_idx.tolist()))
            rev_keep_mask = torch.ones(rev_edge_index.size(1), dtype=torch.bool)
            for i in range(rev_edge_index.size(1)):
                l_idx = rev_edge_index[0, i].item()
                r_idx = rev_edge_index[1, i].item()
                if (r_idx, l_idx) in masked_pairs:
                    rev_keep_mask[i] = False
            
            data[('ligand', 'ligand_receptor', 'receptor')].edge_index = rev_edge_index[:, rev_keep_mask]
            data[('ligand', 'ligand_receptor', 'receptor')].edge_attr = rev_edge_attr[rev_keep_mask]
        
        # Prepare mask info
        if negative_pairs:
            negative_receptor_idx = torch.tensor([p[0] for p in negative_pairs], dtype=torch.long)
            negative_ligand_idx = torch.tensor([p[1] for p in negative_pairs], dtype=torch.long)
        else:
            negative_receptor_idx = torch.empty(0, dtype=torch.long)
            negative_ligand_idx = torch.empty(0, dtype=torch.long)
        
        positive_edges = torch.stack([positive_receptor_idx, positive_ligand_idx], dim=0)
        negative_edges = torch.stack([negative_receptor_idx, negative_ligand_idx], dim=0) if len(negative_pairs) > 0 else torch.empty(2, 0, dtype=torch.long)
        
        edge_labels = torch.cat([
            torch.ones(positive_edges.size(1)),
            torch.zeros(negative_edges.size(1))
        ])
        
        all_edges = torch.cat([positive_edges, negative_edges], dim=1)
        
        mask_info = {
            'positive_edges': positive_edges,
            'negative_edges': negative_edges,
            'all_edges': all_edges,  # (receptor_idx, ligand_idx) pairs
            'edge_labels': edge_labels,
        }
        
        return data, mask_info


class MaskInterfaceIntraEdges:
    """
    Mask intra-molecular edges near the binding interface for link prediction.
    
    Masks a fraction of receptor_receptor or ligand_ligand edges where at least
    one endpoint is an interface node (has an inter-molecular edge to the other chain).
    This teaches the model the structural microenvironment around the interface.
    
    Args:
        mask_ratio: Fraction of eligible intra-edges to mask (default: 0.15)
        negative_ratio: Ratio of negative samples to positive samples (default: 1.0)
        skip_deepcopy: If True, skip deep copy (used when called from combined wrapper)
    """
    def __init__(
        self,
        mask_ratio: float = 0.15,
        negative_ratio: float = 1.0,
        skip_deepcopy: bool = False,
    ):
        self.mask_ratio = mask_ratio
        self.negative_ratio = negative_ratio
        self.skip_deepcopy = skip_deepcopy
    
    def _get_interface_nodes(self, data: HeteroData):
        """Get sets of interface nodes for receptor and ligand."""
        receptor_interface = set()
        ligand_interface = set()
        
        if ('receptor', 'receptor_ligand', 'ligand') in data.edge_index_dict:
            inter_ei = data[('receptor', 'receptor_ligand', 'ligand')].edge_index
            receptor_interface = set(inter_ei[0].tolist())
            ligand_interface = set(inter_ei[1].tolist())
        
        return receptor_interface, ligand_interface
    
    def _mask_intra_edges_for_type(
        self,
        data: HeteroData,
        node_type: str,
        edge_type: Tuple,
        interface_nodes: set,
    ) -> Tuple[HeteroData, Dict]:
        """
        Mask intra-edges for a single node type (receptor or ligand).
        
        Only masks edges where at least one endpoint is an interface node.
        
        Returns:
            Tuple of (modified_data, mask_info_for_this_type)
        """
        if edge_type not in data.edge_index_dict or len(interface_nodes) == 0:
            return data, {
                'positive_edges': torch.empty(2, 0, dtype=torch.long),
                'negative_edges': torch.empty(2, 0, dtype=torch.long),
                'all_edges': torch.empty(2, 0, dtype=torch.long),
                'edge_labels': torch.empty(0),
            }
        
        edge_index = data[edge_type].edge_index
        edge_attr = data[edge_type].edge_attr
        num_edges = edge_index.size(1)
        
        # Find edges where at least one endpoint is an interface node (vectorized)
        interface_tensor = torch.tensor(list(interface_nodes), dtype=torch.long)
        src_is_interface = torch.isin(edge_index[0], interface_tensor)
        dst_is_interface = torch.isin(edge_index[1], interface_tensor)
        eligible_mask = src_is_interface | dst_is_interface
        eligible_indices = torch.where(eligible_mask)[0].numpy()
        
        if len(eligible_indices) == 0:
            return data, {
                'positive_edges': torch.empty(2, 0, dtype=torch.long),
                'negative_edges': torch.empty(2, 0, dtype=torch.long),
                'all_edges': torch.empty(2, 0, dtype=torch.long),
                'edge_labels': torch.empty(0),
            }
        
        # Sample edges to mask from eligible ones
        n_mask = max(1, int(len(eligible_indices) * self.mask_ratio))
        mask_edge_indices = np.random.choice(eligible_indices, n_mask, replace=False)
        
        # Get masked edge pairs (positive samples)
        positive_src = edge_index[0, mask_edge_indices]
        positive_dst = edge_index[1, mask_edge_indices]
        
        # Generate negative samples: non-existing intra-edges near interface
        n_negative = int(n_mask * self.negative_ratio)
        
        # Candidate nodes: interface nodes + their 1-hop intra neighbors (vectorized)
        interface_tensor = torch.tensor(list(interface_nodes), dtype=torch.long)
        src_in = torch.isin(edge_index[0], interface_tensor)
        dst_in = torch.isin(edge_index[1], interface_tensor)
        neighbor_mask = src_in | dst_in
        candidate_nodes = torch.unique(edge_index[:, neighbor_mask]).numpy()
        interface_set = set(interface_nodes)
        
        # Build existing edge set for fast lookup (encode as single int)
        n_nodes = int(max(edge_index.max().item() + 1, candidate_nodes.max() + 1)) if len(candidate_nodes) > 0 else 1
        existing_codes = set()
        ei_src = edge_index[0].numpy()
        ei_dst = edge_index[1].numpy()
        for i in range(len(ei_src)):
            s, d = int(ei_src[i]), int(ei_dst[i])
            existing_codes.add(s * n_nodes + d)
            existing_codes.add(d * n_nodes + s)
        
        # Vectorized batch sampling
        negative_pairs = []
        if n_negative > 0 and len(candidate_nodes) > 1:
            batch = max(n_negative * 4, 256)
            remaining = n_negative
            max_rounds = 5
            for _ in range(max_rounds):
                u = np.random.choice(candidate_nodes, size=batch)
                v = np.random.choice(candidate_nodes, size=batch)
                for j in range(batch):
                    if remaining <= 0:
                        break
                    uj, vj = int(u[j]), int(v[j])
                    if uj == vj:
                        continue
                    code = uj * n_nodes + vj
                    if code in existing_codes:
                        continue
                    if uj not in interface_set and vj not in interface_set:
                        continue
                    negative_pairs.append((uj, vj))
                    existing_codes.add(code)
                    existing_codes.add(vj * n_nodes + uj)
                    remaining -= 1
                if remaining <= 0:
                    break
        
        # Remove masked edges from graph
        keep_mask = torch.ones(num_edges, dtype=torch.bool)
        keep_mask[mask_edge_indices] = False
        data[edge_type].edge_index = edge_index[:, keep_mask]
        data[edge_type].edge_attr = edge_attr[keep_mask]
        
        # Prepare mask info
        if negative_pairs:
            neg_arr = np.array(negative_pairs, dtype=np.int64)
            negative_src = torch.from_numpy(neg_arr[:, 0])
            negative_dst = torch.from_numpy(neg_arr[:, 1])
        else:
            negative_src = torch.empty(0, dtype=torch.long)
            negative_dst = torch.empty(0, dtype=torch.long)
        
        positive_edges = torch.stack([positive_src, positive_dst], dim=0)
        negative_edges = (
            torch.stack([negative_src, negative_dst], dim=0) 
            if len(negative_pairs) > 0 
            else torch.empty(2, 0, dtype=torch.long)
        )
        
        edge_labels = torch.cat([
            torch.ones(positive_edges.size(1)),
            torch.zeros(negative_edges.size(1)),
        ])
        
        all_edges = torch.cat([positive_edges, negative_edges], dim=1)
        
        return data, {
            'positive_edges': positive_edges,
            'negative_edges': negative_edges,
            'all_edges': all_edges,  # (src_idx, dst_idx) pairs — same node type
            'edge_labels': edge_labels,
        }
    
    def __call__(self, data: HeteroData) -> Tuple[HeteroData, Dict]:
        """
        Apply intra-edge masking for both receptor and ligand chains.
        
        Returns:
            Tuple of (masked_data, intra_mask_info) where intra_mask_info contains:
                - 'receptor': mask info for receptor_receptor edges
                - 'ligand': mask info for ligand_ligand edges
        """
        if not self.skip_deepcopy:
            data = copy.deepcopy(data)
        
        receptor_interface, ligand_interface = self._get_interface_nodes(data)
        
        # Mask receptor intra-edges
        data, receptor_info = self._mask_intra_edges_for_type(
            data, 'receptor',
            ('receptor', 'receptor_receptor', 'receptor'),
            receptor_interface,
        )
        
        # Mask ligand intra-edges
        data, ligand_info = self._mask_intra_edges_for_type(
            data, 'ligand',
            ('ligand', 'ligand_ligand', 'ligand'),
            ligand_interface,
        )
        
        intra_mask_info = {
            'receptor': receptor_info,
            'ligand': ligand_info,
        }
        
        return data, intra_mask_info


class MaskInterAndIntraEdges:
    """
    Combined masking strategy: mask both inter-molecular and intra-molecular edges.
    
    Args:
        inter_edge_mask_ratio: Fraction of inter-molecular edges to mask
        intra_edge_mask_ratio: Fraction of interface-adjacent intra-edges to mask
        negative_ratio: Ratio of negative edge samples to positive
    """
    def __init__(
        self,
        inter_edge_mask_ratio: float = 0.25,
        intra_edge_mask_ratio: float = 0.15,
        negative_ratio: float = 1.0,
    ):
        self.inter_masker = MaskInterfaceEdges(
            mask_ratio=inter_edge_mask_ratio,
            negative_ratio=negative_ratio,
        )
        self.intra_masker = MaskInterfaceIntraEdges(
            mask_ratio=intra_edge_mask_ratio,
            negative_ratio=negative_ratio,
            skip_deepcopy=True,  # inter_masker already deep-copied
        )
    
    def __call__(self, data: HeteroData) -> Tuple[HeteroData, Dict]:
        """
        Apply both inter and intra edge masking.
        
        Returns:
            Tuple of (masked_data, mask_info) where mask_info contains:
                - 'inter_edge_mask_info': inter-molecular edge mask info
                - 'intra_edge_mask_info': intra-molecular edge mask info (per chain)
        """
        # First mask inter-edges
        data, inter_mask_info = self.inter_masker(data)
        
        # Then mask intra-edges (on data that already has inter-edges masked)
        data, intra_mask_info = self.intra_masker(data)
        
        mask_info = {
            'inter_edge_mask_info': inter_mask_info,
            'intra_edge_mask_info': intra_mask_info,
        }
        
        return data, mask_info


class MaskInterfaceNodesAndEdges:
    """
    Combined masking strategy: mask both interface nodes and edges.
    
    Args:
        node_mask_ratio: Fraction of interface nodes to mask
        edge_mask_ratio: Fraction of inter-molecular edges to mask
        negative_ratio: Ratio of negative edge samples to positive
        num_aa_classes: Number of amino acid classes
        preserve_aa_properties: Whether to keep AA property features for masked nodes
    """
    def __init__(
        self,
        node_mask_ratio: float = 0.15,
        edge_mask_ratio: float = 0.15,
        negative_ratio: float = 1.0,
        num_aa_classes: int = 20,
        preserve_aa_properties: bool = True,
    ):
        self.node_masker = MaskInterfaceNodes(
            mask_ratio=node_mask_ratio,
            num_aa_classes=num_aa_classes,
            preserve_aa_properties=preserve_aa_properties,
        )
        self.edge_masker = MaskInterfaceEdges(
            mask_ratio=edge_mask_ratio,
            negative_ratio=negative_ratio,
        )
    
    def __call__(self, data: HeteroData) -> Tuple[HeteroData, Dict]:
        """
        Apply both node and edge masking.
        
        Returns:
            Tuple of (masked_data, mask_info) where mask_info contains both node and edge mask info
        """
        # First mask nodes
        data, node_mask_info = self.node_masker(data)
        
        # Then mask edges (on the node-masked data)
        data, edge_mask_info = self.edge_masker(data)
        
        # Combine mask info
        mask_info = {
            'node_mask_info': node_mask_info,
            'edge_mask_info': edge_mask_info,
        }
        
        return data, mask_info


def collate_mask_info(mask_info_list: List[Dict], mask_type: str = 'node') -> Dict:
    """
    Collate mask info from multiple samples into a batch.
    
    Args:
        mask_info_list: List of mask_info dicts from transform
        mask_type: Type of masking ('node', 'edge', 'inter_intra_edge', or 'both')
    
    Returns:
        Batched mask info with adjusted indices
    """
    if mask_type == 'node':
        return _collate_node_mask_info(mask_info_list)
    elif mask_type == 'edge':
        return _collate_edge_mask_info(mask_info_list)
    elif mask_type == 'inter_intra_edge':
        inter_infos = [m['inter_edge_mask_info'] for m in mask_info_list]
        intra_infos = [m['intra_edge_mask_info'] for m in mask_info_list]
        return {
            'inter_edge_mask_info': _collate_edge_mask_info(inter_infos),
            'intra_edge_mask_info': _collate_intra_edge_mask_info(intra_infos),
        }
    else:  # both
        node_infos = [m['node_mask_info'] for m in mask_info_list]
        edge_infos = [m['edge_mask_info'] for m in mask_info_list]
        return {
            'node_mask_info': _collate_node_mask_info(node_infos),
            'edge_mask_info': _collate_edge_mask_info(edge_infos),
        }


def _collate_node_mask_info(mask_info_list: List[Dict]) -> Dict:
    """Collate node mask info with batch offsets."""
    batched = {
        'receptor_mask_indices': [],
        'ligand_mask_indices': [],
        'receptor_labels': [],
        'ligand_labels': [],
        'receptor_batch': [],
        'ligand_batch': [],
    }
    
    receptor_offset = 0
    ligand_offset = 0
    
    for batch_idx, info in enumerate(mask_info_list):
        # Adjust indices by offset
        if len(info['receptor_mask_indices']) > 0:
            batched['receptor_mask_indices'].append(info['receptor_mask_indices'] + receptor_offset)
            batched['receptor_labels'].append(info['receptor_labels'])
            batched['receptor_batch'].extend([batch_idx] * len(info['receptor_mask_indices']))
        
        if len(info['ligand_mask_indices']) > 0:
            batched['ligand_mask_indices'].append(info['ligand_mask_indices'] + ligand_offset)
            batched['ligand_labels'].append(info['ligand_labels'])
            batched['ligand_batch'].extend([batch_idx] * len(info['ligand_mask_indices']))
        
        # Update offsets (need to know node counts from somewhere)
        # This will be handled in the data loader by passing node counts
        receptor_offset += len(info.get('all_receptor_interface', [])) + 1  # Approximate
        ligand_offset += len(info.get('all_ligand_interface', [])) + 1
    
    # Concatenate
    batched['receptor_mask_indices'] = torch.cat(batched['receptor_mask_indices']) if batched['receptor_mask_indices'] else torch.empty(0, dtype=torch.long)
    batched['ligand_mask_indices'] = torch.cat(batched['ligand_mask_indices']) if batched['ligand_mask_indices'] else torch.empty(0, dtype=torch.long)
    batched['receptor_labels'] = torch.cat(batched['receptor_labels']) if batched['receptor_labels'] else torch.empty(0, dtype=torch.long)
    batched['ligand_labels'] = torch.cat(batched['ligand_labels']) if batched['ligand_labels'] else torch.empty(0, dtype=torch.long)
    batched['receptor_batch'] = torch.tensor(batched['receptor_batch'], dtype=torch.long)
    batched['ligand_batch'] = torch.tensor(batched['ligand_batch'], dtype=torch.long)
    
    return batched


def _collate_edge_mask_info(mask_info_list: List[Dict]) -> Dict:
    """Collate edge mask info with batch offsets."""
    batched = {
        'all_edges': [],
        'edge_labels': [],
        'edge_batch': [],
    }
    
    receptor_offset = 0
    ligand_offset = 0
    
    for batch_idx, info in enumerate(mask_info_list):
        if info['all_edges'].size(1) > 0:
            adjusted_edges = info['all_edges'].clone()
            adjusted_edges[0] += receptor_offset  # receptor indices
            adjusted_edges[1] += ligand_offset    # ligand indices
            batched['all_edges'].append(adjusted_edges)
            batched['edge_labels'].append(info['edge_labels'])
            batched['edge_batch'].extend([batch_idx] * info['all_edges'].size(1))
    
    batched['all_edges'] = torch.cat(batched['all_edges'], dim=1) if batched['all_edges'] else torch.empty(2, 0, dtype=torch.long)
    batched['edge_labels'] = torch.cat(batched['edge_labels']) if batched['edge_labels'] else torch.empty(0)
    batched['edge_batch'] = torch.tensor(batched['edge_batch'], dtype=torch.long)
    
    return batched


def _collate_intra_edge_mask_info(mask_info_list: List[Dict]) -> Dict:
    """
    Collate intra-edge mask info with batch offsets.
    
    Each mask_info has 'receptor' and 'ligand' sub-dicts, each with
    'all_edges' and 'edge_labels'. Intra-edges use same-type node indices.
    """
    batched = {
        'all_edges': [],
        'edge_labels': [],
        'edge_batch': [],
        'edge_node_type': [],  # 0 for receptor, 1 for ligand
    }
    
    receptor_offset = 0
    ligand_offset = 0
    
    for batch_idx, info in enumerate(mask_info_list):
        n_receptor = info.get('n_receptor', 0)
        n_ligand = info.get('n_ligand', 0)
        
        # Receptor intra-edges
        r_info = info['receptor']
        if r_info['all_edges'].size(1) > 0:
            adjusted = r_info['all_edges'].clone()
            adjusted[0] += receptor_offset
            adjusted[1] += receptor_offset
            batched['all_edges'].append(adjusted)
            batched['edge_labels'].append(r_info['edge_labels'])
            n_edges = r_info['all_edges'].size(1)
            batched['edge_batch'].extend([batch_idx] * n_edges)
            batched['edge_node_type'].extend([0] * n_edges)  # receptor
        
        # Ligand intra-edges
        l_info = info['ligand']
        if l_info['all_edges'].size(1) > 0:
            adjusted = l_info['all_edges'].clone()
            adjusted[0] += ligand_offset
            adjusted[1] += ligand_offset
            batched['all_edges'].append(adjusted)
            batched['edge_labels'].append(l_info['edge_labels'])
            n_edges = l_info['all_edges'].size(1)
            batched['edge_batch'].extend([batch_idx] * n_edges)
            batched['edge_node_type'].extend([1] * n_edges)  # ligand
        
        receptor_offset += n_receptor
        ligand_offset += n_ligand
    
    batched['all_edges'] = torch.cat(batched['all_edges'], dim=1) if batched['all_edges'] else torch.empty(2, 0, dtype=torch.long)
    batched['edge_labels'] = torch.cat(batched['edge_labels']) if batched['edge_labels'] else torch.empty(0)
    batched['edge_batch'] = torch.tensor(batched['edge_batch'], dtype=torch.long)
    batched['edge_node_type'] = torch.tensor(batched['edge_node_type'], dtype=torch.long)
    
    return batched
