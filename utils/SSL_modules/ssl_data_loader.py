"""
SSL Data Loader for Heterogeneous Protein Complex Graphs.

Provides data loading utilities for self-supervised pre-training with:
- Node masking (amino acid prediction at interface)
- Edge masking (link prediction for inter-molecular contacts)
- Combined node + edge masking

Works with HeteroData graphs (receptor-ligand heterogeneous graphs).
"""

import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData, Batch
from typing import Dict, List, Tuple, Optional, Callable, Union

from utils.SSL_modules.ssl_transforms import MaskInterfaceNodes, MaskInterfaceNodesAndEdges, MaskInterAndIntraEdges

from .ssl_edge_enhancements import create_enhanced_masker 


class SSLHeteroGraphDataset(Dataset):
    """
    Dataset for self-supervised learning on heterogeneous protein complex graphs.
    
    Applies masking transforms on-the-fly for node/edge prediction tasks.
    
    Args:
        graphs: List of HeteroData protein complex graphs
        pdb_names: Optional list of PDB identifiers
        mask_type: Type of masking ('node', 'edge', 'inter_intra_edge', or 'both')
        node_mask_ratio: Fraction of interface nodes to mask
        edge_mask_ratio: Fraction of inter-molecular edges to mask
        intra_edge_mask_ratio: Fraction of interface-adjacent intra-edges to mask
        negative_ratio: Ratio of negative to positive edge samples
        num_aa_classes: Number of amino acid classes (default: 20)
        preserve_aa_properties: Whether to keep AA properties for masked nodes
        transform: Additional transform to apply before masking
    """
    def __init__(
        self,
        graphs: List[HeteroData],
        pdb_names: Optional[List[str]] = None,
        mask_type: str = 'edge',  # 'node', 'edge', 'inter_intra_edge', or 'both'
        strategy_type: str = 'dynamic',
        node_mask_ratio: float = 0.15,
        edge_mask_ratio: float = 0.15,
        intra_edge_mask_ratio: float = 0.15,
        negative_ratio: float = 1.0,
        num_aa_classes: int = 20,
        preserve_aa_properties: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.graphs = graphs
        self.pdb_names = pdb_names or [f"graph_{i}" for i in range(len(graphs))]
        self.mask_type = mask_type
        self.transform = transform
        
        # Initialize masker based on type
        if mask_type == 'node':
            self.masker = MaskInterfaceNodes(
                mask_ratio=node_mask_ratio,
                num_aa_classes=num_aa_classes,
                preserve_aa_properties=preserve_aa_properties,
            )
        elif mask_type == 'edge':
            self.masker = create_enhanced_masker(
                strategy=strategy_type, # stratified, curriculum, dynamic, hard_negative
                mask_ratio=edge_mask_ratio,
                negative_ratio=negative_ratio,
            )
        elif mask_type == 'inter_intra_edge':
            self.masker = MaskInterAndIntraEdges(
                inter_edge_mask_ratio=edge_mask_ratio,
                intra_edge_mask_ratio=intra_edge_mask_ratio,
                negative_ratio=negative_ratio,
            )
        else:  # 'both'
            self.masker = MaskInterfaceNodesAndEdges(
                node_mask_ratio=node_mask_ratio,
                edge_mask_ratio=edge_mask_ratio,
                negative_ratio=negative_ratio,
                num_aa_classes=num_aa_classes,
                preserve_aa_properties=preserve_aa_properties,
            )
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Tuple[HeteroData, Dict, str]:
        """
        Get a masked graph with mask info.
        
        Returns:
            Tuple of (masked_graph, mask_info, pdb_name)
        """
        graph = self.graphs[idx]
        pdb_name = self.pdb_names[idx]
        
        if self.transform is not None:
            graph = self.transform(graph)
        
        # Apply masking
        masked_graph, mask_info = self.masker(graph)
        
        # Store original node counts for batch collation
        mask_info['n_receptor'] = graph['receptor'].x.size(0)
        mask_info['n_ligand'] = graph['ligand'].x.size(0)
        
        return masked_graph, mask_info, pdb_name


def ssl_hetero_collate_fn(batch: List[Tuple[HeteroData, Dict, str]], mask_type: str = 'edge'):
    """
    Custom collate function for SSL hetero graph batches.
    
    Properly batches graphs and adjusts mask indices for the batched tensor.
    
    Args:
        batch: List of (masked_graph, mask_info, pdb_name) tuples
        mask_type: Type of masking ('node', 'edge', 'inter_intra_edge', or 'both')
    
    Returns:
        Tuple of (batched_graph, batched_mask_info, pdb_names)
    """
    graphs, mask_infos, pdb_names = zip(*batch)
    
    # Batch graphs using PyG's Batch
    batched_graph = Batch.from_data_list(graphs)
    
    # Collate mask info with proper batch offsets
    batched_mask_info = _collate_mask_info_with_offsets(mask_infos, mask_type)
    
    return batched_graph, batched_mask_info, list(pdb_names)


def _collate_mask_info_with_offsets(mask_infos: List[Dict], mask_type: str) -> Dict:
    """
    Collate mask info with proper node offsets based on actual node counts.
    """
    if mask_type == 'node':
        return _collate_node_mask_with_offsets(mask_infos)
    elif mask_type == 'edge':
        return _collate_edge_mask_with_offsets(mask_infos)
    elif mask_type == 'inter_intra_edge':
        inter_infos = [m['inter_edge_mask_info'] for m in mask_infos]
        intra_infos = [m['intra_edge_mask_info'] for m in mask_infos]
        
        for i, m in enumerate(mask_infos):
            inter_infos[i]['n_receptor'] = m['n_receptor']
            inter_infos[i]['n_ligand'] = m['n_ligand']
            intra_infos[i]['n_receptor'] = m['n_receptor']
            intra_infos[i]['n_ligand'] = m['n_ligand']
        
        return {
            'inter_edge_mask_info': _collate_edge_mask_with_offsets(inter_infos),
            'intra_edge_mask_info': _collate_intra_edge_mask_with_offsets(intra_infos),
        }
    else:  # 'both'
        # Extract node and edge mask infos
        node_infos = [m['node_mask_info'] for m in mask_infos]
        edge_infos = [m['edge_mask_info'] for m in mask_infos]
        
        # Add node counts to sub-infos
        for i, m in enumerate(mask_infos):
            node_infos[i]['n_receptor'] = m['n_receptor']
            node_infos[i]['n_ligand'] = m['n_ligand']
            edge_infos[i]['n_receptor'] = m['n_receptor']
            edge_infos[i]['n_ligand'] = m['n_ligand']
        
        return {
            'node_mask_info': _collate_node_mask_with_offsets(node_infos),
            'edge_mask_info': _collate_edge_mask_with_offsets(edge_infos),
        }


def _collate_node_mask_with_offsets(mask_infos: List[Dict]) -> Dict:
    """Collate node mask info with proper batch offsets."""
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
    
    for batch_idx, info in enumerate(mask_infos):
        n_receptor = info.get('n_receptor', 0)
        n_ligand = info.get('n_ligand', 0)
        
        # Adjust receptor mask indices
        if len(info['receptor_mask_indices']) > 0:
            adjusted_indices = info['receptor_mask_indices'] + receptor_offset
            batched['receptor_mask_indices'].append(adjusted_indices)
            batched['receptor_labels'].append(info['receptor_labels'])
            batched['receptor_batch'].extend([batch_idx] * len(info['receptor_mask_indices']))
        
        # Adjust ligand mask indices
        if len(info['ligand_mask_indices']) > 0:
            adjusted_indices = info['ligand_mask_indices'] + ligand_offset
            batched['ligand_mask_indices'].append(adjusted_indices)
            batched['ligand_labels'].append(info['ligand_labels'])
            batched['ligand_batch'].extend([batch_idx] * len(info['ligand_mask_indices']))
        
        # Update offsets
        receptor_offset += n_receptor
        ligand_offset += n_ligand
    
    # Concatenate tensors
    batched['receptor_mask_indices'] = torch.cat(batched['receptor_mask_indices']) if batched['receptor_mask_indices'] else torch.empty(0, dtype=torch.long)
    batched['ligand_mask_indices'] = torch.cat(batched['ligand_mask_indices']) if batched['ligand_mask_indices'] else torch.empty(0, dtype=torch.long)
    batched['receptor_labels'] = torch.cat(batched['receptor_labels']) if batched['receptor_labels'] else torch.empty(0, dtype=torch.long)
    batched['ligand_labels'] = torch.cat(batched['ligand_labels']) if batched['ligand_labels'] else torch.empty(0, dtype=torch.long)
    batched['receptor_batch'] = torch.tensor(batched['receptor_batch'], dtype=torch.long)
    batched['ligand_batch'] = torch.tensor(batched['ligand_batch'], dtype=torch.long)
    
    return batched


def _collate_edge_mask_with_offsets(mask_infos: List[Dict]) -> Dict:
    """Collate edge mask info with proper batch offsets."""
    batched = {
        'all_edges': [],
        'edge_labels': [],
        'edge_batch': [],
    }
    
    receptor_offset = 0
    ligand_offset = 0
    
    for batch_idx, info in enumerate(mask_infos):
        n_receptor = info.get('n_receptor', 0)
        n_ligand = info.get('n_ligand', 0)
        
        if info['all_edges'].size(1) > 0:
            adjusted_edges = info['all_edges'].clone()
            adjusted_edges[0] += receptor_offset  # receptor indices
            adjusted_edges[1] += ligand_offset    # ligand indices
            batched['all_edges'].append(adjusted_edges)
            batched['edge_labels'].append(info['edge_labels'])
            batched['edge_batch'].extend([batch_idx] * info['all_edges'].size(1))
        
        receptor_offset += n_receptor
        ligand_offset += n_ligand
    
    batched['all_edges'] = torch.cat(batched['all_edges'], dim=1) if batched['all_edges'] else torch.empty(2, 0, dtype=torch.long)
    batched['edge_labels'] = torch.cat(batched['edge_labels']) if batched['edge_labels'] else torch.empty(0)
    batched['edge_batch'] = torch.tensor(batched['edge_batch'], dtype=torch.long)
    
    return batched


def _collate_intra_edge_mask_with_offsets(mask_infos: List[Dict]) -> Dict:
    """
    Collate intra-edge mask info with proper batch offsets.
    
    Each mask_info has 'receptor' and 'ligand' sub-dicts.
    Intra-edges use same-type node indices, so both rows get the same offset.
    """
    batched = {
        'all_edges': [],
        'edge_labels': [],
        'edge_batch': [],
        'edge_node_type': [],  # 0 for receptor, 1 for ligand
    }
    
    receptor_offset = 0
    ligand_offset = 0
    
    for batch_idx, info in enumerate(mask_infos):
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


def get_ssl_train_loader(
    graphs: List[HeteroData],
    pdb_names: Optional[List[str]] = None,
    batch_size: int = 32,
    mask_type: str = 'edge',
    strategy_type: str = 'dynamic',
    node_mask_ratio: float = 0.15,
    edge_mask_ratio: float = 0.15,
    intra_edge_mask_ratio: float = 0.15,
    negative_ratio: float = 1.0,
    shuffle: bool = True,
    num_workers: int = 0,
    prefetch_factor: int = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for SSL pre-training.
    
    Args:
        graphs: List of HeteroData protein complex graphs
        pdb_names: Optional list of PDB identifiers
        batch_size: Batch size
        mask_type: Type of masking ('node', 'edge', 'inter_intra_edge', or 'both')
        node_mask_ratio: Fraction of interface nodes to mask
        edge_mask_ratio: Fraction of inter-molecular edges to mask
        intra_edge_mask_ratio: Fraction of interface-adjacent intra-edges to mask
        negative_ratio: Ratio of negative to positive edge samples
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for SSLHeteroGraphDataset
    
    Returns:
        DataLoader for SSL training
    """
    dataset = SSLHeteroGraphDataset(
        graphs=graphs,
        pdb_names=pdb_names,
        mask_type=mask_type,
        strategy_type=strategy_type,
        node_mask_ratio=node_mask_ratio,
        edge_mask_ratio=edge_mask_ratio,
        intra_edge_mask_ratio=intra_edge_mask_ratio,
        negative_ratio=negative_ratio,
        **kwargs
    )
    
    # Create collate function with mask_type
    def collate_fn(batch):
        return ssl_hetero_collate_fn(batch, mask_type=mask_type)
    
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        if prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = prefetch_factor
    
    loader = DataLoader(**loader_kwargs)
    
    return loader


def load_ssl_data_from_dir(
    pdb_dir: str,
    premium_only: bool = False,
    golden_only: bool = False,
    premium_threshold: float = 1.5,
) -> Tuple[List[HeteroData], List[str]]:
    """
    Load HeteroData graphs from a directory of PKL files for SSL pre-training.
    
    Filters for premium and/or golden samples (positive binders).
    
    Args:
        pdb_dir: Directory containing .pkl files with StructureToHeteroGraph objects
        premium_only: If True, only load premium samples (affinity > threshold)
        golden_only: If True, only load golden samples (affinity == 1)
        premium_threshold: Threshold for premium sample classification
    
    Returns:
        Tuple of (graphs, pdb_names)
    """
    files = [f for f in os.listdir(pdb_dir) if f.endswith('.pkl')]
    
    graphs = []
    pdb_names = []
    
    for f in files:
        pdb_name = f.replace('.pkl', '').lower()
        with open(os.path.join(pdb_dir, f), 'rb') as fh:
            st = pickle.load(fh)
        
        # Get the graph
        graph = st.protein_graph
        affinity = graph.y.item() if hasattr(graph.y, 'item') else graph.y
        
        # Filter based on sample type
        if premium_only and affinity <= premium_threshold:
            continue
        if golden_only and affinity != 1.0:
            continue
        
        # For SSL, we want positive samples (premium or golden)
        # Skip negative samples (affinity == 0)
        if affinity <= 0:
            continue
        
        # Validate graph has interface edges
        if ('receptor', 'receptor_ligand', 'ligand') not in graph.edge_index_dict:
            continue
        
        # Check minimum node counts
        try:
            n_receptor = graph['receptor'].x.size(0)
            n_ligand = graph['ligand'].x.size(0)
            n_interface = graph[('receptor', 'receptor_ligand', 'ligand')].edge_index.size(1)
            if n_receptor < 5 or n_ligand < 5 or n_interface < 3:
                continue
        except Exception:
            continue
        
        graphs.append(graph)
        pdb_names.append(pdb_name)
    
    return graphs, pdb_names


def filter_samples_for_ssl(all_sttgs: Dict, min_interface_edges: int = 3) -> Dict:
    """
    Filter samples for SSL pre-training.
    
    Keeps only samples with:
    - Positive affinity (premium or golden samples)
    - Sufficient interface edges for masking
    
    Args:
        all_sttgs: Dictionary of {pdb_name: structure_object}
        min_interface_edges: Minimum number of interface edges required
    
    Returns:
        Filtered dictionary
    """
    kept = {}
    for k, st in all_sttgs.items():
        try:
            graph = st.protein_graph
            affinity = graph.y.item() if hasattr(graph.y, 'item') else graph.y
            
            # Skip negative samples
            if affinity <= 0:
                continue
            
            # Check interface edges
            if ('receptor', 'receptor_ligand', 'ligand') not in graph.edge_index_dict:
                continue
            
            n_interface = graph[('receptor', 'receptor_ligand', 'ligand')].edge_index.size(1)
            if n_interface < min_interface_edges:
                continue
            
            # Check node counts
            n_receptor = graph['receptor'].x.size(0)
            n_ligand = graph['ligand'].x.size(0)
            if n_receptor < 5 or n_ligand < 5:
                continue
            
            kept[k] = st
        except Exception:
            continue
    
    return kept
