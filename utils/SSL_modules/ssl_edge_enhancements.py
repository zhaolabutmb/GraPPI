"""
Optional enhancements for SSL edge masking strategies.

These provide more sophisticated masking patterns that could improve
the model's learning of interface knowledge.

Edge Feature Format (8-dimensional):
- [0:5]: Normalized distance histogram (5 bins: 0-3Å, 3-6Å, 6-9Å, 9-12Å, 12-15Å)
         Values represent the fraction of atom pairs in each distance bin
- [5:8]: Unit direction vector (x, y, z) from residue i centroid to residue j centroid

Note: These features are computed by compute_edge_features() in utils/gen_graphs_unified.py
"""

import copy
import torch
import numpy as np
from typing import Dict, Tuple, List
from torch_geometric.data import HeteroData


class StratifiedEdgeMasking:
    """
    Mask edges stratified by distance bins to ensure balanced learning.
    
    Uses the distance histogram from edge attributes to classify edges.
    This ensures the model learns about both:
    - Close contacts (strong interactions, 0-6Å)
    - Medium-range interactions (6-12Å)
    - Longer-range interactions (12-15Å)
    
    Args:
        mask_ratio: Overall fraction of edges to mask
        negative_ratio: Ratio of negative to positive samples
    """
    def __init__(
        self,
        mask_ratio: float = 0.15,
        negative_ratio: float = 1.0,
    ):
        self.mask_ratio = mask_ratio
        self.negative_ratio = negative_ratio
    
    def __call__(self, data: HeteroData) -> Tuple[HeteroData, Dict]:
        """Apply stratified edge masking."""
        data = copy.deepcopy(data)
        
        # Check if receptor_ligand edges exist
        if ('receptor', 'receptor_ligand', 'ligand') not in data.edge_index_dict:
            return data, {
                'positive_edges': torch.empty(2, 0),
                'negative_edges': torch.empty(2, 0),
                'all_edges': torch.empty(2, 0),
                'edge_labels': torch.empty(0)
            }
        
        # Get edge attributes: [5 distance histogram bins, 3 unit vector components]
        edge_attr = data[('receptor', 'receptor_ligand', 'ligand')].edge_attr
        edge_index = data[('receptor', 'receptor_ligand', 'ligand')].edge_index
        
        num_edges = edge_index.size(1)
        n_receptor = data['receptor'].x.size(0)
        n_ligand = data['ligand'].x.size(0)
        
        # Infer edge distance category from distance histogram
        # First 5 elements are normalized histogram: [0-3Å, 3-6Å, 6-9Å, 9-12Å, 12-15Å]
        hist = edge_attr[:, :5]  # [num_edges, 5]
        
        # Classify edges based on which distance bin has highest atom density
        bins = {'close': [], 'medium': [], 'far': []}
        for i in range(hist.size(0)):
            # Get weighted average bin index (0-4)
            bin_weights = hist[i]
            avg_bin = torch.sum(torch.arange(5, dtype=torch.float32) * bin_weights).item()
            
            # Map to distance categories
            # avg_bin 0-1.5 = close (0-6Å), 1.5-3.0 = medium (6-12Å), 3.0+ = far (12-15Å)
            if avg_bin < 1.5:
                bins['close'].append(i)
            elif avg_bin < 3.0:
                bins['medium'].append(i)
            else:
                bins['far'].append(i)
        
        # Mask proportionally from each bin
        masked_indices = []
        for bin_name, indices in bins.items():
            if indices:
                n_mask = max(1, int(len(indices) * self.mask_ratio))
                sampled = np.random.choice(indices, min(n_mask, len(indices)), replace=False)
                masked_indices.extend(sampled)
        
        if not masked_indices:
            masked_indices = [0]  # Mask at least one edge
        
        masked_indices = np.array(masked_indices)
        
        # Get masked edge pairs (positive samples)
        positive_receptor_idx = edge_index[0, masked_indices]
        positive_ligand_idx = edge_index[1, masked_indices]
        
        # Create set of existing edges for fast lookup
        existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        
        # Generate negative samples (non-existing edges between interface nodes)
        n_negative = int(len(masked_indices) * self.negative_ratio)
        negative_pairs = []
        
        # Get all interface nodes
        receptor_interface = list(set(edge_index[0].tolist()))
        ligand_interface = list(set(edge_index[1].tolist()))
        
        attempts = 0
        max_attempts = n_negative * 10
        while len(negative_pairs) < n_negative and attempts < max_attempts:
            r_idx = np.random.choice(receptor_interface)
            l_idx = np.random.choice(ligand_interface)
            if (r_idx, l_idx) not in existing_edges:
                negative_pairs.append((r_idx, l_idx))
                existing_edges.add((r_idx, l_idx))
            attempts += 1
        
        # Remove masked edges from graph
        keep_mask = torch.ones(num_edges, dtype=torch.bool)
        keep_mask[masked_indices] = False
        
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
            'all_edges': all_edges,
            'edge_labels': edge_labels,
        }
        
        return data, mask_info


class CurriculumEdgeMasking:
    """
    Gradually increase masking difficulty during training.
    
    Start with easy task (low mask_ratio, easy negatives),
    progress to harder task (high mask_ratio, hard negatives).
    
    Note: This uses the standard MaskInterfaceEdges but with dynamic ratio.
    The ratio increases over training epochs via the step() method.
    
    Args:
        start_ratio: Initial mask ratio (e.g., 0.10)
        end_ratio: Final mask ratio (e.g., 0.25)
        num_steps: Number of curriculum steps
        negative_ratio: Ratio of negative to positive samples
    """
    def __init__(
        self,
        start_ratio: float = 0.10,
        end_ratio: float = 0.25,
        num_steps: int = 100,
        negative_ratio: float = 1.0,
    ):
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.num_steps = num_steps
        self.current_step = 0
        self.negative_ratio = negative_ratio
        
        # Import here to avoid circular imports
        from .ssl_transforms import MaskInterfaceEdges
        self.base_masker = None  # Will be created on first call
    
    def get_current_ratio(self) -> float:
        """Get current mask ratio based on curriculum progress."""
        progress = min(1.0, self.current_step / self.num_steps)
        return self.start_ratio + (self.end_ratio - self.start_ratio) * progress
    
    def step(self):
        """Advance curriculum by one step (call after each epoch)."""
        self.current_step += 1
        # Reset base masker to use new ratio
        self.base_masker = None
    
    def __call__(self, data: HeteroData) -> Tuple[HeteroData, Dict]:
        """Apply curriculum-based edge masking."""
        from .ssl_transforms import MaskInterfaceEdges
        
        # Create/update masker with current ratio
        current_ratio = self.get_current_ratio()
        if self.base_masker is None:
            self.base_masker = MaskInterfaceEdges(
                mask_ratio=current_ratio,
                negative_ratio=self.negative_ratio
            )
        else:
            # Update the ratio
            self.base_masker.mask_ratio = current_ratio
        
        return self.base_masker(data)


class HardNegativeMining:
    """
    Sample negative edges from spatially close but non-binding residue pairs.
    
    This is harder than random negatives because the model needs to learn
    fine-grained chemical/geometric constraints, not just spatial proximity.
    
    Since we don't have explicit distance values, we use the distance histogram
    to identify spatially proximate but non-binding pairs.
    
    Args:
        mask_ratio: Fraction of edges to mask
        hard_negative_ratio: Fraction of negatives to sample using proximity heuristic
        distance_threshold_bin: Which histogram bin to use as threshold (0-4, default 2 = ~8Å)
    """
    def __init__(
        self,
        mask_ratio: float = 0.2,
        hard_negative_ratio: float = 0.5,
        distance_threshold_bin: int = 2,
    ):
        self.mask_ratio = mask_ratio
        self.hard_negative_ratio = hard_negative_ratio
        self.distance_threshold_bin = distance_threshold_bin
    
    def get_proximate_node_pairs(
        self,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        existing_edges: set,
    ) -> List[Tuple[int, int]]:
        """
        Identify node pairs that are spatially close (based on histogram)
        but don't have actual binding edges.
        
        This is a heuristic approach: we look at existing edges and identify
        receptor/ligand nodes that appear in many edges (interface nodes),
        then sample pairs from these nodes that aren't already connected.
        """
        # Get interface nodes (nodes with multiple edges)
        receptor_nodes = edge_index[0].tolist()
        ligand_nodes = edge_index[1].tolist()
        
        from collections import Counter
        receptor_counts = Counter(receptor_nodes)
        ligand_counts = Counter(ligand_nodes)
        
        # High-degree nodes are likely at the interface
        receptor_interface = [n for n, c in receptor_counts.items() if c >= 2]
        ligand_interface = [n for n, c in ligand_counts.items() if c >= 2]
        
        # Sample pairs from interface nodes that aren't connected
        candidates = []
        for r in receptor_interface:
            for l in ligand_interface:
                if (r, l) not in existing_edges:
                    candidates.append((r, l))
        
        return candidates
    
    def __call__(self, data: HeteroData) -> Tuple[HeteroData, Dict]:
        """Apply hard negative mining edge masking."""
        data = copy.deepcopy(data)
        
        # Check if receptor_ligand edges exist
        if ('receptor', 'receptor_ligand', 'ligand') not in data.edge_index_dict:
            return data, {
                'positive_edges': torch.empty(2, 0),
                'negative_edges': torch.empty(2, 0),
                'all_edges': torch.empty(2, 0),
                'edge_labels': torch.empty(0)
            }
        
        edge_index = data[('receptor', 'receptor_ligand', 'ligand')].edge_index
        edge_attr = data[('receptor', 'receptor_ligand', 'ligand')].edge_attr
        num_edges = edge_index.size(1)
        
        # Sample edges to mask (standard random masking)
        n_mask = max(1, int(num_edges * self.mask_ratio))
        mask_edge_indices = np.random.choice(num_edges, n_mask, replace=False)
        
        # Get masked edge pairs (positive samples)
        positive_receptor_idx = edge_index[0, mask_edge_indices]
        positive_ligand_idx = edge_index[1, mask_edge_indices]
        
        # Create set of existing edges
        existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        
        # Generate negative samples: mix of hard and random
        n_negative = int(n_mask * (1.0 / self.hard_negative_ratio))  # Total negatives
        n_hard = int(n_negative * self.hard_negative_ratio)  # Hard negatives
        n_random = n_negative - n_hard  # Random negatives
        
        negative_pairs = []
        
        # Get hard negatives (from interface nodes)
        hard_candidates = self.get_proximate_node_pairs(edge_attr, edge_index, existing_edges)
        if hard_candidates and n_hard > 0:
            n_hard = min(n_hard, len(hard_candidates))
            hard_sampled = [hard_candidates[i] for i in np.random.choice(len(hard_candidates), n_hard, replace=False)]
            negative_pairs.extend(hard_sampled)
            for pair in hard_sampled:
                existing_edges.add(pair)
        
        # Get random negatives (from all interface nodes)
        receptor_interface = list(set(edge_index[0].tolist()))
        ligand_interface = list(set(edge_index[1].tolist()))
        
        attempts = 0
        max_attempts = n_random * 10
        while len(negative_pairs) < n_negative and attempts < max_attempts:
            r_idx = np.random.choice(receptor_interface)
            l_idx = np.random.choice(ligand_interface)
            if (r_idx, l_idx) not in existing_edges:
                negative_pairs.append((r_idx, l_idx))
                existing_edges.add((r_idx, l_idx))
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
            'all_edges': all_edges,
            'edge_labels': edge_labels,
        }
        
        return data, mask_info


def create_enhanced_masker(
    strategy: str = 'dynamic',
    mask_ratio: float = 0.2,
    **kwargs
):
    """
    Factory function to create enhanced edge masking strategies.
    
    Args:
        strategy: 'dynamic' (default), 'stratified', 'curriculum', or 'hard_negative'
        mask_ratio: Base masking ratio
        **kwargs: Strategy-specific arguments
    
    Returns:
        Masking transform
    """
    if strategy == 'stratified':
        return StratifiedEdgeMasking(mask_ratio=mask_ratio, **kwargs)
    elif strategy == 'curriculum':
        return CurriculumEdgeMasking(start_ratio=mask_ratio * 0.7, end_ratio=mask_ratio * 1.5, **kwargs)
    elif strategy == 'hard_negative':
        # Map negative_ratio to hard_negative_ratio for compatibility
        if 'negative_ratio' in kwargs:
            kwargs['hard_negative_ratio'] = kwargs.pop('negative_ratio')
        return HardNegativeMining(mask_ratio=mask_ratio, **kwargs)
    else:
        # Use the default dynamic masking
        from .ssl_transforms import MaskInterfaceEdges
        return MaskInterfaceEdges(mask_ratio=mask_ratio, **kwargs)
