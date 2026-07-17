"""
Self-Supervised Learning (SSL) modules for protein complex pre-training.

This package contains:
- ssl_data_loader: Data loading with masking strategies for nodes and edges
- ssl_transforms: Graph transforms for masked prediction tasks
"""

from .ssl_data_loader import (
    SSLHeteroGraphDataset,
    ssl_hetero_collate_fn,
    get_ssl_train_loader,
)
from .ssl_transforms import (
    MaskInterfaceNodes,
    MaskInterfaceEdges,
    MaskInterfaceIntraEdges,
    MaskInterAndIntraEdges,
    MaskInterfaceNodesAndEdges,
)

__all__ = [
    'SSLHeteroGraphDataset',
    'ssl_hetero_collate_fn',
    'get_ssl_train_loader',
    'MaskInterfaceNodes',
    'MaskInterfaceEdges',
    'MaskInterfaceIntraEdges',
    'MaskInterAndIntraEdges',
    'MaskInterfaceNodesAndEdges',
]
