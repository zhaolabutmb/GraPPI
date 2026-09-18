"""
Structure Student Models (variant 2-1): individual monomer structure graphs +
cross-attention, distilled against the frozen GraPPI teacher.

Unlike the sequence student (ESM only, no edges), this student consumes each
protein's own intra-chain homogeneous graph (node features + the same 8-dim
distance/direction edge features GraPPI uses), processed by a stack of GATv2
(graph attention v2) layers that natively consume edge features, followed by
the same cross-attention exchange mechanism as the sequence student so the two
towers can exchange interface context.

Output contract matches the sequence student and the teacher: a dict
{'receptor': [N_rec, H], 'ligand': [N_lig, H]}. See
README_STUDENT_DISTILLATION.md for the overall design.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch

from model_collection.StudentModels import CrossEncoderBlock, distillation_loss  # noqa: F401 (re-exported)


class GATv2Layer(nn.Module):
    """One GATv2 layer (edge-feature-aware attention) + residual + LayerNorm."""

    def __init__(self, hidden_dim: int, edge_in_dim: int, heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.conv = GATv2Conv(
            hidden_dim, hidden_dim, heads=heads, concat=False,
            edge_dim=edge_in_dim, dropout=dropout, add_self_loops=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        out = self.conv(x, edge_index, edge_attr=edge_attr)
        return self.norm(x + self.drop(out))


class HomoGNNTower(nn.Module):
    """Stack of GATv2 layers, shared between the receptor and ligand towers."""

    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int,
                 n_layers: int = 3, heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.in_proj = nn.Sequential(nn.LayerNorm(node_in_dim), nn.Linear(node_in_dim, hidden_dim))
        self.layers = nn.ModuleList([
            GATv2Layer(hidden_dim, edge_in_dim, heads=heads, dropout=dropout) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        return h


class StructureStudentEncoder(nn.Module):
    """
    Two-tower structure student: shared homo-GNN encodes each protein's own
    structure graph, then cross-attention blocks let receptor/ligand exchange
    interface context (same mechanism as the sequence student).
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int,
        n_gnn_layers: int = 3,
        gnn_heads: int = 4,
        n_cross_blocks: int = 3,
        n_heads: int = 8,
        dropout: float = 0.2,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gnn = HomoGNNTower(node_in_dim, edge_in_dim, hidden_dim, n_gnn_layers,
                                heads=gnn_heads, dropout=dropout)
        self.rec_role = nn.Parameter(torch.zeros(hidden_dim))
        self.lig_role = nn.Parameter(torch.zeros(hidden_dim))
        self.blocks = nn.ModuleList([
            CrossEncoderBlock(hidden_dim, n_heads, dropout, ffn_mult) for _ in range(n_cross_blocks)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, rec_batch: Batch, lig_batch: Batch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            rec_batch/lig_batch: PyG Batch of per-protein Data objects
                (x, edge_index, edge_attr) for the receptor / ligand towers.
        Returns:
            (rec_dense, lig_dense, rec_pad_mask, lig_pad_mask) — dense
            [B, L_max, H] embeddings + boolean padding masks (True = padded),
            matching the sequence student's interface.
        """
        rec_h = self.gnn(rec_batch.x, rec_batch.edge_index, rec_batch.edge_attr)
        lig_h = self.gnn(lig_batch.x, lig_batch.edge_index, lig_batch.edge_attr)

        rec_dense, rec_valid = to_dense_batch(rec_h, rec_batch.batch)  # [B, Lr, H], [B, Lr]
        lig_dense, lig_valid = to_dense_batch(lig_h, lig_batch.batch)
        rec_mask = ~rec_valid
        lig_mask = ~lig_valid

        rec = rec_dense + self.rec_role
        lig = lig_dense + self.lig_role
        for blk in self.blocks:
            rec, lig = blk(rec, lig, rec_mask, lig_mask)

        return self.out_norm(rec), self.out_norm(lig), rec_mask, lig_mask


def create_structure_student_model(
    node_in_dim: int,
    edge_in_dim: int,
    hidden_dim: int,
    n_gnn_layers: int = 3,
    gnn_heads: int = 4,
    n_cross_blocks: int = 3,
    n_heads: int = 8,
    dropout: float = 0.2,
    ffn_mult: int = 4,
) -> StructureStudentEncoder:
    """Factory for the structure student encoder."""
    return StructureStudentEncoder(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=hidden_dim,
        n_gnn_layers=n_gnn_layers,
        gnn_heads=gnn_heads,
        n_cross_blocks=n_cross_blocks,
        n_heads=n_heads,
        dropout=dropout,
        ffn_mult=ffn_mult,
    )
