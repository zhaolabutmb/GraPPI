"""
Sequence Student Models for GraPPI knowledge distillation.

The student is a *drop-in replacement* for the frozen GraPPI (EdgeEnhancedHGT)
encoder: given the two proteins' per-residue ESM embeddings it produces a
per-residue embedding dict {'receptor': [N_rec, H], 'ligand': [N_lig, H]} that
matches the teacher's final-layer output. No graph edges are consumed — only the
two sequences (their precomputed ESM residue embeddings).

Architecture: a small cross-encoder. Each block runs per-protein self-attention
followed by cross-attention to the partner protein, letting the two towers
exchange interface information without ever seeing the complex.

See README_STUDENT_DISTILLATION.md for the full design.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mult, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossEncoderBlock(nn.Module):
    """Self-attention per protein + cross-attention to the partner (pre-norm)."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        mha = lambda: nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.rec_self, self.lig_self = mha(), mha()
        self.rec_cross, self.lig_cross = mha(), mha()

        self.n_rec_self, self.n_lig_self = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.n_rec_cross_q, self.n_lig_cross_q = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.n_rec_cross_kv, self.n_lig_cross_kv = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.n_rec_ffn, self.n_lig_ffn = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)

        self.rec_ffn = _FeedForward(hidden_dim, ffn_mult, dropout)
        self.lig_ffn = _FeedForward(hidden_dim, ffn_mult, dropout)

    def forward(
        self,
        rec: torch.Tensor, lig: torch.Tensor,
        rec_mask: torch.Tensor, lig_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # rec/lig: [B, L, H]; masks: [B, L] with True = padding
        r = self.n_rec_self(rec)
        rec = rec + self.rec_self(r, r, r, key_padding_mask=rec_mask, need_weights=False)[0]
        l = self.n_lig_self(lig)
        lig = lig + self.lig_self(l, l, l, key_padding_mask=lig_mask, need_weights=False)[0]

        rq, lkv = self.n_rec_cross_q(rec), self.n_lig_cross_kv(lig)
        rec = rec + self.rec_cross(rq, lkv, lkv, key_padding_mask=lig_mask, need_weights=False)[0]
        lq, rkv = self.n_lig_cross_q(lig), self.n_rec_cross_kv(rec)
        lig = lig + self.lig_cross(lq, rkv, rkv, key_padding_mask=rec_mask, need_weights=False)[0]

        rec = rec + self.rec_ffn(self.n_rec_ffn(rec))
        lig = lig + self.lig_ffn(self.n_lig_ffn(lig))
        return rec, lig


class SequenceStudentEncoder(nn.Module):
    """ESM cross-encoder that mimics the frozen GraPPI encoder's node embeddings."""

    def __init__(
        self,
        esm_dim: int,
        hidden_dim: int,
        n_blocks: int = 3,
        n_heads: int = 8,
        dropout: float = 0.2,
        ffn_mult: int = 4,
        input_dropout: float = 0.0,
    ):
        super().__init__()
        self.esm_dim = esm_dim
        self.hidden_dim = hidden_dim

        self.in_drop = nn.Dropout(input_dropout)
        self.in_proj = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, hidden_dim))
        # Learned role bias to distinguish receptor from ligand towers.
        self.rec_role = nn.Parameter(torch.zeros(hidden_dim))
        self.lig_role = nn.Parameter(torch.zeros(hidden_dim))

        self.blocks = nn.ModuleList([
            CrossEncoderBlock(hidden_dim, n_heads, dropout, ffn_mult) for _ in range(n_blocks)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        rec_x: torch.Tensor, lig_x: torch.Tensor,
        rec_mask: torch.Tensor, lig_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rec_x/lig_x: [B, L, esm_dim] padded ESM residue embeddings.
            rec_mask/lig_mask: [B, L] boolean, True where padded.
        Returns:
            (rec_out, lig_out): each [B, L, hidden_dim].
        """
        rec = self.in_proj(self.in_drop(rec_x)) + self.rec_role
        lig = self.in_proj(self.in_drop(lig_x)) + self.lig_role
        for blk in self.blocks:
            rec, lig = blk(rec, lig, rec_mask, lig_mask)
        return self.out_norm(rec), self.out_norm(lig)


def create_student_model(
    esm_dim: int,
    hidden_dim: int,
    n_blocks: int = 3,
    n_heads: int = 8,
    dropout: float = 0.2,
    ffn_mult: int = 4,
    input_dropout: float = 0.0,
) -> SequenceStudentEncoder:
    """Factory for the sequence student encoder."""
    return SequenceStudentEncoder(
        esm_dim=esm_dim,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        n_heads=n_heads,
        dropout=dropout,
        ffn_mult=ffn_mult,
        input_dropout=input_dropout,
    )


def distillation_loss(
    s_rec: torch.Tensor, s_lig: torch.Tensor,
    t_rec: torch.Tensor, t_lig: torch.Tensor,
    rec_mask: torch.Tensor, lig_mask: torch.Tensor,
    mse_weight: float = 1.0,
    cosine_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Per-residue distillation loss over real (non-padded) residues only.

    L = cosine_weight * (1 - cos) + mse_weight * MSE, averaged over all
    receptor+ligand residues in the batch.

    Args:
        s_*/t_*: student/teacher embeddings [B, L, H].
        rec_mask/lig_mask: [B, L] boolean, True where padded.
    """
    valid_rec = ~rec_mask  # [B, L] True = real residue
    valid_lig = ~lig_mask
    s = torch.cat([s_rec[valid_rec], s_lig[valid_lig]], dim=0)  # [M, H]
    t = torch.cat([t_rec[valid_rec], t_lig[valid_lig]], dim=0)  # [M, H]

    s = s.float()
    t = t.float()
    cos = 1.0 - F.cosine_similarity(s, t, dim=-1).mean()
    mse = F.mse_loss(s, t)
    loss = cosine_weight * cos + mse_weight * mse
    return loss, {'cosine': cos.item(), 'mse': mse.item(), 'cos_sim': (1.0 - cos).item()}


class InterEdgePredictor(nn.Module):
    """
    Auxiliary edge-prediction head shared by both students. Scores a
    receptor-residue / ligand-residue pair as in-contact from their embeddings,
    mirroring the teacher's edge feature [h_i ‖ h_j ‖ |h_i−h_j| ‖ h_i⊙h_j].
    Training-time only — discarded at inference (the drop-in dict is unchanged).
    """

    def __init__(self, hidden_dim: int, edge_hidden: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        eh = edge_hidden or hidden_dim
        self.net = nn.Sequential(
            nn.Linear(4 * hidden_dim, eh),
            nn.LayerNorm(eh),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(eh, 1),
        )

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([h_i, h_j, (h_i - h_j).abs(), h_i * h_j], dim=-1)
        return self.net(feat).squeeze(-1)  # [E]


def edge_prediction_loss(
    edge_head: InterEdgePredictor,
    s_rec: torch.Tensor, s_lig: torch.Tensor,
    edge_batch: torch.Tensor, edge_ri: torch.Tensor,
    edge_li: torch.Tensor, edge_labels: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Binary contact-prediction loss over the sampled inter-edges. Endpoint
    embeddings are gathered from the dense [B, L, H] student tensors via
    (edge_batch, edge_ri/edge_li).
    """
    if edge_labels.numel() == 0:
        zero = s_rec.sum() * 0.0
        return zero, {
            'edge_bce': 0.0,
            'edge_acc': 0.0,
            'edge_logits': torch.empty(0),
            'edge_labels': torch.empty(0),
        }
    h_i = s_rec[edge_batch, edge_ri].float()  # [E, H]
    h_j = s_lig[edge_batch, edge_li].float()  # [E, H]
    logits = edge_head(h_i, h_j)
    loss = F.binary_cross_entropy_with_logits(logits, edge_labels)
    with torch.no_grad():
        acc = ((logits > 0).float() == edge_labels).float().mean().item()
    return loss, {
        'edge_bce': loss.item(),
        'edge_acc': acc,
        'edge_logits': logits.detach().float().cpu(),
        'edge_labels': edge_labels.detach().float().cpu(),
    }
