"""
Inter-molecular (interface) edge labels for student distillation.

The teacher's SSL pretext is edge prediction; the natural auxiliary distillation
target for the students is to predict the *inter-molecular* (receptor<->ligand)
contact edges from their own per-residue embeddings. These are hard labels read
straight from the complex graph (`receptor_ligand` edges) — ground-truth
interface contacts, no teacher inference needed.

Shared by both the sequence and structure students. Provides:
  * extract_inter_edges: pull positive contact pairs from a complex HeteroData.
  * build_inter_edge_cache: cache positives (and role-swapped positives) to disk.
  * load_inter_edges: reload a cached positive-edge tensor.
  * assemble_edge_batch: positives + sampled negatives -> batched edge tensors.
"""

import os
from typing import List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

_INTER_REL = ('receptor', 'receptor_ligand', 'ligand')


def extract_inter_edges(g: HeteroData) -> torch.Tensor:
    """
    Return positive interface contact pairs as LongTensor [2, E_pos] with
    row 0 = receptor residue index, row 1 = ligand residue index.
    """
    if _INTER_REL in g.edge_types and hasattr(g[_INTER_REL], 'edge_index'):
        return g[_INTER_REL].edge_index.detach().cpu().long().clone()
    return torch.empty(2, 0, dtype=torch.long)


def build_inter_edge_cache(
    sample_names: List[str],
    graphs: List[HeteroData],
    cache_dir: str,
    also_swapped: bool = False,
) -> None:
    """
    Cache positive interface edges to `{cache_dir}/{name}.edges.pt`. If
    `also_swapped`, also cache role-swapped positives to `{name}__swap.edges.pt`
    (receptor<->ligand roles exchanged: just swap the two index rows).

    Teacher-independent (edges are ground truth from the graph). Resumable.
    """
    os.makedirs(cache_dir, exist_ok=True)
    todo = [(n, g) for n, g in zip(sample_names, graphs)
            if not os.path.exists(os.path.join(cache_dir, f"{n}.edges.pt"))]
    if not todo:
        print(f"Inter-edge cache complete ({len(sample_names)} samples already cached).")
    else:
        print(f"Building inter-edge cache: {len(todo)} / {len(sample_names)} -> {cache_dir}")
        for name, g in tqdm(todo, desc="Inter-edge cache"):
            pos = extract_inter_edges(g)
            torch.save(pos, os.path.join(cache_dir, f"{name}.edges.pt"))
            if also_swapped:
                torch.save(pos.flip(0), os.path.join(cache_dir, f"{name}__swap.edges.pt"))


def load_inter_edges(key: str, cache_dir: str) -> torch.Tensor:
    """Load cached positive edges for a sample key (empty tensor if missing)."""
    path = os.path.join(cache_dir, f"{key}.edges.pt")
    if not os.path.exists(path):
        return torch.empty(2, 0, dtype=torch.long)
    return torch.load(path, weights_only=False)


def assemble_edge_batch(
    inter_edges_list: List[torch.Tensor],
    n_rec_list: List[int],
    n_lig_list: List[int],
    negative_ratio: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build batched edge-prediction targets: all positives + uniformly sampled
    negatives per complex.

    Returns (edge_batch, edge_rec_idx, edge_lig_idx, edge_labels), each 1-D and
    aligned; edge_batch indexes the sample within the batch so endpoint
    embeddings can be gathered from dense [B, L, H] tensors.
    """
    b_idx, ri, li, labels = [], [], [], []
    for b, (pos, n_rec, n_lig) in enumerate(zip(inter_edges_list, n_rec_list, n_lig_list)):
        p = pos.size(1)
        if p == 0:
            continue
        b_idx.append(torch.full((p,), b, dtype=torch.long))
        ri.append(pos[0].long())
        li.append(pos[1].long())
        labels.append(torch.ones(p))

        n_neg = int(round(negative_ratio * p))
        total = n_rec * n_lig
        if n_neg > 0 and total > p:
            pos_keys = pos[0].long() * n_lig + pos[1].long()
            cand = torch.randint(0, total, (n_neg * 2,))
            cand = cand[~torch.isin(cand, pos_keys)][:n_neg]
            if cand.numel() > 0:
                nr, nl = cand // n_lig, cand % n_lig
                b_idx.append(torch.full((cand.numel(),), b, dtype=torch.long))
                ri.append(nr.long())
                li.append(nl.long())
                labels.append(torch.zeros(cand.numel()))

    if not b_idx:
        z = torch.empty(0, dtype=torch.long)
        return z, z, z, torch.empty(0)
    return torch.cat(b_idx), torch.cat(ri), torch.cat(li), torch.cat(labels)
