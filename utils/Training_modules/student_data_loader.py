"""
Data utilities for sequence-student knowledge distillation.

Provides:
  * build_teacher_cache: precompute frozen GraPPI final-layer embeddings for all
    complexes and store them on disk (per-name .pt, float16).
  * StudentDistillDataset: yields (rec_esm, lig_esm, teacher_rec, teacher_lig) per
    complex; ESM slices are read lazily from the in-memory graphs, teacher targets
    lazily from the disk cache.
  * collate_student / pad_and_mask: dynamic per-batch padding with padding masks.
  * BucketBatchSampler: length-bucketed batching to minimize padding overhead.
"""

import os
import random
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm

from utils.Training_modules.edge_label_utils import load_inter_edges, assemble_edge_batch


# Edge relation keys of the GraPPI hetero graph.
_REL = {
    'rr': ('receptor', 'receptor_receptor', 'receptor'),
    'll': ('ligand', 'ligand_ligand', 'ligand'),
    'rl': ('receptor', 'receptor_ligand', 'ligand'),
    'lr': ('ligand', 'ligand_receptor', 'receptor'),
}


def swap_hetero_roles(g: HeteroData) -> HeteroData:
    """
    Return a role-swapped copy of a complex graph (receptor <-> ligand).

    The GraPPI encoder is asymmetric (type-specific weights), so the teacher's
    output on this swapped graph is a *different, valid* target — used for
    order-swap augmentation. Swapping is a pure relabel: node/edge index tensors
    are reused unchanged under the new type names.
    """
    s = HeteroData()
    s['receptor'].x = g['ligand'].x
    s['ligand'].x = g['receptor'].x
    src_of = {'rr': 'll', 'll': 'rr', 'rl': 'lr', 'lr': 'rl'}  # new <- old
    for dst_key, src_key in src_of.items():
        src_et = _REL[src_key]
        if src_et in g.edge_types:
            dst_et = _REL[dst_key]
            s[dst_et].edge_index = g[src_et].edge_index
            if hasattr(g[src_et], 'edge_attr') and g[src_et].edge_attr is not None:
                s[dst_et].edge_attr = g[src_et].edge_attr
    return s


# ============================================================================
# Teacher embedding cache
# ============================================================================
@torch.no_grad()
def _encode_and_save(encoder, names, graphs, cache_dir, device, batch_size, amp_ctx):
    loader = GeoDataLoader(graphs, batch_size=batch_size, shuffle=False)
    graph_idx = 0
    for batch_data in tqdm(loader, desc="Teacher cache"):
        batch_data = batch_data.to(device)
        with amp_ctx:
            x_hgt = encoder(batch_data)  # {'receptor': [N_r, H], 'ligand': [N_l, H]}
        n_graphs = batch_data.num_graphs
        per_graph: Dict[int, Dict[str, torch.Tensor]] = {i: {} for i in range(n_graphs)}
        for nt in ('receptor', 'ligand'):
            batch_indices = batch_data[nt].batch.cpu()
            embs = x_hgt[nt].float().cpu()
            for i in range(n_graphs):
                per_graph[i][nt] = embs[batch_indices == i].half()
        for i in range(n_graphs):
            torch.save(per_graph[i], os.path.join(cache_dir, f"{names[graph_idx + i]}.pt"))
        graph_idx += n_graphs


@torch.no_grad()
def build_teacher_cache(
    encoder,
    sample_names: List[str],
    graphs: List,
    cache_dir: str,
    device: torch.device,
    batch_size: int = 16,
    use_amp: bool = True,
    also_swapped: bool = False,
) -> None:
    """
    Run the frozen encoder on every complex graph and cache its final-layer
    per-residue embeddings to `{cache_dir}/{name}.pt` as float16.

    If `also_swapped`, additionally cache the teacher output on the role-swapped
    graph as `{cache_dir}/{name}__swap.pt` (for order-swap augmentation).

    Resumable: names already present in the cache are skipped.
    """
    os.makedirs(cache_dir, exist_ok=True)
    amp_ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
               if (use_amp and device.type == 'cuda') else nullcontext())
    encoder.eval()

    todo_names, todo_graphs = [], []
    for name, g in zip(sample_names, graphs):
        if not os.path.exists(os.path.join(cache_dir, f"{name}.pt")):
            todo_names.append(name)
            todo_graphs.append(g)
    if todo_graphs:
        print(f"Building teacher cache: {len(todo_graphs)} / {len(sample_names)} to compute "
              f"-> {cache_dir}")
        _encode_and_save(encoder, todo_names, todo_graphs, cache_dir, device, batch_size, amp_ctx)
    else:
        print(f"Teacher cache complete ({len(sample_names)} samples already cached).")

    if also_swapped:
        sw_names, sw_graphs = [], []
        for name, g in zip(sample_names, graphs):
            if not os.path.exists(os.path.join(cache_dir, f"{name}__swap.pt")):
                sw_names.append(f"{name}__swap")
                sw_graphs.append(swap_hetero_roles(g))
        if sw_graphs:
            print(f"Building SWAPPED teacher cache: {len(sw_graphs)} to compute.")
            _encode_and_save(encoder, sw_names, sw_graphs, cache_dir, device, batch_size, amp_ctx)
        else:
            print("Swapped teacher cache complete.")


# ============================================================================
# Dataset + collation
# ============================================================================
class StudentDistillDataset(Dataset):
    """
    One item per complex: (rec_esm, lig_esm, teacher_rec, teacher_lig, name).

    ESM slices are views into the in-memory graph node features (no copy);
    teacher targets are loaded from the disk cache (or a preloaded RAM dict).

    swap_augment: with 50% probability per access, feed the role-swapped inputs
    (ligand->receptor tower, receptor->ligand tower) paired with the swapped
    teacher target `{name}__swap.pt`. Requires a swapped cache; only applied to
    complexes that have one.
    """

    def __init__(
        self,
        sample_names: List[str],
        graphs: List,
        cache_dir: str,
        esm_dim: int,
        max_residues_per_protein: Optional[int] = 1000,
        swap_augment: bool = False,
        preload: bool = False,
    ):
        self.cache_dir = cache_dir
        self.esm_dim = esm_dim
        self.swap_augment = swap_augment
        self.preload = preload
        self.items: List[Tuple[str, object]] = []
        self.swap_ok = set()
        n_skipped = 0
        for name, g in zip(sample_names, graphs):
            n_rec = g['receptor'].x.size(0)
            n_lig = g['ligand'].x.size(0)
            if max_residues_per_protein and (n_rec > max_residues_per_protein
                                             or n_lig > max_residues_per_protein):
                n_skipped += 1
                continue
            if not os.path.exists(os.path.join(cache_dir, f"{name}.pt")):
                continue
            self.items.append((name, g))
            if swap_augment and os.path.exists(os.path.join(cache_dir, f"{name}__swap.pt")):
                self.swap_ok.add(name)
        if n_skipped:
            print(f"  StudentDistillDataset: skipped {n_skipped} complexes over "
                  f"{max_residues_per_protein} residues/protein.")

        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}
        if preload:
            for name, _ in tqdm(self.items, desc="Preloading teacher cache"):
                self._cache[name] = torch.load(os.path.join(cache_dir, f"{name}.pt"))
                if name in self.swap_ok:
                    self._cache[f"{name}__swap"] = torch.load(
                        os.path.join(cache_dir, f"{name}__swap.pt"))

    def _load_target(self, key: str) -> Dict[str, torch.Tensor]:
        if self.preload:
            return self._cache[key]
        return torch.load(os.path.join(self.cache_dir, f"{key}.pt"))

    def lengths(self) -> List[int]:
        """max(N_rec, N_lig) per item, for length bucketing."""
        return [max(g['receptor'].x.size(0), g['ligand'].x.size(0)) for _, g in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        name, g = self.items[idx]
        rec = g['receptor'].x[:, :self.esm_dim].float()
        lig = g['ligand'].x[:, :self.esm_dim].float()

        if self.swap_augment and name in self.swap_ok and random.random() < 0.5:
            tgt = self._load_target(f"{name}__swap")
            pos = load_inter_edges(f"{name}__swap", self.cache_dir)
            # swapped roles: old ligand feeds the receptor tower and vice versa
            return lig, rec, tgt['receptor'].float(), tgt['ligand'].float(), pos, name

        tgt = self._load_target(name)
        pos = load_inter_edges(name, self.cache_dir)
        return rec, lig, tgt['receptor'].float(), tgt['ligand'].float(), pos, name


def pad_and_mask(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of [N_i, D] tensors to [B, L_max, D] and build a [B, L_max]
    boolean padding mask (True where padded).
    """
    lengths = [t.size(0) for t in tensors]
    padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)  # [B, L_max, D]
    b, l_max = padded.size(0), padded.size(1)
    mask = torch.ones(b, l_max, dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = False
    return padded, mask


def collate_student(batch, negative_ratio: float = 1.0):
    recs, ligs, t_recs, t_ligs, pos_edges, names = zip(*batch)
    rec_x, rec_mask = pad_and_mask(list(recs))
    lig_x, lig_mask = pad_and_mask(list(ligs))
    t_rec, _ = pad_and_mask(list(t_recs))
    t_lig, _ = pad_and_mask(list(t_ligs))
    n_recs = [r.size(0) for r in recs]
    n_ligs = [l.size(0) for l in ligs]
    e_batch, e_ri, e_li, e_lab = assemble_edge_batch(
        list(pos_edges), n_recs, n_ligs, negative_ratio)
    return rec_x, lig_x, rec_mask, lig_mask, t_rec, t_lig, e_batch, e_ri, e_li, e_lab, list(names)


class BucketBatchSampler(Sampler):
    """Group similar-length complexes into batches to reduce padding waste."""

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
        bucket_mult: int = 50,
        seed: int = 42,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = max(batch_size, batch_size * bucket_mult)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _make_batches(self) -> List[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            idx = [idx[i] for i in torch.randperm(len(idx), generator=g).tolist()]
        batches: List[List[int]] = []
        for i in range(0, len(idx), self.bucket_size):
            chunk = sorted(idx[i:i + self.bucket_size], key=lambda j: self.lengths[j])
            for j in range(0, len(chunk), self.batch_size):
                batches.append(chunk[j:j + self.batch_size])
        if self.shuffle:
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in order]
        return batches

    def __iter__(self):
        yield from self._make_batches()

    def __len__(self) -> int:
        n = len(self.lengths)
        return (n + self.batch_size - 1) // self.batch_size
