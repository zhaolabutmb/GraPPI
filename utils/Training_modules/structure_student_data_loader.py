"""
Data utilities for the structure student (variant 2-1): individual monomer
homo-graph inputs + cross-attention, distilled against the frozen GraPPI
teacher.

Reuses:
  * utils/Training_modules/homo_graph_cache.py — hetero->homo conversion/cache.
  * utils/Training_modules/student_data_loader.py — teacher embedding cache
    (build_teacher_cache, swap_hetero_roles), pad_and_mask, BucketBatchSampler.
    The teacher cache is keyed only by pdb name + teacher checkpoint, so the
    sequence student and structure student can share the same cache directory.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from utils.Training_modules.homo_graph_cache import load_homo_pair
from utils.Training_modules.student_data_loader import pad_and_mask  # noqa: F401 (re-exported)
from utils.Training_modules.edge_label_utils import load_inter_edges, assemble_edge_batch


class StructureStudentDistillDataset(Dataset):
    """
    One item per complex: (rec_graph, lig_graph, teacher_rec, teacher_lig, name).

    Homo graphs are loaded from the on-disk cache built by
    `build_homo_graph_cache`; teacher targets from the teacher cache built by
    `build_teacher_cache`. Optional role-swap augmentation mirrors the
    sequence student's (swaps receptor/ligand graphs + the swapped teacher
    target `{name}__swap.pt`).
    """

    def __init__(
        self,
        sample_names: List[str],
        homo_cache_dir: str,
        teacher_cache_dir: str,
        max_residues_per_protein: Optional[int] = 1000,
        swap_augment: bool = False,
    ):
        self.homo_cache_dir = homo_cache_dir
        self.teacher_cache_dir = teacher_cache_dir
        self.swap_augment = swap_augment
        self.items: List[str] = []
        self.swap_ok = set()
        n_skipped = 0

        for name in sample_names:
            homo_path = os.path.join(homo_cache_dir, f"{name}.pt")
            teacher_path = os.path.join(teacher_cache_dir, f"{name}.pt")
            if not (os.path.exists(homo_path) and os.path.exists(teacher_path)):
                continue
            pair = torch.load(homo_path, weights_only=False)
            n_rec = pair['receptor'].num_nodes
            n_lig = pair['ligand'].num_nodes
            if max_residues_per_protein and (n_rec > max_residues_per_protein
                                             or n_lig > max_residues_per_protein):
                n_skipped += 1
                continue
            self.items.append(name)
            if swap_augment and os.path.exists(
                    os.path.join(teacher_cache_dir, f"{name}__swap.pt")):
                self.swap_ok.add(name)

        if n_skipped:
            print(f"  StructureStudentDistillDataset: skipped {n_skipped} complexes "
                  f"over {max_residues_per_protein} residues/protein.")

        self._length_cache: Dict[str, int] = {}

    def lengths(self) -> List[int]:
        """max(N_rec, N_lig) per item, for length bucketing."""
        out = []
        for name in self.items:
            pair = torch.load(os.path.join(self.homo_cache_dir, f"{name}.pt"), weights_only=False)
            out.append(max(pair['receptor'].num_nodes, pair['ligand'].num_nodes))
        return out

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        name = self.items[idx]
        pair = torch.load(os.path.join(self.homo_cache_dir, f"{name}.pt"), weights_only=False)
        rec_g, lig_g = pair['receptor'], pair['ligand']

        if self.swap_augment and name in self.swap_ok and random.random() < 0.5:
            tgt = torch.load(os.path.join(self.teacher_cache_dir, f"{name}__swap.pt"))
            pos = load_inter_edges(f"{name}__swap", self.teacher_cache_dir)
            # swapped roles: old ligand graph feeds the receptor tower and vice versa
            return lig_g, rec_g, tgt['receptor'].float(), tgt['ligand'].float(), pos, name

        tgt = torch.load(os.path.join(self.teacher_cache_dir, f"{name}.pt"))
        pos = load_inter_edges(name, self.teacher_cache_dir)
        return rec_g, lig_g, tgt['receptor'].float(), tgt['ligand'].float(), pos, name


def collate_structure_student(batch, negative_ratio: float = 1.0):
    rec_graphs, lig_graphs, t_recs, t_ligs, pos_edges, names = zip(*batch)
    rec_batch = Batch.from_data_list(list(rec_graphs))
    lig_batch = Batch.from_data_list(list(lig_graphs))
    t_rec, _ = pad_and_mask(list(t_recs))
    t_lig, _ = pad_and_mask(list(t_ligs))
    n_recs = [g.num_nodes for g in rec_graphs]
    n_ligs = [g.num_nodes for g in lig_graphs]
    e_batch, e_ri, e_li, e_lab = assemble_edge_batch(
        list(pos_edges), n_recs, n_ligs, negative_ratio)
    return rec_batch, lig_batch, t_rec, t_lig, e_batch, e_ri, e_li, e_lab, list(names)
