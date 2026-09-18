"""
Hetero -> Homo graph conversion for the structure student (variant 2-1).

GraPPI complex graphs (HeteroData) already store each protein's intra-chain
edges (`receptor_receptor`, `ligand_ligand`) using node-local indices (0..N-1
within that protein) — so extracting a standalone homogeneous graph per
protein is a lossless slice of existing tensors, no reindexing required.

Caveat: since receptor and ligand come from the same complex PDB, these are
BOUND-conformation homo graphs (see README_STUDENT_DISTILLATION.md, "bound vs
unbound" risk). They're a fast, easy starting point; swapping in unbound /
AlphaFold-predicted monomer structures later only changes how the cache is
built, not how it's consumed downstream.
"""

import os
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data, HeteroData
from tqdm import tqdm

_INTRA_REL = {
    'receptor': ('receptor', 'receptor_receptor', 'receptor'),
    'ligand': ('ligand', 'ligand_ligand', 'ligand'),
}


def hetero_to_homo_pair(g: HeteroData) -> Dict[str, Data]:
    """
    Slice a complex HeteroData into two standalone homogeneous graphs, one per
    protein, keeping only intra-chain (self) edges. No node relabeling needed:
    `receptor_receptor` / `ligand_ligand` edge_index are already local to their
    protein's node range.
    """
    out: Dict[str, Data] = {}
    for nt in ('receptor', 'ligand'):
        et = _INTRA_REL[nt]
        x = g[nt].x
        if et in g.edge_types and hasattr(g[et], 'edge_index'):
            edge_index = g[et].edge_index
            edge_attr = g[et].edge_attr if hasattr(g[et], 'edge_attr') else None
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = None
        out[nt] = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))
    return out


def build_homo_graph_cache(
    sample_names: List[str],
    graphs: List[HeteroData],
    cache_dir: str,
) -> None:
    """
    Convert and cache homo graphs to `{cache_dir}/{name}.pt`, each file holding
    `{'receptor': Data, 'ligand': Data}`. Resumable: existing files are skipped.
    """
    os.makedirs(cache_dir, exist_ok=True)
    todo = [(n, g) for n, g in zip(sample_names, graphs)
            if not os.path.exists(os.path.join(cache_dir, f"{n}.pt"))]
    if not todo:
        print(f"Homo graph cache complete ({len(sample_names)} samples already cached).")
        return

    print(f"Building homo graph cache: {len(todo)} / {len(sample_names)} to convert "
          f"-> {cache_dir}")
    for name, g in tqdm(todo, desc="Hetero->Homo"):
        pair = hetero_to_homo_pair(g)
        torch.save(pair, os.path.join(cache_dir, f"{name}.pt"))


def load_homo_pair(name: str, cache_dir: str) -> Dict[str, Data]:
    """Load one cached {'receptor': Data, 'ligand': Data} pair by sample name."""
    return torch.load(os.path.join(cache_dir, f"{name}.pt"), weights_only=False)
