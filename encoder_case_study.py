"""
Encoder Case Study: Embedding Quality Assessment & B-factor PDB Generation.

Compares embeddings from a **trained** SSL encoder vs an **untrained**
encoder (same architecture, no training) to assess whether SSL pre-training
produces meaningfully structured residue representations.

Up to five conditions are compared:
  - **trained**: SSL-pretrained encoder weights.
  - **untrained**: same architecture, randomly initialised (no training).
  - **shuffled**: trained encoder embeddings with residue order randomly
    permuted within each chain (destroys spatial structure, preserves marginal
    distribution).  Averaged over *N* independent shuffles.
  - **esm** (optional, ``--compare2esm``): raw ESM-2 (650M, dim=1280) per-residue
    embeddings — no graph encoder, just the protein language model.
  - **esm480** (optional, ``--compare2esm``): raw ESM-2 (150M, dim=480)
    per-residue embeddings.

Two quantitative tests are run for each PDB complex:
  1. Interface-enrichment test:
       - Compute per-residue 1-D scores via PCA/Isomap (sign-oriented so that
         interface residues have higher scores).
       - Report mean score for interface vs non-interface residues
         and a Mann-Whitney U p-value.
  2. Embedding-norm separation:
       - Compute L2 norm per residue.
       - Report mean L2 for interface vs non-interface, plus Mann-Whitney U.

Additionally, for each PDB and encoder, the script writes a B-factor-colored
PDB file where B-factor = normalised 1-D embedding score, suitable for
visualisation in PyMOL (``spectrum b``).

Usage:
    python encoder_case_study.py \\
        --pretrained_path ../GraPPI_data/trained_data_ssl_finetune/2layers_10hdim_esm480/ \\
        --pdb_names 6isc 1us7 2vn5 4bi8 \\
        --cuda_id 0 \\
        --dr_method pca \\
        --output_dir ../GraPPI_data/trained_data_ssl_finetune/bfactor_pdbs_dir
    python encoder_case_study.py --pretrained_path ../GraPPI_data/trained_data_ssl_finetune/2layers_10hdim_esm480/ --pdb_names 6isc 1us7 2vn5 4bi8 --cuda_id 0 --dr_method pca --output_dir ../GraPPI_data/trained_data_ssl_finetune/bfactor_pdbs_dir/ --n_shuffle 20
"""

import argparse
import csv
import os
import pickle
import json
import random

import numpy as np
import torch
from Bio import PDB as BioPDB
from scipy.stats import mannwhitneyu
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model_collection.EMP_HGT import EdgeEnhancedHGT
from utils.Training_modules.common_utils import (
    load_ssl_config,
    MODEL_INIT_DIM, EDGE_IN_DIM, METADATA,
)
from utils.Quantity_compute import dim_reduce_1d


# ── constants ────────────────────────────────────────────────────────────────

STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'SEC', 'PYL',
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../GraPPI/
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)                 # ..


# ── helpers ──────────────────────────────────────────────────────────────────

def build_encoder(ssl_config, device):
    """Return a freshly constructed (untrained) encoder on *device*."""
    node_in_dim = MODEL_INIT_DIM[ssl_config['embedding_type']]
    hidden_dim = 2 ** ssl_config['hidden_dim_power']
    encoder = EdgeEnhancedHGT(
        node_in_dim=node_in_dim,
        edge_in_dim=EDGE_IN_DIM,
        hidden_dim=hidden_dim,
        metadata=METADATA,
        hgt_heads=ssl_config['hgt_heads'],
        hgt_layers=ssl_config['num_layers'],
        message_style=ssl_config.get('message_style', 'gated_src'),
        use_checkpoint=False,
    )
    return encoder.to(device).eval()


def compute_embeddings(encoder, sthg_dict, target_pdbs, device, batch_size=50):
    """Return {pdb_name: {'receptor': np.array, 'ligand': np.array}}."""
    graphs = []
    names = []
    for name in target_pdbs:
        if name not in sthg_dict:
            print(f"  Warning: {name} not in loaded graphs, skipping")
            continue
        graphs.append(sthg_dict[name].protein_graph)
        names.append(name)

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, drop_last=False)
    embeddings = {}
    graph_idx = 0

    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Encoding", leave=False):
            batch_data = batch_data.to(device)
            x_hgt = encoder(batch_data)
            num_in_batch = batch_data.num_graphs
            for nt in ['receptor', 'ligand']:
                batch_idx = batch_data[nt].batch.cpu().numpy()
                embs = x_hgt[nt].float().cpu().numpy()
                for i in range(num_in_batch):
                    name = names[graph_idx + i]
                    embeddings.setdefault(name, {})[nt] = embs[batch_idx == i]
            graph_idx += num_in_batch

    return embeddings


def get_interface_mask(sthg):
    """Return boolean masks (receptor_mask, ligand_mask) where True = interface."""
    pg = sthg.protein_graph
    edge_data = pg[('receptor', 'receptor_ligand', 'ligand')].edge_index
    iface_rec = set(edge_data[0].tolist())
    iface_lig = set(edge_data[1].tolist())

    n_rec = pg['receptor'].x.shape[0]
    n_lig = pg['ligand'].x.shape[0]

    rec_mask = np.zeros(n_rec, dtype=bool)
    lig_mask = np.zeros(n_lig, dtype=bool)
    for idx in iface_rec:
        if idx < n_rec:
            rec_mask[idx] = True
    for idx in iface_lig:
        if idx < n_lig:
            lig_mask[idx] = True

    return rec_mask, lig_mask


def sign_orient_scores(scores, interface_mask):
    """Flip sign of scores so that interface residues have higher mean."""
    if interface_mask.sum() == 0 or (~interface_mask).sum() == 0:
        return scores
    if scores[interface_mask].mean() < scores[~interface_mask].mean():
        scores = -scores
    return scores


def resolve_pdb_path(raw_path):
    """Resolve a (possibly relative) PDB path stored in row_of_df['paths']."""
    # 1. Try relative to script dir (GraPPI/)
    p = os.path.normpath(os.path.join(SCRIPT_DIR, raw_path))
    if os.path.isfile(p):
        return p
    # 2. Try relative to project dir (..)
    p = os.path.normpath(os.path.join(PROJECT_DIR, raw_path))
    if os.path.isfile(p):
        return p
    # 3. Strip all leading '../' and try from project dir
    stripped = raw_path
    while stripped.startswith('../') or stripped.startswith('..\\'): 
        stripped = stripped[3:]
    p = os.path.normpath(os.path.join(PROJECT_DIR, stripped))
    if os.path.isfile(p):
        return p
    return None


def std_residues(chain):
    """Return list of standard-AA residues from a Bio.PDB Chain (same filter as graph builder)."""
    out = []
    for res in chain.get_residues():
        hetfield = res.get_id()[0]
        resname = res.get_resname().strip()
        if hetfield == ' ' or resname in STANDARD_AA:
            out.append(res)
    return out


def clean_and_renumber_chain(chain, start=1):
    """Strip non-standard residues (HETATM: water, ions, ligands) and renumber.

    Keeps only standard amino-acid residues (see ``std_residues``) and
    renumbers them sequentially starting from *start*, collapsing any gaps
    left by truncated/missing regions.  Modifies *chain* in place.

    Returns the next available residue number, so numbering can continue
    across chains (e.g. chain A -> 1..200, chain B -> 201..250).
    """
    keep = std_residues(chain)
    # Detach every residue first to avoid id collisions during renumbering.
    for res in list(chain.get_residues()):
        chain.detach_child(res.get_id())
    # Re-add the standard residues with clean sequential numbering from *start*.
    for offset, res in enumerate(keep):
        res.id = (' ', start + offset, ' ')
        chain.add(res)
    return start + len(keep)



# ── target PDB resolution ────────────────────────────────────────────────────

def resolve_target_pdbs(pretrained_path, pdb_names, val_ratio, seed=42):
    """Build a deduplicated list of PDB names from explicit names and/or val set.

    Args:
        pretrained_path: directory containing ssl_train_val_split.json
        pdb_names: list of explicitly specified PDB names, or None
        val_ratio: fraction of validation set to sample (0 = skip)
        seed: random seed for reproducible sampling

    Returns:
        List of unique PDB name strings.
    """
    has_explicit = pdb_names is not None and len(pdb_names) > 0
    has_val = val_ratio is not None and val_ratio > 0

    if not has_explicit and not has_val:
        raise ValueError(
            "No PDBs specified. Use --pdb_names and/or --val_set_sampled_ratio.")

    target_set = set()

    # Explicit PDB names
    if has_explicit:
        for n in pdb_names:
            target_set.add(n.lower())

    # Sample from validation set
    if has_val:
        split_path = os.path.join(pretrained_path, 'ssl_train_val_split.json')
        if not os.path.isfile(split_path):
            raise FileNotFoundError(
                f"ssl_train_val_split.json not found at {split_path}. "
                "Cannot sample validation PDBs.")
        with open(split_path, 'r') as f:
            ssl_split = json.load(f)
        val_names = [n.lower() for n in ssl_split['val_names']]
        n_sample = max(1, int(len(val_names) * val_ratio))
        if val_ratio >= 1.0:
            sampled = val_names
        else:
            rng = random.Random(seed)
            sampled = rng.sample(val_names, min(n_sample, len(val_names)))
        n_overlap = len(target_set & set(sampled))
        target_set.update(sampled)
        print(f"Val set: {len(val_names)} total, sampled {len(sampled)} "
              f"(ratio={val_ratio}), overlap with explicit={n_overlap}")

    target_list = sorted(target_set)
    print(f"Target PDBs to analyse: {len(target_list)}")
    return target_list


# ── ESM embedding extraction ─────────────────────────────────────────────────

def load_esm_embeddings(pdb_root, esm_type, target_pdbs, dist=8):
    """Load raw ESM embeddings for *target_pdbs* and return in standard format.

    Args:
        pdb_root: root directory of curated database
        esm_type: 'esm' (dim 1280) or 'esm480' (dim 480)
        target_pdbs: list of PDB name strings
        dist: distance threshold used in graph construction (default 8)

    Returns:
        emb_dict: {pdb_name: {'receptor': np.array, 'ligand': np.array}}
    """
    # Load unmut dict
    unmut_path = os.path.join(pdb_root, f'unmut_seq_dicts_{esm_type}_{dist}A_hg.pkl')
    esm_dict = {}
    if os.path.isfile(unmut_path):
        with open(unmut_path, 'rb') as f:
            esm_dict = pickle.load(f)
        print(f"Loaded ESM dict ({esm_type}) with {len(esm_dict)} entries from {unmut_path}")
    else:
        print(f"  Warning: unmut ESM dict not found at {unmut_path}")

    # Load dimer dict as fallback for PDBs not in unmut
    dimer_path = os.path.join(pdb_root, f'dimer_seq_dicts_{esm_type}_{dist}A_hg.pkl')
    dimer_dict = {}
    if os.path.isfile(dimer_path):
        with open(dimer_path, 'rb') as f:
            dimer_dict = pickle.load(f)
        print(f"Loaded dimer ESM dict ({esm_type}) with {len(dimer_dict)} entries from {dimer_path}")

    if not esm_dict and not dimer_dict:
        print(f"  Warning: no ESM dicts found for {esm_type}, skipping")
        return None

    emb_out = {}
    for pdb_n in target_pdbs:
        if pdb_n in esm_dict:
            entry = esm_dict[pdb_n]
        elif pdb_n in dimer_dict:
            entry = dimer_dict[pdb_n]
        else:
            continue
        rec_emb = entry['receptor']['seq_emb']
        lig_emb = entry['ligand']['seq_emb']
        # Convert torch tensors to numpy if needed
        if hasattr(rec_emb, 'numpy'):
            rec_emb = rec_emb.float().numpy()
        if hasattr(lig_emb, 'numpy'):
            lig_emb = lig_emb.float().numpy()
        emb_out[pdb_n] = {'receptor': np.asarray(rec_emb, dtype=np.float32),
                          'ligand':   np.asarray(lig_emb, dtype=np.float32)}
    print(f"  Extracted {esm_type} embeddings for {len(emb_out)}/{len(target_pdbs)} target PDBs")
    return emb_out


# ── shuffled-embedding control ───────────────────────────────────────────────

def shuffle_embeddings(emb_dict, rng=None):
    """Return a copy of *emb_dict* with residue embeddings permuted per chain.

    For each PDB, independently shuffle the row order of the receptor and
    ligand embedding matrices.  This destroys positional (spatial) structure
    while preserving the marginal distribution of embedding vectors.
    """
    if rng is None:
        rng = np.random.default_rng()
    shuffled = {}
    for pdb_n, chains in emb_dict.items():
        shuffled[pdb_n] = {}
        for chain_key, arr in chains.items():
            idx = rng.permutation(len(arr))
            shuffled[pdb_n][chain_key] = arr[idx]
    return shuffled


# ── statistical comparison ───────────────────────────────────────────────────

def _score_one_condition(emb_dict, pdb_n, n_rec, full_mask, dr_method):
    """Compute DR-enrichment and L2-norm stats for one condition on one PDB."""
    emb_rec = emb_dict[pdb_n]['receptor']
    emb_lig = emb_dict[pdb_n]['ligand']
    emb_all = np.vstack([emb_rec, emb_lig])

    scores = dim_reduce_1d(dr_method, emb_all)
    scores = sign_orient_scores(scores, full_mask)

    chain_scores = {'receptor': scores[:n_rec], 'ligand': scores[n_rec:]}

    iface_scores = scores[full_mask]
    non_iface_scores = scores[~full_mask]
    if len(iface_scores) > 0 and len(non_iface_scores) > 0:
        _, pval = mannwhitneyu(iface_scores, non_iface_scores, alternative='greater')
    else:
        pval = float('nan')

    norms = np.linalg.norm(emb_all, axis=1)
    iface_norms = norms[full_mask]
    non_iface_norms = norms[~full_mask]
    if len(iface_norms) > 0 and len(non_iface_norms) > 0:
        _, norm_pval = mannwhitneyu(iface_norms, non_iface_norms, alternative='greater')
    else:
        norm_pval = float('nan')

    stats = {
        'dr_method': dr_method,
        'n_interface': int(full_mask.sum()),
        'n_non_interface': int((~full_mask).sum()),
        'dr_iface_mean': float(np.nanmean(iface_scores)),
        'dr_non_iface_mean': float(np.nanmean(non_iface_scores)),
        'dr_mann_whitney_p': float(pval),
        'l2_iface_mean': float(np.nanmean(iface_norms)),
        'l2_non_iface_mean': float(np.nanmean(non_iface_norms)),
        'l2_mann_whitney_p': float(norm_pval),
    }
    return stats, chain_scores


def _averaged_shuffled_stats(emb_src, pdb_n, n_rec, full_mask, dr_method, n_shuffle):
    """Average DR/L2 stats over *n_shuffle* independent residue permutations."""
    accum = None
    for s in range(n_shuffle):
        rng = np.random.default_rng(seed=s)
        emb_shuf = shuffle_embeddings(emb_src, rng=rng)
        stats_s, _ = _score_one_condition(
            emb_shuf, pdb_n, n_rec, full_mask, dr_method)
        if accum is None:
            accum = {k: v for k, v in stats_s.items()}
        else:
            for k in accum:
                if isinstance(accum[k], float):
                    accum[k] += stats_s[k]
    for k in accum:
        if isinstance(accum[k], float):
            accum[k] /= n_shuffle
    accum['n_shuffle'] = n_shuffle
    return accum


def run_comparison(emb_trained, emb_untrained, sthg_dict, pdb_names, dr_method,
                   n_shuffle=10, extra_embs=None, emb_base=None,
                   label_trained='GraPPI',
                   label_untrained='untrained GraPPI',
                   label_shuffled='shuffled GraPPI',
                   label_shuffled_base='shuffled GraPPI base'):
    """Run interface-enrichment and norm-separation tests for all conditions.

    Args:
        emb_trained: trained encoder embeddings
        emb_untrained: untrained encoder embeddings
        sthg_dict: loaded StructureToHeteroGraph objects
        pdb_names: list of PDB names to analyse
        dr_method: dimensionality reduction method
        n_shuffle: number of shuffle repetitions
        extra_embs: optional dict of {label: emb_dict} for additional conditions
                    (e.g. {'GraPPI base': {...}, 'esm': {...}})
        emb_base: optional GraPPI base encoder embeddings.  If provided, a
            shuffled control of these embeddings is added as an extra condition
            (averaged over *n_shuffle* permutations).
        label_trained: condition label for the trained encoder (default 'GraPPI')
        label_untrained: condition label for the untrained encoder
            (default 'untrained GraPPI')
        label_shuffled: condition label for the shuffled control
            (default 'shuffled GraPPI')
        label_shuffled_base: condition label for the shuffled GraPPI base control
            (default 'shuffled GraPPI base')

    Returns (results_dict, dr_scores_dict).
    dr_scores_dict maps  pdb -> condition_label -> {'receptor': array, 'ligand': array}
    """
    results = {}
    dr_scores = {}

    for pdb_n in pdb_names:
        if pdb_n not in emb_trained or pdb_n not in emb_untrained:
            continue
        if pdb_n not in sthg_dict:
            continue

        rec_mask, lig_mask = get_interface_mask(sthg_dict[pdb_n])
        full_mask = np.concatenate([rec_mask, lig_mask])
        n_rec = rec_mask.shape[0]

        pdb_results = {}
        dr_scores[pdb_n] = {}

        # Trained & untrained
        for label, emb_dict in [(label_trained, emb_trained), (label_untrained, emb_untrained)]:
            stats, chain_scores = _score_one_condition(
                emb_dict, pdb_n, n_rec, full_mask, dr_method)
            pdb_results[label] = stats
            dr_scores[pdb_n][label] = chain_scores

        # Shuffled GraPPI (average over n_shuffle independent permutations)
        pdb_results[label_shuffled] = _averaged_shuffled_stats(
            emb_trained, pdb_n, n_rec, full_mask, dr_method, n_shuffle)

        # Shuffled GraPPI base (only if base embeddings supplied)
        if emb_base is not None and pdb_n in emb_base:
            pdb_results[label_shuffled_base] = _averaged_shuffled_stats(
                emb_base, pdb_n, n_rec, full_mask, dr_method, n_shuffle)

        # Extra conditions (e.g. raw ESM embeddings)
        if extra_embs:
            for label, emb_dict in extra_embs.items():
                if pdb_n not in emb_dict:
                    continue
                stats, chain_scores = _score_one_condition(
                    emb_dict, pdb_n, n_rec, full_mask, dr_method)
                pdb_results[label] = stats
                dr_scores[pdb_n][label] = chain_scores

        results[pdb_n] = pdb_results

    return results, dr_scores


def _collect_labels(results):
    """Return ordered list of condition labels present in results."""
    preferred = [
        'GraPPI', 'GraPPI base', 'untrained GraPPI', 'untrained GraPPI base',
        'shuffled GraPPI', 'shuffled GraPPI base', 'esm', 'esm480',
        'GraPPI interf_mut', 'GraPPI base interf_mut',
        # legacy labels (backward compatibility)
        'trained', 'base', 'untrained', 'shuffled',
    ]
    seen = set()
    for pdb_res in results.values():
        seen.update(pdb_res.keys())
    ordered = [l for l in preferred if l in seen]
    ordered += [l for l in seen if l not in preferred]
    return ordered


def print_results(results):
    """Pretty-print comparison table."""
    labels = _collect_labels(results)
    print(f"\n{'='*102}")
    print(f"{'PDB':<8} {'Condition':<18} {'DR iface':>10} {'DR other':>10} {'DR p-val':>10} "
          f"{'L2 iface':>10} {'L2 other':>10} {'L2 p-val':>10}")
    print(f"{'-'*102}")
    for pdb_n, pdb_res in results.items():
        for label in labels:
            if label not in pdb_res:
                continue
            r = pdb_res[label]
            print(f"{pdb_n:<8} {label:<18} "
                  f"{r['dr_iface_mean']:>10.3f} {r['dr_non_iface_mean']:>10.3f} "
                  f"{r['dr_mann_whitney_p']:>10.2e} "
                  f"{r['l2_iface_mean']:>10.3f} {r['l2_non_iface_mean']:>10.3f} "
                  f"{r['l2_mann_whitney_p']:>10.2e}")
    print(f"{'='*102}")


def save_results_csv(results, csv_path):
    """Save the results table to a CSV file."""
    fieldnames = [
        'pdb', 'condition', 'dr_method', 'n_interface', 'n_non_interface',
        'dr_iface_mean', 'dr_non_iface_mean', 'dr_mann_whitney_p',
        'l2_iface_mean', 'l2_non_iface_mean', 'l2_mann_whitney_p',
    ]
    labels = _collect_labels(results)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pdb_n, pdb_res in results.items():
            for label in labels:
                if label not in pdb_res:
                    continue
                r = pdb_res[label]
                row = {'pdb': pdb_n, 'condition': label}
                for key in fieldnames[2:]:
                    row[key] = r.get(key, '')
                writer.writerow(row)
    print(f"Results table saved to {csv_path}")


# ── visualisation helpers (for notebook use) ─────────────────────────────────

def cosine_similarity_matrix(emb_dict, pdb_n):
    """Compute N×N cosine similarity matrix for a single PDB complex.

    Args:
        emb_dict: {pdb: {'receptor': np.array, 'ligand': np.array}}
        pdb_n: PDB name string

    Returns:
        sim_matrix: (N, N) cosine similarity matrix (receptor rows first)
        n_rec: number of receptor residues (for drawing chain boundary)
    """
    rec = emb_dict[pdb_n]['receptor']
    lig = emb_dict[pdb_n]['ligand']
    emb_all = np.vstack([rec, lig])
    norms = np.linalg.norm(emb_all, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    emb_normed = emb_all / norms
    sim_matrix = emb_normed @ emb_normed.T
    return sim_matrix, rec.shape[0]


def correlation_distance_matrix(emb_dict, pdb_n):
    """Compute N×N correlation distance matrix (1 - Pearson r) for a PDB complex.

    Unlike cosine similarity, Pearson correlation first mean-centres each
    embedding vector, so it is sensitive to distributional shape rather
    than just direction.  Values range from 0 (identical profile) to 2
    (perfectly anti-correlated).

    Args:
        emb_dict: {pdb: {'receptor': np.array, 'ligand': np.array}}
        pdb_n: PDB name string

    Returns:
        corr_dist: (N, N) correlation distance matrix (receptor rows first)
        n_rec: number of receptor residues
    """
    rec = emb_dict[pdb_n]['receptor']
    lig = emb_dict[pdb_n]['ligand']
    emb_all = np.vstack([rec, lig])  # (N, D)
    # Mean-centre each row
    emb_centred = emb_all - emb_all.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(emb_centred, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    emb_normed = emb_centred / norms
    pearson_r = emb_normed @ emb_normed.T  # (N, N)
    return 1.0 - pearson_r, rec.shape[0]


def compute_contact_map(sthg, distance_cutoff=6.0):
    """Compute a binary residue-residue contact map from 3-D coordinates.

    Two residues are in contact if the minimum heavy-atom distance between
    them is <= *distance_cutoff* Å.

    Args:
        sthg: StructureToHeteroGraph object (must have AA_df with atom_coords)
        distance_cutoff: contact distance threshold in Angstroms

    Returns:
        contact_matrix: (N, N) binary numpy array (1 = contact)
        n_rec: number of receptor residues (boundary index)
    """
    df = sthg.AA_df
    n = len(df)
    n_rec = (df['chain_type'] == 'receptor').sum()
    contact = np.zeros((n, n), dtype=np.float32)

    coords_list = df['atom_coords'].tolist()

    for i in range(n):
        ci = np.array(coords_list[i], dtype=np.float32)
        for j in range(i + 1, n):
            cj = np.array(coords_list[j], dtype=np.float32)
            # min heavy-atom distance
            diffs = ci[:, None, :] - cj[None, :, :]       # (Ai, Aj, 3)
            dists = np.sqrt((diffs ** 2).sum(axis=2))      # (Ai, Aj)
            if dists.min() <= distance_cutoff:
                contact[i, j] = 1.0
                contact[j, i] = 1.0

    return contact, n_rec


def cross_chain_nn_precision(emb_dict, sthg_dict, pdb_names, k=10):
    """For each interface residue, fraction of K-NN from the partner chain.

    Measures whether the embedding space places interface residues closer to
    their cross-chain interaction partners than to same-chain residues.

    Args:
        emb_dict: {pdb: {'receptor': np.array, 'ligand': np.array}}
        sthg_dict: loaded StructureToHeteroGraph objects
        pdb_names: list of PDB names
        k: number of nearest neighbours

    Returns:
        results: {pdb: float} — mean cross-chain fraction over interface residues
    """
    from sklearn.neighbors import NearestNeighbors

    precisions = {}
    for pdb_n in pdb_names:
        if pdb_n not in emb_dict or pdb_n not in sthg_dict:
            continue
        rec = emb_dict[pdb_n]['receptor']
        lig = emb_dict[pdb_n]['ligand']
        n_rec = rec.shape[0]
        emb_all = np.vstack([rec, lig])

        rec_mask, lig_mask = get_interface_mask(sthg_dict[pdb_n])
        full_mask = np.concatenate([rec_mask, lig_mask])
        chain_labels = np.array([0] * n_rec + [1] * lig.shape[0])

        iface_idx = np.where(full_mask)[0]
        if len(iface_idx) == 0:
            continue

        nn = NearestNeighbors(
            n_neighbors=min(k + 1, len(emb_all)), metric='cosine')
        nn.fit(emb_all)
        _, indices = nn.kneighbors(emb_all[iface_idx])

        cross_fracs = []
        for row_i, row_nn in enumerate(indices):
            query_chain = chain_labels[iface_idx[row_i]]
            neighbours = row_nn[1:k + 1]  # skip self
            cross = (chain_labels[neighbours] != query_chain).sum()
            cross_fracs.append(cross / len(neighbours))
        precisions[pdb_n] = float(np.mean(cross_fracs))
    return precisions


def embed_2d(emb_dict, pdb_n, method='umap', random_state=42):
    """Project residue embeddings of one PDB to 2-D for scatter plotting.

    Args:
        emb_dict: {pdb: {'receptor': np.array, 'ligand': np.array}}
        pdb_n: PDB name
        method: 'umap', 'tsne', or 'pca'
        random_state: reproducibility seed

    Returns:
        coords_2d: (N, 2) array
        n_rec: number of receptor residues (boundary index)
    """
    rec = emb_dict[pdb_n]['receptor']
    lig = emb_dict[pdb_n]['ligand']
    emb_all = np.vstack([rec, lig])

    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        coords_2d = reducer.fit_transform(emb_all)
    elif method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=2, metric='cosine',
                            random_state=random_state, n_neighbors=15,
                            min_dist=0.1)
        coords_2d = reducer.fit_transform(emb_all)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, metric='cosine',
                       random_state=random_state, perplexity=30)
        coords_2d = reducer.fit_transform(emb_all)
    else:
        raise ValueError(f"Unknown 2D projection method: {method}")

    return coords_2d, rec.shape[0]


# ── attention extraction wrapper ─────────────────────────────────────────────

def extract_attention_scores(encoder, sthg, device='cpu', all_layers=False):
    """Extract per-node attention scores from HGT layers.

    The encoder is deep-copied internally so the original model is unmodified.

    Args:
        encoder: EdgeEnhancedHGT model
        sthg: StructureToHeteroGraph object for one PDB
        device: torch device string or torch.device
        all_layers: if False (default), return attention from the last HGT layer
            only (original behaviour).  If True, return a dict keyed by layer
            index (0 = shallowest) where each value is
            {'node_attention': ..., 'edge_type_attention': ...}.

    Returns (all_layers=False — default):
        node_attention: {'receptor': {'total': tensor, 'incoming': tensor,
                                       'outgoing': tensor},
                         'ligand': {...}}
        edge_type_attention: {edge_type_str: {mean, std, max, min, num_edges}}

    Returns (all_layers=True):
        {layer_idx: {'node_attention': node_attention,
                     'edge_type_attention': edge_type_attention}}
    """
    import copy
    from model_collection.HGTConv_with_attention import (
        extract_attention_from_model, extract_attention_all_layers,
        aggregate_attention_to_nodes,
    )

    model_copy = copy.deepcopy(encoder).to(device).eval()
    loader = DataLoader([sthg.protein_graph], batch_size=1, shuffle=False)
    batch_data = next(iter(loader)).to(device)

    if not all_layers:
        attention_dict = extract_attention_from_model(model_copy, batch_data, device)
        node_attention, edge_type_attention = aggregate_attention_to_nodes(
            attention_dict, batch_data)
        del model_copy
        return node_attention, edge_type_attention
    else:
        layer_attn_list = extract_attention_all_layers(model_copy, batch_data, device)
        del model_copy
        results = {}
        for layer_idx, attention_dict in enumerate(layer_attn_list):
            node_attention, edge_type_attention = aggregate_attention_to_nodes(
                attention_dict, batch_data)
            results[layer_idx] = {
                'node_attention': node_attention,
                'edge_type_attention': edge_type_attention,
            }
        return results


# ── B-factor PDB writing ────────────────────────────────────────────────────

def normalize_scores_to_bfactor(scores):
    """Linearly scale scores to [0, 100] range for B-factor coloring."""
    smin, smax = scores.min(), scores.max()
    if smax - smin < 1e-12:
        return np.full_like(scores, 50.0)
    return 100.0 * (scores - smin) / (smax - smin)


def write_bfactor_pdbs(dr_scores, sthg_dict, output_dir, dr_method, strip_extra_chains=False):
    """Write B-factor colored PDB files for each PDB × encoder combination.

    Args:
        dr_scores: {pdb -> encoder_label -> {'receptor': array, 'ligand': array}}
        sthg_dict: {pdb -> StructureToHeteroGraph object}
        output_dir: directory to write output PDBs
        dr_method: DR method name (for filename)
        strip_extra_chains: if True, remove chains not in receptor/ligand from output
    """
    os.makedirs(output_dir, exist_ok=True)

    parser = BioPDB.PDBParser(QUIET=True, PERMISSIVE=True)
    io_obj = BioPDB.PDBIO()
    written = 0

    for pdb_n, encoder_scores in dr_scores.items():
        sthg = sthg_dict[pdb_n]
        row = sthg.row_of_df

        # Resolve PDB file path
        if 'mut_paths' in row and row.get('mut_paths') and str(row.get('mut_paths', '')) != 'nan':
            raw_path = row['mut_paths']
        else:
            raw_path = row['paths']
        pdb_path = resolve_pdb_path(raw_path)

        if pdb_path is None:
            print(f"  Warning: PDB file not found for {pdb_n} (raw path: {raw_path}), skipping B-factor write")
            continue

        # Parse chain IDs
        raw_rec = row.get('Receptor Chains', '')
        raw_lig = row.get('Ligand Chains', '')
        # Handle both comma-separated and space-separated formats
        receptor_chains = [c.strip() for c in str(raw_rec).replace(',', ' ').split() if c.strip()]
        ligand_chains = [c.strip() for c in str(raw_lig).replace(',', ' ').split() if c.strip()]

        for label, chain_scores in encoder_scores.items():
            scores_rec = chain_scores['receptor']
            scores_lig = chain_scores['ligand']

            # Use raw scores (no normalization) to match values in stats table
            scores_rec_final = scores_rec
            scores_lig_final = scores_lig

            # Load PDB structure
            struct = parser.get_structure(pdb_n, pdb_path)
            model0 = struct[0]

            # Remove non-target chains if requested
            if strip_extra_chains:
                target_chain_ids = set(receptor_chains) | set(ligand_chains)
                chains_to_remove = [c.id for c in model0.get_chains() if c.id not in target_chain_ids]
                for cid in chains_to_remove:
                    model0.detach_child(cid)

            # Reset all B-factors to 0 (get_unpacked_list reaches all
            # alternate conformations that get_atoms() would skip)
            for model in struct:
                for chain in model:
                    for res in chain:
                        for atom in res.get_unpacked_list():
                            atom.bfactor = 0.0

            # Assign ligand chain scores
            lig_ptr = 0
            for chain_id in ligand_chains:
                if chain_id not in model0:
                    print(f"  Warning: chain {chain_id} not found in {pdb_n} (ligand)")
                    continue
                residues = std_residues(model0[chain_id])
                for res in residues:
                    if lig_ptr >= len(scores_lig_final):
                        break
                    for atom in res.get_unpacked_list():
                        atom.bfactor = float(scores_lig_final[lig_ptr])
                    lig_ptr += 1

            # Assign receptor chain scores
            rec_ptr = 0
            for chain_id in receptor_chains:
                if chain_id not in model0:
                    print(f"  Warning: chain {chain_id} not found in {pdb_n} (receptor)")
                    continue
                residues = std_residues(model0[chain_id])
                for res in residues:
                    if rec_ptr >= len(scores_rec_final):
                        break
                    for atom in res.get_unpacked_list():
                        atom.bfactor = float(scores_rec_final[rec_ptr])
                    rec_ptr += 1

            # Strip HETATM (water/ions/ligands) so only the 20 standard amino
            # acids remain, and renumber residues sequentially — continuously
            # across chains (e.g. chain A 1..200, chain B 201..250), collapsing
            # gaps left by truncated regions.
            next_resnum = 1
            for chain in list(model0.get_chains()):
                next_resnum = clean_and_renumber_chain(chain, start=next_resnum)
                if len(list(chain.get_residues())) == 0:
                    model0.detach_child(chain.id)

            # Save
            out_name = f"{pdb_n}_{label}_{dr_method}_bfactor.pdb"
            out_path = os.path.join(output_dir, out_name)
            io_obj.set_structure(struct)
            io_obj.save(out_path)
            written += 1
            print(f"  Saved {out_path}  (rec={rec_ptr}, lig={lig_ptr} residues assigned)")

    print(f"\nWrote {written} B-factor PDB files to {output_dir}")
    print("Visualise in PyMOL:  spectrum b, blue_white_red")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Encoder case study: untrained vs trained baseline + B-factor PDBs')
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='Directory containing ssl_edge_best.pt and ssl_edge_config.json')
    parser.add_argument('--pdb_root', type=str, default='../GraPPI_data/curated_db',
                        help='Root directory for curated PDB database')
    parser.add_argument('--pdb_names', nargs='+', default=None,
                        help='PDB names to analyse (optional if --val_set_sampled_ratio > 0)')
    parser.add_argument('--val_set_sampled_ratio', type=float, default=0,
                        help='Fraction of SSL validation set to sample (0=off, 0.2=20%%, 1.0=all)')
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--dr_method', type=str, default='pca',
                        choices=['pca', 'isomap', 'umap', 'tsne'],
                        help='Dimensionality reduction method')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./bfactor_pdbs',
                        help='Directory to write B-factor colored PDB files')
    parser.add_argument('--save_json', type=str, default=None,
                        help='Path to save results as JSON (optional)')
    parser.add_argument('--save_csv', type=str, default=None,
                        help='Path to save results table as CSV (optional)')
    parser.add_argument('--skip_bfactor', action='store_true',
                        help='Skip B-factor PDB generation (stats only)')
    parser.add_argument('--n_shuffle', type=int, default=10,
                        help='Number of independent shuffles for the shuffled-embedding control')
    parser.add_argument('--compare2esm', action='store_true',
                        help='Also evaluate raw ESM and ESM480 embeddings as additional baselines')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible validation set sampling')
    parser.add_argument('--strip_extra_chains', action='store_true',
                        help='Remove non-receptor/ligand chains from output B-factor PDBs')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')

    # ── Load SSL config ──
    ssl_config = load_ssl_config(args.pretrained_path)
    emb_type = ssl_config['embedding_type']
    print(f"SSL config: {ssl_config['num_layers']} layers, "
          f"hidden_dim=2^{ssl_config['hidden_dim_power']}, "
          f"embedding_type={emb_type}")

    # ── Resolve target PDB names ──
    target_pdbs = resolve_target_pdbs(
        args.pretrained_path, args.pdb_names, args.val_set_sampled_ratio, seed=args.seed)

    # ── Load only the target PDB graphs (no full-database scan) ──
    # Map embedding_type to directory suffix: base->"", esm->"_esm", esm480->"_esm480"
    emb_suffix = '' if emb_type == 'base' else f'_{emb_type}'
    search_dirs = [
        os.path.join(args.pdb_root, f'pos_sthg{emb_suffix}_8A'),
        os.path.join(args.pdb_root, f'testset{emb_suffix}'),
    ]
    # Also try base-feature directories as fallback (won't work for non-base models,
    # but avoids silent failure if the typed directory doesn't exist)
    if emb_suffix:
        search_dirs += [
            os.path.join(args.pdb_root, 'pos_sthg_8A'),
            os.path.join(args.pdb_root, 'testset'),
        ]
    print(f"Graph search dirs (embedding_type={emb_type}): {search_dirs}")

    sthg_dict = {}
    for pdb_n in target_pdbs:
        if pdb_n in sthg_dict:
            continue
        for d in search_dirs:
            pkl_path = os.path.join(d, f'{pdb_n}.pkl')
            if os.path.isfile(pkl_path):
                with open(pkl_path, 'rb') as f:
                    sthg_dict[pdb_n] = pickle.load(f)
                break

    found = [p for p in target_pdbs if p in sthg_dict]
    missing = [p for p in target_pdbs if p not in sthg_dict]
    if missing:
        print(f"Warning: {len(missing)} PDBs not found (first 10): {missing[:10]}")
    if not found:
        print("No target PDBs found. Exiting.")
        return
    print(f"Loaded {len(found)} target PDBs")

    # ── Build trained encoder ──
    print("\n--- Trained encoder ---")
    trained_encoder = build_encoder(ssl_config, device)
    ckpt_path = os.path.join(args.pretrained_path, 'ssl_edge_best.pt')
    state = torch.load(ckpt_path, map_location='cpu')
    trained_encoder.load_state_dict(state['encoder_state_dict'])
    trained_encoder.eval()
    print(f"Loaded weights from {ckpt_path}")

    emb_trained = compute_embeddings(
        trained_encoder, sthg_dict, found, device, args.batch_size)
    del trained_encoder
    torch.cuda.empty_cache()

    # ── Build untrained encoder (same architecture, no training) ──
    print("\n--- Untrained encoder ---")
    untrained_encoder = build_encoder(ssl_config, device)
    # No weight loading — uses random initialisation
    print("Using untrained encoder (random initialisation, no training)")

    emb_untrained = compute_embeddings(
        untrained_encoder, sthg_dict, found, device, args.batch_size)
    del untrained_encoder
    torch.cuda.empty_cache()

    # ── Load ESM baselines if requested ──
    extra_embs = {}
    if args.compare2esm:
        print("\n--- Loading raw ESM baselines ---")
        for esm_type in ['esm', 'esm480']:
            esm_emb = load_esm_embeddings(args.pdb_root, esm_type, found)
            if esm_emb is not None and len(esm_emb) > 0:
                extra_embs[esm_type] = esm_emb
        if not extra_embs:
            print("  Warning: no ESM embeddings loaded; continuing without ESM baselines")

    # ── Run comparison (trained vs untrained vs shuffled [vs esm vs esm480]) ──
    results, dr_scores = run_comparison(
        emb_trained, emb_untrained, sthg_dict, found, args.dr_method,
        n_shuffle=args.n_shuffle,
        extra_embs=extra_embs if extra_embs else None)
    print_results(results)

    # ── Write B-factor PDBs ──
    if not args.skip_bfactor:
        print(f"\n--- Writing B-factor PDB files ---")
        write_bfactor_pdbs(dr_scores, sthg_dict, args.output_dir, args.dr_method,
                           strip_extra_chains=args.strip_extra_chains)

    # ── Save JSON ──
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_json}")

    # ── Save CSV ──
    if args.save_csv:
        save_results_csv(results, args.save_csv)


if __name__ == '__main__':
    main()
