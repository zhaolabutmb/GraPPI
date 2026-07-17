"""
Common utility functions shared across training scripts.
This module contains functions used by both SSL pretraining and fine-tuning.
"""
import os
import pickle
import random
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from scipy.stats import pearsonr, spearmanr
import json
from typing import Optional, Dict
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Optional, List, Tuple

# -----------------------------
# Constants
# -----------------------------
MODEL_INIT_DIM = {
    'base': 25,
    'esm': 1285,
    'esm480':485,
    '+esm': 25,    # +esm uses base features; ESM added after encoding
    '+esm480': 25, # +esm480 uses base features; ESM added after encoding
    'only_esm': 25, # only_esm uses base features; ESM replaces encoder output
    'only_esm480': 25, # only_esm480 uses base features; ESM replaces encoder output
}

EDGE_IN_DIM = 8

METADATA = (
    ['receptor', 'ligand'],
    [
        ('receptor', 'receptor_receptor', 'receptor'),
        ('ligand', 'ligand_ligand', 'ligand'),
        ('receptor', 'receptor_ligand', 'ligand'),
        ('ligand', 'ligand_receptor', 'receptor'),
    ],
)


# -----------------------------
# Utility Functions
# -----------------------------
def set_seed(seed: int):
    """Set random seed for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sttgs_from_dir(pdb_dir: str):
    """Load all .pkl files from a directory."""
    files = [f for f in os.listdir(pdb_dir) if f.endswith('.pkl')]
    all_sttgs = {}
    for f in tqdm(files,total=len(files),desc="PDB graphs loading"):
        with open(os.path.join(pdb_dir, f), 'rb') as fh:
            all_sttgs[f.replace('.pkl', '').lower()] = pickle.load(fh)
    return all_sttgs


def get_db_path_ssl(pdb_root: str, dist: str, embedding_type: str = 'base') -> str:
    """Get database directory path for SSL (positive samples only)."""
    if embedding_type in ['base', '+esm', '+esm480', 'only_esm']:
        return os.path.join(pdb_root, f'pos_sthg_{dist}A')
    else:
        return os.path.join(pdb_root, f'pos_sthg_{embedding_type}_{dist}A')


def get_db_path_finetune(pdb_root: str, dist: str, embedding_type: str = 'base', if_mut: bool = False, if_only_reg: bool = False) -> str:
    """Get database directory path for fine-tuning (all samples)."""
    if if_mut:
        # Mutant database
        prefix = 'mutant_sthg'
    else:
        if if_only_reg:
            # Only regression samples
            prefix = 'unmut_sthg'
        else:
            # All samples (premium, golden, negative)
            prefix = 'all_sthg'
    if embedding_type in ['base','+esm','+esm480','only_esm','only_esm480']:
        return os.path.join(pdb_root, f'{prefix}_{dist}A')
    elif embedding_type in ['esm','esm480']:
        return os.path.join(pdb_root, f'{prefix}_{embedding_type}_{dist}A')


def filter_samples_finetune(all_sttgs: dict, min_nodes: int = 5):
    """Filter samples for fine-tuning by minimum node counts."""
    kept = {}
    for k, st in all_sttgs.items():
        try:
            graph = st.protein_graph
            n_receptor = graph['receptor'].x.size(0)
            n_ligand = graph['ligand'].x.size(0)
            
            # Check for interface edges
            if ('receptor', 'receptor_ligand', 'ligand') not in graph.edge_index_dict:
                continue
            n_interface = graph[('receptor', 'receptor_ligand', 'ligand')].edge_index.size(1)
            
            if n_receptor >= min_nodes and n_ligand >= min_nodes and n_interface >= 3:
                kept[k] = st
        except Exception:
            continue
    return kept


def get_sample_type(affinity: float, premium_threshold: float = 1.5) -> str:
    """Determine sample type based on affinity value."""
    if affinity > premium_threshold:
        return 'premium'
    elif affinity == 1.0:
        return 'golden'
    elif affinity == 0:
        return 'negative'
    else:
        return 'unknown'


# -----------------------------
# Metrics Functions
# -----------------------------
def compute_edge_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute metrics for edge prediction."""
    if len(labels) == 0:
        return {'accuracy': 0.0, 'auc': 0.0}
    
    # Convert to float32 before numpy (bfloat16 not supported by numpy)
    probs = torch.sigmoid(logits).float().cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    labels_np = labels.float().cpu().numpy().astype(int)
    
    accuracy = accuracy_score(labels_np, preds)
    
    if len(np.unique(labels_np)) > 1:
        auc = roc_auc_score(labels_np, probs)
    else:
        auc = 0.5
    
    return {'accuracy': accuracy, 'auc': auc}


def compute_classification_metrics(y_true, y_pred_probs, threshold=0.5):
    """Compute classification metrics."""
    y_pred = (np.array(y_pred_probs) >= threshold).astype(int)
    y_true = np.array(y_true).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if len(np.unique(y_true)) > 1:
        metrics['auc'] = roc_auc_score(y_true, y_pred_probs)
    else:
        metrics['auc'] = 0.0
    
    return metrics


def compute_regression_metrics(y_true, y_pred):
    """Compute regression metrics."""
    metrics = {'rp': 0.0, 'sp': 0.0, 'mae': 0.0}
    
    if len(y_true) > 1:
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        
        # Check for constant arrays to avoid warnings
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics['rp'], _ = pearsonr(y_true_np, y_pred_np)
                if np.isnan(metrics['rp']):
                    metrics['rp'] = 0.0
        except:
            metrics['rp'] = 0.0
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics['sp'], _ = spearmanr(y_true_np, y_pred_np)
                if np.isnan(metrics['sp']):
                    metrics['sp'] = 0.0
        except:
            metrics['sp'] = 0.0
        
        metrics['mae'] = np.mean(np.abs(y_true_np - y_pred_np))
    
    return metrics

def load_ssl_config(pretrained_path: str) -> Optional[Dict]:
    """Load SSL configuration from checkpoint directory."""
    if pretrained_path is None or not os.path.exists(pretrained_path):
        return None
    
    checkpoint_dir = os.path.dirname(pretrained_path) if os.path.isfile(pretrained_path) else pretrained_path
    ssl_config_path = os.path.join(checkpoint_dir, 'ssl_edge_config.json')
    
    if os.path.exists(ssl_config_path):
        with open(ssl_config_path, 'r') as f:
            return json.load(f)
    return None


def stratified_train_test_split(
    samples: List[Tuple], 
    test_ratio: float = 0.2, 
    seed: int = 42
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Stratified train/test split based on sample type (premium/golden/negative).
    
    Args:
        samples: List of (name, graph, affinity, sample_type) tuples
        test_ratio: Fraction of samples for test set
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_samples, test_samples)
    """
    random.seed(seed)
    
    # Group by sample type
    by_type = defaultdict(list)
    for s in samples:
        by_type[s[3]].append(s)
    
    train_samples = []
    test_samples = []
    
    for sample_type, type_samples in by_type.items():
        random.shuffle(type_samples)
        n_test = max(1, int(len(type_samples) * test_ratio))
        test_samples.extend(type_samples[:n_test])
        train_samples.extend(type_samples[n_test:])
    
    return train_samples, test_samples


def save_split_info(
    train_samples: List[Tuple],
    test_samples: List[Tuple],
    save_dir: str,
    prefix: str = "simple_test"
):
    """
    Save train/test sample names for reproducibility.
    
    Args:
        train_samples: List of training samples
        test_samples: List of test samples
        save_dir: Directory to save split info
        prefix: Filename prefix
    """
    split_info = {
        'train_samples': [s[0] for s in train_samples],
        'test_samples': [s[0] for s in test_samples],
        'train_count': len(train_samples),
        'test_count': len(test_samples),
        'train_by_type': defaultdict(list),
        'test_by_type': defaultdict(list),
    }
    
    for s in train_samples:
        split_info['train_by_type'][s[3]].append(s[0])
    for s in test_samples:
        split_info['test_by_type'][s[3]].append(s[0])
    
    # Convert defaultdict to dict for JSON
    split_info['train_by_type'] = dict(split_info['train_by_type'])
    split_info['test_by_type'] = dict(split_info['test_by_type'])
    
    save_path = os.path.join(save_dir, f"{prefix}_split_info.json")
    with open(save_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Split info saved to: {save_path}")
    return save_path



def upsample_premium_samples(
    samples: List[Tuple],
    premium_upsample: int = 1,
    premium_threshold: float = 1.5,
) -> List[Tuple]:
    """
    Upsample premium samples to address class imbalance.
    
    Premium samples (those with binding affinity > premium_threshold) are duplicated
    to give the regression loss more gradient signal during training.
    
    Args:
        samples: List of (name, graph, affinity, sample_type) tuples
        premium_upsample: Multiplier for premium samples (1 = no upsampling, 2 = double, 3 = triple, etc.)
        premium_threshold: Threshold to identify premium samples
    
    Returns:
        List of samples with premium samples duplicated
    """
    if premium_upsample <= 1:
        return samples
    
    upsampled = []
    premium_count = 0
    non_premium_count = 0
    
    for s in samples:
        name, graph, affinity, sample_type = s
        if sample_type == 'premium' or affinity > premium_threshold:
            # Duplicate premium samples
            for _ in range(premium_upsample):
                upsampled.append(s)
            premium_count += 1
        else:
            upsampled.append(s)
            non_premium_count += 1
    
    print(f"Upsampling: {premium_count} premium samples x{premium_upsample} = {premium_count * premium_upsample}")
    print(f"Total samples after upsampling: {len(upsampled)} (was {len(samples)})")
    

# ============================================================================
# SEEDED GROUP K-FOLD
# ============================================================================
def seeded_group_kfold(names, n_folds, seed):
    """
    Split samples into *n_folds* by PDB group, with deterministic
    shuffling controlled by *seed*.

    The PDB group for each sample is ``name.split('_')[0]``.
    Unique groups are sorted, then shuffled with the given seed,
    and assigned to folds round-robin.  Same seed + same set of
    groups always produces the same fold assignments.

    Parameters
    ----------
    names : array-like of str
        Sample names (e.g. ``['1a22_0', '1a22_1', '3sgb_0', ...]``).
    n_folds : int
    seed : int

    Returns
    -------
    folds : list of (train_idx, val_idx) numpy arrays
    """
    names = np.asarray(names)
    groups = np.array([n.split('_')[0] for n in names])
    unique_groups = np.sort(np.unique(groups))

    # n_folds=1 → train on everything, no validation
    if n_folds == 1:
        return [(np.arange(len(names)), np.array([], dtype=int))]

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)

    group_to_fold = {g: i % n_folds for i, g in enumerate(unique_groups)}

    folds = []
    for fold in range(n_folds):
        val_mask = np.array([group_to_fold[g] == fold for g in groups])
        folds.append((np.where(~val_mask)[0], np.where(val_mask)[0]))

    return folds


def seeded_kfold(names, n_folds, seed):
    """
    Standard random KFold (sample-level, not group-aware) with
    deterministic shuffling controlled by *seed*.

    Parameters
    ----------
    names : array-like of str
    n_folds : int
    seed : int

    Returns
    -------
    folds : list of (train_idx, val_idx) numpy arrays
    """
    n = len(names)

    # n_folds=1 → train on everything, no validation
    if n_folds == 1:
        return [(np.arange(n), np.array([], dtype=int))]

    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[:n % n_folds] += 1

    folds = []
    current = 0
    for size in fold_sizes:
        val_idx = indices[current:current + size]
        train_idx = np.concatenate([indices[:current], indices[current + size:]])
        folds.append((train_idx, val_idx))
        current += size

    return folds


def get_fold_splits(names, n_folds, seed, strategy='group'):
    """
    Dispatcher: pick fold-split strategy by name.

    Parameters
    ----------
    names : array-like of str
    n_folds : int
    seed : int
    strategy : str
        'group' → seeded_group_kfold (PDB-based grouping)
        'random' → seeded_kfold (standard random shuffle)

    Returns
    -------
    folds : list of (train_idx, val_idx) numpy arrays
    """
    if strategy == 'group':
        return seeded_group_kfold(names, n_folds, seed)
    elif strategy == 'random':
        return seeded_kfold(names, n_folds, seed)
    else:
        raise ValueError(f"Unknown fold_strategy '{strategy}'. Use 'group' or 'random'.")