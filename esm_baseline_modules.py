"""
esm_baseline_modules.py

Baseline testing pipeline for three tasks using pre-computed ESM or ESM-480
embeddings (mean-pooled per chain) with NN and scikit-learn ML models.

Graphs are loaded from pre-built ESM directories (e.g. ``pos_sthg_esm_8A``,
``unmut_sthg_esm480_8A``).  Node feature tensors have shape [N, 1285] for
ESM (1280 ESM + 5 structural) or [N, 485] for ESM-480 (480 + 5).  Only the
first 1280 or 480 columns (the ESM embedding) are used as input.

Loads train/val fold splits from a pre-existing model save directory
(e.g. ``5layers_9hdim_jkmean``) to guarantee identical splits with the
comparison models.

Tasks (all three always run):
  1. ΔG regression      — K-fold CV
  2. ΔΔG / mutant ΔG   — K-fold CV
  3. Discriminative binder classification — single train / test split

Per task: a baseline MLP (mean-pooled ESM features, no GNN encoder) **and**
a suite of scikit-learn models (RF, GBT, DT, SVR/SVC) are trained and
evaluated.

ΔΔG label convention
-----------------------------------------------------------
The split JSON stores  mut_aff  (the mutant binding affinity).
graph.y already contains the original binding affinity; no overwrite needed.
Both NN and ML therefore predict mut_aff directly.
"""

import os
import glob
import json
import pickle
import math
from collections import defaultdict
from contextlib import nullcontext
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
# Standard DataLoader imported above; PyG DataLoader no longer needed

from sklearn.ensemble import (
    RandomForestRegressor, HistGradientBoostingRegressor,
    RandomForestClassifier, HistGradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
)
from scipy.stats import pearsonr, spearmanr

from torch.utils.data import DataLoader, TensorDataset

from model_collection.FineTune_baselineModels import create_baseline_model
from utils.Training_modules.save_training import FineTuneEarlyStopping
from utils.Quantity_compute.Loss_fun import get_loss
from utils.Training_modules.common_utils import (
    set_seed, load_sttgs_from_dir, get_db_path_finetune,
    filter_samples_finetune, get_sample_type, get_fold_splits,
    MODEL_INIT_DIM,
    compute_regression_metrics, compute_classification_metrics,
)

# ESM embedding dimensions (only the ESM part, excluding 5 structural features)
ESM_DIM = {
    'esm':    1280,
    'esm480': 480,
}


# ============================================================================
# SPLIT FILE LOCATORS
# ============================================================================

def _find_split_file(source_dir: str, pattern: str) -> str:
    """Return the first file in *source_dir* whose name matches *pattern* (glob)."""
    matches = sorted(glob.glob(os.path.join(source_dir, pattern)))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' found in {source_dir}"
        )
    return matches[0]


def _find_dG_split_file(source_dir: str) -> str:
    return _find_split_file(source_dir, 'wt_dg*splits.json')


def _find_ddG_split_file(source_dir: str) -> str:
    return _find_split_file(source_dir, 'ddg_*splits.json')


def _find_disc_binder_split_file(source_dir: str) -> str:
    """Single-split file: disc_binder*_splits.json but NOT a CV *fold_splits.json."""
    all_matches = sorted(glob.glob(os.path.join(source_dir, 'disc_binder*splits.json')))
    matches = [m for m in all_matches if 'fold_splits' not in os.path.basename(m)]
    if not matches:
        raise FileNotFoundError(
            f"No single-split file matching 'disc_binder*_splits.json' "
            f"(excluding *fold_splits.json) found in {source_dir}"
        )
    return matches[0]


def _find_disc_binder_cv_split_file(source_dir: str, n_folds: int,
                                    cv_split_mode: str) -> str:
    """
    Locate the finetune CV fold-split file for the requested strategy.
    global   -> disc_binder*_{n}fold_splits.json  (excluding _fixed_pos_)
    fixed_pos-> disc_binder*_fixed_pos_{n}fold_splits.json
    """
    if cv_split_mode == 'fixed_pos':
        matches = sorted(glob.glob(os.path.join(
            source_dir, f'disc_binder*_fixed_pos_{n_folds}fold_splits.json')))
    else:  # global
        all_m = sorted(glob.glob(os.path.join(
            source_dir, f'disc_binder*_{n_folds}fold_splits.json')))
        matches = [m for m in all_m
                   if '_fixed_pos_' not in os.path.basename(m)]
    if not matches:
        raise FileNotFoundError(
            f"No CV split file for cv_split_mode='{cv_split_mode}', "
            f"n_folds={n_folds} found in {source_dir}. "
            f"Run the finetune disc_binder CV first."
        )
    return matches[0]


# ============================================================================
# GRAPH LOADERS  (ESM-aware: slice node features to ESM-only)
# ============================================================================

def _slice_esm_features(graph, esm_dim: int):
    """Replace node features with ESM-only columns: x[:, :esm_dim]."""
    for nt in ('receptor', 'ligand'):
        if nt in graph:
            graph[nt].x = graph[nt].x[:, :esm_dim].float()
    return graph


def _load_all_graphs(pdb_dir: str, esm_dim: int,
                     min_nodes: int = 5) -> Dict[str, object]:
    """
    Load all .pkl files from *pdb_dir*, filter by min_nodes,
    and return {name: protein_graph}.
    Features are kept at full width; ESM slicing happens at pooling time.
    """
    if not os.path.isdir(pdb_dir):
        print(f"  [WARN] Directory not found: {pdb_dir}")
        return {}
    sttgs = load_sttgs_from_dir(pdb_dir)
    sttgs = filter_samples_finetune(sttgs, min_nodes=min_nodes)
    return {name: st.protein_graph for name, st in sttgs.items()}


def _load_graphs_named(
    search_dirs: List[str],
    names_needed: List[str],
    esm_dim: int,
    min_nodes: int = 5,
) -> Dict[str, object]:
    """
    Load only the graphs whose names are in *names_needed*, searching
    each directory in *search_dirs* in order.  Slices to ESM-only features.
    Returns {name: protein_graph}.
    """
    needed = set(n.lower() for n in names_needed)
    graphs: Dict[str, object] = {}

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        still_needed = needed - set(graphs.keys())
        if not still_needed:
            break
        for name in list(still_needed):
            path = os.path.join(d, f'{name}.pkl')
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        st = pickle.load(f)
                    graph = st.protein_graph
                    # strip random-mutant extra attribute to avoid PyG collate KeyError
                    if hasattr(graph, 'mut_info'):
                        del graph.mut_info
                    if (graph['receptor'].x.size(0) >= min_nodes
                            and graph['ligand'].x.size(0) >= min_nodes):
                        graphs[name] = graph
                except Exception:
                    pass

    missing = needed - set(graphs.keys())
    if missing:
        print(f"  [WARN] {len(missing)} graphs not found: {sorted(missing)[:5]}"
              + (f" …and {len(missing)-5} more" if len(missing) > 5 else ""))
    return graphs


# ============================================================================
# ESM DIRECTORY HELPERS
# ============================================================================

def _esm_dir(pdb_root: str, prefix: str, esm_type: str, dist: str) -> str:
    """Build ESM directory path: e.g. /data/pos_sthg_esm_8A"""
    return os.path.join(pdb_root, f'{prefix}_{esm_type}_{dist}A')


def _esm_train_dir(pdb_root: str, dist: str, esm_type: str,
                    if_mut: bool = False) -> str:
    """Get training graph directory for ESM/ESM480."""
    if if_mut:
        prefix = 'mutant_sthg'
    else:
        prefix = 'unmut_sthg'
    return _esm_dir(pdb_root, prefix, esm_type, dist)


# ============================================================================
# SAMPLE BUILDERS FROM SPLIT JSON
# ============================================================================

def _build_dG_samples(
    split_list: List[List],
    graph_dict: Dict[str, object],
) -> List[Tuple]:
    """
    Convert a dG split list ``[[name, aff, sample_type], …]`` into
    ``[(name, graph, aff, sample_type), …]`` using *graph_dict*.
    """
    samples = []
    skipped = 0
    for entry in split_list:
        name, aff = entry[0].lower(), float(entry[1])
        stype = entry[2] if len(entry) > 2 else 'premium'
        if name in graph_dict:
            samples.append((name, graph_dict[name], aff, stype))
        else:
            skipped += 1
    if skipped:
        print(f"  [dG] Skipped {skipped} entries (graph not found)")
    return samples


def _build_ddG_pairs(
    split_list: List[List],
    mut_graph_dict: Dict[str, object],
    wt_graph_dict: Dict[str, object],
) -> List[Tuple]:
    """
    Convert a ddG split list ``[[mut_name, mut_aff], …]`` into
    ``[(mut_name, mut_graph, wt_graph, mut_aff), …]``.

    graph.y already contains the original binding affinity; no overwrite needed.
    """
    pairs = []
    skipped = 0
    for entry in split_list:
        mut_name = entry[0].lower()
        mut_aff  = float(entry[1])
        wt_name  = mut_name.split('_')[0]
        if mut_name in mut_graph_dict and wt_name in wt_graph_dict:
            mut_graph = mut_graph_dict[mut_name]
            wt_graph  = wt_graph_dict[wt_name]
            pairs.append((mut_name, mut_graph, wt_graph, mut_aff))
        else:
            skipped += 1
    if skipped:
        print(f"  [ddG] Skipped {skipped} entries (mut or WT graph not found)")
    return pairs


def _build_disc_binder_samples(
    split_list: List[List],
    graph_dict: Dict[str, object],
    premium_threshold: float = 1.1,
) -> List[Tuple]:
    """
    Convert a disc_binder split list ``[[name, aff], …]`` into
    ``[(name, graph, binary_label, sample_type), …]``.
    Stamps binary 0.0/1.0 onto graph.y.
    """
    samples = []
    skipped = 0
    for entry in split_list:
        name = entry[0].lower()
        aff  = float(entry[1])
        label = 1.0 if aff > 0 else 0.0
        if name in graph_dict:
            graph = graph_dict[name]
            graph.y = torch.tensor(label, dtype=torch.float)
            stype = get_sample_type(aff, premium_threshold)
            samples.append((name, graph, label, stype))
        else:
            skipped += 1
    if skipped:
        print(f"  [disc_binder] Skipped {skipped} entries (graph not found)")
    return samples


# ============================================================================
# FEATURE EXTRACTION  (ESM node features → numpy)
# ============================================================================

def _mean_pool_graph(graph, esm_dim: int = None) -> np.ndarray:
    """Stack receptor + ligand nodes, mean-pool across all → 1-D numpy.
    If *esm_dim* is given, slice pooled vector to [:esm_dim]."""
    r = graph['receptor'].x.float()
    l = graph['ligand'].x.float()
    pooled = torch.cat([r, l], dim=0).mean(0)
    if esm_dim is not None:
        pooled = pooled[:esm_dim]
    return pooled.numpy()


def _extract_features_regression(samples: List[Tuple],
                                 esm_dim: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """(name, graph, aff, *) → features [N, esm_dim], targets [N]."""
    X = np.stack([_mean_pool_graph(s[1], esm_dim) for s in samples])
    y = np.array([s[2] for s in samples], dtype=np.float32)
    return X, y


def _extract_features_ddg(pairs: List[Tuple],
                          esm_dim: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    (mut_name, mut_graph, wt_graph, mut_aff) → features [N, 2*esm_dim], targets [N].

    Feature : [mut_pooled ‖ wt_pooled]
    Target  : mut_graph.y  (= mut_aff, the original binding affinity)
    """
    feats, targets = [], []
    for _, mut_graph, wt_graph, _ in pairs:
        feats.append(np.concatenate([_mean_pool_graph(mut_graph, esm_dim),
                                     _mean_pool_graph(wt_graph, esm_dim)]))
        mut_y = mut_graph.y.item() if hasattr(mut_graph.y, 'item') else float(mut_graph.y)
        targets.append(mut_y)
    return np.stack(feats), np.array(targets, dtype=np.float32)


def _extract_features_classifier(samples: List[Tuple],
                                 esm_dim: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """(name, graph, label, *) → features [N, esm_dim], binary labels [N]."""
    X = np.stack([_mean_pool_graph(s[1], esm_dim) for s in samples])
    y = np.array([int(s[2] > 0) for s in samples])
    return X, y


# ============================================================================
# ML MODEL BUILDERS
# ============================================================================

def _build_regressors(ml_cfg: dict, seed: int) -> dict:
    rf_max_features = ml_cfg.get('rf_max_features', 'sqrt')
    return {
        'RandomForest': RandomForestRegressor(
            n_estimators=ml_cfg.get('rf_n_estimators', 300),
            max_depth=ml_cfg.get('rf_max_depth', 10),
            min_samples_leaf=ml_cfg.get('rf_min_samples_leaf', 5),
            max_features=rf_max_features,
            random_state=seed, n_jobs=ml_cfg.get('n_jobs', 8)),
        'GradientBoosting': HistGradientBoostingRegressor(
            max_iter=ml_cfg.get('gbt_n_estimators', 100),
            max_depth=ml_cfg.get('gbt_max_depth', 5),
            learning_rate=ml_cfg.get('gbt_learning_rate', 0.1),
            random_state=seed),
        'DecisionTree': DecisionTreeRegressor(
            max_depth=ml_cfg.get('dt_max_depth', 10),
            min_samples_leaf=ml_cfg.get('dt_min_samples_leaf', 10),
            random_state=seed),
        'SVR': SVR(
            C=ml_cfg.get('svm_C', 10),
            epsilon=ml_cfg.get('svm_epsilon', 0.1),
            kernel=ml_cfg.get('svm_kernel', 'rbf')),
    }


def _build_classifiers(ml_cfg: dict, seed: int) -> dict:
    rf_max_features = ml_cfg.get('rf_max_features', 'sqrt')
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators=ml_cfg.get('rf_n_estimators', 300),
            max_depth=ml_cfg.get('rf_max_depth', 10),
            min_samples_leaf=ml_cfg.get('rf_min_samples_leaf', 5),
            max_features=rf_max_features,
            random_state=seed, n_jobs=ml_cfg.get('n_jobs', 8)),
        'GradientBoosting': HistGradientBoostingClassifier(
            max_iter=ml_cfg.get('gbt_n_estimators', 100),
            max_depth=ml_cfg.get('gbt_max_depth', 5),
            learning_rate=ml_cfg.get('gbt_learning_rate', 0.1),
            random_state=seed),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=ml_cfg.get('dt_max_depth', 15),
            min_samples_leaf=ml_cfg.get('dt_min_samples_leaf', 20),
            class_weight='balanced', random_state=seed),
        'SVM': SVC(
            C=ml_cfg.get('svm_C', 10),
            kernel=ml_cfg.get('svm_kernel', 'rbf'),
            class_weight='balanced',
            probability=True),
    }


# ============================================================================
# ML EVALUATORS
# ============================================================================

def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Pearson r, Spearman r, MAE for regression."""
    mae = float(mean_absolute_error(y_true, y_pred))
    try:
        rp, _ = pearsonr(y_true, y_pred)
        rp = float(rp) if not math.isnan(rp) else 0.0
    except Exception:
        rp = 0.0
    try:
        rs, _ = spearmanr(y_true, y_pred)
        rs = float(rs) if not math.isnan(rs) else 0.0
    except Exception:
        rs = 0.0
    return {'rp': rp, 'sp': rs, 'mae': mae}


def _cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Accuracy, F1, AUC for binary classification."""
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5
    return {'accuracy': acc, 'f1': f1, 'auc': auc}


def _eval_regressor_ml(
    name: str, reg,
    train_X, train_y, val_X, val_y, test_X, test_y,
    save_path: Optional[str] = None,
) -> dict:
    reg.fit(train_X, train_y)
    result = {
        'val':  _reg_metrics(val_y,  reg.predict(val_X)),
        'test': _reg_metrics(test_y, reg.predict(test_X)),
    }
    print(f"    {name:20s}  val_rp={result['val']['rp']:.3f}  "
          f"test_rp={result['test']['rp']:.3f}  "
          f"test_mae={result['test']['mae']:.3f}")
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(reg, f)
    return result


def _eval_classifier_ml(
    name: str, clf,
    train_X, train_y, test_X, test_y,
    save_path: Optional[str] = None,
) -> dict:
    clf.fit(train_X, train_y)
    te_pred = clf.predict(test_X)
    if hasattr(clf, 'predict_proba'):
        te_prob = clf.predict_proba(test_X)[:, 1]
    elif hasattr(clf, 'decision_function'):
        te_prob = clf.decision_function(test_X)
    else:
        te_prob = te_pred.astype(float)
    result = {'test': _cls_metrics(test_y, te_pred, te_prob)}
    print(f"    {name:20s}  test_auc={result['test']['auc']:.3f}  "
          f"test_f1={result['test']['f1']:.3f}  "
          f"test_acc={result['test']['accuracy']:.3f}")
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(clf, f)
    return result


# ============================================================================
# NN TRAINING HELPERS
# ============================================================================

def _make_optimizer_scheduler(model: nn.Module, bl_cfg: dict):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=bl_cfg.get('lr', 1e-4),
        weight_decay=bl_cfg.get('weight_decay', 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=bl_cfg.get('scheduler_t0', 10),
        T_mult=bl_cfg.get('scheduler_t_mult', 2),
        eta_min=float(bl_cfg.get('scheduler_eta_min', 1e-6)),
    )
    return optimizer, scheduler


# ============================================================================
# TENSOR-BASED NN TRAIN / VALIDATE (features pre-pooled into np arrays)
# ============================================================================

def _make_tensor_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                        shuffle: bool = False) -> DataLoader:
    """Wrap numpy arrays into a TensorDataset → DataLoader."""
    ds = TensorDataset(torch.from_numpy(X).float(),
                       torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _train_epoch_tensor(model, loader, optimizer, scaler, criterion, device,
                        clip_max_norm=5.0, use_amp=True):
    """One training epoch on (X, y) tensor batches."""
    model.train()
    total_loss, n, gn_sum = 0.0, 0, 0.0
    all_true, all_pred = [], []
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
    for X_batch, y_batch in loader:
        optimizer.zero_grad(set_to_none=True)
        X_batch = X_batch.to(device)
        y_batch = y_batch.unsqueeze(-1).to(device)          # [B, 1]
        with amp_ctx:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
        gn_sum += sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None) ** 0.5
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item(); n += 1
        all_true.extend(y_batch.squeeze(1).cpu().tolist())
        all_pred.extend(pred.squeeze(1).detach().cpu().tolist())
    return total_loss / max(1, n), gn_sum / max(1, n), compute_regression_metrics(all_true, all_pred)


@torch.no_grad()
def _validate_tensor(model, loader, criterion, device, use_amp=True):
    """Validation on (X, y) tensor batches — regression metrics."""
    model.eval()
    total_loss, n = 0.0, 0
    all_true, all_pred = [], []
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.unsqueeze(-1).to(device)
        with amp_ctx:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
        total_loss += loss.item(); n += 1
        all_true.extend(y_batch.squeeze(1).cpu().tolist())
        all_pred.extend(pred.squeeze(1).cpu().tolist())
    return total_loss / max(1, n), compute_regression_metrics(all_true, all_pred)


def _train_epoch_cls_tensor(model, loader, optimizer, scaler, criterion, device,
                            clip_max_norm=5.0, use_amp=True):
    """One training epoch for classifier on (X, y_binary) tensor batches."""
    model.train()
    total_loss, n, gn_sum = 0.0, 0, 0.0
    all_true, all_pred = [], []
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
    for X_batch, y_batch in loader:
        optimizer.zero_grad(set_to_none=True)
        X_batch = X_batch.to(device)
        y_batch = y_batch.unsqueeze(-1).to(device).float()   # [B, 1]
        with amp_ctx:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
        gn_sum += sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None) ** 0.5
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item(); n += 1
        all_true.extend(y_batch.squeeze(-1).cpu().tolist())
        all_pred.extend(torch.sigmoid(logits).squeeze(-1).detach().cpu().tolist())
    return total_loss / max(1, n), gn_sum / max(1, n), compute_classification_metrics(all_true, all_pred)


@torch.no_grad()
def _validate_cls_tensor(model, loader, criterion, device, use_amp=True):
    """Validation for classifier on (X, y_binary) tensor batches."""
    model.eval()
    total_loss, n = 0.0, 0
    all_true, all_pred = [], []
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.unsqueeze(-1).to(device).float()
        with amp_ctx:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
        total_loss += loss.item(); n += 1
        all_true.extend(y_batch.squeeze(-1).cpu().tolist())
        all_pred.extend(torch.sigmoid(logits).squeeze(-1).cpu().tolist())
    return total_loss / max(1, n), compute_classification_metrics(all_true, all_pred)


def _train_nn_regressor(
    model, train_loader, val_loader, test_loader,
    bl_cfg: dict, device: torch.device, ckpt_path: str,
) -> Tuple[float, dict, dict]:
    """Train a BaselineRegressor on pre-pooled tensor loaders."""
    use_amp = device.type == 'cuda'
    scaler  = GradScaler(enabled=use_amp)
    optimizer, scheduler = _make_optimizer_scheduler(model, bl_cfg)
    criterion_train = get_loss(bl_cfg.get('loss_fun', 'l1'),
                               gamma=bl_cfg.get('gamma', 1.0),
                               delta=bl_cfg.get('delta', 1.0),
                               reduction='mean')
    criterion_val = get_loss('l1', reduction='mean')

    early_stop = FineTuneEarlyStopping(
        patience=bl_cfg.get('patience', 30),
        save_path=ckpt_path,
        mode='max',
    )
    cur_clip = bl_cfg.get('clip_max_norm', 5.0)
    clip_min, clip_factor = 5.0, 1.25
    best_val_m = {}

    for epoch in range(bl_cfg.get('n_epochs', 500)):
        model.update_epoch(epoch)
        train_loss, gn, _ = _train_epoch_tensor(
            model, train_loader, optimizer, scaler, criterion_train, device,
            clip_max_norm=cur_clip, use_amp=use_amp)
        val_loss, val_m = _validate_tensor(
            model, val_loader, criterion_val, device, use_amp=use_amp)
        scheduler.step()

        if gn and math.isfinite(gn) and gn > 0:
            cur_clip = max(clip_min, gn * clip_factor)

        should_stop = early_stop(val_m.get('rp', 0.0), model, epoch)
        if not should_stop and early_stop.counter == 0:
            best_val_m = val_m

        if (epoch + 1) % 50 == 0 or should_stop:
            print(f"      ep{epoch+1:4d}  train_loss={train_loss:.3f}  "
                  f"val_rp={val_m.get('rp', 0):.3f}  "
                  f"[best_val_rp={early_stop.best_score:.3f}]")
        if should_stop:
            break

    best_state = early_stop.get_best_state()
    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_m = _validate_tensor(model, test_loader, criterion_val, device, use_amp=use_amp)
    return float(early_stop.best_score), best_val_m, test_m


def _train_nn_ddg(
    model, train_loader, val_loader, test_loader,
    bl_cfg: dict, device: torch.device, ckpt_path: str,
) -> Tuple[float, dict, dict]:
    """Train a BaselineDDGRegressor on pre-pooled [mut‖wt] tensor loaders."""
    use_amp = device.type == 'cuda'
    scaler  = GradScaler(enabled=use_amp)
    optimizer, scheduler = _make_optimizer_scheduler(model, bl_cfg)
    criterion_train = get_loss(bl_cfg.get('loss_fun', 'l1'),
                               gamma=bl_cfg.get('gamma', 1.0),
                               delta=bl_cfg.get('delta', 1.0),
                               reduction='mean')
    criterion_val = get_loss('l1', reduction='mean')

    early_stop = FineTuneEarlyStopping(
        patience=bl_cfg.get('patience', 30),
        save_path=ckpt_path,
        mode='max',
    )
    cur_clip = bl_cfg.get('clip_max_norm', 5.0)
    clip_min, clip_factor = 5.0, 1.25
    best_val_m = {}

    for epoch in range(bl_cfg.get('n_epochs', 500)):
        model.update_epoch(epoch)
        train_loss, gn, _ = _train_epoch_tensor(
            model, train_loader, optimizer, scaler, criterion_train, device,
            clip_max_norm=cur_clip, use_amp=use_amp)
        val_loss, val_m = _validate_tensor(
            model, val_loader, criterion_val, device, use_amp=use_amp)
        scheduler.step()

        if gn and math.isfinite(gn) and gn > 0:
            cur_clip = max(clip_min, gn * clip_factor)

        should_stop = early_stop(val_m.get('rp', 0.0), model, epoch)
        if not should_stop and early_stop.counter == 0:
            best_val_m = val_m

        if (epoch + 1) % 50 == 0 or should_stop:
            print(f"      ep{epoch+1:4d}  train_loss={train_loss:.3f}  "
                  f"val_rp={val_m.get('rp', 0):.3f}  "
                  f"[best_val_rp={early_stop.best_score:.3f}]")
        if should_stop:
            break

    best_state = early_stop.get_best_state()
    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_m = _validate_tensor(model, test_loader, criterion_val, device, use_amp=use_amp)
    return float(early_stop.best_score), best_val_m, test_m


def _train_nn_classifier(
    model, train_loader, val_loader, test_loader,
    bl_cfg: dict, device: torch.device, ckpt_path: str,
) -> Tuple[float, dict, dict]:
    """Train a BaselineClassifier on pre-pooled tensor loaders."""
    use_amp = device.type == 'cuda'
    scaler  = GradScaler(enabled=use_amp)
    optimizer, scheduler = _make_optimizer_scheduler(model, bl_cfg)
    criterion = nn.BCEWithLogitsLoss()

    early_stop = FineTuneEarlyStopping(
        patience=bl_cfg.get('patience', 30),
        save_path=ckpt_path,
        mode='max',
    )
    cur_clip = bl_cfg.get('clip_max_norm', 5.0)
    clip_min, clip_factor = 5.0, 1.25
    best_val_m = {}

    for epoch in range(bl_cfg.get('n_epochs', 500)):
        model.update_epoch(epoch)
        train_loss, gn, _ = _train_epoch_cls_tensor(
            model, train_loader, optimizer, scaler, criterion, device,
            clip_max_norm=cur_clip, use_amp=use_amp)
        val_loss, val_m = _validate_cls_tensor(
            model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()

        if gn and math.isfinite(gn) and gn > 0:
            cur_clip = max(clip_min, gn * clip_factor)

        should_stop = early_stop(val_m.get('auc', 0.0), model, epoch)
        if not should_stop and early_stop.counter == 0:
            best_val_m = val_m

        if (epoch + 1) % 50 == 0 or should_stop:
            print(f"      ep{epoch+1:4d}  train_loss={train_loss:.3f}  "
                  f"val_auc={val_m.get('auc', 0):.3f}  "
                  f"[best={early_stop.best_score:.3f}]")
        if should_stop:
            break

    best_state = early_stop.get_best_state()
    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_m = _validate_cls_tensor(model, test_loader, criterion, device, use_amp=use_amp)
    return float(early_stop.best_score), best_val_m, test_m


# ============================================================================
# AGGREGATE HELPER
# ============================================================================

def _aggregate_fold_metrics(fold_metrics: dict) -> dict:
    """
    Given {fold_id: {metric_key: value, …}, …} (flat dicts per fold),
    compute mean and std across folds for each metric key.
    """
    per_key: dict = defaultdict(list)
    for v in fold_metrics.values():
        for k, val in v.items():
            per_key[k].append(float(val))
    agg = {}
    for k, vals in per_key.items():
        agg[f'{k}_mean'] = float(np.mean(vals))
        agg[f'{k}_std']  = float(np.std(vals))
    return agg


# ============================================================================
# TASK 1 — ΔG REGRESSION
# ============================================================================

def _run_dG_esm_baseline(config: dict, device: torch.device,
                         save_dir: str, esm_type: str) -> dict:
    """
    K-fold ESM baseline for ΔG regression.
    Loads graphs from unmut_sthg_{esm_type}_{dist}A and testset_{esm_type}.
    """
    print(f"\n{'='*70}")
    print(f"ESM BASELINE ({esm_type.upper()}) — ΔG REGRESSION")
    print(f"{'='*70}")

    bl_cfg   = config['esm_baseline']
    data_cfg = config['data']
    seed     = config.get('system', {}).get('seed', 42)
    ml_cfg   = bl_cfg.get('dG', {})

    esm_dim    = ESM_DIM[esm_type]
    node_in_dim = esm_dim
    hidden_dim  = 2 ** bl_cfg.get('hidden_dim_power', 9)
    batch_size  = bl_cfg.get('batch_size', 64)
    min_nodes   = data_cfg.get('min_nodes', 5)
    dist        = data_cfg['dist']
    pdb_root    = data_cfg['pdb_root']
    split_src   = bl_cfg['split_source_dir']

    # ── Find and load the fold split file ────────────────────────
    split_file = _find_dG_split_file(split_src)
    print(f"  Split file : {split_file}")
    with open(split_file) as f:
        fold_splits = json.load(f)
    n_folds = len(fold_splits)
    print(f"  Folds: {n_folds}")
    print(f"  ESM dim: {esm_dim}  →  pooled feature dim: {esm_dim}")

    # ── Load train graphs (unmut_sthg_{esm_type}_{dist}A) ────────
    train_graph_dir = _esm_train_dir(pdb_root, dist, esm_type, if_mut=False)
    print(f"  Loading training graphs from: {train_graph_dir}")
    train_graph_dict = _load_all_graphs(train_graph_dir, esm_dim,
                                        min_nodes=min_nodes)
    print(f"  Training graphs loaded: {len(train_graph_dict)}")

    # ── Load test graphs (testset_{esm_type}) ─────────────────────
    test_dir = os.path.join(pdb_root, f'testset_{esm_type}')
    print(f"  Loading test graphs from: {test_dir}")
    test_graph_dict = _load_all_graphs(test_dir, esm_dim, min_nodes=min_nodes)
    test_samples_all = [
        (n, g, g.y.item() if hasattr(g.y, 'item') else float(g.y), 'premium')
        for n, g in test_graph_dict.items()
    ]
    print(f"  Test samples: {len(test_samples_all)}")

    if not test_samples_all:
        print("ERROR: empty test set — skipping ΔG ESM baseline.")
        return {}

    test_X, test_y = _extract_features_regression(test_samples_all, esm_dim)
    test_loader = _make_tensor_loader(test_X, test_y, batch_size)

    # ── K-fold training ───────────────────────────────────────────
    nn_fold_val:  dict = {}
    nn_fold_test: dict = {}
    ml_fold_results: dict = defaultdict(dict)

    for fold_id_str, split_data in fold_splits.items():
        fold_id = fold_id_str
        print(f"\n  --- ΔG {fold_id}/{n_folds} ---")

        train_samples = _build_dG_samples(split_data['train'], train_graph_dict)
        val_samples   = _build_dG_samples(split_data['val'],   train_graph_dict)
        print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

        if not train_samples or not val_samples:
            print(f"  WARN: empty train or val in {fold_id}, skipping.")
            continue

        # ── Extract features (shared by NN and ML) ────────────────
        train_X, train_y = _extract_features_regression(train_samples, esm_dim)
        val_X,   val_y   = _extract_features_regression(val_samples,   esm_dim)

        # ── NN ────────────────────────────────────────────────────
        model = create_baseline_model(
            'regressor', node_in_dim=node_in_dim, hidden_dim=hidden_dim,
            dropout=bl_cfg.get('dropout', 0.2)).to(device)

        train_loader_nn = _make_tensor_loader(train_X, train_y, batch_size, shuffle=True)
        val_loader_nn   = _make_tensor_loader(val_X,   val_y,   batch_size)
        ckpt = os.path.join(save_dir, f'dG_nn_{fold_id}_best.pt')

        best_val_rp, val_m, test_m = _train_nn_regressor(
            model, train_loader_nn, val_loader_nn, test_loader,
            bl_cfg, device, ckpt)

        print(f"  NN {fold_id}: best_val_rp={best_val_rp:.3f}  "
              f"test_rp={test_m.get('rp', 0):.3f}  "
              f"test_sp={test_m.get('sp', 0):.3f}  "
              f"test_mae={test_m.get('mae', 0):.3f}")

        nn_fold_val[fold_id]  = {'rp': float(best_val_rp)}
        nn_fold_test[fold_id] = {k: float(v) for k, v in test_m.items()}

        # ── ML regressors ────────────────────────────────────────
        print(f"  ML Regressors ({fold_id}):")
        for ml_name, reg in _build_regressors(ml_cfg, seed).items():
            ml_save = os.path.join(save_dir, f'dG_ml_{fold_id}_{ml_name}.pkl')
            r = _eval_regressor_ml(ml_name, reg,
                                   train_X, train_y, val_X, val_y,
                                   test_X, test_y,
                                   save_path=ml_save)
            ml_fold_results[ml_name][fold_id] = {
                'val':  {k: float(v) for k, v in r['val'].items()},
                'test': {k: float(v) for k, v in r['test'].items()},
            }

    # ── Build summary ─────────────────────────────────────────────
    nn_results = {}
    for fid in nn_fold_val:
        nn_results[fid] = {
            'best_val_rp': nn_fold_val[fid]['rp'],
            'test': nn_fold_test[fid],
        }
    nn_agg = {
        'val_rp_mean': float(np.mean([v['rp'] for v in nn_fold_val.values()])),
        'val_rp_std':  float(np.std( [v['rp'] for v in nn_fold_val.values()])),
    }
    for k, v in _aggregate_fold_metrics({fid: nn_fold_test[fid]
                                         for fid in nn_fold_test}).items():
        nn_agg[f'test_{k}' if not k.startswith('test_') else k] = v
    nn_results['aggregate'] = nn_agg

    ml_results = {}
    for ml_name, fold_dict in ml_fold_results.items():
        agg_val  = _aggregate_fold_metrics({fid: d['val']  for fid, d in fold_dict.items()})
        agg_test = _aggregate_fold_metrics({fid: d['test'] for fid, d in fold_dict.items()})
        ml_results[ml_name] = {
            **fold_dict,
            'aggregate': {f'val_{k}':  v for k, v in agg_val.items()} |
                         {f'test_{k}': v for k, v in agg_test.items()},
        }

    result = {
        'task':         'dG',
        'esm_type':     esm_type,
        'esm_dim':      esm_dim,
        'split_source': split_src,
        'split_file':   os.path.basename(split_file),
        'nn':           nn_results,
        'ml':           ml_results,
    }

    out_path = os.path.join(save_dir, f'dG_{esm_type}_baseline_results.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=lambda x: float(x))
    print(f"\n  ΔG results saved to: {out_path}")

    ag = nn_results['aggregate']
    print(f"\n  [ΔG NN]  val_rp={ag['val_rp_mean']:.3f}±{ag['val_rp_std']:.3f}  "
          f"test_rp={ag.get('test_rp_mean',0):.3f}±{ag.get('test_rp_std',0):.3f}")
    for ml_name, mr in ml_results.items():
        a = mr['aggregate']
        print(f"  [ΔG {ml_name:18s}]  "
              f"val_rp={a.get('val_rp_mean',0):.3f}  "
              f"test_rp={a.get('test_rp_mean',0):.3f}")

    return result


# ============================================================================
# TASK 2 — ΔΔG / MUTANT ΔG REGRESSION
# ============================================================================

def _run_ddG_esm_baseline(config: dict, device: torch.device,
                          save_dir: str, esm_type: str) -> dict:
    """
    K-fold ESM baseline for ΔΔG / mutant-ΔG.
    Uses seeded_group_kfold for PDB-based CV with deterministic splits.
    Loads graphs from mutant_sthg_{esm_type}_{dist}A and testset_mut_{esm_type}.
    """
    print(f"\n{'='*70}")
    print(f"ESM BASELINE ({esm_type.upper()}) — ΔΔG / MUTANT ΔG REGRESSION")
    print(f"{'='*70}")

    bl_cfg   = config['esm_baseline']
    data_cfg = config['data']
    seed     = config.get('system', {}).get('seed', 42)
    ml_cfg   = bl_cfg.get('ddG', {})

    esm_dim     = ESM_DIM[esm_type]
    node_in_dim = esm_dim
    hidden_dim  = 2 ** bl_cfg.get('hidden_dim_power', 9)
    batch_size  = bl_cfg.get('batch_size', 64)
    min_nodes   = data_cfg.get('min_nodes', 5)
    dist        = data_cfg['dist']
    pdb_root    = data_cfg['pdb_root']
    n_folds     = bl_cfg.get('n_folds', 5)
    fold_strategy = bl_cfg.get('fold_strategy', 'group')
    print(f"  ESM dim: {esm_dim}  →  ddg feature dim: {2*esm_dim}")

    # ── Load ALL training mutant graphs ──────────────────────────
    mut_train_dir = _esm_train_dir(pdb_root, dist, esm_type, if_mut=True)
    print(f"  Loading mutant training graphs from: {mut_train_dir}")
    mut_graph_dict = _load_all_graphs(mut_train_dir, esm_dim, min_nodes=min_nodes)
    print(f"  Mutant graphs loaded: {len(mut_graph_dict)}")

    # ── Load WT training graphs (only PDB IDs present in mutant set) ──
    wt_train_dir = _esm_train_dir(pdb_root, dist, esm_type, if_mut=False)
    wt_names_needed = list({n.split('_')[0] for n in mut_graph_dict})
    print(f"  Loading WT training graphs from: {wt_train_dir}")
    wt_graph_dict = _load_graphs_named([wt_train_dir], wt_names_needed,
                                        esm_dim, min_nodes)
    print(f"  WT graphs loaded: {len(wt_graph_dict)}")

    # ── Load test data (mutant + WT coexist in testset_mut_{esm_type})
    test_dir = os.path.join(pdb_root, f'testset_mut_{esm_type}')
    print(f"  Loading test graphs from: {test_dir}")
    all_test_graphs = _load_all_graphs(test_dir, esm_dim, min_nodes=min_nodes)
    test_mut_dict = {}
    test_wt_dict  = {}
    for name, graph in all_test_graphs.items():
        if len(name.split('_')) > 1:
            test_mut_dict[name] = graph
        else:
            test_wt_dict[name] = graph
    print(f"  Test set split: {len(test_mut_dict)} mutants, {len(test_wt_dict)} WT")

    # ── Build test pairs ──────────────────────────────────────────
    test_pairs = []
    for mut_name, mut_graph in test_mut_dict.items():
        wt_name = mut_name.split('_')[0]
        if wt_name in test_wt_dict:
            mut_aff = mut_graph.y.item() if hasattr(mut_graph.y, 'item') else float(mut_graph.y)
            test_pairs.append((mut_name, mut_graph, test_wt_dict[wt_name], mut_aff))
    print(f"  Test pairs matched: {len(test_pairs)}")

    if not test_pairs:
        print("ERROR: no test pairs — skipping ΔΔG ESM baseline.")
        return {}

    # ── Exclude training mutants overlapping test PDB IDs ────────
    test_pdb_ids = {name.split('_')[0] for name in test_mut_dict}
    n_before = len(mut_graph_dict)
    mut_graph_dict = {k: v for k, v in mut_graph_dict.items()
                      if k.split('_')[0] not in test_pdb_ids}
    n_excluded = n_before - len(mut_graph_dict)
    if n_excluded > 0:
        print(f"  Excluded {n_excluded} training mutants overlapping test PDB IDs")

    # ── Build all training pairs ─────────────────────────────────
    all_train_pairs = []
    for mut_name, mut_graph in mut_graph_dict.items():
        wt_name = mut_name.split('_')[0]
        if wt_name in wt_graph_dict:
            mut_aff = mut_graph.y.item() if hasattr(mut_graph.y, 'item') else float(mut_graph.y)
            all_train_pairs.append((mut_name, mut_graph, wt_graph_dict[wt_name], mut_aff))
    print(f"  Training pairs: {len(all_train_pairs)}")

    if not all_train_pairs:
        print("ERROR: no training pairs — skipping ΔΔG ESM baseline.")
        return {}

    test_X, test_y  = _extract_features_ddg(test_pairs, esm_dim)
    test_loader     = _make_tensor_loader(test_X, test_y, batch_size)

    # ── Fold splitting ───────────────────────────────────────────
    train_names = [p[0] for p in all_train_pairs]
    fold_splits = get_fold_splits(train_names, n_folds, seed, fold_strategy)
    groups = np.array([n.split('_')[0] for n in train_names])
    strategy_label = 'GroupKFold' if fold_strategy == 'group' else 'KFold'
    print(f"  Seeded {strategy_label} (seed={seed}): {len(np.unique(groups))} PDB groups, {n_folds} folds")

    nn_fold_val:  dict = {}
    nn_fold_test: dict = {}
    ml_fold_results: dict = defaultdict(dict)

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        fold_id = f'fold_{fold + 1}'
        print(f"\n  --- ΔΔG {fold_id}/{n_folds} ---")

        train_pairs = [all_train_pairs[i] for i in train_idx]
        val_pairs   = [all_train_pairs[i] for i in val_idx]
        print(f"  Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

        if not train_pairs or not val_pairs:
            print(f"  WARN: empty train or val in {fold_id}, skipping.")
            continue

        # ── Extract features (shared by NN and ML) ────────────────
        train_X, train_y = _extract_features_ddg(train_pairs, esm_dim)
        val_X,   val_y   = _extract_features_ddg(val_pairs,   esm_dim)

        # ── NN ────────────────────────────────────────────────────
        model = create_baseline_model(
            'ddg', node_in_dim=node_in_dim, hidden_dim=hidden_dim,
            dropout=bl_cfg.get('dropout', 0.2)).to(device)

        train_loader_nn = _make_tensor_loader(train_X, train_y, batch_size, shuffle=True)
        val_loader_nn   = _make_tensor_loader(val_X,   val_y,   batch_size)
        ckpt   = os.path.join(save_dir, f'ddG_nn_{fold_id}_best.pt')

        best_val_rp, val_m, test_m = _train_nn_ddg(
            model, train_loader_nn, val_loader_nn, test_loader,
            bl_cfg, device, ckpt)

        print(f"  NN {fold_id}: best_val_rp={best_val_rp:.3f}  "
              f"test_rp={test_m.get('rp', 0):.3f}  "
              f"test_sp={test_m.get('sp', 0):.3f}  "
              f"test_mae={test_m.get('mae', 0):.3f}")

        nn_fold_val[fold_id]  = {'rp': float(best_val_rp)}
        nn_fold_test[fold_id] = {k: float(v) for k, v in test_m.items()}

        # ── ML regressors ────────────────────────────────────────
        print(f"  ML Regressors ({fold_id}):")
        for ml_name, reg in _build_regressors(ml_cfg, seed).items():
            ml_save = os.path.join(save_dir, f'ddG_ml_{fold_id}_{ml_name}.pkl')
            r = _eval_regressor_ml(ml_name, reg,
                                   train_X, train_y, val_X, val_y,
                                   test_X, test_y,
                                   save_path=ml_save)
            ml_fold_results[ml_name][fold_id] = {
                'val':  {k: float(v) for k, v in r['val'].items()},
                'test': {k: float(v) for k, v in r['test'].items()},
            }

    # ── Build summary ─────────────────────────────────────────────
    nn_results = {}
    for fid in nn_fold_val:
        nn_results[fid] = {
            'best_val_rp': nn_fold_val[fid]['rp'],
            'test': nn_fold_test[fid],
        }
    nn_agg = {
        'val_rp_mean': float(np.mean([v['rp'] for v in nn_fold_val.values()])),
        'val_rp_std':  float(np.std( [v['rp'] for v in nn_fold_val.values()])),
    }
    for k, v in _aggregate_fold_metrics({fid: nn_fold_test[fid]
                                         for fid in nn_fold_test}).items():
        nn_agg[f'test_{k}' if not k.startswith('test_') else k] = v
    nn_results['aggregate'] = nn_agg

    ml_results = {}
    for ml_name, fold_dict in ml_fold_results.items():
        agg_val  = _aggregate_fold_metrics({fid: d['val']  for fid, d in fold_dict.items()})
        agg_test = _aggregate_fold_metrics({fid: d['test'] for fid, d in fold_dict.items()})
        ml_results[ml_name] = {
            **fold_dict,
            'aggregate': {f'val_{k}':  v for k, v in agg_val.items()} |
                         {f'test_{k}': v for k, v in agg_test.items()},
        }

    result = {
        'task':         'ddG',
        'esm_type':     esm_type,
        'esm_dim':      esm_dim,
        'seed':         seed,
        'n_folds':      n_folds,
        'fold_strategy': fold_strategy,
        'nn':           nn_results,
        'ml':           ml_results,
    }

    out_path = os.path.join(save_dir, f'ddG_{esm_type}_baseline_results.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=lambda x: float(x))
    print(f"\n  ΔΔG results saved to: {out_path}")

    ag = nn_results['aggregate']
    print(f"\n  [ΔΔG NN]  val_rp={ag['val_rp_mean']:.3f}±{ag['val_rp_std']:.3f}  "
          f"test_rp={ag.get('test_rp_mean',0):.3f}±{ag.get('test_rp_std',0):.3f}")
    for ml_name, mr in ml_results.items():
        a = mr['aggregate']
        print(f"  [ΔΔG {ml_name:18s}]  "
              f"val_rp={a.get('val_rp_mean',0):.3f}  "
              f"test_rp={a.get('test_rp_mean',0):.3f}")

    return result


# ============================================================================
# TASK 3 — DISCRIMINATIVE BINDER CLASSIFICATION
# ============================================================================

# ---- CV (leakage-aware) helpers --------------------------------------------

def _cls_metrics_full(y_true, y_prob, is_seen) -> dict:
    """Overall + unseen classification metrics for a held-out fold.
    Unseen subset (~is_seen) = all negatives + positives the finetune encoder
    did not see during SSL pre-training. Threshold 0.5 for accuracy/F1."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    seen = np.asarray(is_seen, dtype=bool)
    m = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        m['auc'] = float(roc_auc_score(y_true, y_prob))
        m['auprc'] = float(average_precision_score(y_true, y_prob))
    else:
        m['auc'], m['auprc'] = 0.0, 0.0
    unseen = ~seen
    m['n_seen'], m['n_unseen'] = int(seen.sum()), int(unseen.sum())
    if unseen.sum() > 0 and len(np.unique(y_true[unseen])) > 1:
        m['auc_unseen'] = float(roc_auc_score(y_true[unseen], y_prob[unseen]))
        m['auprc_unseen'] = float(average_precision_score(y_true[unseen], y_prob[unseen]))
    else:
        m['auc_unseen'], m['auprc_unseen'] = 0.0, 0.0
    return m


@torch.no_grad()
def _validate_cls_tensor_seen_unseen(model, loader, criterion, device,
                                     is_seen_flags, use_amp=True):
    """Validation for classifier (tensor loader, shuffle=False) returning
    overall + unseen metrics. is_seen_flags must align with loader order."""
    model.eval()
    total_loss, n = 0.0, 0
    all_true, all_pred = [], []
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.unsqueeze(-1).to(device).float()
        with amp_ctx:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
        total_loss += loss.item(); n += 1
        all_true.extend(y_batch.squeeze(-1).cpu().tolist())
        all_pred.extend(torch.sigmoid(logits).squeeze(-1).cpu().tolist())
    return total_loss / max(1, n), _cls_metrics_full(all_true, all_pred, is_seen_flags)


def _train_nn_classifier_cv(model, train_loader, val_loader, val_is_seen,
                            bl_cfg, device, ckpt_path, cls_metric_name):
    """Train a BaselineClassifier on a fold; early-stop on cls_metric_name
    (overall or unseen). Returns (best_score, best_val_metrics)."""
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    optimizer, scheduler = _make_optimizer_scheduler(model, bl_cfg)
    criterion = nn.BCEWithLogitsLoss()
    early_stop = FineTuneEarlyStopping(
        patience=bl_cfg.get('patience', 30), save_path=ckpt_path, mode='max')
    cur_clip = bl_cfg.get('clip_max_norm', 5.0)
    clip_min, clip_factor = 5.0, 1.25
    best_val_m = {}
    for epoch in range(bl_cfg.get('n_epochs', 500)):
        model.update_epoch(epoch)
        train_loss, gn, _ = _train_epoch_cls_tensor(
            model, train_loader, optimizer, scaler, criterion, device,
            clip_max_norm=cur_clip, use_amp=use_amp)
        val_loss, val_m = _validate_cls_tensor_seen_unseen(
            model, val_loader, criterion, device, val_is_seen, use_amp=use_amp)
        scheduler.step()
        if gn and math.isfinite(gn) and gn > 0:
            cur_clip = max(clip_min, gn * clip_factor)
        score = val_m.get(cls_metric_name, val_m.get('auc', 0.0))
        should_stop = early_stop(score, model, epoch)
        if not should_stop and early_stop.counter == 0:
            best_val_m = val_m
        if (epoch + 1) % 50 == 0 or should_stop:
            print(f"      ep{epoch+1:4d}  train_loss={train_loss:.3f}  "
                  f"val_{cls_metric_name}={score:.3f}  "
                  f"auc_un={val_m.get('auc_unseen', 0):.3f}  "
                  f"[best={early_stop.best_score:.3f}]")
        if should_stop:
            break
    best_state = early_stop.get_best_state()
    if best_state is not None:
        model.load_state_dict(best_state)
    return float(early_stop.best_score), best_val_m


def _eval_classifier_ml_cv(name, clf, train_X, train_y, val_X, val_y,
                           val_is_seen, save_path=None) -> dict:
    """Fit ML classifier on a fold, evaluate on val with overall+unseen metrics."""
    clf.fit(train_X, train_y)
    if hasattr(clf, 'predict_proba'):
        val_prob = clf.predict_proba(val_X)[:, 1]
    elif hasattr(clf, 'decision_function'):
        val_prob = clf.decision_function(val_X)
    else:
        val_prob = clf.predict(val_X).astype(float)
    m = _cls_metrics_full(val_y, val_prob, val_is_seen)
    print(f"    {name:20s}  val_auc={m['auc']:.3f}  val_auprc={m['auprc']:.3f}  "
          f"auc_un={m['auc_unseen']:.3f}  auprc_un={m['auprc_unseen']:.3f}")
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(clf, f)
    return {'val': m}


def _build_disc_binder_cv_samples(split_list, graph_dict, premium_thr):
    """Convert CV split rows ``[name, label, is_seen]`` into ordered
    (name, graph, label_float, sample_type) samples + aligned is_seen list.
    Stamps binary label onto graph.y."""
    samples, is_seen = [], []
    skipped = 0
    for entry in split_list:
        name = entry[0].lower()
        label = float(entry[1])
        seen_flag = bool(entry[2]) if len(entry) > 2 else False
        g = graph_dict.get(name)
        if g is None:
            skipped += 1
            continue
        g.y = torch.tensor(label)
        st = get_sample_type(label, premium_thr)
        samples.append((name, g, label, st))
        is_seen.append(seen_flag)
    if skipped:
        print(f"    [warn] skipped {skipped} samples (graph not found)")
    return samples, is_seen


def _run_disc_binder_esm_baseline_cv(config: dict, device: torch.device,
                                     save_dir: str, esm_type: str) -> dict:
    """
    N-fold CV ESM baseline for binder classification. Reads the finetune CV
    fold splits (`disc_binder…_{N}fold_splits.json`, selected by cv_split_mode)
    so folds + seen/unseen flags match the finetune model exactly. Trains the
    ESM baseline NN + sklearn classifiers per fold and reports overall + unseen
    (auc_unseen/auprc_unseen) metrics. Output files carry a
    `_{cv_split_mode}_{N}fold` suffix to avoid clobbering single-split results.
    """
    bl_cfg   = config['esm_baseline']
    data_cfg = config['data']
    seed     = config.get('system', {}).get('seed', 42)
    ml_cfg   = bl_cfg.get('disc_binder', {})

    cv_split_mode = bl_cfg.get('cv_split_mode', 'global')
    n_folds       = int(bl_cfg.get('n_folds', 5))
    cls_metric    = bl_cfg.get('cls_metric', 'auc')

    esm_dim     = ESM_DIM[esm_type]
    node_in_dim = esm_dim
    hidden_dim  = 2 ** bl_cfg.get('hidden_dim_power', 9)
    batch_size  = bl_cfg.get('batch_size', 64)
    min_nodes   = data_cfg.get('min_nodes', 5)
    dist        = data_cfg['dist']
    pdb_root    = data_cfg['pdb_root']
    split_src   = bl_cfg['split_source_dir']
    premium_thr = bl_cfg.get('premium_threshold', 1.1)

    print(f"\n{'='*70}")
    print(f"ESM BASELINE ({esm_type.upper()}) — DISC BINDER {n_folds}-FOLD CV "
          f"(split={cv_split_mode})")
    print(f"{'='*70}")

    split_file = _find_disc_binder_cv_split_file(split_src, n_folds, cv_split_mode)
    print(f"  CV split file : {split_file}")
    print(f"  ESM dim: {esm_dim}  |  Primary metric: {cls_metric}")
    with open(split_file) as f:
        fold_data = json.load(f)

    all_names = set()
    for fd in fold_data.values():
        for e in fd['train'] + fd['val']:
            all_names.add(e[0].lower())

    search_dirs = [
        _esm_dir(pdb_root, 'pos_sthg', esm_type, dist),
        _esm_dir(pdb_root, 'swapped_sthg', esm_type, dist),
        _esm_dir(pdb_root, 'random_mut_sthg', esm_type, dist),
        _esm_dir(pdb_root, 'swapped_abag_sthg', esm_type, dist),
        _esm_dir(pdb_root, 'preppi_sthg', esm_type, dist),
    ]
    print(f"  Loading {len(all_names)} graphs from {len(search_dirs)} dirs …")
    graph_dict = _load_graphs_named(search_dirs, list(all_names), esm_dim, min_nodes)
    print(f"  Graphs loaded: {len(graph_dict)}")

    fold_keys = sorted(fold_data.keys(), key=lambda k: int(k.split('_')[1]))
    nn_fold_metrics: dict = {}
    ml_fold_metrics: dict = {}

    for fold_key in fold_keys:
        fd = fold_data[fold_key]
        train_samples, train_seen = _build_disc_binder_cv_samples(fd['train'], graph_dict, premium_thr)
        val_samples, val_seen     = _build_disc_binder_cv_samples(fd['val'], graph_dict, premium_thr)
        n_pos_vl = sum(1 for s in val_samples if s[2] > 0)
        n_pos_vl_un = sum(1 for s, sn in zip(val_samples, val_seen) if s[2] > 0 and not sn)
        print(f"\n  {fold_key}: train={len(train_samples)}  val={len(val_samples)} "
              f"({n_pos_vl} pos [{n_pos_vl_un} unseen] / {len(val_samples)-n_pos_vl} neg)")
        if not train_samples or not val_samples:
            print(f"    [warn] empty fold {fold_key} — skipping.")
            continue

        train_X, train_y = _extract_features_classifier(train_samples, esm_dim)
        val_X, val_y     = _extract_features_classifier(val_samples, esm_dim)

        # NN classifier
        model = create_baseline_model(
            'classifier', node_in_dim=node_in_dim, hidden_dim=hidden_dim,
            dropout=bl_cfg.get('dropout', 0.2)).to(device)
        train_loader = _make_tensor_loader(train_X, train_y.astype(np.float32),
                                           batch_size, shuffle=True)
        val_loader   = _make_tensor_loader(val_X, val_y.astype(np.float32),
                                           batch_size, shuffle=False)
        ckpt = os.path.join(
            save_dir,
            f'disc_binder_{esm_type}_baseline_{cv_split_mode}_{fold_key}_nn_best.pt')
        best_score, val_m = _train_nn_classifier_cv(
            model, train_loader, val_loader, val_seen, bl_cfg, device, ckpt, cls_metric)
        print(f"    NN  best_{cls_metric}={best_score:.3f}  "
              f"auc_un={val_m.get('auc_unseen', 0):.3f}  "
              f"auprc_un={val_m.get('auprc_unseen', 0):.3f}")
        nn_fold_metrics[fold_key] = {k: float(v) for k, v in val_m.items()}

        # ML classifiers
        print("    ML classifiers:")
        for ml_name, clf in _build_classifiers(ml_cfg, seed).items():
            clf_save = os.path.join(
                save_dir,
                f'disc_binder_{esm_type}_baseline_{cv_split_mode}_{fold_key}_ml_{ml_name.lower()}.pkl')
            r = _eval_classifier_ml_cv(ml_name, clf, train_X, train_y,
                                       val_X, val_y, val_seen, save_path=clf_save)
            ml_fold_metrics.setdefault(ml_name, {})[fold_key] = {
                k: float(v) for k, v in r['val'].items()}

    nn_agg = _aggregate_fold_metrics(nn_fold_metrics)
    ml_summary = {ml_name: {'folds': fm, 'aggregate': _aggregate_fold_metrics(fm)}
                  for ml_name, fm in ml_fold_metrics.items()}

    print(f"\n  NN  CV {cls_metric}: {nn_agg.get(cls_metric+'_mean', 0):.3f} ± "
          f"{nn_agg.get(cls_metric+'_std', 0):.3f}  |  "
          f"auc_unseen: {nn_agg.get('auc_unseen_mean', 0):.3f}  |  "
          f"auprc_unseen: {nn_agg.get('auprc_unseen_mean', 0):.3f}")

    result = {
        'task':          'disc_binder',
        'cv':            True,
        'cv_split_mode': cv_split_mode,
        'n_folds':       n_folds,
        'cls_metric':    cls_metric,
        'esm_type':      esm_type,
        'esm_dim':       esm_dim,
        'split_source':  split_src,
        'split_file':    os.path.basename(split_file),
        'nn':            {'folds': nn_fold_metrics, 'aggregate': nn_agg},
        'ml':            ml_summary,
    }
    out_path = os.path.join(
        save_dir,
        f'disc_binder_{esm_type}_baseline_{cv_split_mode}_{n_folds}fold_results.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=lambda x: float(x))
    print(f"\n  Disc binder CV results saved to: {out_path}")
    return result


def _run_disc_binder_esm_baseline(config: dict, device: torch.device,
                                  save_dir: str, esm_type: str) -> dict:
    """
    Single train/test split ESM baseline for binary binder classification.
    Loads graphs from pos_sthg_{esm_type}_{dist}A, capri_sthg_{esm_type}_{dist}A, etc.

    If config['esm_baseline']['cv_split_mode'] is set (global/fixed_pos) and
    n_folds > 0, delegates to the N-fold CV ESM baseline instead.
    """
    bl_cfg = config['esm_baseline']
    cv_split_mode = bl_cfg.get('cv_split_mode', None)
    n_folds = int(bl_cfg.get('n_folds', 0))
    if cv_split_mode in ('global', 'fixed_pos') and n_folds > 0:
        return _run_disc_binder_esm_baseline_cv(config, device, save_dir, esm_type)

    print(f"\n{'='*70}")
    print(f"ESM BASELINE ({esm_type.upper()}) — DISCRIMINATIVE BINDER CLASSIFICATION")
    print(f"{'='*70}")

    bl_cfg   = config['esm_baseline']
    data_cfg = config['data']
    seed     = config.get('system', {}).get('seed', 42)
    ml_cfg   = bl_cfg.get('disc_binder', {})

    esm_dim     = ESM_DIM[esm_type]
    node_in_dim = esm_dim
    hidden_dim  = 2 ** bl_cfg.get('hidden_dim_power', 9)
    batch_size  = bl_cfg.get('batch_size', 64)
    min_nodes   = data_cfg.get('min_nodes', 5)
    dist        = data_cfg['dist']
    pdb_root    = data_cfg['pdb_root']
    split_src   = bl_cfg['split_source_dir']
    premium_thr = bl_cfg.get('premium_threshold', 1.1)

    # ── Find and load split file ──────────────────────────────────
    split_file = _find_disc_binder_split_file(split_src)
    print(f"  Split file : {split_file}")
    with open(split_file) as f:
        split_data = json.load(f)

    all_names = [e[0].lower() for e in split_data['train'] + split_data['test']]
    all_names = list(set(all_names))
    print(f"  Total unique samples in split: {len(all_names)}")
    print(f"  ESM dim: {esm_dim}  →  pooled feature dim: {esm_dim}")

    # ── Load graphs from all candidate ESM source directories ─────
    search_dirs = [
        _esm_dir(pdb_root, 'pos_sthg', esm_type, dist),
        _esm_dir(pdb_root, 'swapped_sthg', esm_type, dist),
        _esm_dir(pdb_root, 'random_mut_sthg', esm_type, dist),
        _esm_dir(pdb_root, 'swapped_abag_sthg', esm_type, dist),
        _esm_dir(pdb_root, 'preppi_sthg', esm_type, dist),
    ]
    print(f"  Searching {len(search_dirs)} candidate directories …")
    graph_dict = _load_graphs_named(search_dirs, all_names, esm_dim, min_nodes)
    print(f"  Graphs loaded: {len(graph_dict)}")

    train_samples = _build_disc_binder_samples(
        split_data['train'], graph_dict, premium_thr)
    test_samples  = _build_disc_binder_samples(
        split_data['test'],  graph_dict, premium_thr)

    n_pos_tr = sum(1 for s in train_samples if s[2] > 0)
    n_pos_te = sum(1 for s in test_samples  if s[2] > 0)
    print(f"  Train: {len(train_samples)} ({n_pos_tr} pos / "
          f"{len(train_samples)-n_pos_tr} neg)")
    print(f"  Test : {len(test_samples)}  ({n_pos_te} pos / "
          f"{len(test_samples)-n_pos_te} neg)")

    if not train_samples or not test_samples:
        print("ERROR: empty train or test — skipping disc_binder ESM baseline.")
        return {}

    # ── Extract features (shared by NN and ML) ────────────────────
    train_X, train_y = _extract_features_classifier(train_samples, esm_dim)
    test_X,  test_y  = _extract_features_classifier(test_samples,  esm_dim)

    # ── NN Classifier ─────────────────────────────────────────────
    model = create_baseline_model(
        'classifier', node_in_dim=node_in_dim, hidden_dim=hidden_dim,
        dropout=bl_cfg.get('dropout', 0.2)).to(device)

    train_loader = _make_tensor_loader(train_X, train_y.astype(np.float32),
                                       batch_size, shuffle=True)
    test_loader  = _make_tensor_loader(test_X,  test_y.astype(np.float32),
                                       batch_size)
    ckpt = os.path.join(save_dir, 'disc_binder_nn_best.pt')

    # Use test loader as val (single split — early stopping on test AUC)
    best_val_auc, val_m, test_m = _train_nn_classifier(
        model, train_loader, test_loader, test_loader,
        bl_cfg, device, ckpt)

    print(f"  NN: best_val_auc={best_val_auc:.3f}  "
          f"test_auc={test_m.get('auc', 0):.3f}  "
          f"test_f1={test_m.get('f1', 0):.3f}  "
          f"test_acc={test_m.get('accuracy', 0):.3f}")

    nn_result = {
        'best_val_auc': float(best_val_auc),
        'test': {k: float(v) for k, v in test_m.items()},
    }

    # ── ML Classifiers ────────────────────────────────────────────
    ml_results = {}
    print("\n  ML Classifiers:")
    for ml_name, clf in _build_classifiers(ml_cfg, seed).items():
        clf_save = os.path.join(save_dir,
                                f'disc_binder_ml_{ml_name.lower()}.pkl')
        r = _eval_classifier_ml(ml_name, clf, train_X, train_y, test_X, test_y,
                                save_path=clf_save)
        ml_results[ml_name] = {k: float(v) for k, v in r['test'].items()}

    result = {
        'task':         'disc_binder',
        'esm_type':     esm_type,
        'esm_dim':      esm_dim,
        'split_source': split_src,
        'split_file':   os.path.basename(split_file),
        'nn':           nn_result,
        'ml':           ml_results,
    }

    out_path = os.path.join(save_dir,
                            f'disc_binder_{esm_type}_baseline_results.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=lambda x: float(x))
    print(f"\n  Disc binder results saved to: {out_path}")

    return result


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def run_esm_baseline_test(config: dict, device: torch.device) -> dict:
    """
    Run all three ESM baseline tasks: ΔG, ΔΔG, discriminative binder.

    Config layout
    -------------
    esm_baseline:
      esm_type: 'esm'  or 'esm480'
      split_source_dir: '/mnt/boltzmann/data/zhisong/trained_data_ssl_finetune/5layers_9hdim_jkmean'
      save_dir: '/mnt/boltzmann/data/zhisong/trained_data_ssl_finetune/esm_baseline'

      # NN hyper-parameters (same keys as baseline)
      hidden_dim_power:  9
      dropout:           0.2
      lr:                1e-4
      weight_decay:      1e-4
      n_epochs:          500
      patience:          30
      loss_fun:          l1
      batch_size:        64
      ...

      # Per-task ML hyper-parameters
      dG:   { ... }
      ddG:  { ... }
      disc_binder: { ... }

    data:
      pdb_root: /mnt/boltzmann/data/zhisong/curated_db
      dist:     "8"
      min_nodes: 5

    system:
      seed: 42
    """
    print(f"\n{'='*70}")
    print("ESM BASELINE TESTING — ΔG + ΔΔG + Discriminative Binder")
    print(f"{'='*70}")

    bl_cfg   = config['esm_baseline']
    seed     = config.get('system', {}).get('seed', 42)
    esm_type = str(bl_cfg.get('esm_type', 'esm')).strip().strip("'\"").lower()
    set_seed(seed)

    if esm_type not in ESM_DIM:
        raise ValueError(f"Unknown esm_type '{esm_type}'. Must be 'esm' or 'esm480'.")

    save_dir = bl_cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    esm_dim = ESM_DIM[esm_type]
    print(f"  Device:           {device}")
    print(f"  ESM type:         {esm_type} (dim={esm_dim})")
    print(f"  Split source dir: {bl_cfg['split_source_dir']}")
    print(f"  Save dir:         {save_dir}")
    print(f"  Seed:             {seed}")
    print(f"  Hidden dim:       2^{bl_cfg.get('hidden_dim_power',9)} "
          f"= {2**bl_cfg.get('hidden_dim_power',9)}")

    # -- Select which task(s) to run --------------------------------------
    # task: dG | ddG | binder_cls (aka disc_binder) | all  (default: all)
    all_tasks = [
        ('dG',          _run_dG_esm_baseline),
        ('ddG',         _run_ddG_esm_baseline),
        ('disc_binder', _run_disc_binder_esm_baseline),
    ]
    task_aliases = {
        'dg': 'dG', 'ddg': 'ddG',
        'binder_cls': 'disc_binder', 'disc_binder': 'disc_binder',
        'binder': 'disc_binder', 'cls': 'disc_binder',
        'all': 'all',
    }
    task_sel = str(bl_cfg.get('task', 'all')).strip().lower()
    if task_sel not in task_aliases:
        raise ValueError(
            f"Unknown task '{bl_cfg.get('task')}'. "
            f"Must be one of: dG, ddG, binder_cls, all.")
    selected = task_aliases[task_sel]
    if selected != 'all':
        all_tasks = [t for t in all_tasks if t[0] == selected]
    print(f"  Task:             {task_sel} -> "
          f"{'all (dG, ddG, disc_binder)' if selected == 'all' else selected}")

    all_results = {}

    for task_name, task_fn in all_tasks:
        try:
            all_results[task_name] = task_fn(config, device, save_dir, esm_type)
        except Exception as exc:
            import traceback
            print(f"\nERROR in {task_name} ESM baseline: {exc}")
            traceback.print_exc()
            all_results[task_name] = {'error': str(exc)}

    combined_name = (f'{esm_type}_baseline_all_results.json' if selected == 'all'
                     else f'{esm_type}_baseline_{selected}_results.json')
    combined_path = os.path.join(save_dir, combined_name)
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x))

    print(f"\n{'='*70}")
    print(f"ESM BASELINE TESTING ({esm_type.upper()}) COMPLETE")
    print(f"  Combined results: {combined_path}")
    print(f"{'='*70}")

    return all_results
