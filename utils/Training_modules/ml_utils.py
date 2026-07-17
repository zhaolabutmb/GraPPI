"""
Shared utilities for ML fine-tuning on pretrained encoder embeddings.

Provides:
- mean_pool_graph: Mean-pool receptor + ligand node embeddings to a 1D vector
- ML model builders: regressors (RF, GBT, DT, SVR), classifiers (RF, GBT, DT, SVC)
- Evaluation helpers: fit + predict + compute metrics
- Metric functions: Pearson r / Spearman r / MAE, Accuracy / F1 / AUC
- Fold result aggregation
"""

import math
import os
import json
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import (
    RandomForestRegressor, HistGradientBoostingRegressor,
    RandomForestClassifier, HistGradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, f1_score,
    roc_auc_score, average_precision_score,
)


# ============================================================================
# POOLING
# ============================================================================

def mean_pool_graph(graph) -> np.ndarray:
    """
    Stack receptor + ligand node features (precomputed embeddings in graph.x),
    mean-pool across all nodes → 1-D numpy array [dim].

    This is identical to the baseline pooling strategy, operating on
    whatever features are stored in graph[node_type].x (raw or encoder output).
    """
    r = graph['receptor'].x.float()
    l = graph['ligand'].x.float()
    return torch.cat([r, l], dim=0).mean(0).numpy()


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_regression(samples: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
    """(name, graph, affinity, *) → features [N, dim], targets [N]."""
    X = np.stack([mean_pool_graph(s[1]) for s in samples])
    y = np.array([s[2] for s in samples], dtype=np.float32)
    return X, y


def extract_features_ddg(pairs: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
    """
    (mut_name, mut_graph, wt_graph, mut_aff) → features [N, 2*dim], targets [N].

    Feature : [mut_pooled ‖ wt_pooled]
    Target  : mut_graph.y  (= mut_aff, the original binding affinity)
    """
    feats, targets = [], []
    for _, mut_graph, wt_graph, _ in pairs:
        feats.append(np.concatenate([mean_pool_graph(mut_graph),
                                     mean_pool_graph(wt_graph)]))
        mut_y = mut_graph.y.item() if hasattr(mut_graph.y, 'item') else float(mut_graph.y)
        targets.append(mut_y)
    return np.stack(feats), np.array(targets, dtype=np.float32)


def extract_features_classifier(samples: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
    """(name, graph, label, *) → features [N, dim], binary labels [N]."""
    X = np.stack([mean_pool_graph(s[1]) for s in samples])
    y = np.array([int(s[2] > 0) for s in samples])
    return X, y


# ============================================================================
# ML MODEL BUILDERS
# ============================================================================

def build_regressors(ml_cfg: dict, seed: int) -> dict:
    """Build a dict of sklearn regressors (RF, GBT, DT, SVR)."""
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


def build_classifiers(ml_cfg: dict, seed: int) -> dict:
    """Build a dict of sklearn classifiers (RF, GBT, DT, SVC)."""
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
# METRICS
# ============================================================================

def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


def cls_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                y_prob: np.ndarray) -> dict:
    """Accuracy, F1, AUC for binary classification."""
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5
    return {'accuracy': acc, 'f1': f1, 'auc': auc}


# ============================================================================
# SEEN / UNSEEN HELPERS (for leakage-aware CV evaluation)
# ============================================================================

def load_ssl_pos_split(pretrained_path: str) -> Tuple[Set[str], Set[str]]:
    """
    Load ssl_train_val_split.json next to the SSL checkpoint and return
    (seen_pos_names, unseen_pos_names) as lowercased name sets.
    Positives in SSL train_names were seen by the encoder during
    pre-training; positives in val_names were not. All negatives are
    unseen by definition.
    """
    ssl_dir = os.path.dirname(pretrained_path) if os.path.isfile(pretrained_path) else pretrained_path
    split_path = os.path.join(ssl_dir, 'ssl_train_val_split.json')
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"ssl_train_val_split.json not found at {split_path}. "
            "Required to identify seen/unseen positives for disc_binder CV."
        )
    with open(split_path, 'r') as f:
        ssl_split = json.load(f)
    seen = set(n.lower() for n in ssl_split['train_names'])
    unseen = set(n.lower() for n in ssl_split['val_names'])
    return seen, unseen


def cls_metrics_full(y_true, y_prob, is_seen) -> dict:
    """
    Full classification metrics for a held-out fold, including overall AUPRC
    and unseen-only AUROC/AUPRC.  *is_seen* must be aligned with y_true/y_prob;
    the unseen subset = (~is_seen) = all negatives + positives the encoder did
    not see during SSL pre-training.
    Threshold for accuracy/F1 is 0.5.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    seen = np.asarray(is_seen, dtype=bool)

    m = cls_metrics(y_true, y_pred, y_prob)
    if len(np.unique(y_true)) > 1:
        m['auprc'] = float(average_precision_score(y_true, y_prob))
    else:
        m['auprc'] = 0.0

    unseen = ~seen
    m['n_seen'] = int(seen.sum())
    m['n_unseen'] = int(unseen.sum())
    if unseen.sum() > 0 and len(np.unique(y_true[unseen])) > 1:
        m['auc_unseen'] = float(roc_auc_score(y_true[unseen], y_prob[unseen]))
        m['auprc_unseen'] = float(average_precision_score(y_true[unseen], y_prob[unseen]))
    else:
        m['auc_unseen'] = 0.0
        m['auprc_unseen'] = 0.0
    return m


# ============================================================================
# ML EVALUATORS
# ============================================================================

def eval_regressor_ml(
    name: str, reg,
    train_X, train_y, val_X, val_y, test_X, test_y,
    save_path: Optional[str] = None,
) -> dict:
    """Fit a regressor on train, evaluate on val + test, optionally save."""
    reg.fit(train_X, train_y)
    result = {
        'val':  reg_metrics(val_y, reg.predict(val_X)),
        'test': reg_metrics(test_y, reg.predict(test_X)),
    }
    print(f"    {name:20s}  val_rp={result['val']['rp']:.3f}  "
          f"test_rp={result['test']['rp']:.3f}  "
          f"test_mae={result['test']['mae']:.3f}")
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(reg, f)
    return result


def eval_classifier_ml(
    name: str, clf,
    train_X, train_y, test_X, test_y,
    save_path: Optional[str] = None,
) -> dict:
    """Fit a classifier on train, evaluate on test, optionally save."""
    clf.fit(train_X, train_y)
    te_pred = clf.predict(test_X)
    if hasattr(clf, 'predict_proba'):
        te_prob = clf.predict_proba(test_X)[:, 1]
    elif hasattr(clf, 'decision_function'):
        te_prob = clf.decision_function(test_X)
    else:
        te_prob = te_pred.astype(float)
    result = {'test': cls_metrics(test_y, te_pred, te_prob)}
    print(f"    {name:20s}  test_auc={result['test']['auc']:.3f}  "
          f"test_f1={result['test']['f1']:.3f}  "
          f"test_acc={result['test']['accuracy']:.3f}")
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(clf, f)
    return result


def eval_classifier_ml_cv(
    name: str, clf,
    train_X, train_y, val_X, val_y, val_is_seen,
    save_path: Optional[str] = None,
) -> dict:
    """
    Fit a classifier on a training fold, evaluate on the held-out validation
    fold with full (overall + unseen) metrics.  Returns {'val': {...}}.
    """
    clf.fit(train_X, train_y)
    if hasattr(clf, 'predict_proba'):
        val_prob = clf.predict_proba(val_X)[:, 1]
    elif hasattr(clf, 'decision_function'):
        val_prob = clf.decision_function(val_X)
    else:
        val_prob = clf.predict(val_X).astype(float)
    result = {'val': cls_metrics_full(val_y, val_prob, val_is_seen)}
    print(f"    {name:20s}  val_auc={result['val']['auc']:.3f}  "
          f"val_auprc={result['val']['auprc']:.3f}  "
          f"auc_un={result['val']['auc_unseen']:.3f}  "
          f"auprc_un={result['val']['auprc_unseen']:.3f}")
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(clf, f)
    return result


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_fold_metrics(fold_metrics: dict) -> dict:
    """
    Given {fold_id: {metric_key: value, …}, …} (flat dicts per fold),
    compute mean and std across folds for each metric key.
    Returns {f"{key}_mean": …, f"{key}_std": …, …}.
    """
    per_key: dict = defaultdict(list)
    for v in fold_metrics.values():
        for k, val in v.items():
            per_key[k].append(float(val))
    agg = {}
    for k, vals in per_key.items():
        agg[f'{k}_mean'] = float(np.mean(vals))
        agg[f'{k}_std'] = float(np.std(vals))
    return agg
