"""
ML fine-tuning for discriminative binder classification using pretrained
encoder embeddings.

Pipeline:
1. Load pretrained encoder (frozen, eval mode)
2. Load classifier data (binder vs non-binder, single train/test split)
3. Precompute node-level embeddings
4. Free encoder from GPU memory
5. Mean-pool embeddings per graph → numpy feature vector
6. Train sklearn ML classifiers (RF, GBT, DT, SVC) on training set
7. Evaluate on test set
8. Save results JSON

Save directory: same encoder folder or jk mode modified folder.
Result file:   ml_disc_binder[_jk{mode}]_results.json
"""

import os
import json
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold

from model_collection.FineTuneModels import (
    load_pretrained_encoder, precompute_embeddings,
    get_pool_input_dim,
)
from utils.Training_modules.common_utils import (
    set_seed, get_sample_type, load_ssl_config,
    MODEL_INIT_DIM, EDGE_IN_DIM, METADATA,
)
from utils.Training_modules.classifier_data_loader import (
    load_classifier_data, load_classifier_data_cv,
)
from utils.Training_modules.ml_utils import (
    extract_features_classifier, build_classifiers,
    eval_classifier_ml, eval_classifier_ml_cv,
    load_ssl_pos_split, aggregate_fold_metrics,
)


# ============================================================================
# DISCRIMINATIVE BINDER ML CV (leakage-aware)
# ============================================================================
def _build_ml_fold_splits(samples, pool_labels, cv_split_mode, n_folds, seed,
                          seen_pos_names, unseen_pos_names):
    """
    Return a list of (train_idx, val_idx) tuples over *samples* according to
    cv_split_mode ('global' StratifiedKFold or 'fixed_pos' KFold-on-negatives).
    Mirrors _run_disc_binder_cv in finetune_disc_binder_module.py exactly so
    folds are identical to the NN CV run (same seed, same pool).
    """
    labels_arr = np.array([int(pool_labels[s[0]]) for s in samples])
    n_pos = int((labels_arr == 1).sum())

    if cv_split_mode == 'global':
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return [(np.asarray(tr), np.asarray(vl))
                for tr, vl in skf.split(samples, labels_arr)]

    # fixed_pos
    pos_seen_idx = np.array(
        [i for i, s in enumerate(samples)
         if pool_labels[s[0]] == 1 and s[0] in seen_pos_names], dtype=int)
    pos_unseen_idx = np.array(
        [i for i, s in enumerate(samples)
         if pool_labels[s[0]] == 1 and s[0] in unseen_pos_names], dtype=int)
    neg_idx = np.array(
        [i for i, s in enumerate(samples) if pool_labels[s[0]] == 0], dtype=int)

    n_excluded_pos = n_pos - len(pos_seen_idx) - len(pos_unseen_idx)
    if n_excluded_pos > 0:
        print(f"  [WARN] {n_excluded_pos} positive(s) not in either SSL "
              f"train_names or val_names are dropped from fixed_pos CV.")
    if len(pos_unseen_idx) == 0:
        raise ValueError("fixed_pos CV requires at least one unseen positive "
                         "(SSL val_names) but found none.")
    if len(neg_idx) < n_folds:
        raise ValueError(f"fixed_pos CV needs at least n_folds={n_folds} "
                         f"negatives, got {len(neg_idx)}.")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_splits = []
    for neg_tr_local, neg_vl_local in kf.split(neg_idx):
        tr = np.concatenate([pos_seen_idx, neg_idx[neg_tr_local]])
        vl = np.concatenate([pos_unseen_idx, neg_idx[neg_vl_local]])
        fold_splits.append((tr, vl))
    return fold_splits


def _run_ml_disc_binder_cv(
    config: Dict,
    device: torch.device,
    pretrained_path: str,
    final_save_dir: str,
    embedding_type: str,
    hidden_dim: int,
    use_amp: bool,
    use_jk: bool,
    jk_mode: str,
    jk_suffix: str,
    n_folds: int,
):
    """
    N-fold CV for ML disc_binder classification on pretrained encoder
    embeddings.  Mirrors the NN _run_disc_binder_cv folds exactly, then trains
    sklearn classifiers per fold and reports overall + unseen metrics.

    cv_split_mode (config['disc_binder']['cv_split_mode']):
      * 'global'    — StratifiedKFold over the pool (leak-prone; use unseen
                      metrics for a leak-free signal).
      * 'fixed_pos' — SSL-train positives pinned to train, SSL-val positives
                      pinned to val, negatives KFold-rotated.
    """
    test_cfg = config['disc_binder']
    data_cfg = config['data']
    sys_cfg = config['system']
    ml_cfg_all = config.get('ml', {})
    ml_cfg = ml_cfg_all.get('disc_binder', ml_cfg_all)
    seed = sys_cfg.get('seed', 42)
    batch_size = test_cfg['batch_size']

    cv_split_mode = test_cfg.get('cv_split_mode', 'global')
    if cv_split_mode not in ('global', 'fixed_pos'):
        raise ValueError(f"Unknown cv_split_mode={cv_split_mode!r}. "
                         "Use 'global' or 'fixed_pos'.")
    cls_metric_name = test_cfg.get('cls_metric', 'auc')

    if embedding_type in ['+esm', '+esm480', 'only_esm', 'only_esm480']:
        raise ValueError("+esm/only_esm embedding types are not supported "
                         "for ML classifier CV.")

    # ---------------------------------------------------------------
    # Load CV pool + SSL seen/unseen split
    # ---------------------------------------------------------------
    pool_sttgs, pool_labels = load_classifier_data_cv(
        data_cfg=data_cfg,
        save_dir=final_save_dir,
        seed=seed,
        min_nodes=data_cfg.get('min_nodes', 5),
    )
    for name, st in pool_sttgs.items():
        st.protein_graph.y = torch.tensor(float(pool_labels[name]))

    samples = []
    for name, st in pool_sttgs.items():
        graph = st.protein_graph
        affinity = graph.y.item() if hasattr(graph.y, 'item') else graph.y
        sample_type = get_sample_type(affinity, test_cfg.get('premium_threshold', 1.1))
        samples.append((name, graph, affinity, sample_type))
    if len(samples) == 0:
        raise ValueError("CV pool is empty.")

    labels_arr = np.array([int(pool_labels[s[0]]) for s in samples])
    n_pos = int((labels_arr == 1).sum())
    n_neg = int((labels_arr == 0).sum())

    seen_pos_names, unseen_pos_names = load_ssl_pos_split(pretrained_path)
    sample_is_seen = {
        s[0]: (int(pool_labels[s[0]]) == 1 and s[0] in seen_pos_names)
        for s in samples
    }
    n_pos_seen = sum(1 for s in samples
                     if pool_labels[s[0]] == 1 and s[0] in seen_pos_names)
    n_pos_unseen = sum(1 for s in samples
                       if pool_labels[s[0]] == 1 and s[0] in unseen_pos_names)
    print(f"\nCV pool: {len(samples)} total ({n_pos} pos / {n_neg} neg)")
    print(f"  SSL coverage of positives: {n_pos_seen} seen / "
          f"{n_pos_unseen} unseen / {n_pos - n_pos_seen - n_pos_unseen} not in SSL split")
    print(f"  CV split mode: {cv_split_mode}  |  primary metric: {cls_metric_name}")

    # ---------------------------------------------------------------
    # Load encoder, precompute embeddings for the whole pool, free encoder
    # ---------------------------------------------------------------
    node_in_dim = MODEL_INIT_DIM[embedding_type]
    encoder = load_pretrained_encoder(
        checkpoint_path=pretrained_path,
        node_in_dim=node_in_dim,
        edge_in_dim=EDGE_IN_DIM,
        metadata=METADATA,
        hidden_dim=hidden_dim,
        num_hgt_layers=test_cfg['num_layers'],
        hgt_heads=test_cfg['hgt_heads'],
        device=device,
        message_style=test_cfg.get('message_style', 'gated_src'),
    )
    encoder.eval()
    print("\n--- Precomputing CV pool embeddings ---")
    precompute_embeddings(
        encoder, [s[0] for s in samples], [s[1] for s in samples], device,
        batch_size=batch_size, embedding_type=embedding_type, esm_dict=None,
        use_amp=use_amp, use_jk=use_jk, jk_mode=jk_mode,
    )
    del encoder
    torch.cuda.empty_cache()
    print("Encoder freed from memory.\n")

    # ---------------------------------------------------------------
    # Build folds (identical to NN CV) and run ML per fold
    # ---------------------------------------------------------------
    fold_splits = _build_ml_fold_splits(
        samples, pool_labels, cv_split_mode, n_folds, seed,
        seen_pos_names, unseen_pos_names,
    )

    mode_suffix = '' if cv_split_mode == 'global' else f"_{cv_split_mode}"
    base_name = f"ml_disc_binder{jk_suffix}{mode_suffix}"

    print(f"{'='*60}")
    print(f"ML Disc Binder {n_folds}-Fold CV  (split mode: {cv_split_mode})")
    print(f"{'='*60}")

    ml_fold_metrics: Dict[str, Dict[str, dict]] = {}
    fold_split_log: Dict[str, dict] = {}

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        fold_id = fold + 1
        fold_key = f'fold_{fold_id}'
        train_fold = [samples[i] for i in train_idx]
        val_fold = [samples[i] for i in val_idx]
        val_is_seen = [bool(sample_is_seen[s[0]]) for s in val_fold]

        n_pos_vl = sum(1 for s in val_fold if pool_labels[s[0]] == 1)
        n_pos_vl_unseen = sum(1 for s in val_fold
                              if pool_labels[s[0]] == 1 and not sample_is_seen[s[0]])
        print(f"\n  {fold_key}: train={len(train_fold)}  val={len(val_fold)} "
              f"({n_pos_vl} pos [{n_pos_vl_unseen} unseen] / "
              f"{len(val_fold)-n_pos_vl} neg)")

        fold_split_log[fold_key] = {
            'train': [[s[0], int(pool_labels[s[0]]), int(sample_is_seen[s[0]])]
                      for s in train_fold],
            'val':   [[s[0], int(pool_labels[s[0]]), int(sample_is_seen[s[0]])]
                      for s in val_fold],
        }

        train_X, train_y = extract_features_classifier(train_fold)
        val_X, val_y = extract_features_classifier(val_fold)

        for ml_name, clf in build_classifiers(ml_cfg, seed).items():
            clf_save = os.path.join(
                final_save_dir, f'{base_name}_{fold_key}_{ml_name}.pkl')
            r = eval_classifier_ml_cv(
                ml_name, clf, train_X, train_y, val_X, val_y, val_is_seen,
                save_path=clf_save,
            )
            ml_fold_metrics.setdefault(ml_name, {})[fold_key] = {
                k: float(v) for k, v in r['val'].items()
            }

    # ---------------------------------------------------------------
    # Aggregate & save
    # ---------------------------------------------------------------
    ml_summary = {}
    for ml_name, fm in ml_fold_metrics.items():
        agg = aggregate_fold_metrics(fm)
        ml_summary[ml_name] = {'folds': fm, 'aggregate': agg}
        print(f"  {ml_name:18s} | val {cls_metric_name}: "
              f"{agg.get(cls_metric_name + '_mean', 0.0):.3f} ± "
              f"{agg.get(cls_metric_name + '_std', 0.0):.3f} | "
              f"auc_unseen: {agg.get('auc_unseen_mean', 0.0):.3f} | "
              f"auprc_unseen: {agg.get('auprc_unseen_mean', 0.0):.3f}")

    results = {
        'task': 'ml_disc_binder',
        'cv': True,
        'cv_split_mode': cv_split_mode,
        'n_folds': n_folds,
        'cls_metric': cls_metric_name,
        'seed': seed,
        'embedding_type': embedding_type,
        'num_layers': test_cfg['num_layers'],
        'hidden_dim_power': test_cfg['hidden_dim_power'],
        'use_jk': use_jk,
        'jk_mode': jk_mode if use_jk else None,
        'n_pos_seen_total': n_pos_seen,
        'n_pos_unseen_total': n_pos_unseen,
        'n_neg_total': n_neg,
        'ml': ml_summary,
    }

    result_path = os.path.join(final_save_dir, f"{base_name}_{n_folds}fold_results.json")
    split_path = os.path.join(final_save_dir, f"{base_name}_{n_folds}fold_splits.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    with open(split_path, 'w') as f:
        json.dump(fold_split_log, f, indent=2)

    print(f"\n  Results: {result_path}")
    print(f"  Splits : {split_path}")
    print(f"{'='*60}\n")
    return results


# ============================================================================
# DISCRIMINATIVE BINDER ML FINE-TUNING
# ============================================================================
def run_ml_disc_binder_finetuning(
    config: Dict,
    device: torch.device,
    ssl_checkpoint_path: Optional[str] = None,
):
    """
    Run discriminative binder classification using sklearn ML models
    on pretrained encoder embeddings.

    Mirrors finetune_disc_binder_module.py for data loading and precomputation,
    then replaces the NN training loop with sklearn ML classifiers.

    Args:
        config: Full configuration dictionary (uses 'disc_binder' + 'data' sections)
        device: Torch device
        ssl_checkpoint_path: Path to SSL checkpoint (overrides config if provided)
    """
    print(f"\n{'='*80}")
    print("DISC BINDER ML FINE-TUNING (Encoder Embeddings + sklearn)")
    print(f"{'='*80}\n")

    test_cfg = config['disc_binder']
    data_cfg = config['data']
    sys_cfg = config['system']
    ml_cfg_all = config.get('ml', {})
    ml_cfg = ml_cfg_all.get('disc_binder', ml_cfg_all)  # per-task or fallback to flat

    # ---------------------------------------------------------------
    # Determine pretrained path
    # ---------------------------------------------------------------
    if ssl_checkpoint_path is not None:
        pretrained_path = ssl_checkpoint_path
    else:
        pretrained_path = test_cfg.get('pretrained_path')

    if pretrained_path is None:
        raise ValueError(
            "pretrained_path is required. "
            "Provide via config['disc_binder']['pretrained_path'] or ssl_checkpoint_path."
        )
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

    # Load SSL config if available
    ssl_config = load_ssl_config(pretrained_path)
    if ssl_config is not None:
        print("Loaded SSL config, overriding architecture parameters:")
        print(f"  num_layers: {ssl_config['num_layers']}")
        print(f"  hidden_dim_power: {ssl_config['hidden_dim_power']}")
        print(f"  hgt_heads: {ssl_config['hgt_heads']}")
        print(f"  message_style: {ssl_config.get('message_style', 'gated_src')}")
        print(f"  strategy_type: {ssl_config.get('strategy_type', 'dynamic')}")

        test_cfg['num_layers'] = ssl_config['num_layers']
        test_cfg['hidden_dim_power'] = ssl_config['hidden_dim_power']
        test_cfg['hgt_heads'] = ssl_config['hgt_heads']
        test_cfg['message_style'] = ssl_config.get('message_style', 'gated_src')
        test_cfg['strategy_type'] = ssl_config.get('strategy_type', 'dynamic')

    # ---------------------------------------------------------------
    # Setup directories
    # ---------------------------------------------------------------
    embedding_type = data_cfg['embedding_type']
    local_dir = (
        f"{test_cfg['num_layers']}layers_{test_cfg['hidden_dim_power']}hdim"
        + (f'_{embedding_type}' if embedding_type in ['esm', 'esm480'] else '')
        + (f"_{test_cfg['strategy_type']}" if test_cfg.get('strategy_type', 'dynamic') != 'dynamic' else '')
        + (f"_additive" if test_cfg.get('message_style', 'gated_src') == 'additive' else '')
    )
    # JK-Net config
    use_jk = test_cfg.get('use_jk', False)
    jk_mode = test_cfg.get('jk_mode', 'mean')
    jk_suffix = f"_jk{jk_mode}" if use_jk else ""
    local_dir += jk_suffix
    final_save_dir = os.path.join(test_cfg['save_dir'], local_dir)
    os.makedirs(final_save_dir, exist_ok=True)

    # JK-Net config (optional, default off)
    use_jk = test_cfg.get('use_jk', False)
    jk_mode = test_cfg.get('jk_mode', 'mean')
    jk_suffix = f"_jk{jk_mode}" if use_jk else ""

    use_amp = device.type == 'cuda'
    hidden_dim = 2 ** test_cfg['hidden_dim_power']

    print(f"Device: {device}")
    print(f"Task: ML disc_binder classification")
    print(f"Embedding Type: {embedding_type}")
    print(f"JK-Net: {'enabled (mode=' + jk_mode + ')' if use_jk else 'disabled'}")
    print(f"Pre-trained path: {pretrained_path}")
    print(f"Save directory: {final_save_dir}")

    # ---------------------------------------------------------------
    # CV mode: trigger when cv_split_mode is set + n_folds > 0
    # ---------------------------------------------------------------
    n_folds = int(test_cfg.get('n_folds', 0))
    cv_split_mode = test_cfg.get('cv_split_mode', None)
    if n_folds > 0 and cv_split_mode in ('global', 'fixed_pos'):
        print(f"n_folds        : {n_folds}  (CV mode, split={cv_split_mode})")
        return _run_ml_disc_binder_cv(
            config=config,
            device=device,
            pretrained_path=pretrained_path,
            final_save_dir=final_save_dir,
            embedding_type=embedding_type,
            hidden_dim=hidden_dim,
            use_amp=use_amp,
            use_jk=use_jk,
            jk_mode=jk_mode,
            jk_suffix=jk_suffix,
            n_folds=n_folds,
        )

    # ---------------------------------------------------------------
    # Load classifier data (single train/test split)
    # ---------------------------------------------------------------
    ssl_save_dir = os.path.dirname(pretrained_path) if os.path.isfile(pretrained_path) else pretrained_path
    train_sttgs, test_sttgs, train_labels, test_labels = load_classifier_data(
        data_cfg=data_cfg,
        ssl_save_dir=ssl_save_dir,
        save_dir=final_save_dir,
        seed=sys_cfg.get('seed', 42),
        min_nodes=data_cfg.get('min_nodes', 5),
    )

    # Stamp binary label into graph.y
    for name, st in train_sttgs.items():
        st.protein_graph.y = torch.tensor(float(train_labels[name]))
    for name, st in test_sttgs.items():
        st.protein_graph.y = torch.tensor(float(test_labels[name]))

    # ESM not yet supported for classifier task
    esm_dict = None
    if embedding_type in ['+esm', '+esm480', 'only_esm', 'only_esm480']:
        raise ValueError(f"+esm/only_esm embedding types are not yet supported for classifier task.")

    # ---------------------------------------------------------------
    # Convert to samples
    # ---------------------------------------------------------------
    train_samples, test_samples = [], []
    for name, st in train_sttgs.items():
        graph = st.protein_graph
        affinity = graph.y.item() if hasattr(graph.y, 'item') else graph.y
        sample_type = get_sample_type(affinity, test_cfg.get('premium_threshold', 1.1))
        train_samples.append((name, graph, affinity, sample_type))
    for name, st in test_sttgs.items():
        graph = st.protein_graph
        affinity = graph.y.item() if hasattr(graph.y, 'item') else graph.y
        sample_type = get_sample_type(affinity, test_cfg.get('premium_threshold', 1.1))
        test_samples.append((name, graph, affinity, sample_type))

    n_pos_train = sum(1 for s in train_samples if s[2] > 0)
    n_neg_train = sum(1 for s in train_samples if s[2] <= 0)
    n_pos_test = sum(1 for s in test_samples if s[2] > 0)
    n_neg_test = sum(1 for s in test_samples if s[2] <= 0)
    print(f"\nTrain: {len(train_samples)} ({n_pos_train} pos / {n_neg_train} neg)")
    print(f"Test:  {len(test_samples)} ({n_pos_test} pos / {n_neg_test} neg)")

    if not train_samples:
        raise ValueError("No training samples loaded.")

    # ---------------------------------------------------------------
    # Load encoder & precompute embeddings
    # ---------------------------------------------------------------
    node_in_dim = MODEL_INIT_DIM[embedding_type]
    encoder = load_pretrained_encoder(
        checkpoint_path=pretrained_path,
        node_in_dim=node_in_dim,
        edge_in_dim=EDGE_IN_DIM,
        metadata=METADATA,
        hidden_dim=hidden_dim,
        num_hgt_layers=test_cfg['num_layers'],
        hgt_heads=test_cfg['hgt_heads'],
        device=device,
        message_style=test_cfg.get('message_style', 'gated_src'),
    )

    train_names = [s[0] for s in train_samples]
    train_graphs = [s[1] for s in train_samples]
    test_names = [s[0] for s in test_samples]
    test_graphs = [s[1] for s in test_samples]

    encoder.eval()
    print("\n--- Precomputing train embeddings ---")
    precompute_embeddings(
        encoder, train_names, train_graphs, device,
        batch_size=test_cfg['batch_size'],
        embedding_type=embedding_type, esm_dict=esm_dict,
        use_amp=use_amp, use_jk=use_jk, jk_mode=jk_mode,
    )
    print("--- Precomputing test embeddings ---")
    precompute_embeddings(
        encoder, test_names, test_graphs, device,
        batch_size=test_cfg['batch_size'],
        embedding_type=embedding_type, esm_dict=esm_dict,
        use_amp=use_amp, use_jk=use_jk, jk_mode=jk_mode,
    )

    # Free encoder
    del encoder
    torch.cuda.empty_cache()
    print("Encoder freed from memory.\n")

    # ---------------------------------------------------------------
    # Mean-pool → numpy features
    # ---------------------------------------------------------------
    train_X, train_y = extract_features_classifier(train_samples)
    test_X, test_y = extract_features_classifier(test_samples)
    print(f"Feature dimension: {train_X.shape[1]}")

    # ---------------------------------------------------------------
    # Train & evaluate ML classifiers
    # ---------------------------------------------------------------
    seed = sys_cfg.get('seed', 42)

    print(f"\n{'='*60}")
    print("ML Disc Binder Classification")
    print(f"{'='*60}")
    print("  ML Classifiers:")

    ml_results = {}
    for ml_name, clf in build_classifiers(ml_cfg, seed).items():
        save_path = os.path.join(final_save_dir, f'ml_disc_binder{jk_suffix}_{ml_name}.pkl')
        r = eval_classifier_ml(
            ml_name, clf, train_X, train_y, test_X, test_y,
            save_path=save_path,
        )
        ml_results[ml_name] = {
            'test': {k: float(v) for k, v in r['test'].items()},
        }

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    results = {
        'task': 'ml_disc_binder',
        'embedding_type': embedding_type,
        'num_layers': test_cfg['num_layers'],
        'hidden_dim_power': test_cfg['hidden_dim_power'],
        'use_jk': use_jk,
        'jk_mode': jk_mode if use_jk else None,
        'feature_dim': int(train_X.shape[1]),
        'n_train': len(train_samples),
        'n_test': len(test_samples),
        'train_pos_neg': f"{n_pos_train}/{n_neg_train}",
        'test_pos_neg': f"{n_pos_test}/{n_neg_test}",
        'ml': ml_results,
    }

    result_name = f"ml_disc_binder{jk_suffix}_results.json"
    result_path = os.path.join(final_save_dir, result_name)
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save split info
    split_log = {
        'train': [[s[0], float(s[2])] for s in train_samples],
        'test': [[s[0], float(s[2])] for s in test_samples],
    }
    split_name = f"ml_disc_binder{jk_suffix}_splits.json"
    split_path = os.path.join(final_save_dir, split_name)
    with open(split_path, 'w') as f:
        json.dump(split_log, f, indent=2)

    print(f"\n  Results: {result_path}")
    print(f"  Splits:  {split_path}")
    print(f"{'='*60}\n")

    return results


# ============================================================================
# MAIN FOR STANDALONE EXECUTION
# ============================================================================
if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description='Disc Binder ML Fine-tuning (Encoder Embeddings + sklearn)',
    )
    parser.add_argument('-config', type=str, required=True,
                        help='Path to YAML configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['system']['seed'])
    device = torch.device(f"cuda:{config['system']['cuda_id']}"
                          if torch.cuda.is_available() else 'cpu')

    run_ml_disc_binder_finetuning(config, device)
