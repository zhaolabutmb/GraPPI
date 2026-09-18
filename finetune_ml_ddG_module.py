"""
ML fine-tuning for ΔΔG regression using pretrained encoder embeddings.

Pipeline:
1. Load pretrained encoder (frozen, eval mode)
2. Load mutant + wild-type training graphs, pair them
3. Load test mutant/WT pairs
4. Precompute node-level embeddings (with WT graph deduplication)
5. Free encoder from GPU memory
6. Mean-pool each graph, build per-pair features [mut_pooled ‖ wt_pooled]
7. Train sklearn ML models (RF, GBT, DT, SVR) with 5-fold KFold CV
8. Evaluate each fold on held-out test set
9. Aggregate results (mean ± std) and save JSON

Save directory: same encoder folder (no JK suffix in folder name).
Result file:   ml_ddg[_jk{mode}]_5fold_results.json
"""

import pickle
import os
import json
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch

from model_collection.FineTuneModels import (
    load_pretrained_encoder, precompute_embeddings,
    get_pool_input_dim, build_embedder,
)
from utils.Training_modules.common_utils import (
    set_seed, load_sttgs_from_dir, load_ssl_config,
    filter_samples_finetune, get_fold_splits,
    MODEL_INIT_DIM, EDGE_IN_DIM, METADATA,
)
from utils.Training_modules.ml_utils import (
    extract_features_ddg, build_regressors,
    eval_regressor_ml, aggregate_fold_metrics,
    reg_metrics,
)


# ============================================================================
# ΔΔG ML FINE-TUNING WITH CROSS-VALIDATION + TEST
# ============================================================================
def run_ml_ddG_finetuning(
    config: Dict,
    device: torch.device,
    ssl_checkpoint_path: Optional[str] = None,
):
    """
    Run ΔΔG regression using sklearn ML models on pretrained encoder embeddings.

    Mirrors finetune_ddG_module.py for data loading, pairing, precomputation,
    then replaces the NN training loop with sklearn ML models.

    Args:
        config: Full configuration dictionary (uses 'mutation' + 'data' sections)
        device: Torch device
        ssl_checkpoint_path: Path to SSL checkpoint (overrides config if provided)
    """
    print(f"\n{'='*80}")
    print("ΔΔG ML FINE-TUNING (Encoder Embeddings + sklearn, CV + Test)")
    print(f"{'='*80}\n")

    mut_cfg = config['mutation']
    data_cfg = config['data']
    sys_cfg = config['system']
    ml_cfg_all = config.get('ml', {})
    ml_cfg = ml_cfg_all.get('mutation', ml_cfg_all)  # per-task or fallback to flat

    # ---------------------------------------------------------------
    # Determine pretrained path
    # ---------------------------------------------------------------
    if ssl_checkpoint_path is not None:
        pretrained_path = ssl_checkpoint_path
    else:
        pretrained_path = mut_cfg.get('pretrained_path')

    if pretrained_path is None:
        raise ValueError(
            "pretrained_path is required. "
            "Provide via config['mutation']['pretrained_path'] or ssl_checkpoint_path."
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

        mut_cfg['num_layers'] = ssl_config['num_layers']
        mut_cfg['hidden_dim_power'] = ssl_config['hidden_dim_power']
        mut_cfg['hgt_heads'] = ssl_config['hgt_heads']
        mut_cfg['message_style'] = ssl_config.get('message_style', 'gated_src')
        mut_cfg['strategy_type'] = ssl_config.get('strategy_type', 'dynamic')

    # ---------------------------------------------------------------
    # Setup directories (NO jk suffix in folder name)
    # ---------------------------------------------------------------
    #load embedding type from ssl config if available, otherwise use data config
    if ssl_config['embedding_type'] != data_cfg['embedding_type']:
        embedding_type = ssl_config['embedding_type']
    else:
        embedding_type = data_cfg['embedding_type']
    local_dir = (
        f"{mut_cfg['num_layers']}layers_{mut_cfg['hidden_dim_power']}hdim"
        + (f'_{embedding_type}' if embedding_type in ['esm', 'esm480'] else '')
        + (f"_{mut_cfg['strategy_type']}" if mut_cfg.get('strategy_type', 'dynamic') != 'dynamic' else '')
        + (f"_additive" if mut_cfg.get('message_style', 'gated_src') == 'additive' else '')
    )
    # JK-Net config
    use_jk = mut_cfg.get('use_jk', False)
    jk_mode = mut_cfg.get('jk_mode', 'mean')
    jk_suffix = f"_jk{jk_mode}" if use_jk else ""
    local_dir += jk_suffix
    final_save_dir = os.path.join(mut_cfg['save_dir'], local_dir)
    os.makedirs(final_save_dir, exist_ok=True)

    # JK-Net config (optional, default off)
    use_jk = mut_cfg.get('use_jk', False)
    jk_mode = mut_cfg.get('jk_mode', 'mean')
    jk_suffix = f"_jk{jk_mode}" if use_jk else ""

    use_amp = device.type == 'cuda'
    hidden_dim = 2 ** mut_cfg['hidden_dim_power']

    print(f"Device: {device}")
    print(f"Task: ML ΔΔG regression")
    print(f"Embedding Type: {embedding_type}")
    print(f"JK-Net: {'enabled (mode=' + jk_mode + ')' if use_jk else 'disabled'}")
    print(f"Pre-trained path: {pretrained_path}")
    print(f"Save directory: {final_save_dir}")

    # ---------------------------------------------------------------
    # Load ΔΔG data (same logic as finetune_ddG_module.py)
    # ---------------------------------------------------------------
    dist = data_cfg['dist']

    # --- Load mutant training graphs ---
    mut_train_dir = (
        f'{data_cfg["pdb_root"]}/mutant_sthg'
        f'{"" if embedding_type == "base" else "_" + embedding_type}_{dist}A'
    )
    print(f"\nLoading mutant training graphs from: {mut_train_dir}")
    mut_sttgs_train = load_sttgs_from_dir(mut_train_dir)
    print(f"  Loaded {len(mut_sttgs_train)} mutant training graphs")

    # --- Load wild-type training graphs (only PDB IDs in mutant set) ---
    wt_train_dir = (
        f'{data_cfg["pdb_root"]}/unmut_sthg'
        f'{"" if embedding_type == "base" else "_" + embedding_type}_{dist}A'
    )
    mut_pdb_ids_train = set(name.split('_')[0] for name in mut_sttgs_train.keys())
    print(f"Loading wild-type training graphs from: {wt_train_dir}")
    wt_sttgs_train = {}
    for pdb_id in mut_pdb_ids_train:
        pkl_path = os.path.join(wt_train_dir, f'{pdb_id}.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as fh:
                wt_sttgs_train[pdb_id] = pickle.load(fh)
    print(f"  Loaded {len(wt_sttgs_train)} wild-type training graphs")

    # --- Load test graphs ---
    if embedding_type in ['esm', 'esm480']:
        test_dir = f'{data_cfg["pdb_root"]}/testset_mut_{embedding_type}'
    else:
        test_dir = f'{data_cfg["pdb_root"]}/testset_mut'
    print(f"\nLoading test graphs from: {test_dir}")
    all_sttgs_test = load_sttgs_from_dir(test_dir)
    print(f"  Loaded {len(all_sttgs_test)} test graphs total")

    # Separate test into mutant vs wild-type
    wt_sttgs_test, mut_sttgs_test = {}, {}
    for name, sttg in all_sttgs_test.items():
        if len(name.split('_')) > 1:
            mut_sttgs_test[name] = sttg
        else:
            wt_sttgs_test[name] = sttg
    print(f"  Test split: {len(mut_sttgs_test)} mutants, {len(wt_sttgs_test)} wild-types")

    # --- Exclude training mutants overlapping test PDB IDs ---
    test_pdb_ids = set(name.split('_')[0] for name in mut_sttgs_test.keys())
    n_before = len(mut_sttgs_train)
    mut_sttgs_train = {
        k: v for k, v in mut_sttgs_train.items()
        if k.split('_')[0] not in test_pdb_ids
    }
    n_excluded = n_before - len(mut_sttgs_train)
    if n_excluded > 0:
        print(f"\nExcluded {n_excluded} training mutants overlapping test PDB IDs")
    print(f"Remaining training mutants: {len(mut_sttgs_train)}")

    # --- Filter by min nodes ---
    mut_sttgs_train = filter_samples_finetune(mut_sttgs_train, min_nodes=data_cfg.get('min_nodes', 5))
    wt_sttgs_train = filter_samples_finetune(wt_sttgs_train, min_nodes=data_cfg.get('min_nodes', 5))
    mut_sttgs_test = filter_samples_finetune(mut_sttgs_test, min_nodes=data_cfg.get('min_nodes', 5))
    wt_sttgs_test = filter_samples_finetune(wt_sttgs_test, min_nodes=data_cfg.get('min_nodes', 5))

    # ---------------------------------------------------------------
    # Pair mutant ↔ wild-type
    # ---------------------------------------------------------------
    def pair_mutant_wildtype(mut_sttgs, wt_sttgs, label=""):
        paired, unmatched = [], []
        for mut_name, mut_sttg in mut_sttgs.items():
            pdb_id = mut_name.split('_')[0]
            if pdb_id in wt_sttgs:
                mut_graph = mut_sttg.protein_graph
                wt_graph = wt_sttgs[pdb_id].protein_graph
                mut_aff = mut_graph.y.item() if hasattr(mut_graph.y, 'item') else float(mut_graph.y)
                paired.append((mut_name, mut_graph, wt_graph, mut_aff))
            else:
                unmatched.append(mut_name)
        if unmatched:
            print(f"  [{label}] {len(unmatched)} mutants have no matching wild-type")
        return paired, unmatched

    print("\n--- Pairing training mutant ↔ wild-type ---")
    train_pairs, _ = pair_mutant_wildtype(mut_sttgs_train, wt_sttgs_train, "Train")
    print(f"  Matched training pairs: {len(train_pairs)}")

    print("--- Pairing test mutant ↔ wild-type ---")
    test_pairs, _ = pair_mutant_wildtype(mut_sttgs_test, wt_sttgs_test, "Test")
    print(f"  Matched test pairs: {len(test_pairs)}")

    if not train_pairs:
        print("Error: No training pairs found.")
        return
    if not test_pairs:
        print("Error: No test pairs found.")
        return

    # ---------------------------------------------------------------
    # Prepare graph lists
    # ---------------------------------------------------------------
    train_mut_graphs = [p[1] for p in train_pairs]
    train_wt_graphs = [p[2] for p in train_pairs]
    train_names = [p[0] for p in train_pairs]
    test_mut_graphs = [p[1] for p in test_pairs]
    test_wt_graphs = [p[2] for p in test_pairs]
    test_names = [p[0] for p in test_pairs]

    # ---------------------------------------------------------------
    # Load encoder (GraPPI SSL or sequence student) & precompute embeddings
    # ---------------------------------------------------------------
    node_in_dim = MODEL_INIT_DIM[embedding_type]
    encoder_type = mut_cfg.get('encoder_type', 'ssl')
    batch_size = mut_cfg['batch_size']
    precompute_fn, hidden_dim, use_jk = build_embedder(
        pretrained_path, device,
        encoder_type=encoder_type,
        embedding_type=embedding_type,
        node_in_dim=node_in_dim,
        edge_in_dim=EDGE_IN_DIM,
        metadata=METADATA,
        hidden_dim=hidden_dim,
        num_hgt_layers=mut_cfg['num_layers'],
        hgt_heads=mut_cfg['hgt_heads'],
        message_style=mut_cfg.get('message_style', 'gated_src'),
        use_jk=use_jk,
        jk_mode=jk_mode,
        batch_size=batch_size,
        use_amp=use_amp,
    )

    # Train mutant embeddings
    print("\n--- Precomputing train mutant embeddings ---")
    precompute_fn(train_names, train_mut_graphs)

    # Train WT embeddings (deduplicated)
    seen_wt_ids = set()
    unique_wt_names, unique_wt_graphs = [], []
    for name, graph in zip(train_names, train_wt_graphs):
        pdb_id = name.split('_')[0]
        if pdb_id not in seen_wt_ids:
            seen_wt_ids.add(pdb_id)
            unique_wt_names.append(pdb_id)
            unique_wt_graphs.append(graph)
    print(f"--- Precomputing train WT embeddings ({len(unique_wt_graphs)} unique) ---")
    precompute_fn(unique_wt_names, unique_wt_graphs)

    # Test mutant embeddings
    print("--- Precomputing test mutant embeddings ---")
    precompute_fn(test_names, test_mut_graphs)

    # Test WT embeddings (deduplicated)
    seen_wt_ids_test = set()
    unique_test_wt_names, unique_test_wt_graphs = [], []
    for name, graph in zip(test_names, test_wt_graphs):
        pdb_id = name.split('_')[0]
        if pdb_id not in seen_wt_ids_test:
            seen_wt_ids_test.add(pdb_id)
            unique_test_wt_names.append(pdb_id)
            unique_test_wt_graphs.append(graph)
    print(f"--- Precomputing test WT embeddings ({len(unique_test_wt_graphs)} unique) ---")
    precompute_fn(unique_test_wt_names, unique_test_wt_graphs)

    torch.cuda.empty_cache()
    print("Embeddings precomputed.\n")

    # ---------------------------------------------------------------
    # Mean-pool → numpy features for ML
    # ---------------------------------------------------------------
    test_X, test_y = extract_features_ddg(test_pairs)
    print(f"Feature dimension: {test_X.shape[1]} (2 × {test_X.shape[1] // 2})")

    # ---------------------------------------------------------------
    # N-fold CV
    # ---------------------------------------------------------------
    n_folds = mut_cfg.get('n_folds', 5)
    seed = sys_cfg.get('seed', 42)
    fold_strategy = mut_cfg.get('fold_strategy', 'group')

    train_names_list = [p[0] for p in train_pairs]
    fold_splits = get_fold_splits(train_names_list, n_folds, seed, fold_strategy)

    groups = np.array([p[0].split('_')[0] for p in train_pairs])
    unique_groups = np.unique(groups)
    ml_fold_results = defaultdict(dict)
    fold_split_log = {}

    strategy_label = f"Seeded {'GroupKFold' if fold_strategy == 'group' else 'KFold'}"
    print(f"\n{'='*60}")
    print(f"Starting {n_folds}-Fold {strategy_label} CV (seed={seed}) — ML ΔΔG Regression")
    print(f"  {len(unique_groups)} unique PDB groups across {len(train_pairs)} samples")
    print(f"{'='*60}\n")

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        fold_id = f'fold_{fold + 1}'
        val_groups = set(groups[val_idx]) if len(val_idx) > 0 else set()
        print(f"\n--- {fold_id}/{n_folds}  (val groups: {len(val_groups)}) ---")

        train_fold_pairs = [train_pairs[i] for i in train_idx]
        val_fold_pairs = [train_pairs[i] for i in val_idx]

        fold_split_log[fold_id] = {
            'train': [[p[0], float(p[3])] for p in train_fold_pairs],
            'val': [[p[0], float(p[3])] for p in val_fold_pairs],
        }

        train_X, train_y = extract_features_ddg(train_fold_pairs)

        has_val = len(val_fold_pairs) > 0
        if has_val:
            val_X, val_y = extract_features_ddg(val_fold_pairs)
            print(f"  Train: {len(train_fold_pairs)}, Val: {len(val_fold_pairs)}, Test: {len(test_pairs)}")
        else:
            print(f"  Train: {len(train_fold_pairs)} (100% — no val), Test: {len(test_pairs)}")

        print(f"  ML Regressors:")

        for ml_name, reg in build_regressors(ml_cfg, seed).items():
            save_path = os.path.join(final_save_dir, f'ml_ddg{jk_suffix}_{fold_id}_{ml_name}.pkl')

            if has_val:
                r = eval_regressor_ml(
                    ml_name, reg, train_X, train_y, val_X, val_y, test_X, test_y,
                    save_path=save_path,
                )
                ml_fold_results[ml_name][fold_id] = {
                    'val': {k: float(v) for k, v in r['val'].items()},
                    'test': {k: float(v) for k, v in r['test'].items()},
                }
            else:
                # n_folds=1: train on all, evaluate on test only
                reg.fit(train_X, train_y)
                test_pred = reg.predict(test_X)
                test_m = reg_metrics(test_y, test_pred)
                print(f"    {ml_name:20s}  test_rp={test_m['rp']:.3f}  "
                      f"test_mae={test_m['mae']:.3f}")
                if save_path:
                    with open(save_path, 'wb') as f:
                        pickle.dump(reg, f)
                ml_fold_results[ml_name][fold_id] = {
                    'val': {},
                    'test': {k: float(v) for k, v in test_m.items()},
                }

    # ---------------------------------------------------------------
    # Aggregate and save results
    # ---------------------------------------------------------------
    results = {
        'task': 'ml_ddG',
        'embedding_type': embedding_type,
        'num_layers': mut_cfg['num_layers'],
        'hidden_dim_power': mut_cfg['hidden_dim_power'],
        'use_jk': use_jk,
        'jk_mode': jk_mode if use_jk else None,
        'feature_dim': int(test_X.shape[1]),
        'n_folds': n_folds,
        'fold_strategy': fold_strategy,
        'n_train_pairs': len(train_pairs),
        'n_test_pairs': len(test_pairs),
        'ml': {},
    }

    print(f"\n{'='*60}")
    print(f"ML ΔΔG {n_folds}-Fold CV Results")
    print(f"{'='*60}")

    for ml_name, fold_dict in ml_fold_results.items():
        agg_val = aggregate_fold_metrics(
            {fid: d['val'] for fid, d in fold_dict.items()})
        agg_test = aggregate_fold_metrics(
            {fid: d['test'] for fid, d in fold_dict.items()})
        results['ml'][ml_name] = {
            **fold_dict,
            'aggregate': {
                **{f'val_{k}': v for k, v in agg_val.items()},
                **{f'test_{k}': v for k, v in agg_test.items()},
            },
        }
        print(f"  {ml_name:20s}  "
              f"val_rp={agg_val.get('rp_mean', 0):.3f}±{agg_val.get('rp_std', 0):.3f}  "
              f"test_rp={agg_test.get('rp_mean', 0):.3f}±{agg_test.get('rp_std', 0):.3f}")

    # Save results
    result_name = f"ml_ddg{jk_suffix}_{n_folds}fold_results.json"
    result_path = os.path.join(final_save_dir, result_name)
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save splits
    split_name = f"ml_ddg{jk_suffix}_{n_folds}fold_splits.json"
    split_path = os.path.join(final_save_dir, split_name)
    with open(split_path, 'w') as f:
        json.dump(fold_split_log, f, indent=2)

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
        description='ΔΔG ML Fine-tuning (Encoder Embeddings + sklearn)',
    )
    parser.add_argument('-config', type=str, required=True,
                        help='Path to YAML configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['system']['seed'])
    device = torch.device(f"cuda:{config['system']['cuda_id']}"
                          if torch.cuda.is_available() else 'cpu')

    run_ml_ddG_finetuning(config, device)
