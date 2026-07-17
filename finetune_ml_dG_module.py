"""
ML fine-tuning for ΔG regression using pretrained encoder embeddings.

Pipeline:
1. Load pretrained encoder (frozen, eval mode)
2. Precompute node-level embeddings for all graphs (CV + test)
3. Free encoder from GPU memory
4. Mean-pool embeddings per graph: cat([receptor.x, ligand.x], dim=0).mean(0)
5. Train sklearn ML models (RF, GBT, DT, SVR) with 5-fold stratified CV
6. Evaluate each fold on held-out test set
7. Aggregate results (mean ± std) and save JSON

Save directory: same encoder folder or jk mode modified folder.
Result file:   ml_dg[_jk{mode}]_5fold_results.json
"""

import pickle
import os
import json
import random
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from model_collection.FineTuneModels import (
    load_pretrained_encoder, precompute_embeddings,
    get_pool_input_dim,
)
from utils.Training_modules.common_utils import (
    set_seed, load_sttgs_from_dir, get_db_path_finetune,
    filter_samples_finetune, load_ssl_config,
    MODEL_INIT_DIM, EDGE_IN_DIM, METADATA,
)
from utils.Training_modules.ml_utils import (
    extract_features_regression, build_regressors,
    eval_regressor_ml, aggregate_fold_metrics,
)


# ============================================================================
# ΔG ML FINE-TUNING WITH CROSS-VALIDATION + TEST
# ============================================================================
def run_ml_dG_finetuning(
    config: Dict,
    device: torch.device,
    ssl_checkpoint_path: Optional[str] = None,
):
    """
    Run ΔG regression using sklearn ML models on pretrained encoder embeddings.

    Pipeline (mirrors finetune_dG_module.py steps 1-5, replaces step 6 with ML):
    1. Load pretrained encoder, precompute embeddings (with optional JK)
    2. Free encoder from GPU
    3. Mean-pool each graph → numpy feature vector
    4. 5-fold StratifiedKFold CV with held-out test evaluation per fold
    5. Aggregate and save results

    Args:
        config: Full configuration dictionary (uses 'dg_reg' + 'data' sections)
        device: Torch device (needed for encoder forward pass)
        ssl_checkpoint_path: Path to SSL checkpoint (overrides config if provided)
    """
    print(f"\n{'='*80}")
    print("ΔG ML FINE-TUNING (Encoder Embeddings + sklearn, CV + Test)")
    print(f"{'='*80}\n")

    finetune_cfg = config['dg_reg']
    data_cfg = config['data']
    sys_cfg = config['system']
    ml_cfg_all = config.get('ml', {})
    ml_cfg = ml_cfg_all.get('dg_reg', ml_cfg_all)  # per-task or fallback to flat

    # ---------------------------------------------------------------
    # Determine pretrained path
    # ---------------------------------------------------------------
    if ssl_checkpoint_path is not None:
        pretrained_path = ssl_checkpoint_path
    else:
        pretrained_path = finetune_cfg.get('pretrained_path')

    if pretrained_path is None:
        raise ValueError(
            "pretrained_path is required. "
            "Provide via config['dg_reg']['pretrained_path'] or ssl_checkpoint_path."
        )
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

    # Load SSL config if available (overrides architecture params)
    ssl_config = load_ssl_config(pretrained_path)
    if ssl_config is not None:
        print("Loaded SSL config, overriding architecture parameters:")
        print(f"  num_layers: {ssl_config['num_layers']}")
        print(f"  hidden_dim_power: {ssl_config['hidden_dim_power']}")
        print(f"  hgt_heads: {ssl_config['hgt_heads']}")
        print(f"  message_style: {ssl_config.get('message_style', 'gated_src')}")
        print(f"  strategy_type: {ssl_config.get('strategy_type', 'dynamic')}")

        finetune_cfg['num_layers'] = ssl_config['num_layers']
        finetune_cfg['hidden_dim_power'] = ssl_config['hidden_dim_power']
        finetune_cfg['hgt_heads'] = ssl_config['hgt_heads']
        finetune_cfg['message_style'] = ssl_config.get('message_style', 'gated_src')
        finetune_cfg['strategy_type'] = ssl_config.get('strategy_type', 'dynamic')

    # ---------------------------------------------------------------
    # Setup directories
    # ---------------------------------------------------------------
    embedding_type = data_cfg['embedding_type']
    local_dir = (
        f"{finetune_cfg['num_layers']}layers_{finetune_cfg['hidden_dim_power']}hdim"
        + (f'_{embedding_type}' if embedding_type in ['esm', 'esm480'] else '')
        + (f"_{finetune_cfg['strategy_type']}" if finetune_cfg.get('strategy_type', 'dynamic') != 'dynamic' else '')
        + (f"_additive" if finetune_cfg.get('message_style', 'gated_src') == 'additive' else '')
    )
    # JK-Net config
    use_jk = finetune_cfg.get('use_jk', False)
    jk_mode = finetune_cfg.get('jk_mode', 'mean')
    jk_suffix = f"_jk{jk_mode}" if use_jk else ""
    local_dir += jk_suffix
    final_save_dir = os.path.join(finetune_cfg['save_dir'], local_dir)

    os.makedirs(final_save_dir, exist_ok=True)

    # JK-Net config (optional, default off)
    use_jk = finetune_cfg.get('use_jk', False)
    jk_mode = finetune_cfg.get('jk_mode', 'mean')
    jk_suffix = f"_jk{jk_mode}" if use_jk else ""

    use_amp = device.type == 'cuda'
    hidden_dim = 2 ** finetune_cfg['hidden_dim_power']

    print(f"Device: {device}")
    print(f"Task: ML ΔG regression")
    print(f"Embedding Type: {embedding_type}")
    print(f"JK-Net: {'enabled (mode=' + jk_mode + ')' if use_jk else 'disabled'}")
    print(f"Pre-trained path: {pretrained_path}")
    print(f"Save directory: {final_save_dir}")

    # ---------------------------------------------------------------
    # Load CV graph data
    # ---------------------------------------------------------------
    pdb_dir = get_db_path_finetune(
        data_cfg['pdb_root'], data_cfg['dist'], embedding_type,
        if_mut=False, if_only_reg=True,
    )
    print(f"Loading CV data from: {pdb_dir}")
    all_sttgs = load_sttgs_from_dir(pdb_dir)
    all_sttgs = filter_samples_finetune(all_sttgs, min_nodes=data_cfg.get('min_nodes', 5))

    # ---------------------------------------------------------------
    # Load test set
    # ---------------------------------------------------------------
    if embedding_type in ['base', '+esm', '+esm480', 'only_esm', 'only_esm480']:
        test_sttgs = load_sttgs_from_dir(f'{data_cfg["pdb_root"]}/testset')
    elif embedding_type in ['esm', 'esm480']:
        test_sttgs = load_sttgs_from_dir(f'{data_cfg["pdb_root"]}/testset_{embedding_type}')
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    print(f"Loaded {len(test_sttgs)} test samples")

    # ---------------------------------------------------------------
    # Load ESM dict if needed
    # ---------------------------------------------------------------
    esm_dict = None
    if embedding_type in ['+esm', '+esm480', 'only_esm', 'only_esm480']:
        prefix = 'unmut'
        esm_suffix = embedding_type[1:] if embedding_type.startswith('+') else 'esm'
        dict_path = f'{data_cfg["pdb_root"]}/{prefix}_seq_dicts_{esm_suffix}_{data_cfg["dist"]}A_hg.pkl'
        print(f"Loading ESM embeddings from: {dict_path}")
        with open(dict_path, 'rb') as f:
            esm_dict = pickle.load(f)
        print(f"  Loaded ESM dicts for {len(esm_dict)} samples")

    # ---------------------------------------------------------------
    # Convert settings to samples
    # ---------------------------------------------------------------
    all_samples = []
    for name, st in all_sttgs.items():
        graph = st.protein_graph
        affinity = graph.y.item() if hasattr(graph.y, 'item') else graph.y
        all_samples.append((name, graph, affinity, 'premium'))
    random.shuffle(all_samples)

    test_samples = []
    for name, st in test_sttgs.items():
        graph = st.protein_graph
        affinity = graph.y.item() if hasattr(graph.y, 'item') else graph.y
        test_samples.append((name, graph, affinity, 'premium'))

    print(f"CV samples: {len(all_samples)}, Test samples: {len(test_samples)}")

    if not all_samples:
        raise ValueError("No CV samples loaded.")
    if not test_samples:
        raise ValueError("No test samples loaded.")

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
        num_hgt_layers=finetune_cfg['num_layers'],
        hgt_heads=finetune_cfg['hgt_heads'],
        device=device,
        message_style=finetune_cfg.get('message_style', 'gated_src'),
    )

    all_names = [s[0] for s in all_samples]
    all_graphs = [s[1] for s in all_samples]
    test_names = [s[0] for s in test_samples]
    test_graphs = [s[1] for s in test_samples]

    encoder.eval()
    print("\n--- Precomputing CV embeddings ---")
    precompute_embeddings(
        encoder, all_names, all_graphs, device,
        batch_size=finetune_cfg['batch_size'],
        embedding_type=embedding_type, esm_dict=esm_dict,
        use_amp=use_amp, use_jk=use_jk, jk_mode=jk_mode,
    )
    print("--- Precomputing test embeddings ---")
    precompute_embeddings(
        encoder, test_names, test_graphs, device,
        batch_size=finetune_cfg['batch_size'],
        embedding_type=embedding_type, esm_dict=esm_dict,
        use_amp=use_amp, use_jk=use_jk, jk_mode=jk_mode,
    )

    # Free encoder and ESM dict
    del encoder
    if esm_dict is not None:
        del esm_dict
    torch.cuda.empty_cache()
    print("Encoder freed from memory.\n")

    # ---------------------------------------------------------------
    # Mean-pool embeddings → numpy features
    # ---------------------------------------------------------------
    test_X, test_y = extract_features_regression(test_samples)
    print(f"Feature dimension: {test_X.shape[1]}")

    # ---------------------------------------------------------------
    # 5-fold Stratified CV
    # ---------------------------------------------------------------
    labels = [1 if s[2] > 0 else 0 for s in all_samples]
    n_splits = finetune_cfg.get('n_folds', 5)
    seed = sys_cfg['seed']
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    ml_fold_results = defaultdict(dict)  # {model_name: {fold_id: {val: …, test: …}}}
    fold_split_log = {}

    print(f"\n{'='*60}")
    print(f"Starting {n_splits}-Fold CV — ML ΔG Regression")
    print(f"{'='*60}\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, labels)):
        fold_id = f'fold_{fold + 1}'
        print(f"\n--- {fold_id}/{n_splits} ---")

        train_fold = [all_samples[i] for i in train_idx]
        val_fold = [all_samples[i] for i in val_idx]

        fold_split_log[fold_id] = {
            'train': [[s[0], float(s[2]), s[3]] for s in train_fold],
            'val': [[s[0], float(s[2]), s[3]] for s in val_fold],
        }

        train_X, train_y = extract_features_regression(train_fold)
        val_X, val_y = extract_features_regression(val_fold)

        print(f"  Train: {len(train_fold)}, Val: {len(val_fold)}, Test: {len(test_samples)}")
        print(f"  ML Regressors:")

        for ml_name, reg in build_regressors(ml_cfg, seed).items():
            save_path = os.path.join(final_save_dir, f'ml_dg{jk_suffix}_{fold_id}_{ml_name}.pkl')
            r = eval_regressor_ml(
                ml_name, reg, train_X, train_y, val_X, val_y, test_X, test_y,
                save_path=save_path,
            )
            ml_fold_results[ml_name][fold_id] = {
                'val': {k: float(v) for k, v in r['val'].items()},
                'test': {k: float(v) for k, v in r['test'].items()},
            }

    # ---------------------------------------------------------------
    # Aggregate and save results
    # ---------------------------------------------------------------
    results = {
        'task': 'ml_dG',
        'embedding_type': embedding_type,
        'num_layers': finetune_cfg['num_layers'],
        'hidden_dim_power': finetune_cfg['hidden_dim_power'],
        'use_jk': use_jk,
        'jk_mode': jk_mode if use_jk else None,
        'feature_dim': int(test_X.shape[1]),
        'n_folds': n_splits,
        'ml': {},
    }

    print(f"\n{'='*60}")
    print(f"ML ΔG {n_splits}-Fold CV Results")
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

    # Save results JSON (jk mode in filename, not in folder)
    result_name = f"ml_dg{jk_suffix}_{n_splits}fold_results.json"
    result_path = os.path.join(final_save_dir, result_name)
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save fold splits
    split_name = f"ml_dg{jk_suffix}_{n_splits}fold_splits.json"
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
        description='ΔG ML Fine-tuning (Encoder Embeddings + sklearn)',
    )
    parser.add_argument('-config', type=str, required=True,
                        help='Path to YAML configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['system']['seed'])
    device = torch.device(f"cuda:{config['system']['cuda_id']}"
                          if torch.cuda.is_available() else 'cpu')

    run_ml_dG_finetuning(config, device)
