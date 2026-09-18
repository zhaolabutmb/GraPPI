import pickle
import os
import json
import random
import time
import math
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
from torch.amp import GradScaler
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

from model_collection.FineTuneModels import (
    load_pretrained_encoder, precompute_embeddings,
    get_pool_input_dim, create_poolhead_model, build_embedder,
)
from utils.Training_modules.save_training import FineTuneEarlyStopping
from utils.Quantity_compute.Loss_fun import get_loss
from utils.Training_modules.common_utils import (
    set_seed, load_sttgs_from_dir, get_db_path_finetune,
    filter_samples_finetune, load_ssl_config,
    MODEL_INIT_DIM, EDGE_IN_DIM, METADATA
)
from utils.Training_modules.finetune_training import (
    train_epoch_regressor, validate_regressor,
)


# ============================================================================
# ΔG FINE-TUNING WITH CROSS-VALIDATION + TEST (precomputed-embedding pipeline)
# ============================================================================
def run_dG_finetuning(config: Dict, device: torch.device, ssl_checkpoint_path: Optional[str] = None):
    """
    Run ΔG regression fine-tuning with cross-validation on training data
    and held-out test evaluation per fold.

    Pipeline:
    1. Load pretrained encoder (frozen, eval mode)
    2. Precompute node-level embeddings for all graphs (CV + test)
    3. For +esm/+esm480: concatenate ESM embeddings at node level
       For only_esm: replace encoder output with ESM embeddings
    4. Store enriched embeddings in graph.x (replaces original features)
    5. Free encoder from memory
    6. For each CV fold:
       a. Create pool+head model and train on fold train split
       b. Validate on fold val split (early stopping on val Pearson r)
       c. Evaluate best model on held-out test set
    7. Report per-fold and aggregate test Pearson R

    Args:
        config: Full configuration dictionary
        device: Torch device
        ssl_checkpoint_path: Path to SSL checkpoint (overrides config if provided)
    """
    print(f"\n{'='*80}")
    print("ΔG FINE-TUNING (Cross-Validation + Test, Precomputed Embeddings)")
    print(f"{'='*80}\n")

    finetune_cfg = config['dg_reg']
    data_cfg = config['data']
    sys_cfg = config['system']

    # ---------------------------------------------------------------
    # Determine pretrained path (required for this pipeline)
    # ---------------------------------------------------------------
    if ssl_checkpoint_path is not None:
        pretrained_path = ssl_checkpoint_path
    else:
        pretrained_path = finetune_cfg.get('pretrained_path')

    if pretrained_path is None:
        raise ValueError(
            "pretrained_path is required for the precomputed-embedding pipeline. "
            "Provide via config['dg_reg']['pretrained_path'] or ssl_checkpoint_path argument."
        )
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

    # Load SSL config if available
    ssl_config = load_ssl_config(pretrained_path)
    if ssl_config is not None:
        print("Loaded SSL config, overriding architecture parameters:")
        print(f"  SSL embedding_type: {ssl_config['embedding_type']}")
        print(f"  finetune embedding_type: {data_cfg['embedding_type']}")
        print(f"  num_layers: {ssl_config['num_layers']}")
        print(f"  hidden_dim_power: {ssl_config['hidden_dim_power']}")
        print(f"  hgt_heads: {ssl_config['hgt_heads']}")
        print(f"  dropout: {ssl_config['dropout']}")
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
    encoder_type = finetune_cfg.get('encoder_type', 'ssl')
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

    use_amp = device.type == 'cuda'
    hidden_dim = 2 ** finetune_cfg['hidden_dim_power']

    print(f"\nDevice: {device}")
    print(f"Fine-tune Task: regressor (ΔG prediction)")
    print(f"Embedding Type: {embedding_type}")
    print(f"Pool Mode: {finetune_cfg.get('pool_mode', 'cross_attn')}")
    print(f"Strategy Type: {finetune_cfg.get('strategy_type', 'dynamic')}")
    print(f"Pre-trained path: {pretrained_path}")
    print(f"Save directory: {final_save_dir}")

    # ---------------------------------------------------------------
    # Load CV graph data (training/val pool)
    # ---------------------------------------------------------------
    pdb_dir = get_db_path_finetune(
        data_cfg['pdb_root'], data_cfg['dist'], embedding_type,
        if_mut=False,
        if_only_reg=True,
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
    # Load ESM dict if needed (for +esm / +esm480 / only_esm / only_esm480)
    # ---------------------------------------------------------------
    esm_dict = None
    if embedding_type in ['+esm', '+esm480', 'only_esm', 'only_esm480']:
        prefix = 'unmut'
        esm_suffix = embedding_type[1:] if embedding_type.startswith('+') else 'esm'  # 'esm' or 'esm480'
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
        sample_type = 'premium'

        all_samples.append((name, graph, affinity, sample_type))

    random.shuffle(all_samples)

    # Count sample types
    type_counts = defaultdict(int)
    for s in all_samples:
        type_counts[s[3]] += 1

    print(f"\nCV samples: {len(all_samples)}")
    print(f"Premium: {type_counts['premium']}, Golden: {type_counts['golden']}, Negative: {type_counts['negative']}")

    # Convert test settings to samples
    test_samples = []
    for name, st in test_sttgs.items():
        graph = st.protein_graph
        affinity = graph.y.item() if hasattr(graph.y, 'item') else graph.y
        sample_type = 'premium'  # Test samples are all premium by definition
        test_samples.append((name, graph, affinity, sample_type))

    print(f"Test samples: {len(test_samples)}")

    if len(all_samples) == 0:
        raise ValueError("No CV samples loaded.")
    if len(test_samples) == 0:
        raise ValueError("No test samples loaded.")

    # ---------------------------------------------------------------
    # Load encoder (GraPPI SSL or sequence student) & precompute embeddings
    # ---------------------------------------------------------------
    node_in_dim = MODEL_INIT_DIM[embedding_type]

    precompute_fn, hidden_dim, use_jk = build_embedder(
        pretrained_path, device,
        encoder_type=encoder_type,
        embedding_type=embedding_type,
        node_in_dim=node_in_dim,
        edge_in_dim=EDGE_IN_DIM,
        metadata=METADATA,
        hidden_dim=hidden_dim,
        num_hgt_layers=finetune_cfg['num_layers'],
        hgt_heads=finetune_cfg['hgt_heads'],
        message_style=finetune_cfg.get('message_style', 'gated_src'),
        use_jk=use_jk,
        jk_mode=jk_mode,
        batch_size=finetune_cfg['batch_size'],
        use_amp=use_amp,
    )

    # Precompute embeddings for CV samples
    all_names = [s[0] for s in all_samples]
    all_graphs = [s[1] for s in all_samples]
    print("\n--- Precomputing CV embeddings ---")
    precompute_fn(all_names, all_graphs, esm_dict=esm_dict)

    # Precompute embeddings for test samples
    test_names = [s[0] for s in test_samples]
    test_graphs = [s[1] for s in test_samples]
    print("--- Precomputing test embeddings ---")
    precompute_fn(test_names, test_graphs, esm_dict=esm_dict)

    # Free ESM dict from memory
    if esm_dict is not None:
        del esm_dict
    torch.cuda.empty_cache()
    print("Embeddings precomputed. Training with precomputed embeddings.\n")

    # ---------------------------------------------------------------
    # Create test loader (shared across folds)
    # ---------------------------------------------------------------
    test_graphs_list = [s[1] for s in test_samples]
    test_loader = DataLoader(test_graphs_list, batch_size=finetune_cfg['batch_size'], shuffle=False)

    # ---------------------------------------------------------------
    # Stratified labels for CV
    # ---------------------------------------------------------------
    labels = [1 if s[2] > 0 else 0 for s in all_samples]

    # CV setup
    n_splits = finetune_cfg.get('n_folds', 5)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=sys_cfg['seed'])

    pool_input_dim = get_pool_input_dim(
        hidden_dim, embedding_type,
        use_jk=use_jk, jk_mode=jk_mode,
        num_hgt_layers=finetune_cfg['num_layers'],
    )

    # Storage for all-folds results
    all_fold_logs = {}
    fold_split_log = {}
    fold_test_rp = {}
    fold_test_mae = {}

    print(f"\n{'='*60}")
    print(f"Starting {n_splits}-Fold CV for ΔG Regression + Test Evaluation")
    print(f"{'='*60}\n")

    # ---------------------------------------------------------------
    # Run cross-validation
    # ---------------------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_samples, labels)):
        fold_id = fold + 1
        print(f"\n{'='*60}")
        print(f"FOLD {fold_id}/{n_splits}")
        print(f"{'='*60}")

        # Build datasets
        train_fold_samples = [all_samples[i] for i in train_idx]
        val_fold_samples = [all_samples[i] for i in val_idx]

        # Record fold splits (before upsampling, so each sample appears exactly once)
        fold_split_log[f'fold_{fold_id}'] = {
            'train': [[s[0], float(s[2]), s[3]] for s in train_fold_samples],
            'val':   [[s[0], float(s[2]), s[3]] for s in val_fold_samples],
        }

        train_graphs = [s[1] for s in train_fold_samples]
        val_graphs = [s[1] for s in val_fold_samples]

        train_loader = DataLoader(train_graphs, batch_size=finetune_cfg['batch_size'], shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=finetune_cfg['batch_size'], shuffle=False)

        print(f"  Train: {len(train_fold_samples)}, Val: {len(val_fold_samples)}, Test: {len(test_samples)}")

        # ---------------------------------------------------------------
        # Create pool+head model (no encoder, fresh per fold)
        # ---------------------------------------------------------------
        # Dynamic ft_hdim: if not set in config, use pool_input_dim // 8 (min 256)
        ft_hdim = finetune_cfg.get('ft_hdim', None)
        if ft_hdim is None:
            ft_hdim = max(256, pool_input_dim // 8)
        ft_hdim = min(ft_hdim, pool_input_dim // 2) # e.g. ft_hdim = 512, while pool_input_dim or pool_input_dim//8 = 1024
        model = create_poolhead_model(
            model_type='regressor',
            pool_input_dim=pool_input_dim,
            hidden_dim=ft_hdim,
            metadata=METADATA,
            dropout=finetune_cfg['dropout'],
            pool_mode=finetune_cfg.get('pool_mode', 'cross_attn'),
            cross_attn_queries=finetune_cfg.get('cross_attn_queries', 2),
            cross_attn_heads=finetune_cfg.get('cross_attn_heads', 4),
            regressor_layers=finetune_cfg.get('regressor_layers', 2),
            use_reg_adaptor=finetune_cfg.get('use_reg_adaptor', True),
            reg_adaptor_layers=finetune_cfg.get('reg_adaptor_layers', 2),
        ).to(device)

        # Print model info (first fold only)
        if fold == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Pool+Head model parameters: {total_params:,} total, {trainable_params:,} trainable")
            print(f"Pool input dim: {pool_input_dim} (ft_hdim={ft_hdim})")
            if use_jk:
                print(f"JK-Net enabled: mode={jk_mode}")

        # ---------------------------------------------------------------
        # Optimizer
        # ---------------------------------------------------------------
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=finetune_cfg['lr'],
            weight_decay=finetune_cfg['weight_decay'],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=finetune_cfg.get('scheduler_t0', 10),
            T_mult=finetune_cfg.get('scheduler_t_mult', 2),
            eta_min=float(finetune_cfg.get('scheduler_eta_min', 1e-6))
        )

        scaler = GradScaler(enabled=use_amp)

        # ---------------------------------------------------------------
        # Loss criteria
        # ---------------------------------------------------------------
        criterion_reg = get_loss(
            finetune_cfg.get('loss_fun', 'l1'),
            gamma=finetune_cfg.get('gamma', 1.0),
            delta=finetune_cfg.get('delta', 1.0),
            reduction='mean'
        )
        criterion_reg_val = get_loss('l1', reduction='mean')

        # ---------------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------------
        checkpoint_name = (
            f"wt_dg"
            + (f"_{embedding_type}" if embedding_type not in ['base', 'esm'] else '')
            + ('' if finetune_cfg.get('pool_mode', 'cross_attn') == 'cross_attn' else f"_{finetune_cfg['pool_mode']}")
            + jk_suffix
            + f"_fold{fold_id}_best.pt"
        )
        checkpoint_path = os.path.join(final_save_dir, checkpoint_name)

        early_stopping = FineTuneEarlyStopping(
            patience=finetune_cfg['patience'],
            save_path=checkpoint_path,
            mode='max',  # maximize Pearson r
        )

        # Dynamic clipping state (reset per fold)
        clip_min: float = 5.0
        clip_factor: float = 1.25
        current_clip: float = finetune_cfg.get('clip_max_norm', 5.0)

        # Fold log
        fold_log = {
            'train_loss': [], 'val_loss': [],
            'train_rp': [], 'val_rp': [],
            'grad_norm': [], 'lr': [],
        }

        reg_metric_name = finetune_cfg.get('reg_metric', 'rp')

        # ---------------------------------------------------------------
        # Training loop
        # ---------------------------------------------------------------
        for epoch in range(finetune_cfg['n_epochs']):
            start_time = time.time()
            model.update_epoch(epoch)

            # Train
            train_loss, grad_norm, train_metrics = train_epoch_regressor(
                model, train_loader, optimizer, scaler, criterion_reg, device,
                clip_max_norm=current_clip, use_amp=use_amp,
            )
            # Validate
            val_loss, val_metrics = validate_regressor(
                model, val_loader, criterion_reg_val, device, use_amp=use_amp,
            )

            score = val_metrics[reg_metric_name]

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Update dynamic clip for next epoch
            if grad_norm is not None and math.isfinite(grad_norm) and grad_norm > 0:
                current_clip = max(clip_min, grad_norm * clip_factor)

            # Log
            fold_log['train_loss'].append(float(train_loss))
            fold_log['val_loss'].append(float(val_loss))
            fold_log['train_rp'].append(float(train_metrics['rp']))
            fold_log['val_rp'].append(float(val_metrics['rp']))
            fold_log['grad_norm'].append(float(grad_norm) if grad_norm is not None else 0.0)
            fold_log['lr'].append(float(current_lr))

            epoch_time = time.time() - start_time

            print(f"  Epoch {epoch+1:3d} | "
                  f"Loss T/V: {train_loss:.3f}/{val_loss:.3f} | "
                  f"rp T/V: {train_metrics['rp']:.3f}/{val_metrics['rp']:.3f} | "
                  f"Time: {epoch_time:.1f}s")

            # Early stopping on val Pearson r
            stop = early_stopping(score, model, epoch)
            if stop:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(best rp={early_stopping.best_score:.3f} "
                      f"at epoch {early_stopping.best_epoch+1})")
                break

        # --- Restore best model and evaluate on test set ---
        best_state = early_stopping.get_best_state()
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), checkpoint_path)

        test_loss, test_metrics = validate_regressor(
            model, test_loader, criterion_reg_val, device, use_amp=use_amp,
        )

        fold_test_rp[f'fold_{fold_id}'] = float(test_metrics['rp'])
        fold_test_mae[f'fold_{fold_id}'] = float(test_metrics['mae'])
        fold_log['test_rp'] = float(test_metrics['rp'])
        fold_log['test_mae'] = float(test_metrics['mae'])
        fold_log['test_loss'] = float(test_loss)
        fold_log['best_val_rp'] = float(early_stopping.best_score)
        fold_log['best_epoch'] = int(early_stopping.best_epoch + 1)

        all_fold_logs[f'fold_{fold_id}'] = fold_log

        print(f"  Fold {fold_id} — Best val rp: {early_stopping.best_score:.3f}, "
              f"Test rp: {test_metrics['rp']:.3f}, Test MAE: {test_metrics['mae']:.3f}")

        # Clean up fold model
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Aggregate and save results
    # ---------------------------------------------------------------
    test_rps = [v for v in fold_test_rp.values()]
    mean_test_rp = float(np.mean(test_rps))
    std_test_rp = float(np.std(test_rps))
    test_maes = [v for v in fold_test_mae.values()]
    mean_test_mae = float(np.mean(test_maes))
    std_test_mae = float(np.std(test_maes))

    print(f"\n{'='*60}")
    print(f"ΔG {n_splits}-Fold CV + Test Complete")
    print(f"{'='*60}")
    print(f"  Test Pearson r per fold: {[f'{v:.3f}' for v in test_rps]}")
    print(f"  Mean ± Std: {mean_test_rp:.3f} ± {std_test_rp:.3f}")

    # Save all-folds training log
    log_name = (
        f"finetune_regressor"
        + (f"_{embedding_type}" if embedding_type not in ['base', 'esm'] else '')
        + ('' if finetune_cfg.get('pool_mode', 'cross_attn') == 'cross_attn' else f"_{finetune_cfg['pool_mode']}")
        + jk_suffix
        + f"_{n_splits}fold_log.json"
    )
    log_path = os.path.join(final_save_dir, log_name)
    with open(log_path, 'w') as f:
        json.dump(all_fold_logs, f, indent=2)

    # Save fold split info
    split_name = (
        f"wt_dg"
        + (f"_{embedding_type}" if embedding_type not in ['base', 'esm'] else '')
        + ('' if finetune_cfg.get('pool_mode', 'cross_attn') == 'cross_attn' else f"_{finetune_cfg['pool_mode']}")
        + jk_suffix
        + f"_{n_splits}fold_splits.json"
    )
    split_path = os.path.join(final_save_dir, split_name)
    with open(split_path, 'w') as f:
        json.dump(fold_split_log, f, indent=2)

    # Save summary
    summary = {
        'n_folds': n_splits,
        'embedding_type': embedding_type,
        'num_layers': finetune_cfg['num_layers'],
        'hidden_dim_power': finetune_cfg['hidden_dim_power'],
        'pool_mode': finetune_cfg.get('pool_mode', 'cross_attn'),
        'use_jk': use_jk,
        'jk_mode': jk_mode if use_jk else None,
        'fold_test_rp': fold_test_rp,
        'mean_test_rp': mean_test_rp,
        'std_test_rp': std_test_rp,
        'fold_test_mae': fold_test_mae,
        'mean_test_mae': mean_test_mae,
        'std_test_mae': std_test_mae,
    }
    summary_name = (
        f"wt_dg"
        + (f"_{embedding_type}" if embedding_type not in ['base', 'esm'] else '')
        + ('' if finetune_cfg.get('pool_mode', 'cross_attn') == 'cross_attn' else f"_{finetune_cfg['pool_mode']}")
        + jk_suffix
        + f"_{n_splits}fold_summary.json"
    )
    summary_path = os.path.join(final_save_dir, summary_name)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Training logs: {log_path}")
    print(f"  Fold splits:   {split_path}")
    print(f"  Summary:        {summary_path}")
    print(f"  Checkpoints:    {final_save_dir}/wt_dg_*_fold*_best.pt")
    print(f"{'='*60}\n")

    return summary


# ============================================================================
# MAIN FOR STANDALONE EXECUTION
# ============================================================================
if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description='ΔG Regression Fine-tuning with N-fold CV + Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Example:
    python finetune_dG_module.py -config config_finetune.yaml
        """
    )
    parser.add_argument('-config', type=str, required=True,
                       help='Path to YAML configuration file')

    args = parser.parse_args()

    # Load config
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config['system']['seed'])

    # Setup device
    device = torch.device(f"cuda:{config['system']['cuda_id']}"
                         if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*80}")
    print("ΔG REGRESSION FINE-TUNING PIPELINE")
    print(f"{'='*80}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Seed: {config['system']['seed']}")

    # Run dG fine-tuning
    run_dG_finetuning(config, device)

    print(f"\n{'='*80}")
    print("ALL TRAINING COMPLETE!")
    print(f"{'='*80}\n")
