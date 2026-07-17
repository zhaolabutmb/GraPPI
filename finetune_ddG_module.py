import os
import pickle
import json
import time
import math
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch_geometric.loader import DataLoader

from model_collection.FineTuneModels import (
    load_pretrained_encoder, precompute_embeddings,
    get_pool_input_dim, create_poolhead_model,
)
from utils.Training_modules.save_training import FineTuneEarlyStopping
from utils.Quantity_compute.Loss_fun import get_loss

# Import shared utilities
from utils.Training_modules.common_utils import (
    set_seed, load_sttgs_from_dir, load_ssl_config,
    filter_samples_finetune, get_fold_splits,
    MODEL_INIT_DIM, EDGE_IN_DIM, METADATA
)
from utils.Training_modules.finetune_training import (
    train_epoch_ddg, validate_ddg,
)

# ============================================================================
# MAIN ddG FINE-TUNING FUNCTION
# ============================================================================
def run_ddg_finetuning(config: Dict, device: torch.device, ssl_checkpoint_path: Optional[str] = None):
    """
    Run ddG fine-tuning phase.

    This function:
    - Loads pretrained SSL encoder
    - Trains on premium, golden, and negative samples (wild-type)
    - Tests on mutation samples
    - Excludes wild-type PDBs that appear in mutant test set from training
    - No cross-validation (single train/test split)
    
    Args:
        config: Configuration dictionary with 'mutation' section
        device: Torch device
    """
    print(f"\n{'='*80}")
    print("MUTATION FINE-TUNING")
    print(f"{'='*80}\n")
    
    mut_cfg = config['mutation']
    data_cfg = config['data']
    #sys_cfg = config['system']
    
    # Determine pretrained path
    if ssl_checkpoint_path is not None:
        pretrained_path = ssl_checkpoint_path
    else:
        pretrained_path = mut_cfg.get('pretrained_path') 

    # Load SSL config if available
    ssl_config = load_ssl_config(pretrained_path)
    if ssl_config is not None:
        print("Loaded SSL config, overriding architecture parameters:")
        print(f"  SSL embedding_type: {ssl_config['embedding_type']}")
        print(f"  test embedding_type: {data_cfg['embedding_type']}")
        print(f"  num_layers: {ssl_config['num_layers']}")
        print(f"  hidden_dim_power: {ssl_config['hidden_dim_power']}")
        print(f"  hgt_heads: {ssl_config['hgt_heads']}")
        print(f"  dropout: {ssl_config['dropout']}")
        print(f"  message_style: {ssl_config.get('message_style', 'gated_src')}")
        print(f"  strategy_type: {ssl_config.get('strategy_type', 'dynamic')}")
        
        mut_cfg['num_layers'] = ssl_config['num_layers']
        mut_cfg['hidden_dim_power'] = ssl_config['hidden_dim_power']
        mut_cfg['hgt_heads'] = ssl_config['hgt_heads']
        mut_cfg['message_style'] = ssl_config.get('message_style', 'gated_src')
        mut_cfg['strategy_type'] = ssl_config.get('strategy_type', 'dynamic')
    
    # Setup directories
    #load embedding type from ssl config if available, otherwise use data config
    if ssl_config['embedding_type'] != data_cfg['embedding_type']:
        embedding_type = ssl_config['embedding_type']
    else:
        embedding_type = data_cfg['embedding_type']
    local_dir = (
        f"{mut_cfg['num_layers']}layers_{mut_cfg['hidden_dim_power']}hdim"
        + (f'_{embedding_type}' if embedding_type in ['esm', 'esm480'] else '')
        + (f"_{mut_cfg['strategy_type']}" if mut_cfg['strategy_type'] != 'dynamic' else '')
        + (f"_additive" if mut_cfg.get('message_style', 'gated_src') == 'additive' else '')
    )
    # JK-Net config
    use_jk = mut_cfg.get('use_jk', False)
    jk_mode = mut_cfg.get('jk_mode', 'mean')
    jk_suffix = f"_jk{jk_mode}" if use_jk else ""
    local_dir += jk_suffix
    final_save_dir = os.path.join(mut_cfg['save_dir'], local_dir)
    os.makedirs(final_save_dir, exist_ok=True)
    
    use_amp = device.type == 'cuda'
    
    print(f"\nDevice: {device}")
    print(f"Task: regression model (ΔΔG prediction)")
    print(f"Pre-trained path: {pretrained_path}")
    print(f"Save directory: {final_save_dir}")
    
    # ---------------------------------------------------------------
    # Load ΔΔG data: mutant graphs + paired wild-type graphs
    # ΔΔG = mutant_affinity − wild-type_affinity
    # ---------------------------------------------------------------
    dist = data_cfg['dist']
    
    # --- Load mutant training graphs ---
    mut_train_dir = f'{data_cfg["pdb_root"]}/mutant_sthg{"" if embedding_type == "base" else "_"+embedding_type}_{dist}A'
    print(f"\nLoading mutant training graphs from: {mut_train_dir}")
    mut_sttgs_train = load_sttgs_from_dir(mut_train_dir)
    print(f"  Loaded {len(mut_sttgs_train)} mutant training graphs")
    
    # --- Load wild-type training graphs (only PDB IDs present in mutant set) ---
    wt_train_dir = f'{data_cfg["pdb_root"]}/unmut_sthg{"" if embedding_type == "base" else "_"+embedding_type}_{dist}A'
    mut_pdb_ids_train = set(name.split('_')[0] for name in mut_sttgs_train.keys())
    print(f"Loading wild-type training graphs from: {wt_train_dir}")
    print(f"  (selecting {len(mut_pdb_ids_train)} unique PDB IDs from mutant set)")
    wt_sttgs_train = {}
    for pdb_id in mut_pdb_ids_train:
        pkl_path = os.path.join(wt_train_dir, f'{pdb_id}.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as fh:
                wt_sttgs_train[pdb_id] = pickle.load(fh)
    print(f"  Loaded {len(wt_sttgs_train)} wild-type training graphs")
    
    # --- Load test graphs (mutant + wild-type coexist in the same folder) ---
    if embedding_type in ['esm', 'esm480']:
        test_dir = f'{data_cfg["pdb_root"]}/testset_mut_{embedding_type}'
    else:
        test_dir = f'{data_cfg["pdb_root"]}/testset_mut'
    print(f"\nLoading test graphs from: {test_dir}")
    all_sttgs_test = load_sttgs_from_dir(test_dir)
    print(f"  Loaded {len(all_sttgs_test)} test graphs total")
    
    # Separate test set into mutant vs wild-type graphs.
    # Mutant names follow the pattern: xxxx_nnn  (pdb_id + '_' + mutation_index)
    # Wild-type names are just the pdb_id (no underscore suffix)
    wt_sttgs_test = {}
    mut_sttgs_test = {}
    for name, sttg in all_sttgs_test.items():
        parts = name.split('_')
        if len(parts) > 1:
            mut_sttgs_test[name] = sttg
        else:
            wt_sttgs_test[name] = sttg
    print(f"  Test set split: {len(mut_sttgs_test)} mutants, {len(wt_sttgs_test)} wild-types")
    
    # --- Exclude training mutants whose PDB IDs overlap with test set ---
    test_pdb_ids = set(name.split('_')[0] for name in mut_sttgs_test.keys())
    n_before = len(mut_sttgs_train)
    mut_sttgs_train = {
        k: v for k, v in mut_sttgs_train.items()
        if k.split('_')[0] not in test_pdb_ids
    }
    n_excluded = n_before - len(mut_sttgs_train)
    if n_excluded > 0:
        print(f"\nExcluded {n_excluded} training mutants with PDB IDs overlapping test set")
    print(f"Remaining training mutants: {len(mut_sttgs_train)}")
    
    # --- Filter by minimum node/edge counts ---
    mut_sttgs_train = filter_samples_finetune(mut_sttgs_train, min_nodes=data_cfg.get('min_nodes', 5))
    wt_sttgs_train = filter_samples_finetune(wt_sttgs_train, min_nodes=data_cfg.get('min_nodes', 5))
    mut_sttgs_test = filter_samples_finetune(mut_sttgs_test, min_nodes=data_cfg.get('min_nodes', 5))
    wt_sttgs_test = filter_samples_finetune(wt_sttgs_test, min_nodes=data_cfg.get('min_nodes', 5))
    print(f"After filtering — Train mutants: {len(mut_sttgs_train)}, "
          f"Train WT: {len(wt_sttgs_train)}, "
          f"Test mutants: {len(mut_sttgs_test)}, "
          f"Test WT: {len(wt_sttgs_test)}")
    
    # ---------------------------------------------------------------
    # Pair mutant ↔ wild-type
    # ---------------------------------------------------------------
    def pair_mutant_wildtype(mut_sttgs, wt_sttgs, label=""):
        """
        Pair each mutant graph with its corresponding wild-type graph.
        
        For a mutant named 'xxxx_nnn', the wild-type is 'xxxx' in wt_sttgs.
        graph.y already contains the original binding affinity (mut_aff / wt_aff).
        
        Returns
        -------
        paired : list of (mut_name, mut_graph, wt_graph, mut_aff)
        unmatched : list of mutant names with no wild-type match
        """
        paired = []
        unmatched = []
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
            print(f"  [{label}] {len(unmatched)} mutants have no matching wild-type:")
            for name in unmatched[:5]:
                print(f"    - {name}  (looked for WT '{name.split('_')[0]}')")
            if len(unmatched) > 5:
                print(f"    ... and {len(unmatched) - 5} more")
        
        return paired, unmatched
    
    print("\n--- Pairing training mutant ↔ wild-type ---")
    train_pairs, train_unmatched = pair_mutant_wildtype(
        mut_sttgs_train, wt_sttgs_train, label="Train")
    print(f"  Matched training pairs: {len(train_pairs)}")
    
    print("--- Pairing test mutant ↔ wild-type ---")
    test_pairs, test_unmatched = pair_mutant_wildtype(
        mut_sttgs_test, wt_sttgs_test, label="Test")
    print(f"  Matched test pairs: {len(test_pairs)}")
    
    # --- Print mut_aff distribution ---
    train_mut_affs = [p[3] for p in train_pairs]
    test_mut_affs = [p[3] for p in test_pairs]
    
    if train_mut_affs:
        print(f"\nMut Affinity Statistics (Training, n={len(train_mut_affs)}):")
        print(f"  Mean={np.mean(train_mut_affs):.3f}, Std={np.std(train_mut_affs):.3f}, "
              f"Min={np.min(train_mut_affs):.3f}, Max={np.max(train_mut_affs):.3f}")
    if test_mut_affs:
        print(f"Mut Affinity Statistics (Test, n={len(test_mut_affs)}):")
        print(f"  Mean={np.mean(test_mut_affs):.3f}, Std={np.std(test_mut_affs):.3f}, "
              f"Min={np.min(test_mut_affs):.3f}, Max={np.max(test_mut_affs):.3f}")
    
    if len(train_pairs) == 0:
        print("Error: No training pairs found. Check graph database directories.")
        return
    if len(test_pairs) == 0:
        print("Error: No test pairs found. Check test database directory.")
        return
    
    # ---------------------------------------------------------------
    # Prepare graph lists for DataLoader
    # graph.y contains the original binding affinity (mut_aff / wt_aff).
    # The training target (mut_aff) is taken directly from mut_batch.y.
    # ---------------------------------------------------------------
    train_mut_graphs = [p[1] for p in train_pairs]
    train_wt_graphs  = [p[2] for p in train_pairs]
    train_names      = [p[0] for p in train_pairs]
    test_mut_graphs  = [p[1] for p in test_pairs]
    test_wt_graphs   = [p[2] for p in test_pairs]
    test_names       = [p[0] for p in test_pairs]
    
    print(f"\nFinal dataset sizes:")
    print(f"  Training: {len(train_mut_graphs)} mutant–WT pairs")
    print(f"  Test:     {len(test_mut_graphs)} mutant–WT pairs")
    
    # ---------------------------------------------------------------
    # Load pretrained encoder and precompute embeddings
    # ---------------------------------------------------------------
    node_in_dim = MODEL_INIT_DIM[embedding_type]
    hidden_dim = 2 ** mut_cfg['hidden_dim_power']
    
    encoder = load_pretrained_encoder(
        checkpoint_path=pretrained_path,
        node_in_dim=node_in_dim,
        edge_in_dim=EDGE_IN_DIM,
        metadata=METADATA,
        hidden_dim=hidden_dim,
        num_hgt_layers=mut_cfg['num_layers'],
        hgt_heads=mut_cfg['hgt_heads'],
        device=device,
        message_style=mut_cfg.get('message_style', 'gated_src'),
    )
    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters (frozen): {encoder_params:,}")
    
    encoder.eval()
    batch_size = mut_cfg['batch_size']
    
    # Precompute embeddings for all graph sets
    print("\n--- Precomputing train mutant embeddings ---")
    precompute_embeddings(
        encoder, train_names, train_mut_graphs, device,
        batch_size=batch_size, embedding_type=embedding_type, use_amp=use_amp,
        use_jk=use_jk, jk_mode=jk_mode,
    )
    # Wild-type graphs for training: deduplicate before encoding.
    # Multiple mutants share the same WT graph object — encoding it twice
    # would feed already-overwritten .x back into the encoder.
    seen_wt_ids = set()
    unique_wt_names, unique_wt_graphs = [], []
    for name, graph in zip(train_names, train_wt_graphs):
        pdb_id = name.split('_')[0]
        if pdb_id not in seen_wt_ids:
            seen_wt_ids.add(pdb_id)
            unique_wt_names.append(pdb_id)
            unique_wt_graphs.append(graph)
    print(f"--- Precomputing train wild-type embeddings ({len(unique_wt_graphs)} unique of {len(train_wt_graphs)} total) ---")
    precompute_embeddings(
        encoder, unique_wt_names, unique_wt_graphs, device,
        batch_size=batch_size, embedding_type=embedding_type, use_amp=use_amp,
        use_jk=use_jk, jk_mode=jk_mode,
    )
    print("--- Precomputing test mutant embeddings ---")
    precompute_embeddings(
        encoder, test_names, test_mut_graphs, device,
        batch_size=batch_size, embedding_type=embedding_type, use_amp=use_amp,
        use_jk=use_jk, jk_mode=jk_mode,
    )
    # Same deduplication for test wild-type graphs
    seen_wt_ids_test = set()
    unique_test_wt_names, unique_test_wt_graphs = [], []
    for name, graph in zip(test_names, test_wt_graphs):
        pdb_id = name.split('_')[0]
        if pdb_id not in seen_wt_ids_test:
            seen_wt_ids_test.add(pdb_id)
            unique_test_wt_names.append(pdb_id)
            unique_test_wt_graphs.append(graph)
    print(f"--- Precomputing test wild-type embeddings ({len(unique_test_wt_graphs)} unique of {len(test_wt_graphs)} total) ---")
    precompute_embeddings(
        encoder, unique_test_wt_names, unique_test_wt_graphs, device,
        batch_size=batch_size, embedding_type=embedding_type, use_amp=use_amp,
        use_jk=use_jk, jk_mode=jk_mode,
    )
    
    # Free encoder
    del encoder
    torch.cuda.empty_cache()
    print("Encoder freed from memory. Training with precomputed embeddings.\n")
    
    # ---------------------------------------------------------------
    # N-fold cross-validation
    # ---------------------------------------------------------------
    n_folds = mut_cfg.get('n_folds', 5)
    seed = config.get('system', {}).get('seed', 42)
    fold_strategy = mut_cfg.get('fold_strategy', 'group')
    
    fold_splits = get_fold_splits(train_names, n_folds, seed, fold_strategy)
    
    pool_input_dim = get_pool_input_dim(
        hidden_dim, embedding_type,
        use_jk=use_jk, jk_mode=jk_mode,
        num_hgt_layers=mut_cfg['num_layers'],
    )
    
    # Prepare indices and PDB-based groups
    groups = np.array([name.split('_')[0] for name in train_names])
    unique_groups = np.unique(groups)
    strategy_label = f"Seeded {'GroupKFold' if fold_strategy == 'group' else 'KFold'}"
    print(f"  {strategy_label} (seed={seed}): {len(unique_groups)} unique PDB groups across {len(train_names)} samples")
    
    # Test data loaders (shared across folds)
    test_mut_loader = DataLoader(test_mut_graphs, batch_size=batch_size, shuffle=False)
    test_wt_loader  = DataLoader(test_wt_graphs,  batch_size=batch_size, shuffle=False)
    
    # Storage for all-folds results
    all_fold_logs = {}
    fold_split_log = {}
    fold_test_rp = {}
    
    print(f"\n{'='*60}")
    print(f"Starting {n_folds}-Fold CV for ΔΔG Mutation Fine-tuning")
    print(f"{'='*60}\n")
    
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        fold_id = fold + 1
        val_groups = set(groups[val_idx])
        print(f"\n{'='*60}")
        print(f"FOLD {fold_id}/{n_folds}")
        print(f"{'='*60}")
        print(f"  Train: {len(train_idx)} pairs, Val: {len(val_idx)} pairs")
        print(f"  Val PDB groups ({len(val_groups)}): {sorted(val_groups)[:10]}"
              + (f" ...+{len(val_groups)-10} more" if len(val_groups) > 10 else ""))
        
        # Record split info
        fold_split_log[f'fold_{fold_id}'] = {
            'train': [
                [train_names[i], float(train_mut_graphs[i].y.item())]
                for i in train_idx
            ],
            'val': [
                [train_names[i], float(train_mut_graphs[i].y.item())]
                for i in val_idx
            ],
        }
        
        # Fold data loaders (mutant and WT must be in the same order)
        fold_train_mut = [train_mut_graphs[i] for i in train_idx]
        fold_train_wt  = [train_wt_graphs[i]  for i in train_idx]
        fold_val_mut   = [train_mut_graphs[i]  for i in val_idx]
        fold_val_wt    = [train_wt_graphs[i]   for i in val_idx]
        
        fold_train_mut_loader = DataLoader(fold_train_mut, batch_size=batch_size, shuffle=False)
        fold_train_wt_loader  = DataLoader(fold_train_wt,  batch_size=batch_size, shuffle=False)
        fold_val_mut_loader   = DataLoader(fold_val_mut,   batch_size=batch_size, shuffle=False)
        fold_val_wt_loader    = DataLoader(fold_val_wt,    batch_size=batch_size, shuffle=False)
        
        # --- Create fresh model for this fold ---
        # Dynamic ft_hdim: if not set in config, use pool_input_dim // 4 (min 256)
        ft_hdim = mut_cfg.get('ft_hdim', None)
        if ft_hdim is None:
            ft_hdim = max(256, pool_input_dim // 4)
        ft_hdim = min(ft_hdim, pool_input_dim // 2)
        model = create_poolhead_model(
            model_type='ddg_regressor',
            pool_input_dim=pool_input_dim,
            hidden_dim=ft_hdim,
            metadata=METADATA,
            regressor_layers=mut_cfg.get('regressor_layers', 4),
            dropout=mut_cfg['dropout'],
            pool_mode=mut_cfg.get('pool_mode', 'cross_attn'),
            cross_attn_queries=mut_cfg.get('cross_attn_queries', 2),
            cross_attn_heads=mut_cfg.get('cross_attn_heads', 4),
            ddg_input_mode=mut_cfg.get('ddg_input_mode', 'concat_diff'),
        ).to(device)
        
        if fold == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  DDG model: {total_params:,} total params, {trainable_params:,} trainable")
            print(f"  Pool input dim: {pool_input_dim}, ddg_input_mode: {mut_cfg.get('ddg_input_mode', 'concat_diff')}, ft_hdim: {ft_hdim}")
            if use_jk:
                print(f"  JK-Net enabled: mode={jk_mode}")
        
        # Optimizer, scheduler, scaler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=mut_cfg['lr'],
            weight_decay=mut_cfg['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=mut_cfg.get('scheduler_t0', 10),
            T_mult=mut_cfg.get('scheduler_t_mult', 2),
            eta_min=float(mut_cfg.get('scheduler_eta_min', 1e-6)),
        )
        scaler = GradScaler(enabled=use_amp)
        
        # Loss
        criterion_train = get_loss(
            mut_cfg.get('loss_fun', 'l1'),
            gamma=mut_cfg.get('gamma', 1.0),
            delta=mut_cfg.get('delta', 1.0),
            reduction='mean',
        )
        criterion_val = get_loss('l1', reduction='mean')
        
        # Early stopping
        checkpoint_name = (
            f"mut_ddg"
            + (f"_{embedding_type}" if embedding_type not in ['base', 'esm'] else '')
            + ('' if mut_cfg.get('pool_mode', 'cross_attn') == 'cross_attn' else f"_{mut_cfg['pool_mode']}")
            + jk_suffix
            + f"_fold{fold_id}_best.pt"
        )
        checkpoint_path = os.path.join(final_save_dir, checkpoint_name)
        early_stopping = FineTuneEarlyStopping(
            patience=mut_cfg['patience'],
            save_path=checkpoint_path,
            mode='max',  # maximize Pearson r
        )
        
        # Dynamic clipping
        clip_min = 5.0
        clip_factor = 1.25
        current_clip = mut_cfg.get('clip_max_norm', 5.0)
        
        # Fold training log
        fold_log = {
            'train_loss': [], 'val_loss': [],
            'train_rp': [], 'val_rp': [],
            'grad_norm': [], 'lr': [],
        }
        
        # --- Training loop for this fold ---
        for epoch in range(mut_cfg['n_epochs']):
            epoch_start = time.time()
            model.update_epoch(epoch)
            
            # Train
            train_loss, grad_norm, train_metrics = train_epoch_ddg(
                model, fold_train_mut_loader, fold_train_wt_loader,
                optimizer, scaler, criterion_train, device,
                clip_max_norm=current_clip, use_amp=use_amp,
            )
            # Validate
            val_loss, val_metrics = validate_ddg(
                model, fold_val_mut_loader, fold_val_wt_loader,
                criterion_val, device, use_amp=use_amp,
            )
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Dynamic gradient clipping
            if grad_norm is not None and math.isfinite(grad_norm) and grad_norm > 0:
                current_clip = max(clip_min, grad_norm * clip_factor)
            
            # Log
            fold_log['train_loss'].append(float(train_loss))
            fold_log['val_loss'].append(float(val_loss))
            fold_log['train_rp'].append(float(train_metrics['rp']))
            fold_log['val_rp'].append(float(val_metrics['rp']))
            fold_log['grad_norm'].append(float(grad_norm) if grad_norm is not None else 0.0)
            fold_log['lr'].append(float(current_lr))
            
            epoch_time = time.time() - epoch_start
            
            print(f"  Epoch {epoch+1:3d} | "
                  f"Loss T/V: {train_loss:.3f}/{val_loss:.3f} | "
                  f"rp T/V: {train_metrics['rp']:.3f}/{val_metrics['rp']:.3f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Early stopping on val Pearson r
            stop = early_stopping(val_metrics['rp'], model, epoch)
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
        
        test_loss, test_metrics = validate_ddg(
            model, test_mut_loader, test_wt_loader,
            criterion_val, device, use_amp=use_amp,
        )
        fold_test_rp[f'fold_{fold_id}'] = float(test_metrics['rp'])
        fold_log['test_rp'] = float(test_metrics['rp'])
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
    
    print(f"\n{'='*60}")
    print(f"ΔΔG {n_folds}-Fold CV Complete")
    print(f"{'='*60}")
    print(f"  Test Pearson r per fold: {[f'{v:.3f}' for v in test_rps]}")
    print(f"  Mean ± Std: {mean_test_rp:.3f} ± {std_test_rp:.3f}")
    
    # Save all-folds training log
    log_name = f"ddg_{n_folds}fold{jk_suffix}_training_log.json"
    log_path = os.path.join(final_save_dir, log_name)
    with open(log_path, 'w') as f:
        json.dump(all_fold_logs, f, indent=2)
    
    # Save fold split info
    split_name = f"ddg_{n_folds}fold{jk_suffix}_splits.json"
    split_path = os.path.join(final_save_dir, split_name)
    with open(split_path, 'w') as f:
        json.dump(fold_split_log, f, indent=2)
    
    # Save summary
    summary = {
        'n_folds': n_folds,
        'fold_strategy': fold_strategy,
        'embedding_type': embedding_type,
        'ddg_input_mode': mut_cfg.get('ddg_input_mode', 'concat_diff'),
        'num_layers': mut_cfg['num_layers'],
        'hidden_dim_power': mut_cfg['hidden_dim_power'],
        'pool_mode': mut_cfg.get('pool_mode', 'cross_attn'),
        'use_jk': use_jk,
        'jk_mode': jk_mode if use_jk else None,
        'fold_test_rp': fold_test_rp,
        'mean_test_rp': mean_test_rp,
        'std_test_rp': std_test_rp,
    }
    summary_path = os.path.join(final_save_dir, f"ddg_{n_folds}fold{jk_suffix}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Training logs: {log_path}")
    print(f"  Fold splits:   {split_path}")
    print(f"  Summary:        {summary_path}")
    print(f"  Checkpoints:    {final_save_dir}/best_ddg_fold*.pt")
    print(f"{'='*60}\n")
    
    return summary


# ============================================================================
# MAIN FOR STANDALONE EXECUTION
# ============================================================================
if __name__ == '__main__':
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(
        description='ΔΔG Fine-tuning with N-fold CV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Example:
    python finetune_ddg_module.py -config config_ddg.yaml
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
    print("ΔΔG FINE-TUNING PIPELINE")
    print(f"{'='*80}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Seed: {config['system']['seed']}")
    
    # Run ddG fine-tuning
    run_ddg_finetuning(config, device)
    
    print(f"\n{'='*80}")
    print("ALL TRAINING COMPLETE!")
    print(f"{'='*80}\n")
