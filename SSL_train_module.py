import os
import json
import random
import time
import math
from typing import Dict
import numpy as np
import torch
from torch.amp import GradScaler
from utils.SSL_modules import get_ssl_train_loader
from utils.SSL_modules.ssl_data_loader import filter_samples_for_ssl
from model_collection.SSLModels import create_ssl_model
from utils.Training_modules.GPU_tracking import get_gpu_memory_mb

# Import shared utilities and training functions
from utils.Training_modules.common_utils import (
    set_seed, load_sttgs_from_dir, get_db_path_ssl,
    MODEL_INIT_DIM, EDGE_IN_DIM, METADATA
)
from utils.Training_modules.ssl_training import (
    train_epoch_ssl, validate_ssl, SSLEarlyStopping
)

# ============================================================================
# SSL PRE-TRAINING ORCHESTRATION
# ============================================================================
def run_ssl_pretraining(config: Dict, device: torch.device) -> str:
    """
    Run SSL pre-training phase.
    
    Returns:
        Path to the best checkpoint file
    """
    print(f"\n{'='*80}")
    print("PHASE 1: SSL PRE-TRAINING")
    print(f"{'='*80}\n")
    
    ssl_cfg = config['ssl']
    data_cfg = config['data']
    #sys_cfg = config['system']
    
    # Determine mask type: use inter_intra_edge if intra_edge_mask_ratio is configured
    intra_edge_mask_ratio = ssl_cfg.get('intra_edge_mask_ratio', 0.0)
    intra_edge_beta = ssl_cfg.get('intra_edge_beta', 0.7)
    use_intra_edges = intra_edge_mask_ratio > 0.0
    mask_type = 'inter_intra_edge' if use_intra_edges else 'edge'
    
    # Setup directories
    local_dir = (
        f"{ssl_cfg['num_layers']}layers_{ssl_cfg['hidden_dim_power']}hdim"
        + ('_' + data_cfg['embedding_type'] if data_cfg['embedding_type'] in ['esm','esm480'] else '')
        + (f"_{ssl_cfg['strategy_type']}" if ssl_cfg['strategy_type'] != 'dynamic' else '')
        + (f"_additive" if ssl_cfg.get('message_style', 'gated_src') == 'additive' else '')
        #+ (f"_intra{intra_edge_mask_ratio}_b{intra_edge_beta}" if use_intra_edges else '')
    )
    final_save_dir = os.path.join(ssl_cfg['save_dir'], local_dir)
    os.makedirs(final_save_dir, exist_ok=True)
    
    use_amp = device.type == 'cuda'
    
    print(f"Device: {device}")
    print(f"SSL Task: Edge Prediction ({'Inter + Intra' if use_intra_edges else 'Inter-only'})")
    print(f"Inter-edge mask ratio: {ssl_cfg['edge_mask_ratio']}")
    if use_intra_edges:
        print(f"Intra-edge mask ratio: {intra_edge_mask_ratio}")
        print(f"Beta (inter weight): {intra_edge_beta}")
    print(f"Strategy type: {ssl_cfg['strategy_type']}")
    print(f"Save directory: {final_save_dir}")
    
    # Load data
    pdb_dir = get_db_path_ssl(data_cfg['pdb_root'], data_cfg['dist'], data_cfg['embedding_type'])
    print(f"Loading data from: {pdb_dir}")
    
    all_sttgs = load_sttgs_from_dir(pdb_dir)
    all_sttgs = filter_samples_for_ssl(all_sttgs, min_interface_edges=ssl_cfg['min_interface_edges'])
    
    graphs = []
    pdb_names = []
    for name, st in all_sttgs.items():
        graphs.append(st.protein_graph)
        pdb_names.append(name)
    
    print(f"Loaded {len(graphs)} samples for SSL pre-training")
    
    if len(graphs) == 0:
        raise ValueError("No samples loaded. Check data directory.")
    
    # Train/val split
    split_seed = data_cfg.get('split_seed', 42)
    set_seed(split_seed)
    n_samples = len(graphs)
    indices = list(range(n_samples))
    random.shuffle(indices)
    split_idx = int(0.8 * n_samples)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_graphs = [graphs[i] for i in train_indices]
    train_names = [pdb_names[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]
    val_names = [pdb_names[i] for i in val_indices]
    
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")
    
    # Save train/val split info
    split_info = {
        'split_seed': split_seed,
        'n_total': n_samples,
        'n_train': len(train_names),
        'n_val': len(val_names),
        'train_names': train_names,
        'val_names': val_names,
    }
    split_info_path = os.path.join(final_save_dir, 'ssl_train_val_split.json')
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"Train/val split saved to: {split_info_path}")

    # Create data loaders
    train_loader = get_ssl_train_loader(
        graphs=train_graphs,
        pdb_names=train_names,
        batch_size=ssl_cfg['batch_size'],
        mask_type=mask_type,
        strategy_type=ssl_cfg['strategy_type'],
        node_mask_ratio=0.0,
        edge_mask_ratio=ssl_cfg['edge_mask_ratio'],
        intra_edge_mask_ratio=intra_edge_mask_ratio,
        negative_ratio=ssl_cfg['negative_ratio'],
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
    )
    
    val_loader = get_ssl_train_loader(
        graphs=val_graphs,
        pdb_names=val_names,
        batch_size=ssl_cfg['batch_size'],
        mask_type=mask_type,
        strategy_type=ssl_cfg['strategy_type'],
        node_mask_ratio=0.0,
        edge_mask_ratio=ssl_cfg['edge_mask_ratio'],
        intra_edge_mask_ratio=intra_edge_mask_ratio,
        negative_ratio=ssl_cfg['negative_ratio'],
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
    )
    
    # Create model
    node_in_dim = MODEL_INIT_DIM[data_cfg['embedding_type']]
    hidden_dim = 2 ** ssl_cfg['hidden_dim_power']
    
    model = create_ssl_model(
        node_in_dim=node_in_dim,
        edge_in_dim=EDGE_IN_DIM,
        metadata=METADATA,
        hidden_dim=hidden_dim,
        num_hgt_layers=ssl_cfg['num_layers'],
        hgt_heads=ssl_cfg['hgt_heads'],
        dropout=ssl_cfg['dropout'],
        use_checkpoint=ssl_cfg['use_checkpoint'],
        message_style=ssl_cfg.get('message_style', 'gated_src'),
        mask_type=mask_type,
        beta=intra_edge_beta,
    ).to(device)
    
    model_name = 'SSL Inter+Intra Edge Predictor' if use_intra_edges else 'SSL Edge Predictor'
    print(f"\nModel: {model_name}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  HGT layers: {ssl_cfg['num_layers']}")
    print(f"  hgt_heads: {ssl_cfg['hgt_heads']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")    
    print(f"  message_style: {ssl_cfg.get('message_style', 'gated_src')}")
    
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ssl_cfg['lr'],
        weight_decay=ssl_cfg['weight_decay'],
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=ssl_cfg['scheduler_t0'],
        T_mult=2,
        eta_min=ssl_cfg['lr'] * 0.01,
    )
    
    scaler = GradScaler(enabled=use_amp)
    
    # Early stopping
    checkpoint_path = os.path.join(final_save_dir, 'ssl_edge_best.pt')
    early_stopping = SSLEarlyStopping(
        patience=ssl_cfg['patience'],
        min_delta=1e-5,
        save_path=checkpoint_path,
    )
    
    # Training log
    training_log = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': [],
        'grad_norm': [],
        'lr': [],
        'gpu_memory_mb': [],
    }
    
    print(f"\n{'='*60}")
    print("Starting SSL Training")
    print(f"{'='*60}\n")
    
    # Dynamic clipping state
    clip_min: float = 5.0
    clip_factor: float = 1.25
    current_clip: float = ssl_cfg.get('clip_max_norm', 5.0)
    
    for epoch in range(ssl_cfg['n_epochs']):
        start_time = time.time()
        
        # Train
        train_loss, grad_norm, train_metrics = train_epoch_ssl(
            model, train_loader, optimizer, scaler, device,
            clip_max_norm=current_clip,
            use_amp=use_amp,
        )
        
        # Validate
        val_loss, val_metrics = validate_ssl(model, val_loader, device, use_amp=use_amp)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        training_log['train_loss'].append(train_loss)
        training_log['val_loss'].append(val_loss)
        training_log['train_metrics'].append(train_metrics)
        training_log['val_metrics'].append(val_metrics)
        training_log['grad_norm'].append(grad_norm)
        training_log['lr'].append(current_lr)
        training_log['gpu_memory_mb'].append(get_gpu_memory_mb(device))
        
        # Update dynamic clip for next epoch
        if grad_norm is not None and math.isfinite(grad_norm) and grad_norm > 0:
            next_clip = grad_norm * clip_factor
            next_clip = max(clip_min, next_clip)
            current_clip = next_clip
        
        # Early stopping
        early_stopping(val_loss, model, epoch)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        if use_intra_edges:
            print(f"Epoch {epoch+1:3d} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"Inter AUC: {val_metrics.get('inter_edge_auc', 0):.3f} | "
                  f"Intra AUC: {val_metrics.get('intra_edge_auc', 0):.3f} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"GPU: {training_log['gpu_memory_mb'][-1]['max_reserved_mb']} MB")
        else:
            print(f"Epoch {epoch+1:3d} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"AUC: {val_metrics.get('edge_auc', 0):.3f} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"GPU: {training_log['gpu_memory_mb'][-1]['max_reserved_mb']} MB")
        
        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best validation loss: {early_stopping.best_loss:.4f} at epoch {early_stopping.best_epoch+1}")
            break
    
    # Save training log
    log_path = os.path.join(final_save_dir, 'ssl_edge_training_log.json')
    serializable_log = {}
    for k, v in training_log.items():
        if k in ['train_metrics', 'val_metrics']:
            serializable_log[k] = [{kk: float(vv) for kk, vv in m.items()} for m in v]
        else:
            serializable_log[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
    
    with open(log_path, 'w') as f:
        json.dump(serializable_log, f, indent=2)
    
    # Save config
    ssl_config_dict = {
        'embedding_type': data_cfg['embedding_type'],
        'num_layers': ssl_cfg['num_layers'],
        'hidden_dim_power': ssl_cfg['hidden_dim_power'],
        'hgt_heads': ssl_cfg['hgt_heads'],
        'dropout': ssl_cfg['dropout'],
        'message_style': ssl_cfg.get('message_style', 'gated_src'),
        'best_val_loss': early_stopping.best_loss,
        'best_epoch': early_stopping.best_epoch,
        'strategy_type': ssl_cfg['strategy_type'],
        'mask_type': mask_type,
        'intra_edge_mask_ratio': intra_edge_mask_ratio,
        'intra_edge_beta': intra_edge_beta,
    }
    config_path = os.path.join(final_save_dir, 'ssl_edge_config.json')
    with open(config_path, 'w') as f:
        json.dump(ssl_config_dict, f, indent=2)
    
    print(f"\nSSL Pre-training Complete!")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training log: {log_path}")
    print(f"Config: {config_path}")
    
    return checkpoint_path