"""
Sequence-Student Knowledge-Distillation training.

Trains a sequence-only student encoder (frozen ESM residue embeddings + cross
attention) to reproduce the frozen GraPPI encoder's final-layer per-residue
embeddings. The student becomes a drop-in replacement for the GraPPI encoder in
all downstream fine-tuning tasks (use_jk=False).

See README_STUDENT_DISTILLATION.md for the full design.
"""

import os
import json
import math
import random
import time
from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
from functools import partial
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from model_collection.StudentModels import (
    create_student_model, distillation_loss, InterEdgePredictor, edge_prediction_loss,
)
from model_collection.FineTuneModels import load_pretrained_encoder
from utils.Training_modules.common_utils import (
    set_seed, load_sttgs_from_dir, get_db_path_ssl, load_ssl_config,
    MODEL_INIT_DIM, EDGE_IN_DIM, METADATA,
)
from utils.Training_modules.save_training import FineTuneEarlyStopping
from utils.Training_modules.student_data_loader import (
    build_teacher_cache, StudentDistillDataset, collate_student, BucketBatchSampler,
)
from utils.Training_modules.edge_label_utils import build_inter_edge_cache
from utils.SSL_modules.ssl_data_loader import filter_samples_for_ssl

ESM_DIM = {'esm': 1280, 'esm480': 480}


def _run_epoch_student(model, edge_head, loader, optimizer, scaler, device, cfg, use_amp, train: bool):
    model.train(train)
    if edge_head is not None:
        edge_head.train(train)
    total_loss, n_batches = 0.0, 0
    agg = {'cosine': 0.0, 'mse': 0.0, 'cos_sim': 0.0, 'edge_bce': 0.0, 'edge_acc': 0.0}
    amp_ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
               if use_amp else nullcontext())
    w_edge = cfg.get('edge_loss_weight', 0.0)
    grad_ctx = nullcontext() if train else torch.no_grad()
    edge_logits_all, edge_labels_all = [], []

    with grad_ctx:
        for (rec_x, lig_x, rec_mask, lig_mask, t_rec, t_lig,
             e_batch, e_ri, e_li, e_lab, _) in loader:
            rec_x, lig_x = rec_x.to(device), lig_x.to(device)
            rec_mask, lig_mask = rec_mask.to(device), lig_mask.to(device)
            t_rec, t_lig = t_rec.to(device), t_lig.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                s_rec, s_lig = model(rec_x, lig_x, rec_mask, lig_mask)
                loss, parts = distillation_loss(
                    s_rec, s_lig, t_rec, t_lig, rec_mask, lig_mask,
                    mse_weight=cfg.get('mse_weight', 1.0),
                    cosine_weight=cfg.get('cosine_weight', 1.0),
                )
                if edge_head is not None and w_edge > 0:
                    e_batch, e_ri, e_li, e_lab = (
                        e_batch.to(device), e_ri.to(device), e_li.to(device), e_lab.to(device))
                    eloss, eparts = edge_prediction_loss(
                        edge_head, s_rec, s_lig, e_batch, e_ri, e_li, e_lab)
                    loss = loss + w_edge * eloss
                    parts.update(eparts)
                    if eparts['edge_labels'].numel() > 0:
                        edge_logits_all.append(eparts['edge_logits'])
                        edge_labels_all.append(eparts['edge_labels'])

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                params = list(model.parameters())
                if edge_head is not None:
                    params += list(edge_head.parameters())
                torch.nn.utils.clip_grad_norm_(params, cfg.get('clip_max_norm', 5.0))
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            for k in agg:
                agg[k] += parts.get(k, 0.0)
            n_batches += 1

    n = max(1, n_batches)
    metrics = {k: v / n for k, v in agg.items()}
    if edge_labels_all:
        y_true = torch.cat(edge_labels_all).numpy()
        y_score = torch.cat(edge_logits_all).numpy()
        if len(np.unique(y_true)) > 1:
            metrics['edge_auc'] = float(roc_auc_score(y_true, y_score))
            metrics['edge_auprc'] = float(average_precision_score(y_true, y_score))
        else:
            metrics['edge_auc'] = 0.0
            metrics['edge_auprc'] = 0.0
    else:
        metrics['edge_auc'] = 0.0
        metrics['edge_auprc'] = 0.0
    return total_loss / n, metrics


def run_student_training(config: Dict, device: torch.device) -> str:
    """Run sequence-student distillation. Returns path to the best checkpoint."""
    print(f"\n{'='*80}")
    print("PHASE: SEQUENCE STUDENT DISTILLATION")
    print(f"{'='*80}\n")

    st_cfg = config['student']
    data_cfg = config['data']
    use_amp = device.type == 'cuda'

    embedding_type = data_cfg['embedding_type']
    if embedding_type not in ESM_DIM:
        raise ValueError(
            f"Student distillation requires an ESM graph set "
            f"(embedding_type 'esm' or 'esm480'), got '{embedding_type}'.")
    esm_dim = ESM_DIM[embedding_type]

    # ------------------------------------------------------------------
    # 1. Load complex graphs (same pool GraPPI was pretrained on)
    # ------------------------------------------------------------------
    pdb_dir = get_db_path_ssl(data_cfg['pdb_root'], data_cfg['dist'], embedding_type)
    print(f"Loading data from: {pdb_dir}")
    all_sttgs = load_sttgs_from_dir(pdb_dir)
    all_sttgs = filter_samples_for_ssl(
        all_sttgs, min_interface_edges=st_cfg.get('min_interface_edges', 5))

    graphs, names = [], []
    for name, st in all_sttgs.items():
        graphs.append(st.protein_graph)
        names.append(name)
    print(f"Loaded {len(graphs)} complexes.")
    if not graphs:
        raise ValueError("No samples loaded. Check data directory.")

    # ------------------------------------------------------------------
    # 2. Teacher: load frozen GraPPI encoder, build the target cache
    # ------------------------------------------------------------------
    teacher_ckpt = st_cfg['teacher_ckpt']
    ssl_conf = load_ssl_config(teacher_ckpt)
    if ssl_conf is None:
        raise FileNotFoundError(
            f"ssl_edge_config.json not found next to {teacher_ckpt}; needed to "
            "rebuild the teacher architecture.")
    teacher_emb = ssl_conf.get('embedding_type', embedding_type)
    if teacher_emb != embedding_type:
        raise ValueError(
            f"Teacher was trained with embedding_type='{teacher_emb}' but data uses "
            f"'{embedding_type}'. Use a matching ESM-trained GraPPI teacher.")

    hidden_dim = 2 ** ssl_conf['hidden_dim_power']
    num_layers = ssl_conf['num_layers']
    hgt_heads = ssl_conf['hgt_heads']
    message_style = ssl_conf.get('message_style', 'gated_src')

    encoder = load_pretrained_encoder(
        checkpoint_path=teacher_ckpt,
        node_in_dim=MODEL_INIT_DIM[embedding_type],
        edge_in_dim=EDGE_IN_DIM,
        metadata=METADATA,
        hidden_dim=hidden_dim,
        num_hgt_layers=num_layers,
        hgt_heads=hgt_heads,
        device=device,
        message_style=message_style,
    )

    teacher_tag = os.path.basename(os.path.dirname(teacher_ckpt)) or 'teacher'
    cache_dir = os.path.join(st_cfg['teacher_cache_dir'], teacher_tag)
    swap_augment = st_cfg.get('swap_augment', False)
    build_teacher_cache(
        encoder, names, graphs, cache_dir, device,
        batch_size=st_cfg.get('teacher_batch_size', 16), use_amp=use_amp,
        also_swapped=swap_augment,
    )
    del encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Cache ground-truth interface edges (teacher-independent) into the same dir
    if st_cfg.get('edge_loss_weight', 0.0) > 0:
        build_inter_edge_cache(names, graphs, cache_dir, also_swapped=swap_augment)

    # ------------------------------------------------------------------
    # 3. Train/val split (mirror SSL: 80/20, split_seed 42)
    # ------------------------------------------------------------------
    split_seed = data_cfg.get('split_seed', 42)
    set_seed(split_seed)
    indices = list(range(len(graphs)))
    random.shuffle(indices)
    split_idx = int(0.8 * len(graphs))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_names = [names[i] for i in train_idx]
    train_graphs = [graphs[i] for i in train_idx]
    val_names = [names[i] for i in val_idx]
    val_graphs = [graphs[i] for i in val_idx]
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

    max_res = st_cfg.get('max_residues_per_protein', 1000)
    preload_cache = st_cfg.get('preload_cache', False)
    train_ds = StudentDistillDataset(
        train_names, train_graphs, cache_dir, esm_dim, max_res,
        swap_augment=swap_augment, preload=preload_cache)
    val_ds = StudentDistillDataset(
        val_names, val_graphs, cache_dir, esm_dim, max_res,
        swap_augment=False, preload=preload_cache)

    batch_size = st_cfg.get('batch_size', 16)
    num_workers = st_cfg.get('num_workers', 4)
    train_sampler = BucketBatchSampler(
        train_ds.lengths(), batch_size, shuffle=True, seed=split_seed)
    val_sampler = BucketBatchSampler(
        val_ds.lengths(), batch_size, shuffle=False, seed=split_seed)
    neg_ratio = st_cfg.get('edge_negative_ratio', 1.0)
    collate_fn = partial(collate_student, negative_ratio=neg_ratio)
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=collate_fn,
        num_workers=num_workers)
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler, collate_fn=collate_fn,
        num_workers=num_workers)

    # ------------------------------------------------------------------
    # 4. Student model
    # ------------------------------------------------------------------
    student = create_student_model(
        esm_dim=esm_dim,
        hidden_dim=hidden_dim,
        n_blocks=st_cfg.get('n_blocks', 3),
        n_heads=st_cfg.get('n_heads', 8),
        dropout=st_cfg.get('dropout', 0.2),
        ffn_mult=st_cfg.get('ffn_mult', 4),
        input_dropout=st_cfg.get('input_dropout', 0.0),
    ).to(device)
    print(f"\nStudent: esm_dim={esm_dim}, hidden_dim={hidden_dim}, "
          f"n_blocks={st_cfg.get('n_blocks', 3)}, n_heads={st_cfg.get('n_heads', 8)}")
    print(f"  Parameters: {sum(p.numel() for p in student.parameters()):,}")

    # Auxiliary interface edge-prediction head (training-time only, discarded at inference)
    edge_loss_weight = st_cfg.get('edge_loss_weight', 0.0)
    edge_head = None
    if edge_loss_weight > 0:
        edge_head = InterEdgePredictor(
            hidden_dim, edge_hidden=st_cfg.get('edge_hidden'),
            dropout=st_cfg.get('edge_dropout', 0.1)).to(device)
        print(f"  Edge head params: {sum(p.numel() for p in edge_head.parameters()):,} "
              f"(edge_loss_weight={edge_loss_weight}, neg_ratio={neg_ratio})")

    # ------------------------------------------------------------------
    # 5. Save dir + config
    # ------------------------------------------------------------------
    local_dir = (
        f"student_{st_cfg.get('n_blocks', 3)}blk_{st_cfg.get('n_heads', 8)}h_"
        f"{embedding_type}_from_{teacher_tag}"
    )
    final_save_dir = os.path.join(st_cfg['save_dir'], local_dir)
    os.makedirs(final_save_dir, exist_ok=True)
    checkpoint_path = os.path.join(final_save_dir, 'student_best.pt')

    student_config = {
        'esm_dim': esm_dim,
        'hidden_dim': hidden_dim,
        'n_blocks': st_cfg.get('n_blocks', 3),
        'n_heads': st_cfg.get('n_heads', 8),
        'dropout': st_cfg.get('dropout', 0.2),
        'ffn_mult': st_cfg.get('ffn_mult', 4),
        'input_dropout': st_cfg.get('input_dropout', 0.0),
        'embedding_type': embedding_type,
        'teacher_ckpt': teacher_ckpt,
        'teacher_hidden_dim': hidden_dim,
        'max_residues_per_protein': max_res,
    }
    with open(os.path.join(final_save_dir, 'student_config.json'), 'w') as f:
        json.dump(student_config, f, indent=2)

    split_info = {
        'split_seed': split_seed,
        'n_total': len(graphs),
        'train_names': train_names,
        'val_names': val_names,
    }
    with open(os.path.join(final_save_dir, 'student_train_val_split.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    # ------------------------------------------------------------------
    # 6. Optimizer / scheduler / early stopping
    # ------------------------------------------------------------------
    train_params = list(student.parameters())
    if edge_head is not None:
        train_params += list(edge_head.parameters())
    optimizer = torch.optim.AdamW(
        train_params,
        lr=st_cfg.get('lr', 3e-4),
        weight_decay=st_cfg.get('weight_decay', 1e-3),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=st_cfg.get('scheduler_t0', 10),
        T_mult=st_cfg.get('scheduler_t_mult', 2),
        eta_min=st_cfg.get('lr', 3e-4) * 0.01,
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)
    early_stopping = FineTuneEarlyStopping(
        patience=st_cfg.get('patience', 30),
        min_delta=1e-6,
        save_path=checkpoint_path,
        mode='min',
    )

    training_log = {'train_loss': [], 'val_loss': [], 'val_cos_sim': [], 'lr': []}

    print(f"\n{'='*60}\nStarting Student Distillation\n{'='*60}\n")
    for epoch in range(st_cfg.get('n_epochs', 1000)):
        start = time.time()
        train_sampler.set_epoch(epoch)

        train_loss, train_parts = _run_epoch_student(
            student, edge_head, train_loader, optimizer, scaler, device, st_cfg, use_amp, train=True)
        val_loss, val_parts = _run_epoch_student(
            student, edge_head, val_loader, optimizer, scaler, device, st_cfg, use_amp, train=False)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        training_log['train_loss'].append(train_loss)
        training_log['val_loss'].append(val_loss)
        training_log['val_cos_sim'].append(val_parts['cos_sim'])
        training_log['lr'].append(lr)

        stop = early_stopping(val_loss, student, epoch)
        edge_str = (f" | val edge_auc/auprc: {val_parts['edge_auc']:.3f}/{val_parts['edge_auprc']:.3f}"
                    if edge_head is not None else "")
        print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"val cos: {val_parts['cos_sim']:.4f} | val mse: {val_parts['mse']:.4f}"
              f"{edge_str} | Time: {time.time()-start:.1f}s")

        if stop:
            print(f"\nEarly stopping at epoch {epoch+1}. "
                  f"Best val loss {early_stopping.best_score:.4f} "
                  f"at epoch {early_stopping.best_epoch+1}.")
            break

    log_path = os.path.join(final_save_dir, 'student_training_log.json')
    with open(log_path, 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in training_log.items()}, f, indent=2)

    print(f"\nStudent Distillation Complete!")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {os.path.join(final_save_dir, 'student_config.json')}")
    return checkpoint_path
