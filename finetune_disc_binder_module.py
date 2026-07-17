import pickle
import os
import json
import time
import math
from contextlib import nullcontext
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

from model_collection.FineTuneModels import (
    load_pretrained_encoder, precompute_embeddings,
    get_pool_input_dim, create_poolhead_model,
)
from utils.Training_modules.save_training import FineTuneEarlyStopping
from utils.Training_modules.common_utils import (
get_sample_type, load_ssl_config,
    set_seed, compute_classification_metrics,
    MODEL_INIT_DIM, EDGE_IN_DIM, METADATA
)
from utils.Training_modules.classifier_data_loader import (
    load_classifier_data, load_classifier_data_cv,
)

from utils.Training_modules.finetune_training import (
    train_epoch_classifier, validate_classifier,
)


# ============================================================================
# Seen/unseen helpers (positive samples encountered during SSL pre-training
# are "seen"; all negatives are unseen by definition).
# ============================================================================
def _load_ssl_pos_split(pretrained_path: str) -> Tuple[Set[str], Set[str]]:
    """Load ssl_train_val_split.json and return (seen_pos_names, unseen_pos_names)."""
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


@torch.no_grad()
def _validate_classifier_seen_unseen(
    model, val_loader, is_seen_flags: List[bool],
    criterion, device, use_amp: bool, label_smoothing: float = 0.0,
):
    """
    Forward over val_loader (must be non-shuffled) once and compute both overall
    and unseen-only metrics. is_seen_flags must be aligned with the loader's
    sample order (i.e. the list of graphs passed to DataLoader).

    Returned metrics dict includes:
        accuracy, precision, recall, f1, auc, auprc,
        auc_unseen, auprc_unseen, n_seen, n_unseen.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_true: List[float] = []
    all_pred: List[float] = []
    amp_ctx = (
        torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
        if use_amp else nullcontext()
    )

    for batch_data in val_loader:
        batch_data = batch_data.to(device)
        y_aff = batch_data.y
        y_hard = (y_aff > 0).float().unsqueeze(-1)
        y_target = (
            y_hard if label_smoothing == 0
            else y_hard * (1.0 - label_smoothing) + 0.5 * label_smoothing
        )
        with amp_ctx:
            logits = model(batch_data)
            loss = criterion(logits, y_target)
        total_loss += loss.item()
        n_batches += 1
        all_true.extend(y_hard.squeeze(-1).cpu().tolist())
        all_pred.extend(torch.sigmoid(logits).squeeze(-1).cpu().tolist())

    avg_loss = total_loss / max(1, n_batches)

    y_true = np.asarray(all_true, dtype=int)
    y_pred = np.asarray(all_pred, dtype=float)
    seen_arr = np.asarray(is_seen_flags, dtype=bool)
    if len(y_true) != len(seen_arr):
        raise RuntimeError(
            f"is_seen_flags length ({len(seen_arr)}) "
            f"!= predictions length ({len(y_true)})"
        )

    metrics = compute_classification_metrics(y_true.tolist(), y_pred.tolist())
    # Overall AUPRC
    if len(np.unique(y_true)) > 1:
        metrics['auprc'] = float(average_precision_score(y_true, y_pred))
    else:
        metrics['auprc'] = 0.0

    # Unseen subset: all negatives + unseen positives
    unseen_mask = ~seen_arr
    metrics['n_seen']   = int(seen_arr.sum())
    metrics['n_unseen'] = int(unseen_mask.sum())
    if metrics['n_unseen'] > 0:
        yt_u = y_true[unseen_mask]
        yp_u = y_pred[unseen_mask]
        if len(np.unique(yt_u)) > 1:
            metrics['auc_unseen']   = float(roc_auc_score(yt_u, yp_u))
            metrics['auprc_unseen'] = float(average_precision_score(yt_u, yp_u))
        else:
            metrics['auc_unseen']   = 0.0
            metrics['auprc_unseen'] = 0.0
    else:
        metrics['auc_unseen']   = 0.0
        metrics['auprc_unseen'] = 0.0

    return avg_loss, metrics


# ============================================================================
# Discriminative Binder CV pipeline (StratifiedKFold on a single pool)
# ============================================================================
def _run_disc_binder_cv(
    test_cfg: Dict,
    data_cfg: Dict,
    sys_cfg: Dict,
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
    N-fold (default 5) CV on a single pool of samples.
    No held-out test set; per-fold val metric drives early stopping and
    per-fold checkpoint saving.

    cv_split_mode (test_cfg['cv_split_mode']):
      * 'global'    (default) — StratifiedKFold over the whole pool. Encoder
                                may have seen most positives during SSL
                                pre-training, so the overall AUC is
                                leak-contaminated; unseen-only metrics
                                (auc_unseen, auprc_unseen) computed on
                                (unseen positives + all negatives) provide
                                a leak-free signal.
      * 'fixed_pos' — Positives are split once using the SSL split
                      (train_names → always train, val_names → always val);
                      negatives are KFold-rotated. In this mode the
                      validation positives are exclusively unseen, so
                      `auc` and `auc_unseen` coincide.
    """
    seed = sys_cfg.get('seed', 42)
    batch_size = test_cfg['batch_size']

    # ---------------------------------------------------------------
    # 1. Load the CV pool (pos + matched negatives)
    # ---------------------------------------------------------------
    pool_sttgs, pool_labels = load_classifier_data_cv(
        data_cfg=data_cfg,
        save_dir=final_save_dir,
        seed=seed,
        min_nodes=data_cfg.get('min_nodes', 5),
    )

    # Reject unsupported embedding types
    if embedding_type in ['+esm', '+esm480', 'only_esm', 'only_esm480']:
        raise ValueError(
            f"+esm/only_esm embedding types are not yet supported for classifier CV.")

    # Stamp binary label into graph.y so downstream code is uniform
    for name, st in pool_sttgs.items():
        st.protein_graph.y = torch.tensor(float(pool_labels[name]))

    # Build sample list (4-tuples for consistency with other tasks)
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
    print(f"\nCV pool: {len(samples)} total ({n_pos} pos / {n_neg} neg)")

    # ---------------------------------------------------------------
    # 1b. Load SSL seen/unseen split for the positives. All negatives are
    #     unseen by definition (they were never used in SSL pre-training).
    # ---------------------------------------------------------------
    cv_split_mode = test_cfg.get('cv_split_mode', 'global')
    if cv_split_mode not in ('global', 'fixed_pos'):
        raise ValueError(
            f"Unknown cv_split_mode={cv_split_mode!r}. Use 'global' or 'fixed_pos'."
        )

    seen_pos_names, unseen_pos_names = _load_ssl_pos_split(pretrained_path)
    sample_is_seen: Dict[str, bool] = {
        s[0]: (int(pool_labels[s[0]]) == 1 and s[0] in seen_pos_names)
        for s in samples
    }
    n_pos_seen   = sum(1 for s in samples
                       if pool_labels[s[0]] == 1 and s[0] in seen_pos_names)
    n_pos_unseen = sum(1 for s in samples
                       if pool_labels[s[0]] == 1 and s[0] in unseen_pos_names)
    n_pos_unknown = n_pos - n_pos_seen - n_pos_unseen
    print(f"  SSL split coverage of positives: "
          f"{n_pos_seen} seen / {n_pos_unseen} unseen / {n_pos_unknown} not in SSL split")
    print(f"  CV split mode: {cv_split_mode}")

    # ---------------------------------------------------------------
    # 2. Load pretrained encoder (frozen) and precompute embeddings
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
    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters (frozen): {encoder_params:,}")

    pool_names = [s[0] for s in samples]
    pool_graphs = [s[1] for s in samples]
    encoder.eval()
    print("\n--- Precomputing CV pool embeddings ---")
    precompute_embeddings(
        encoder, pool_names, pool_graphs, device,
        batch_size=batch_size,
        embedding_type=embedding_type,
        esm_dict=None,
        use_amp=use_amp,
        use_jk=use_jk,
        jk_mode=jk_mode,
    )

    del encoder
    torch.cuda.empty_cache()
    print("Encoder freed from memory. Training with precomputed embeddings.\n")

    # ---------------------------------------------------------------
    # 3. Fold splits per cv_split_mode
    # ---------------------------------------------------------------
    if cv_split_mode == 'global':
        # Original behaviour: StratifiedKFold over the whole pool. Encoder
        # may have seen most positives during SSL pre-training, so the
        # overall AUC is leak-contaminated. Unseen-only metrics computed
        # below give a leak-free signal.
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_splits = [
            (np.asarray(tr), np.asarray(vl))
            for tr, vl in skf.split(samples, labels_arr)
        ]
    else:  # 'fixed_pos'
        # Positives split once via the SSL train/val split: SSL-train
        # positives ("seen") are pinned to every training fold; SSL-val
        # positives ("unseen") are pinned to every validation fold.
        # Negatives are rotated with plain KFold across folds.
        pos_seen_idx = np.array(
            [i for i, s in enumerate(samples)
             if pool_labels[s[0]] == 1 and s[0] in seen_pos_names],
            dtype=int,
        )
        pos_unseen_idx = np.array(
            [i for i, s in enumerate(samples)
             if pool_labels[s[0]] == 1 and s[0] in unseen_pos_names],
            dtype=int,
        )
        neg_idx = np.array(
            [i for i, s in enumerate(samples) if pool_labels[s[0]] == 0],
            dtype=int,
        )
        n_excluded_pos = n_pos - len(pos_seen_idx) - len(pos_unseen_idx)
        if n_excluded_pos > 0:
            print(f"  [WARN] {n_excluded_pos} positive(s) not present in either SSL "
                  f"train_names or val_names are dropped from fixed_pos CV.")
        if len(pos_unseen_idx) == 0:
            raise ValueError(
                "fixed_pos CV requires at least one unseen positive "
                "(SSL val_names) but found none.")
        if len(neg_idx) < n_folds:
            raise ValueError(
                f"fixed_pos CV needs at least n_folds={n_folds} negatives, "
                f"got {len(neg_idx)}.")

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_splits = []
        for neg_tr_local, neg_vl_local in kf.split(neg_idx):
            tr = np.concatenate([pos_seen_idx, neg_idx[neg_tr_local]])
            vl = np.concatenate([pos_unseen_idx, neg_idx[neg_vl_local]])
            fold_splits.append((tr, vl))

    pool_input_dim = get_pool_input_dim(
        hidden_dim, embedding_type,
        use_jk=use_jk, jk_mode=jk_mode,
        num_hgt_layers=test_cfg['num_layers'],
    )

    # Dynamic ft_hdim
    ft_hdim = test_cfg.get('ft_hdim', None)
    if ft_hdim is None:
        ft_hdim = max(256, pool_input_dim // 4)
    ft_hdim = min(ft_hdim, pool_input_dim // 2)

    cls_metric_name = test_cfg.get('cls_metric', 'auc')
    label_smoothing = test_cfg.get('label_smoothing', 0.0)
    criterion_cls = nn.BCEWithLogitsLoss()

    # Checkpoint name template (cv_split_mode appended when non-default so
    # fixed_pos checkpoints / logs don't clobber global ones).
    mode_suffix = '' if cv_split_mode == 'global' else f"_{cv_split_mode}"
    base_name = (
        f"disc_binder"
        + (f"{'_'+embedding_type if embedding_type not in ['base', 'esm'] else ''}")
        + ('' if test_cfg.get('pool_mode', 'cross_attn') == 'cross_attn' else f"_{test_cfg['pool_mode']}")
        + jk_suffix
        + mode_suffix
    )

    print(f"\n{'='*60}")
    _cv_label = 'StratifiedKFold' if cv_split_mode == 'global' else 'FixedPos-KFold(negatives)'
    print(f"Starting {n_folds}-Fold {_cv_label} CV (seed={seed}) — Discriminative Binder")
    print(f"Pool input dim: {pool_input_dim} (ft_hdim={ft_hdim})")
    if use_jk:
        print(f"JK-Net enabled: mode={jk_mode}")
    print(f"{'='*60}")

    # ---------------------------------------------------------------
    # 4. CV loop
    # ---------------------------------------------------------------
    all_fold_logs: Dict[str, Dict] = {}
    fold_split_log: Dict[str, Dict] = {}
    fold_val_score: Dict[str, float] = {}

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        fold_id = fold + 1
        print(f"\n{'='*60}")
        print(f"FOLD {fold_id}/{n_folds}")
        print(f"{'='*60}")

        train_fold = [samples[i] for i in train_idx]
        val_fold = [samples[i] for i in val_idx]

        n_pos_tr = sum(1 for s in train_fold if pool_labels[s[0]] == 1)
        n_neg_tr = sum(1 for s in train_fold if pool_labels[s[0]] == 0)
        n_pos_vl = sum(1 for s in val_fold   if pool_labels[s[0]] == 1)
        n_neg_vl = sum(1 for s in val_fold   if pool_labels[s[0]] == 0)
        n_pos_vl_unseen = sum(1 for s in val_fold
                              if pool_labels[s[0]] == 1 and not sample_is_seen[s[0]])
        print(f"  Train: {len(train_fold)} ({n_pos_tr} pos / {n_neg_tr} neg)  "
              f"Val: {len(val_fold)} ({n_pos_vl} pos [{n_pos_vl_unseen} unseen] / {n_neg_vl} neg)")

        fold_split_log[f'fold_{fold_id}'] = {
            'train': [[s[0], int(pool_labels[s[0]]), int(sample_is_seen[s[0]])]
                      for s in train_fold],
            'val':   [[s[0], int(pool_labels[s[0]]), int(sample_is_seen[s[0]])]
                      for s in val_fold],
        }

        train_graphs = [s[1] for s in train_fold]
        val_graphs   = [s[1] for s in val_fold]
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)

        # is_seen flags aligned with val_loader (shuffle=False preserves order)
        val_is_seen_flags = [bool(sample_is_seen[s[0]]) for s in val_fold]

        # Fresh model per fold
        model = create_poolhead_model(
            model_type='classifier',
            pool_input_dim=pool_input_dim,
            hidden_dim=ft_hdim,
            metadata=METADATA,
            dropout=test_cfg['dropout'],
            pool_mode=test_cfg.get('pool_mode', 'cross_attn'),
            cross_attn_queries=test_cfg.get('cross_attn_queries', 2),
            cross_attn_heads=test_cfg.get('cross_attn_heads', 4),
            classifier_layers=test_cfg.get('classifier_layers', 3),
            cls_grad_scale=test_cfg.get('cls_grad_scale', 1.0),
        ).to(device)

        if fold == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Pool+Head model: {total_params:,} total, {trainable_params:,} trainable")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=test_cfg['lr'],
            weight_decay=test_cfg['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=test_cfg.get('scheduler_t0', 10),
            T_mult=test_cfg.get('scheduler_t_mult', 2),
            eta_min=float(test_cfg.get('scheduler_eta_min', 1e-6)),
        )
        scaler = GradScaler(enabled=use_amp)

        # Per-fold checkpoint
        checkpoint_name = f"{base_name}_fold{fold_id}_best.pt"
        checkpoint_path = os.path.join(final_save_dir, checkpoint_name)
        early_stopping = FineTuneEarlyStopping(
            patience=test_cfg['patience'],
            save_path=checkpoint_path,
            mode='max',  # maximize classification metric (AUC)
        )

        # Dynamic clipping
        clip_min: float = 5.0
        clip_factor: float = 1.25
        current_clip: float = test_cfg.get('clip_max_norm', 5.0)

        fold_log = {
            'train_loss': [], 'val_loss': [],
            'train_metrics': [], 'val_metrics': [],
            'grad_norm': [], 'lr': [],
        }

        for epoch in range(test_cfg['n_epochs']):
            epoch_start = time.time()
            model.update_epoch(epoch)

            train_loss, grad_norm, train_metrics = train_epoch_classifier(
                model, train_loader, optimizer, scaler, criterion_cls, device,
                clip_max_norm=current_clip, use_amp=use_amp,
                label_smoothing=label_smoothing,
            )
            val_loss, val_metrics = _validate_classifier_seen_unseen(
                model, val_loader, val_is_seen_flags,
                criterion_cls, device, use_amp=use_amp,
                label_smoothing=label_smoothing,
            )
            if cls_metric_name not in val_metrics:
                raise KeyError(
                    f"cls_metric={cls_metric_name!r} not produced by validator. "
                    f"Available keys: {sorted(val_metrics.keys())}"
                )
            score = val_metrics[cls_metric_name]

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            if grad_norm is not None and math.isfinite(grad_norm) and grad_norm > 0:
                current_clip = max(clip_min, grad_norm * clip_factor)

            fold_log['train_loss'].append(float(train_loss))
            fold_log['val_loss'].append(float(val_loss))
            fold_log['train_metrics'].append({k: float(v) for k, v in train_metrics.items()})
            fold_log['val_metrics'].append({k: float(v) for k, v in val_metrics.items()})
            fold_log['grad_norm'].append(float(grad_norm) if grad_norm is not None else 0.0)
            fold_log['lr'].append(float(current_lr))

            epoch_time = time.time() - epoch_start
            # train_metrics only has base keys (auc/f1/...); unseen variants
            # exist only on the validation side. Fall back to plain 'auc' for
            # the training-side display when cls_metric is an unseen variant.
            train_disp_key = cls_metric_name if cls_metric_name in train_metrics else 'auc'
            print(f"  Epoch {epoch+1:3d} | "
                  f"Loss T/V: {train_loss:.3f}/{val_loss:.3f} | "
                  f"{train_disp_key}(T)/{cls_metric_name}(V): "
                  f"{train_metrics[train_disp_key]:.3f}/{val_metrics[cls_metric_name]:.3f} | "
                  f"AUC_un/AUPRC_un: "
                  f"{val_metrics['auc_unseen']:.3f}/{val_metrics['auprc_unseen']:.3f} | "
                  f"Time: {epoch_time:.1f}s")

            stop = early_stopping(score, model, epoch)
            if stop:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(best {cls_metric_name}={early_stopping.best_score:.3f} "
                      f"at epoch {early_stopping.best_epoch+1})")
                break

        # Restore best and save
        best_state = early_stopping.get_best_state()
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), checkpoint_path)

        fold_val_score[f'fold_{fold_id}'] = float(early_stopping.best_score)
        fold_log['best_val_score'] = float(early_stopping.best_score)
        fold_log['best_epoch']     = int(early_stopping.best_epoch + 1)
        fold_log['cls_metric']     = cls_metric_name

        # Snapshot val_metrics at best epoch (1-indexed best_epoch was stored)
        best_ep_zero = int(early_stopping.best_epoch)
        if 0 <= best_ep_zero < len(fold_log['val_metrics']):
            best_val_metrics = fold_log['val_metrics'][best_ep_zero]
            fold_log['best_val_metrics'] = best_val_metrics
            print(f"  Fold {fold_id} — Best val {cls_metric_name}: "
                  f"{early_stopping.best_score:.3f} | "
                  f"AUC_un={best_val_metrics.get('auc_unseen', 0.0):.3f} "
                  f"AUPRC_un={best_val_metrics.get('auprc_unseen', 0.0):.3f}  "
                  f"\u2192  {checkpoint_path}")
        else:
            print(f"  Fold {fold_id} — Best val {cls_metric_name}: "
                  f"{early_stopping.best_score:.3f}  \u2192  {checkpoint_path}")

        all_fold_logs[f'fold_{fold_id}'] = fold_log

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # 5. Aggregate & save
    # ---------------------------------------------------------------
    scores = [v for v in fold_val_score.values()]
    mean_score = float(np.mean(scores))
    std_score  = float(np.std(scores))

    # Aggregate unseen metrics across folds (taken at each fold's best epoch)
    unseen_auc_per_fold:   List[float] = []
    unseen_auprc_per_fold: List[float] = []
    for fk in [f'fold_{i+1}' for i in range(n_folds)]:
        bvm = all_fold_logs[fk].get('best_val_metrics', {})
        unseen_auc_per_fold.append(float(bvm.get('auc_unseen', 0.0)))
        unseen_auprc_per_fold.append(float(bvm.get('auprc_unseen', 0.0)))
    mean_auc_unseen   = float(np.mean(unseen_auc_per_fold))
    std_auc_unseen    = float(np.std(unseen_auc_per_fold))
    mean_auprc_unseen = float(np.mean(unseen_auprc_per_fold))
    std_auprc_unseen  = float(np.std(unseen_auprc_per_fold))

    print(f"\n{'='*60}")
    print(f"Discriminative Binder {n_folds}-Fold CV Complete  (split mode: {cv_split_mode})")
    print(f"{'='*60}")
    print(f"  Per-fold val {cls_metric_name}: {[f'{v:.3f}' for v in scores]}")
    print(f"  Mean ± Std ({cls_metric_name}):   {mean_score:.3f} ± {std_score:.3f}")
    print(f"  Mean ± Std (AUC_unseen):   {mean_auc_unseen:.3f} ± {std_auc_unseen:.3f}")
    print(f"  Mean ± Std (AUPRC_unseen): {mean_auprc_unseen:.3f} ± {std_auprc_unseen:.3f}")

    log_path   = os.path.join(final_save_dir, f"{base_name}_{n_folds}fold_log.json")
    split_path = os.path.join(final_save_dir, f"{base_name}_{n_folds}fold_splits.json")
    summary_path = os.path.join(final_save_dir, f"{base_name}_{n_folds}fold_summary.json")

    with open(log_path, 'w') as f:
        json.dump(all_fold_logs, f, indent=2)
    with open(split_path, 'w') as f:
        json.dump(fold_split_log, f, indent=2)
    with open(summary_path, 'w') as f:
        json.dump({
            'n_folds': n_folds,
            'seed': seed,
            'cv_split_mode': cv_split_mode,
            'embedding_type': embedding_type,
            'num_layers': test_cfg['num_layers'],
            'hidden_dim_power': test_cfg['hidden_dim_power'],
            'pool_mode': test_cfg.get('pool_mode', 'cross_attn'),
            'use_jk': use_jk,
            'jk_mode': jk_mode if use_jk else None,
            'cls_metric': cls_metric_name,
            'fold_val_score': fold_val_score,
            'mean_val_score': mean_score,
            'std_val_score': std_score,
            'fold_auc_unseen':   {f'fold_{i+1}': v for i, v in enumerate(unseen_auc_per_fold)},
            'fold_auprc_unseen': {f'fold_{i+1}': v for i, v in enumerate(unseen_auprc_per_fold)},
            'mean_auc_unseen':   mean_auc_unseen,
            'std_auc_unseen':    std_auc_unseen,
            'mean_auprc_unseen': mean_auprc_unseen,
            'std_auprc_unseen':  std_auprc_unseen,
            'n_pos_seen_total':   n_pos_seen,
            'n_pos_unseen_total': n_pos_unseen,
            'n_neg_total':        n_neg,
        }, f, indent=2)

    print(f"\n  Training log: {log_path}")
    print(f"  Fold splits : {split_path}")
    print(f"  Summary     : {summary_path}")
    print(f"  Checkpoints : {final_save_dir}/{base_name}_fold*.pt")

    return [
        os.path.join(final_save_dir, f"{base_name}_fold{i+1}_best.pt")
        for i in range(n_folds)
    ]


# ============================================================================
# Discriminative Binder fine-tuning (precomputed-embedding pipeline)
# ============================================================================
def run_disc_binder_finetuning(
    config: Dict,
    device: torch.device,
    ssl_checkpoint_path: Optional[str] = None,
    n_folds: Optional[int] = None,
):
    """
    Fine-tuning pipeline for binary classification (binder vs non-binder)
    with precomputed encoder embeddings.

    Two modes:
      * n_folds == 0 (default): single train/test split pipeline (SSL split based).
      * n_folds > 0: stratified K-fold CV on a single pool (all positives + all
        SWAP + half/half PrePPI & random_mut for the remainder); no held-out
        test set. Per-fold checkpoints are saved as
        `disc_binder{_embedding}{_pool}{_jkmode}_fold{k}_best.pt`.

    Pipeline (single-split mode):
    1. Load pretrained encoder (frozen, eval mode)
    2. Precompute node-level embeddings for all graphs (train + test)
    3. Store enriched embeddings in graph.x (replaces original features)
    4. Free encoder from memory
    5. Create pool+head classifier model (trainable, no encoder)
    6. Train pool+head on precomputed embeddings
    """
    print(f"{'='*80}\n")
    test_cfg = config['disc_binder']

    data_cfg = config['data']
    sys_cfg = config['system']

    # Resolve n_folds: explicit arg > config > 0 (single-split default)
    if n_folds is None:
        n_folds = int(test_cfg.get('n_folds', 0))

    # ---------------------------------------------------------------
    # Determine pretrained path (required for this pipeline)
    # ---------------------------------------------------------------
    if ssl_checkpoint_path is not None:
        pretrained_path = ssl_checkpoint_path
    else:
        pretrained_path = test_cfg.get('pretrained_path')

    if pretrained_path is None:
        raise ValueError(
            "pretrained_path is required for the precomputed-embedding pipeline. "
            "Provide via config['test']['pretrained_path'] or ssl_checkpoint_path argument."
        )
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

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
        + (f"_{test_cfg['strategy_type']}" if test_cfg['strategy_type'] != 'dynamic' else '')
        + (f"_additive" if test_cfg.get('message_style', 'gated_src') == 'additive' else '')
    )
    # JK-Net config
    use_jk = test_cfg.get('use_jk', False)
    jk_mode = test_cfg.get('jk_mode', 'mean')
    jk_suffix = f"_jk{jk_mode}" if use_jk else ""
    local_dir += jk_suffix
    final_save_dir = os.path.join(test_cfg['save_dir'], local_dir)
    os.makedirs(final_save_dir, exist_ok=True)

    use_amp = device.type == 'cuda'
    hidden_dim = 2 ** test_cfg['hidden_dim_power']

    print(f"\nDevice: {device}")
    print(f"Fine-tune Task: classifier (discriminative binder)")
    print(f"Embedding Type: {embedding_type}")
    print(f"Pool Mode: {test_cfg.get('pool_mode', 'cross_attn')}")
    print(f"Strategy Type: {test_cfg.get('strategy_type', 'dynamic')}")
    print(f"Pre-trained path: {pretrained_path}")
    print(f"Save directory: {final_save_dir}")
    print(f"n_folds        : {n_folds}  ({'CV mode' if n_folds > 0 else 'single split mode'})")

    # ---------------------------------------------------------------
    # CV mode: delegate to dedicated pipeline and return
    # ---------------------------------------------------------------
    if n_folds > 0:
        return _run_disc_binder_cv(
            test_cfg=test_cfg,
            data_cfg=data_cfg,
            sys_cfg=sys_cfg,
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
    # Load graph data (classifier: pos/neg binary loading)
    # ---------------------------------------------------------------
    ssl_save_dir = os.path.dirname(pretrained_path) if os.path.isfile(pretrained_path) else pretrained_path
    train_sttgs, test_sttgs, train_labels, test_labels = load_classifier_data(
        data_cfg=data_cfg,
        ssl_save_dir=ssl_save_dir,
        save_dir=final_save_dir,
        seed=sys_cfg.get('seed', 42),
        min_nodes=data_cfg.get('min_nodes', 5),
    )
    # Stop the program for testing purpose here, to check if the data is loaded correctly
    #breakpoint()
    # Stamp binary label into graph.y so downstream code is uniform
    for name, st in train_sttgs.items():
        st.protein_graph.y = torch.tensor(float(train_labels[name]))
    for name, st in test_sttgs.items():
        st.protein_graph.y = torch.tensor(float(test_labels[name]))

    # ---------------------------------------------------------------
    # Load ESM dict if needed (for +esm / +esm480 / only_esm / only_esm480)
    # ---------------------------------------------------------------
    esm_dict = None
    if embedding_type in ['+esm', '+esm480', 'only_esm', 'only_esm480']:
        raise ValueError(f"+esm/only_esm embedding types are not yet supported for classifier task.")

    # ---------------------------------------------------------------
    # Convert settings to samples (always 4-element tuples)
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

    # Count sample types
    n_pos_train = sum(1 for s in train_samples if s[2] > 0)
    n_neg_train = sum(1 for s in train_samples if s[2] <= 0)
    n_pos_test  = sum(1 for s in test_samples if s[2] > 0)
    n_neg_test  = sum(1 for s in test_samples if s[2] <= 0)
    print(f"\nClassifier train: {len(train_samples)} ({n_pos_train} pos / {n_neg_train} neg)")
    print(f"Classifier test : {len(test_samples)} ({n_pos_test} pos / {n_neg_test} neg)")

    if len(train_samples) == 0:
        raise ValueError("No training samples loaded.")

    # ---------------------------------------------------------------
    # Save split info for reproducibility
    # ---------------------------------------------------------------
    split_log = {
        'train': [[s[0], float(s[2])] for s in train_samples],
        'test':  [[s[0], float(s[2])] for s in test_samples],
    }
    split_name = (
        f"disc_binder"
        + (f"{'_'+embedding_type if embedding_type not in ['base', 'esm'] else ''}")
        + ('' if test_cfg.get('pool_mode', 'cross_attn') == 'cross_attn' else f"_{test_cfg['pool_mode']}")
        + "_splits.json"
    )
    split_path = os.path.join(final_save_dir, split_name)
    with open(split_path, 'w') as f:
        json.dump(split_log, f, indent=2)
    print(f"Split info saved to: {split_path}")

    # ---------------------------------------------------------------
    # Load pretrained encoder (frozen, eval mode)
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

    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters (frozen): {encoder_params:,}")

    # ---------------------------------------------------------------
    # Precompute embeddings in batches (replaces graph.x in place)
    # ---------------------------------------------------------------
    train_names = [s[0] for s in train_samples]
    train_graphs = [s[1] for s in train_samples]
    test_names = [s[0] for s in test_samples]
    test_graphs = [s[1] for s in test_samples]
    encoder.eval()
    print("\n--- Precomputing train set embeddings ---")
    precompute_embeddings(
        encoder, train_names, train_graphs, device,
        batch_size=test_cfg['batch_size'],
        embedding_type=embedding_type,
        esm_dict=esm_dict,
        use_amp=use_amp,
        use_jk=use_jk,
        jk_mode=jk_mode,
    )
    print("--- Precomputing test set embeddings ---")
    precompute_embeddings(
        encoder, test_names, test_graphs, device,
        batch_size=test_cfg['batch_size'],
        embedding_type=embedding_type,
        esm_dict=esm_dict,
        use_amp=use_amp,
        use_jk=use_jk,
        jk_mode=jk_mode,
    )

    # Free encoder from memory
    del encoder
    torch.cuda.empty_cache()
    print("Encoder freed from memory. Training with precomputed embeddings.\n")

    # ---------------------------------------------------------------
    # Create data loaders (graphs now have precomputed x)
    # ---------------------------------------------------------------
    train_graphs = [s[1] for s in train_samples]
    test_graphs = [s[1] for s in test_samples]

    train_loader = DataLoader(train_graphs, batch_size=test_cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=test_cfg['batch_size'], shuffle=False)

    # ---------------------------------------------------------------
    # Create pool+head model (no encoder)
    # ---------------------------------------------------------------
    pool_input_dim = get_pool_input_dim(
        hidden_dim, embedding_type,
        use_jk=use_jk, jk_mode=jk_mode,
        num_hgt_layers=test_cfg['num_layers'],
    )

    # Dynamic ft_hdim: if not set in config, use pool_input_dim // 4 (min 256)
    ft_hdim = test_cfg.get('ft_hdim', None)
    if ft_hdim is None:
        ft_hdim = max(256, pool_input_dim // 4)
    ft_hdim = min(ft_hdim, pool_input_dim // 2)
    model = create_poolhead_model(
        model_type='classifier',
        pool_input_dim=pool_input_dim,
        hidden_dim=ft_hdim,
        metadata=METADATA,
        dropout=test_cfg['dropout'],
        pool_mode=test_cfg.get('pool_mode', 'cross_attn'),
        cross_attn_queries=test_cfg.get('cross_attn_queries', 2),
        cross_attn_heads=test_cfg.get('cross_attn_heads', 4),
        classifier_layers=test_cfg.get('classifier_layers', 3),
        cls_grad_scale=test_cfg.get('cls_grad_scale', 1.0),
    ).to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Pool+Head model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Pool input dim: {pool_input_dim} (ft_hdim={ft_hdim})")
    if use_jk:
        print(f"JK-Net enabled: mode={jk_mode}")

    # ---------------------------------------------------------------
    # Optimizer (all pool+head parameters are trainable)
    # ---------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=test_cfg['lr'],
        weight_decay=test_cfg['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=test_cfg.get('scheduler_t0', 10),
        T_mult=test_cfg.get('scheduler_t_mult', 2),
        eta_min=float(test_cfg.get('scheduler_eta_min', 1e-6))
    )

    scaler = GradScaler(enabled=use_amp)

    # ---------------------------------------------------------------
    # Loss criteria
    # ---------------------------------------------------------------
    criterion_cls = nn.BCEWithLogitsLoss()

    # ---------------------------------------------------------------
    # Early stopping
    # ---------------------------------------------------------------
    checkpoint_name = (
        f"disc_binder"
        + (f"{'_'+embedding_type if embedding_type not in ['base', 'esm'] else ''}")
        + ('' if test_cfg.get('pool_mode', 'cross_attn') == 'cross_attn' else f"_{test_cfg['pool_mode']}")
        + jk_suffix
        + "_best.pt"
    )
    checkpoint_path = os.path.join(final_save_dir, checkpoint_name)

    early_stopping = FineTuneEarlyStopping(
        patience=test_cfg['patience'],
        save_path=checkpoint_path,
        mode='max',  # maximize classification metric (AUC)
    )

    # Dynamic clipping state
    clip_min: float = 5.0
    clip_factor: float = 1.25
    current_clip: float = test_cfg.get('clip_max_norm', 5.0)

    # Training log
    training_log = {
        'train_loss': [], 'test_loss': [],
        'train_metrics': [], 'test_metrics': [],
        'grad_norm': [], 'lr': [],
    }

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Starting Training (classifier, precomputed-embedding pipeline)")
    print(f"{'='*60}")

    cls_metric_name = test_cfg.get('cls_metric', 'auc')
    label_smoothing = test_cfg.get('label_smoothing', 0.0)

    for epoch in range(test_cfg['n_epochs']):
        start_time = time.time()

        model.update_epoch(epoch)

        # Train
        train_loss, grad_norm, train_metrics = train_epoch_classifier(
            model, train_loader, optimizer, scaler, criterion_cls, device,
            clip_max_norm=current_clip, use_amp=use_amp,
            label_smoothing=label_smoothing,
        )
        # Evaluate on test
        test_loss, test_metrics = validate_classifier(
            model, test_loader, criterion_cls, device, use_amp=use_amp,
            label_smoothing=label_smoothing,
        )
        score = test_metrics[cls_metric_name]
        metric_str = (f"{cls_metric_name} Train/Test: "
                      f"{train_metrics[cls_metric_name]:.3f}/{test_metrics[cls_metric_name]:.3f}")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Update dynamic clip for next epoch
        if grad_norm is not None and math.isfinite(grad_norm) and grad_norm > 0:
            next_clip = grad_norm * clip_factor
            next_clip = max(clip_min, next_clip)
            current_clip = next_clip

        # Log
        training_log['train_loss'].append(train_loss)
        training_log['test_loss'].append(test_loss)
        training_log['train_metrics'].append(train_metrics)
        training_log['test_metrics'].append(test_metrics)
        training_log['grad_norm'].append(grad_norm)
        training_log['lr'].append(current_lr)

        # Early stopping
        stop = early_stopping(score, model, epoch)

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1:3d} | "
              f"Loss Train/Test: {train_loss:.3f}/{test_loss:.3f} | "
              f"{metric_str} | "
              f"Time: {epoch_time:.1f}s")

        if stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best score: {early_stopping.best_score:.3f} at epoch {early_stopping.best_epoch+1}")

            # Save best model
            best_state = early_stopping.get_best_state()
            if best_state is not None:
                model.load_state_dict(best_state)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved to: {checkpoint_path}")

            break

    # ---------------------------------------------------------------
    # Save training log
    # ---------------------------------------------------------------
    log_name = (
        f"disc_binder"
        + (f"{'_'+embedding_type if embedding_type not in ['base', 'esm'] else ''}")
        + ('' if test_cfg.get('pool_mode', 'cross_attn') == 'cross_attn' else f"_{test_cfg['pool_mode']}")
        + jk_suffix
        + "_log.json"
    )
    log_path = os.path.join(final_save_dir, log_name)

    serializable_log = {}
    for k, v in training_log.items():
        if k in ['train_metrics', 'test_metrics']:
            serializable_log[k] = [{kk: float(vv) for kk, vv in m.items()} for m in v]
        else:
            serializable_log[k] = [
                float(x) if isinstance(x, (np.floating, float)) else x for x in v
            ]

    with open(log_path, 'w') as f:
        json.dump(serializable_log, f, indent=2)

    print(f"\nDiscriminative Binder Fine-tuning Complete!")
    print(f"Training log: {log_path}")
    print(f"Best model: {checkpoint_path}")

    return checkpoint_path


# ============================================================================
# MAIN FOR STANDALONE EXECUTION
# ============================================================================
if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description='Discriminative Binder (Classifier) Fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Example:
    python finetune_disc_binder_module.py -config config_test.yaml
    python finetune_disc_binder_module.py -config config_test.yaml -nfold 5
        """
    )
    parser.add_argument('-config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('-nfold', type=int, default=None,
                       help='If > 0, run StratifiedKFold CV with this many folds '
                            'on a single pool (no held-out test set). '
                            'If 0 or omitted, use config[disc_binder][n_folds] '
                            '(default 0 → single train/test split pipeline).')

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
    print("DISCRIMINATIVE BINDER FINE-TUNING PIPELINE")
    print(f"{'='*80}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Seed: {config['system']['seed']}")

    # Run discriminative binder fine-tuning
    run_disc_binder_finetuning(config, device, n_folds=args.nfold)

    print(f"\n{'='*80}")
    print("ALL TRAINING COMPLETE!")
    print(f"{'='*80}\n")
