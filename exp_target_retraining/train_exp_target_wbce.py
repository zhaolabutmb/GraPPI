#!/usr/bin/env python3
"""Leave-one-target-out weighted-BCE retraining of the GraPPI binder classifier.

Design (see README.md):

* Leave-one-target-out (LOTO) CV: one fold per protein target. In each fold the
  held-out target is the validation set and all other targets are training data,
  so every target is validated exactly once (15 folds for the experimental set).
* Loss is weighted BCE: ``BCEWithLogitsLoss(pos_weight = N_neg / N_pos)`` with a
  fixed global weight (3275/394 = 8.3122). Labels are hard 0/1.
* Each fold early-stops / selects on its held-out target's AUPRC and reports that
  target's AUROC + AUPRC. Results are aggregated as median + IQR across targets,
  plus a pooled AUROC/AUPRC over all held-out samples. Raw per-sample logits and
  probabilities are saved so any other metric can be recomputed later.

The encoder stays frozen and its embeddings are precomputed once. Everything
reuses the published training internals (``load_pretrained_encoder``,
``precompute_embeddings``, ``create_poolhead_model``, ``FineTuneEarlyStopping``,
``train_epoch_classifier``). The only genuinely new logic is the leave-one-target-out
split, the WBCE criterion wiring, and the AUROC/AUPRC + raw-score reporting.

With --mode esm, the encoder is bypassed and the same classifier head consumes
only the ESM columns already stored in the graph cache (no structural features).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Pure split logic (no torch / graph dependencies, unit-testable)
# ---------------------------------------------------------------------------
def make_loto_folds(
    target_ids: Sequence[str],
    labels: Optional[Sequence[int]] = None,
) -> List[Dict]:
    """Leave-one-target-out (LOTO) folds — one fold per unique target.

    In each fold the held-out target's samples are the validation set and every
    other target's samples are the training set, so every target is validated
    exactly once. Targets are iterated in sorted order for determinism.

    Returns a list of dicts with ``target``, ``train_idx`` and ``val_idx``
    (global index arrays). If ``labels`` is given, raises when holding out a
    target would leave a single-class training pool (guards misuse; this never
    happens on the experimental set because positives span many targets).
    """
    target_ids = list(target_ids)
    unique = sorted(set(target_ids))
    if len(unique) < 2:
        raise ValueError(f'Need >=2 targets for leave-one-target-out, got {unique}.')
    folds: List[Dict] = []
    for target in unique:
        val_idx = np.array([i for i, t in enumerate(target_ids) if t == target], dtype=int)
        train_idx = np.array([i for i, t in enumerate(target_ids) if t != target], dtype=int)
        if labels is not None:
            train_classes = {int(labels[i]) for i in train_idx}
            if len(train_classes) < 2:
                raise ValueError(
                    f'Holding out {target!r} leaves a single-class training pool.')
        folds.append({'target': target, 'train_idx': train_idx, 'val_idx': val_idx})
    return folds


def resolve_pos_weight(spec, n_pos: int, n_neg: int) -> float:
    """Resolve ``train.pos_weight`` ('global' -> N_neg/N_pos, else a float)."""
    if isinstance(spec, str) and spec.strip().lower() == 'global':
        if n_pos == 0:
            raise ValueError('Cannot compute global pos_weight with zero positives.')
        return n_neg / n_pos
    return float(spec)


# ---------------------------------------------------------------------------
# Training orchestration (imports torch lazily so the split logic stays testable)
# ---------------------------------------------------------------------------
def _load_config(path: Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def _abs(data_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (data_dir / p)


def resolve_device(config_cuda_id: int, cuda_id: Optional[int] = None):
    """Choose a visible CUDA device, or retain the default CPU fallback."""
    import torch

    selected_id = config_cuda_id if cuda_id is None else cuda_id
    if selected_id < 0:
        raise ValueError(f'CUDA device ID must be non-negative, got {selected_id}.')
    if not torch.cuda.is_available():
        if cuda_id is not None:
            raise RuntimeError(
                f'CUDA device {cuda_id} was requested, but CUDA is not available.')
        return torch.device('cpu')
    count = torch.cuda.device_count()
    if selected_id >= count:
        raise ValueError(
            f'CUDA device {selected_id} is unavailable: {count} device(s) visible. '
            'Device IDs are relative to CUDA_VISIBLE_DEVICES.')
    return torch.device('cuda', selected_id)


def evaluate(model, graphs, device, batch_size: int):
    """Return (y_true, y_prob, y_logit) numpy arrays for precomputed graphs.

    Order matches the input ``graphs`` list (loader uses ``shuffle=False``), so
    callers can zip the results back to the corresponding sample names. Raw
    logits are returned so any later metric can be recomputed without retraining.
    """
    import torch
    from torch_geometric.loader import DataLoader

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    y_true, y_prob, y_logit = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch).squeeze(-1).float()
            y_logit.extend(logits.cpu().tolist())
            y_prob.extend(torch.sigmoid(logits).cpu().tolist())
            y_true.extend((batch.y > 0).float().cpu().tolist())
    return np.asarray(y_true), np.asarray(y_prob), np.asarray(y_logit)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """AUROC + AUPRC, guarded for the single-class edge case."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = {'n': int(y_true.size),
           'n_pos': int((y_true == 1).sum()),
           'n_neg': int((y_true == 0).sum())}
    if len(np.unique(y_true)) > 1:
        out['auroc'] = float(roc_auc_score(y_true, y_prob))
        out['auprc'] = float(average_precision_score(y_true, y_prob))
    else:
        out['auroc'] = float('nan')
        out['auprc'] = float('nan')
    return out


def prepare_esm_graphs(graphs: Sequence, esm_dim: int) -> int:
    """Reuse cached ESM features, excluding the five structural columns."""
    if esm_dim not in (1280, 480):
        raise ValueError(f'Unsupported ESM dimension: {esm_dim}')
    for graph in graphs:
        for node_type in ('receptor', 'ligand'):
            x = graph[node_type].x
            if x.ndim != 2 or x.shape[1] != esm_dim + 5 or x.shape[0] == 0:
                raise ValueError(
                    f'{graph.name}: expected nonempty {node_type} features with '
                    f'{esm_dim + 5} columns in the ESM graph cache, got {tuple(x.shape)}')
    for graph in graphs:
        for node_type in ('receptor', 'ligand'):
            graph[node_type].x = graph[node_type].x[:, :esm_dim].float()
    return esm_dim


def run(config_path: Path, mode: Optional[str] = None,
        output_dir: Optional[Path] = None, cuda_id: Optional[int] = None) -> dict:
    import torch
    import torch.nn as nn
    from torch.amp import GradScaler
    from torch_geometric.loader import DataLoader

    from utils.Training_modules.common_utils import (
        MODEL_INIT_DIM, EDGE_IN_DIM, METADATA, set_seed, load_ssl_config,
    )
    from model_collection.FineTuneModels import (
        load_pretrained_encoder, precompute_embeddings,
        get_pool_input_dim, create_poolhead_model,
    )
    from utils.Training_modules.save_training import FineTuneEarlyStopping
    from utils.Training_modules.finetune_training import train_epoch_classifier

    cfg = _load_config(config_path)
    sys_cfg, data_cfg = cfg['system'], cfg['data']
    enc_cfg_in, head_cfg = cfg['encoder'], cfg['head']
    train_cfg, out_cfg = cfg['train'], cfg['output']
    mode = mode or enc_cfg_in.get('mode', 'grappi')
    if mode not in ('grappi', 'esm'):
        raise ValueError(f'Unknown encoder mode: {mode!r}')
    esm_dim = int(enc_cfg_in.get('esm_dim', 1280))

    device = resolve_device(int(sys_cfg.get('cuda_id', 0)), cuda_id)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        print(f'Using device: {device}')
    else:
        print('CUDA is unavailable; using CPU.')

    seed = int(sys_cfg.get('seed', 42))
    set_seed(seed)
    data_dir = _abs(REPO_ROOT, data_cfg['data_dir']).resolve()
    out_dir = _abs(REPO_ROOT, out_cfg['dir']).resolve()
    if output_dir is not None:
        out_dir = _abs(REPO_ROOT, str(output_dir)).resolve()
    elif mode == 'esm':
        out_dir = out_dir.with_name(f'esm{esm_dim}_exp_target_loto_wbce')
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- cached graphs ----
    cache_path = _abs(data_dir, data_cfg['graph_cache'])
    if not cache_path.is_file():
        raise FileNotFoundError(
            f'Graph cache not found: {cache_path}\nRun cache_graphs.py first.')
    with open(cache_path, 'rb') as handle:
        cached: Dict[str, object] = pickle.load(handle)
    names = sorted(cached.keys())
    graphs = [cached[n] for n in names]
    target_ids = [str(cached[n].target_id) for n in names]
    labels = [int(cached[n].y.item() > 0) for n in names]
    n_pos, n_neg = int(sum(labels)), int(len(labels) - sum(labels))
    print(f'Loaded {len(names)} cached graphs: {n_pos} pos / {n_neg} neg '
          f'across {len(set(target_ids))} targets.')

    pos_weight = resolve_pos_weight(train_cfg.get('pos_weight', 'global'), n_pos, n_neg)
    print(f'WBCE pos_weight = {pos_weight:.4f}')

    # ---- leave-one-target-out folds ----
    folds = make_loto_folds(target_ids, labels)
    print(f"Leave-one-target-out CV: {len(folds)} folds "
          f"(each of {len(folds)} targets is the validation set exactly once).")

    use_amp = device.type == 'cuda'
    enc_dir = None
    if mode == 'esm':
        embedding_type = 'only_esm' if esm_dim == 1280 else 'only_esm480'
        pool_input_dim = prepare_esm_graphs(graphs, esm_dim)
        print(f'ESM-only baseline: reusing {esm_dim} cached ESM columns; '
              'no GraPPI encoder or ESM forward pass.')
    else:
        enc_dir = _abs(REPO_ROOT, enc_cfg_in['dir']).resolve()
        ckpt = enc_dir / 'ssl_edge_best.pt'
        ssl = load_ssl_config(str(ckpt)) or {}
        embedding_type = enc_cfg_in.get('embedding_type', ssl.get('embedding_type', 'esm'))
        num_layers = int(ssl.get('num_layers', 2))
        hidden_dim = 2 ** int(ssl.get('hidden_dim_power', 10))
        hgt_heads = int(ssl.get('hgt_heads', 4))
        message_style = ssl.get('message_style', 'gated_src')
        use_jk = bool(head_cfg.get('use_jk', True))
        jk_mode = head_cfg.get('jk_mode', 'mean')
        encoder = load_pretrained_encoder(
            checkpoint_path=str(ckpt),
            node_in_dim=MODEL_INIT_DIM[embedding_type],
            edge_in_dim=EDGE_IN_DIM, metadata=METADATA,
            hidden_dim=hidden_dim, num_hgt_layers=num_layers, hgt_heads=hgt_heads,
            device=device, message_style=message_style,
        )
        encoder.eval()
        precompute_embeddings(
            encoder, names, graphs, device,
            batch_size=int(train_cfg.get('batch_size', 50)),
            embedding_type=embedding_type, esm_dict=None,
            use_amp=use_amp, use_jk=use_jk, jk_mode=jk_mode,
        )
        del encoder
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        pool_input_dim = get_pool_input_dim(
            hidden_dim, embedding_type, use_jk=use_jk, jk_mode=jk_mode,
            num_hgt_layers=num_layers,
        )

    def new_head():
        return create_poolhead_model(
            model_type='classifier', pool_input_dim=pool_input_dim,
            hidden_dim=int(head_cfg.get('ft_hdim', 256)), metadata=METADATA,
            dropout=float(head_cfg.get('dropout', 0.4)),
            pool_mode=head_cfg.get('pool_mode', 'cross_attn'),
            cross_attn_queries=int(head_cfg.get('cross_attn_queries', 2)),
            cross_attn_heads=int(head_cfg.get('cross_attn_heads', 4)),
            classifier_layers=int(head_cfg.get('classifier_layers', 2)),
            cls_grad_scale=float(head_cfg.get('cls_grad_scale', 1.0)),
        ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device))
    select_metric = train_cfg.get('select_metric', 'auprc')
    batch_size = int(train_cfg.get('batch_size', 50))
    label_smoothing = float(train_cfg.get('label_smoothing', 0.0))

    fold_records = []
    raw_rows = []  # per-sample held-out scores across all folds
    split_log = {'protocol': 'leave_one_target_out', 'folds': {}}

    for fold, spec in enumerate(folds, start=1):
        held = spec['target']
        train_idx, val_idx = spec['train_idx'], spec['val_idx']
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        val_names = [names[i] for i in val_idx]
        n_vpos = int(sum(labels[i] for i in val_idx))
        n_vneg = int(len(val_idx) - n_vpos)
        split_log['folds'][held] = {
            'n_train': int(len(train_idx)), 'n_val': int(len(val_idx)),
            'val_pos': n_vpos, 'val_neg': n_vneg,
        }
        print(f"\n{'='*56}\nFOLD {fold}/{len(folds)} | held-out target: {held} "
              f"({len(val_idx)} samples, {n_vpos} pos / {n_vneg} neg)\n{'='*56}")
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)

        model = new_head()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(train_cfg.get('lr', 2e-4)),
            weight_decay=float(train_cfg.get('weight_decay', 0.02)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=int(train_cfg.get('scheduler_t0', 10)),
            T_mult=int(train_cfg.get('scheduler_t_mult', 2)),
            eta_min=float(train_cfg.get('scheduler_eta_min', 1e-6)))
        scaler = GradScaler(enabled=use_amp)

        checkpoint_prefix = 'disc_binder' if mode == 'grappi' else 'disc_binder_esm'
        ckpt_path = out_dir / f'{checkpoint_prefix}_exp_target_loto_{held}_best.pt'
        stopper = FineTuneEarlyStopping(
            patience=int(train_cfg.get('patience', 20)),
            save_path=str(ckpt_path), mode='max')

        for epoch in range(int(train_cfg.get('n_epochs', 1000))):
            t0 = time.time()
            if hasattr(model, 'update_epoch'):
                model.update_epoch(epoch)
            train_loss, _, _ = train_epoch_classifier(
                model, train_loader, optimizer, scaler, criterion, device,
                clip_max_norm=float(train_cfg.get('clip_max_norm', 5.0)),
                use_amp=use_amp, label_smoothing=label_smoothing)
            scheduler.step()

            yv, pv, _ = evaluate(model, val_graphs, device, batch_size)
            val_metrics = binary_metrics(yv, pv)
            score = val_metrics[select_metric]
            print(f'  Epoch {epoch+1:3d} | loss {train_loss:.3f} | '
                  f'val AUROC {val_metrics["auroc"]:.3f} AUPRC {val_metrics["auprc"]:.3f} | '
                  f'{time.time()-t0:.1f}s')
            if stopper(score, model, epoch):
                print(f'  Early stop at epoch {epoch+1} '
                      f'(best val {select_metric}={stopper.best_score:.3f})')
                break

        best_state = stopper.get_best_state()
        if best_state is not None:
            model.load_state_dict(best_state)

        yv, pv, lv = evaluate(model, val_graphs, device, batch_size)
        rec = {
            'fold': fold,
            'target': held,
            'checkpoint': str(ckpt_path),
            'best_epoch': int(stopper.best_epoch + 1),
            'best_val_select': float(stopper.best_score),
            'select_metric': select_metric,
            'metrics': binary_metrics(yv, pv),
        }
        fold_records.append(rec)
        for nm, yt, pr, lg in zip(val_names, yv.tolist(), pv.tolist(), lv.tolist()):
            raw_rows.append({'name': nm, 'target_id': held, 'label': int(yt),
                             'logit': float(lg), 'prob': float(pr)})
        print(f"  {held}: AUROC {rec['metrics']['auroc']:.4f} "
              f"AUPRC {rec['metrics']['auprc']:.4f} (best epoch {rec['best_epoch']})")
        del model, optimizer, scheduler, scaler
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    def agg(metric):
        vals = np.array([r['metrics'][metric] for r in fold_records
                         if not np.isnan(r['metrics'][metric])], dtype=float)
        if vals.size == 0:
            return {'median': float('nan'), 'iqr': float('nan'),
                    'q25': float('nan'), 'q75': float('nan'),
                    'mean': float('nan'), 'std': float('nan'), 'n': 0}
        q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        return {'median': float(np.median(vals)), 'iqr': q75 - q25,
                'q25': q25, 'q75': q75, 'mean': float(vals.mean()),
                'std': float(vals.std()), 'n': int(vals.size)}

    # Pooled metrics over all held-out samples (robust for tiny-positive targets,
    # analogous to the paper's pooled PR/ROC panels).
    pooled_y = np.array([r['label'] for r in raw_rows], dtype=float)
    pooled_p = np.array([r['prob'] for r in raw_rows], dtype=float)
    pooled = binary_metrics(pooled_y, pooled_p)

    summary = {
        'protocol': 'leave_one_target_out',
        'seed': seed,
        'device': str(device),
        'n_folds': len(folds),
        'n_targets': len(folds),
        'pos_weight': pos_weight,
        'encoder_mode': mode,
        'embedding_type': embedding_type,
        'encoder_dir': str(enc_dir) if enc_dir is not None else None,
        'graph_cache': str(cache_path),
        'pool_input_dim': pool_input_dim,
        'head_config': head_cfg,
        'select_metric': select_metric,
        'label_smoothing': label_smoothing,
        'n_pos_total': n_pos, 'n_neg_total': n_neg,
        'per_target': fold_records,
        'across_targets': {'auroc': agg('auroc'), 'auprc': agg('auprc')},
        'pooled': pooled,
    }
    with open(out_dir / 'exp_target_loto_summary.json', 'w') as handle:
        json.dump(summary, handle, indent=2)
    with open(out_dir / 'exp_target_loto_splits.json', 'w') as handle:
        json.dump(split_log, handle, indent=2)

    # Raw per-sample scores so any later metric (F1, precision@recall, custom
    # thresholds, pooled curves) can be recomputed without retraining.
    raw_csv = out_dir / 'exp_target_loto_raw_scores.csv'
    with open(raw_csv, 'w', newline='') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['name', 'target_id', 'label', 'logit', 'prob'])
        writer.writeheader()
        writer.writerows(raw_rows)

    # Tidy per-target table for the report.
    pt_csv = out_dir / 'exp_target_loto_per_target.csv'
    with open(pt_csv, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['target', 'n', 'n_pos', 'n_neg', 'auroc', 'auprc', 'best_epoch'])
        for r in fold_records:
            m = r['metrics']
            writer.writerow([r['target'], m['n'], m['n_pos'], m['n_neg'],
                             f"{m['auroc']:.6f}", f"{m['auprc']:.6f}", r['best_epoch']])

    a_roc, a_prc = summary['across_targets']['auroc'], summary['across_targets']['auprc']
    print(f"\n{'='*56}\nLeave-one-target-out over {len(fold_records)} targets")
    print(f"  AUROC  median {a_roc['median']:.4f} (IQR {a_roc['q25']:.4f}-{a_roc['q75']:.4f}) "
          f"| mean {a_roc['mean']:.4f} +/- {a_roc['std']:.4f}")
    print(f"  AUPRC  median {a_prc['median']:.4f} (IQR {a_prc['q25']:.4f}-{a_prc['q75']:.4f}) "
          f"| mean {a_prc['mean']:.4f} +/- {a_prc['std']:.4f}")
    print(f"  Pooled AUROC {pooled['auroc']:.4f} | AUPRC {pooled['auprc']:.4f} "
          f"({pooled['n']} samples, {pooled['n_pos']} pos)")
    print(f"Wrote summary     -> {out_dir / 'exp_target_loto_summary.json'}")
    print(f"Wrote raw scores  -> {raw_csv}")
    print(f"Wrote per-target  -> {pt_csv}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).with_name('config_exp_target_wbce.yaml'))
    parser.add_argument('--mode', choices=('grappi', 'esm'),
                        help='Encoder mode; esm reuses cached ESM features without the GNN.')
    parser.add_argument('--cuda-id', type=int,
                        help='Visible CUDA device index; overrides system.cuda_id in the config.')
    parser.add_argument('--output-dir', type=Path,
                        help='Override the mode-specific output directory.')
    args = parser.parse_args()
    run(args.config, mode=args.mode, output_dir=args.output_dir, cuda_id=args.cuda_id)


if __name__ == '__main__':
    main()
