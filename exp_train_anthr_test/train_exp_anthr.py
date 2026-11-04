#!/usr/bin/env python3
"""Train the GraPPI binder classifier on the full experimental set, test on Anthropic.

Design (see README.md):

* Training pool: all 3,669 labeled designs in ``exp_binder_data`` — no held-out
  target, no internal split. The LOTO protocol in ``exp_target_retraining`` is a
  separate study; this run uses every experimental sample.
* Test set: the ``anthr_binder_data`` designs that have a prepared structure
  (1,189 of 1,320 labeled). The 131 without a released ``designed.cif`` are
  skipped, not scored as negatives.
* Loss is weighted BCE: ``BCEWithLogitsLoss(pos_weight = N_neg / N_pos)`` with
  the weight computed on the TRAINING set only (3275/394 = 8.3122). Hard 0/1
  labels.
* The encoder (``2layers_10hdim_esm``) stays frozen; its embeddings are
  precomputed once for both datasets. Only the cross-attention pooling and the
  MLP head are trained.
* ``--mode esm`` bypasses the GNN and feeds the same head the 1,280-dim ESM
  columns already in the graph caches — identical head config, identical folds,
  identical loss, so the two rows are comparable.

LIMITATION, stated up front: ``train.select_on: test`` means the Anthropic
labels drive early stopping and best-epoch selection. The reported Anthropic
metrics are therefore optimistic and are NOT a blind-test estimate. This
mirrors the known limitation of the LOTO study. Set ``select_on: train`` for a
selection-free run (fixed epoch budget on the training loss) if a blind number
is needed later.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _path in (str(REPO_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from anthr_eval import (  # noqa: E402
    attach_metadata, binary_metrics, load_test_metadata, print_summary,
    summarize, write_json, write_scores, write_stratum_tables,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def resolve_pos_weight(spec, n_pos: int, n_neg: int) -> float:
    """Resolve ``train.pos_weight`` ('global' -> N_neg/N_pos, else a float)."""
    if isinstance(spec, str) and spec.strip().lower() == 'global':
        if n_pos == 0:
            raise ValueError('Cannot compute global pos_weight with zero positives.')
        return n_neg / n_pos
    return float(spec)


def check_disjoint(train_names: Sequence[str], test_names: Sequence[str]) -> List[str]:
    """Design ids present in both datasets (expected empty; reported if not)."""
    return sorted(set(train_names) & set(test_names))


def _load_config(path: Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def _abs(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


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


def load_cache(path: Path):
    """Load a {name: graph} pickle; return (names, graphs, targets, labels)."""
    if not path.is_file():
        raise FileNotFoundError(f'Graph cache not found: {path}')
    with open(path, 'rb') as handle:
        cached: Dict[str, object] = pickle.load(handle)
    if not cached:
        raise ValueError(f'Graph cache is empty: {path}')
    names = sorted(cached.keys())
    graphs = [cached[n] for n in names]
    targets = [str(cached[n].target_id) for n in names]
    labels = [int(float(cached[n].y) > 0) for n in names]
    return names, graphs, targets, labels


def prepare_esm_graphs(graphs: Sequence, esm_dim: int) -> int:
    """Reuse cached ESM features, dropping the five structural columns."""
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


def evaluate(model, graphs, device, batch_size: int):
    """Return (y_true, y_prob, y_logit) in the order of ``graphs``."""
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


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
def run(config_path: Path, mode: Optional[str] = None,
        output_dir: Optional[Path] = None, cuda_id: Optional[int] = None,
        seed: Optional[int] = None) -> dict:
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
    train_data_cfg, test_data_cfg = data_cfg['train'], data_cfg['test']

    mode = mode or enc_cfg_in.get('mode', 'grappi')
    if mode not in ('grappi', 'esm'):
        raise ValueError(f'Unknown encoder mode: {mode!r}')
    esm_dim = int(enc_cfg_in.get('esm_dim', 1280))

    device = resolve_device(int(sys_cfg.get('cuda_id', 0)), cuda_id)
    print(f'Using device: {device}' if device.type == 'cuda'
          else 'CUDA is unavailable; using CPU.')

    run_seed = int(sys_cfg.get('seed', 42)) if seed is None else int(seed)
    set_seed(run_seed)

    out_dir = _abs(REPO_ROOT, out_cfg['dir']).resolve()
    if output_dir is not None:
        out_dir = _abs(REPO_ROOT, str(output_dir)).resolve()
    elif mode == 'esm':
        out_dir = out_dir.with_name(f'esm{esm_dim}_exp_train_anthr_test')
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- caches ----
    train_dir = _abs(REPO_ROOT, train_data_cfg['data_dir']).resolve()
    test_dir = _abs(REPO_ROOT, test_data_cfg['data_dir']).resolve()
    train_cache = _abs(train_dir, train_data_cfg['graph_cache'])
    test_cache = _abs(test_dir, test_data_cfg['graph_cache'])

    tr_names, tr_graphs, tr_targets, tr_labels = load_cache(train_cache)
    te_names, te_graphs, te_targets, te_labels = load_cache(test_cache)
    n_pos, n_neg = int(sum(tr_labels)), int(len(tr_labels) - sum(tr_labels))
    print(f'Train (exp):  {len(tr_names)} graphs, {n_pos} pos / {n_neg} neg, '
          f'{len(set(tr_targets))} targets.')
    print(f'Test (anthr): {len(te_names)} graphs, {int(sum(te_labels))} pos / '
          f'{int(len(te_labels) - sum(te_labels))} neg, '
          f'{len(set(te_targets))} targets.')

    overlap = check_disjoint(tr_names, te_names)
    if overlap:
        print(f'[warn] {len(overlap)} design id(s) appear in both datasets: '
              f'{overlap[:5]}')
    shared_targets = sorted(set(tr_targets) & set(te_targets))
    if shared_targets:
        print(f'[note] target names common to both sets: {shared_targets}')

    # ---- test-set stratification metadata ----
    strata = list(test_data_cfg.get('strata', ['target']))
    meta = load_test_metadata(
        _abs(test_dir, test_data_cfg.get('labels_csv', 'labels.csv')),
        test_data_cfg.get('id_column', 'binder_id'), strata)

    pos_weight = resolve_pos_weight(train_cfg.get('pos_weight', 'global'), n_pos, n_neg)
    print(f'WBCE pos_weight = {pos_weight:.4f} (training set only)')

    use_amp = device.type == 'cuda'
    enc_dir = None
    if mode == 'esm':
        embedding_type = 'only_esm' if esm_dim == 1280 else 'only_esm480'
        prepare_esm_graphs(tr_graphs, esm_dim)
        pool_input_dim = prepare_esm_graphs(te_graphs, esm_dim)
        print(f'ESM-only baseline: reusing {esm_dim} cached ESM columns for both '
              'datasets; no GraPPI encoder and no ESM forward pass.')
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
        print(f'Frozen encoder: {enc_dir.name} '
              f'({num_layers} HGT layers, hidden {hidden_dim}, {embedding_type})')
        encoder = load_pretrained_encoder(
            checkpoint_path=str(ckpt),
            node_in_dim=MODEL_INIT_DIM[embedding_type],
            edge_in_dim=EDGE_IN_DIM, metadata=METADATA,
            hidden_dim=hidden_dim, num_hgt_layers=num_layers, hgt_heads=hgt_heads,
            device=device, message_style=message_style,
        )
        encoder.eval()
        batch_size = int(train_cfg.get('batch_size', 50))
        for split_name, split_names, split_graphs in (
            ('train', tr_names, tr_graphs), ('test', te_names, te_graphs),
        ):
            print(f'  precomputing {split_name} embeddings '
                  f'({len(split_graphs)} graphs)...')
            precompute_embeddings(
                encoder, split_names, split_graphs, device,
                batch_size=batch_size, embedding_type=embedding_type,
                esm_dict=None, use_amp=use_amp, use_jk=use_jk, jk_mode=jk_mode,
            )
        del encoder
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        pool_input_dim = get_pool_input_dim(
            hidden_dim, embedding_type, use_jk=use_jk, jk_mode=jk_mode,
            num_hgt_layers=num_layers,
        )

    # ---- head ----
    model = create_poolhead_model(
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
    metric_key = 'ap' if select_metric in ('auprc', 'ap') else select_metric
    select_on = str(train_cfg.get('select_on', 'test')).lower()
    if select_on not in ('test', 'train'):
        raise ValueError(f"train.select_on must be 'test' or 'train', got {select_on!r}")
    batch_size = int(train_cfg.get('batch_size', 50))
    label_smoothing = float(train_cfg.get('label_smoothing', 0.0))

    if select_on == 'test':
        print('[limitation] Early stopping uses the Anthropic labels; the reported '
              'Anthropic metrics are NOT a blind-test estimate.')

    train_loader = DataLoader(tr_graphs, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg.get('lr', 2e-4)),
        weight_decay=float(train_cfg.get('weight_decay', 0.02)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=int(train_cfg.get('scheduler_t0', 10)),
        T_mult=int(train_cfg.get('scheduler_t_mult', 2)),
        eta_min=float(train_cfg.get('scheduler_eta_min', 1e-6)))
    scaler = GradScaler(enabled=use_amp)

    prefix = 'anthr_test' if mode == 'grappi' else 'anthr_test_esm'
    ckpt_path = out_dir / f'{prefix}_best.pt'
    stopper = FineTuneEarlyStopping(
        patience=int(train_cfg.get('patience', 20)),
        save_path=str(ckpt_path), mode='max')

    epoch_log: List[dict] = []
    for epoch in range(int(train_cfg.get('n_epochs', 1000))):
        t0 = time.time()
        if hasattr(model, 'update_epoch'):
            model.update_epoch(epoch)
        train_loss, _, _ = train_epoch_classifier(
            model, train_loader, optimizer, scaler, criterion, device,
            clip_max_norm=float(train_cfg.get('clip_max_norm', 5.0)),
            use_amp=use_amp, label_smoothing=label_smoothing)
        scheduler.step()

        yt, pt, _ = evaluate(model, te_graphs, device, batch_size)
        test_metrics = binary_metrics(yt, pt)
        score = (test_metrics[metric_key] if select_on == 'test'
                 else -float(train_loss))
        epoch_log.append({'epoch': epoch + 1, 'train_loss': float(train_loss),
                          'test_auroc': test_metrics['auroc'],
                          'test_ap': test_metrics['ap'],
                          'seconds': time.time() - t0})
        print(f'  Epoch {epoch+1:3d} | loss {train_loss:.4f} | '
              f'anthr AUROC {test_metrics["auroc"]:.4f} AP {test_metrics["ap"]:.4f} | '
              f'{time.time()-t0:.1f}s')
        if stopper(score, model, epoch):
            print(f'  Early stop at epoch {epoch+1} '
                  f'(best {select_on} score {stopper.best_score:.4f})')
            break

    best_state = stopper.get_best_state()
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- final scoring ----
    yt, pt, lt = evaluate(model, te_graphs, device, batch_size)
    rows = [{'name': n, 'label': int(y), 'logit': float(l), 'prob': float(p)}
            for n, y, p, l in zip(te_names, yt.tolist(), pt.tolist(), lt.tolist())]
    attach_metadata(rows, meta, strata)
    summary_metrics = summarize(rows, strata)

    ytr, ptr, _ = evaluate(model, tr_graphs, device, batch_size)
    train_fit = binary_metrics(ytr, ptr)

    n_labeled_test = len(meta)
    coverage = (f'{len(rows)}/{n_labeled_test} labeled Anthropic designs scored '
                f'({n_labeled_test - len(rows)} without a prepared structure, skipped)')

    write_scores(out_dir / f'{prefix}_raw_scores.csv', rows, strata)
    write_stratum_tables(out_dir, prefix, summary_metrics)
    with open(out_dir / f'{prefix}_epoch_log.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_log[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_log)

    summary = {
        'protocol': 'train_on_full_exp__test_on_anthropic',
        'limitation': (
            'Early stopping / best-epoch selection used the Anthropic labels '
            f'(select_on={select_on}); reported Anthropic metrics are optimistic '
            'and are not a blind-test estimate.'),
        'seed': run_seed,
        'device': str(device),
        'encoder_mode': mode,
        'embedding_type': embedding_type,
        'encoder_dir': str(enc_dir) if enc_dir is not None else None,
        'train_cache': str(train_cache),
        'test_cache': str(test_cache),
        'pool_input_dim': pool_input_dim,
        'head_config': head_cfg,
        'train_config': train_cfg,
        'pos_weight': pos_weight,
        'select_metric': select_metric,
        'select_on': select_on,
        'best_epoch': int(stopper.best_epoch + 1),
        'best_select_score': float(stopper.best_score),
        'n_epochs_run': len(epoch_log),
        'checkpoint': str(ckpt_path),
        'n_train': len(tr_names), 'n_train_pos': n_pos, 'n_train_neg': n_neg,
        'n_train_targets': len(set(tr_targets)),
        'coverage': coverage,
        'shared_target_names': shared_targets,
        'overlapping_design_ids': overlap,
        'train_fit_at_best_epoch': train_fit,
        'test_metrics': summary_metrics,
    }
    write_json(out_dir / f'{prefix}_summary.json', summary)

    print_summary(f'{mode.upper()} trained on {len(tr_names)} exp designs '
                  f'-> Anthropic test', summary_metrics, coverage)
    print(f"  training-set fit at best epoch: AUROC {train_fit['auroc']:.4f} "
          f"AP {train_fit['ap']:.4f}")
    print(f'  outputs -> {out_dir}')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).with_name('config_exp_train_anthr_test.yaml'))
    parser.add_argument('--mode', choices=('grappi', 'esm'),
                        help='Encoder mode; esm reuses cached ESM features without the GNN.')
    parser.add_argument('--cuda-id', type=int,
                        help='Visible CUDA device index; overrides system.cuda_id.')
    parser.add_argument('--seed', type=int, help='Overrides system.seed.')
    parser.add_argument('--output-dir', type=Path,
                        help='Override the mode-specific output directory.')
    args = parser.parse_args()
    run(args.config, mode=args.mode, output_dir=args.output_dir,
        cuda_id=args.cuda_id, seed=args.seed)


if __name__ == '__main__':
    main()
