#!/usr/bin/env python3
"""Score the Anthropic set with the ORIGINAL published GraPPI classifier head.

This re-runs the published ``2layers_10hdim_esm_jkmean`` classifier through the
same graph cache, the same frozen encoder, and the same metric code as the
newly trained heads, so all three score sets in this study come from one
preprocessing path and one manifest. (``anthr_binder_data/results/`` already
holds an equivalent run made through the ``GraPPI.py`` CLI; this reproduces it
inside the comparison pipeline rather than relying on cross-pipeline alignment.)

Fold selection uses the head's ORIGINAL training ``fold_val_score``, never the
Anthropic labels. Pin a specific checkpoint with ``--head-checkpoint``.

No training happens here, so these numbers ARE a genuine held-out evaluation of
the published model — unlike the trained heads in ``train_exp_anthr.py``, whose
early stopping sees the Anthropic labels.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _path in (str(REPO_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from anthr_eval import (  # noqa: E402
    attach_metadata, load_test_metadata, print_summary, summarize,
    write_json, write_scores, write_stratum_tables,
)
from train_exp_anthr import (  # noqa: E402
    _abs, _load_config, evaluate, load_cache, resolve_device,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run(config_path: Path, head_checkpoint: Optional[Path] = None,
        output_dir: Optional[Path] = None, cuda_id: Optional[int] = None) -> dict:
    import torch

    from utils.Training_modules.common_utils import (
        MODEL_INIT_DIM, EDGE_IN_DIM, METADATA, load_ssl_config,
    )
    from model_collection.FineTuneModels import (
        load_pretrained_encoder, precompute_embeddings, get_pool_input_dim,
    )
    from utils.inference_utils import (
        build_poolhead_model, load_torch_head_state, select_best_fold_by_key,
    )

    cfg = _load_config(config_path)
    enc_cfg_in, head_cfg = cfg['encoder'], cfg['head']
    orig_cfg, out_cfg = cfg['original_head'], cfg['output']
    test_data_cfg = cfg['data']['test']

    device = resolve_device(int(cfg['system'].get('cuda_id', 0)), cuda_id)
    print(f'Using device: {device}' if device.type == 'cuda'
          else 'CUDA is unavailable; using CPU.')

    out_dir = _abs(REPO_ROOT, out_cfg['dir']).resolve()
    if output_dir is not None:
        out_dir = _abs(REPO_ROOT, str(output_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- head checkpoint selection (never uses the Anthropic labels) ----
    head_dir = _abs(REPO_ROOT, orig_cfg['dir']).resolve()
    if head_checkpoint is not None:
        head_path = head_checkpoint.resolve()
        selection = 'explicit --head-checkpoint'
        fold, fold_score = None, None
    else:
        summary_path = head_dir / orig_cfg.get('summary',
                                               'disc_binder_jkmean_5fold_summary.json')
        fold, fold_score, _ = select_best_fold_by_key(
            str(summary_path), score_key=orig_cfg.get('select_key', 'fold_val_score'))
        head_path = head_dir / orig_cfg['template'].format(fold=fold)
        selection = (f"maximum original training "
                     f"{orig_cfg.get('select_key', 'fold_val_score')}")
        print(f'Selected fold {fold} ({fold_score:.4f}) by {selection}.')
    if not head_path.is_file():
        raise FileNotFoundError(f'Head checkpoint not found: {head_path}')
    print(f'Head checkpoint: {head_path}')

    # ---- test cache ----
    test_dir = _abs(REPO_ROOT, test_data_cfg['data_dir']).resolve()
    test_cache = _abs(test_dir, test_data_cfg['graph_cache'])
    names, graphs, targets, labels = load_cache(test_cache)
    print(f'Test (anthr): {len(names)} graphs, {int(sum(labels))} pos / '
          f'{int(len(labels) - sum(labels))} neg, {len(set(targets))} targets.')

    strata = list(test_data_cfg.get('strata', ['target']))
    meta = load_test_metadata(
        _abs(test_dir, test_data_cfg.get('labels_csv', 'labels.csv')),
        test_data_cfg.get('id_column', 'binder_id'), strata)

    # ---- frozen encoder (same one the head was trained on) ----
    enc_dir = _abs(REPO_ROOT, enc_cfg_in['dir']).resolve()
    enc_ckpt = enc_dir / 'ssl_edge_best.pt'
    ssl = load_ssl_config(str(enc_ckpt)) or {}
    embedding_type = enc_cfg_in.get('embedding_type', ssl.get('embedding_type', 'esm'))
    num_layers = int(ssl.get('num_layers', 2))
    hidden_dim = 2 ** int(ssl.get('hidden_dim_power', 10))
    use_jk = bool(head_cfg.get('use_jk', True))
    jk_mode = head_cfg.get('jk_mode', 'mean')
    batch_size = int(cfg['train'].get('batch_size', 50))

    encoder = load_pretrained_encoder(
        checkpoint_path=str(enc_ckpt),
        node_in_dim=MODEL_INIT_DIM[embedding_type],
        edge_in_dim=EDGE_IN_DIM, metadata=METADATA,
        hidden_dim=hidden_dim, num_hgt_layers=num_layers,
        hgt_heads=int(ssl.get('hgt_heads', 4)), device=device,
        message_style=ssl.get('message_style', 'gated_src'),
    )
    encoder.eval()
    precompute_embeddings(
        encoder, names, graphs, device, batch_size=batch_size,
        embedding_type=embedding_type, esm_dict=None,
        use_amp=(device.type == 'cuda'), use_jk=use_jk, jk_mode=jk_mode,
    )
    del encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    pool_input_dim = get_pool_input_dim(
        hidden_dim, embedding_type, use_jk=use_jk, jk_mode=jk_mode,
        num_hgt_layers=num_layers)

    # ---- published head architecture, straight from the repo's model config ----
    arch_path = _abs(REPO_ROOT, orig_cfg.get('arch_yaml',
                                             'model_configs/config_disc_binder.yaml'))
    with open(arch_path) as handle:
        arch_cfg = yaml.safe_load(handle)[orig_cfg.get('arch_key', 'disc_binder')]
    model = build_poolhead_model('bind_score', arch_cfg, pool_input_dim, device)
    load_torch_head_state(model, str(head_path), device)

    y, prob, logit = evaluate(model, graphs, device, batch_size)
    rows = [{'name': n, 'label': int(yy), 'logit': float(l), 'prob': float(p)}
            for n, yy, p, l in zip(names, y.tolist(), prob.tolist(), logit.tolist())]
    attach_metadata(rows, meta, strata)
    metrics = summarize(rows, strata)

    prefix = 'anthr_test_original_head'
    coverage = (f'{len(rows)}/{len(meta)} labeled Anthropic designs scored '
                f'({len(meta) - len(rows)} without a prepared structure, skipped)')
    write_scores(out_dir / f'{prefix}_raw_scores.csv', rows, strata)
    write_stratum_tables(out_dir, prefix, metrics)

    summary = {
        'protocol': 'published_head_inference_on_anthropic',
        'note': ('No training and no label-driven selection: this is a genuine '
                 'held-out evaluation of the published classifier.'),
        'device': str(device),
        'encoder_dir': str(enc_dir),
        'encoder_sha256': sha256(enc_ckpt),
        'head_dir': str(head_dir),
        'head_checkpoint': str(head_path),
        'head_sha256': sha256(head_path),
        'head_selection': selection,
        'head_fold': fold,
        'head_fold_score': fold_score,
        'head_arch_config': arch_cfg,
        'embedding_type': embedding_type,
        'pool_input_dim': pool_input_dim,
        'test_cache': str(test_cache),
        'coverage': coverage,
        'git_commit': git_commit(REPO_ROOT),
        'test_metrics': metrics,
    }
    write_json(out_dir / f'{prefix}_summary.json', summary)

    print_summary('Published head -> Anthropic test', metrics, coverage)
    print(f'  outputs -> {out_dir}')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).with_name('config_exp_train_anthr_test.yaml'))
    parser.add_argument('--head-checkpoint', type=Path,
                        help='Pin a specific .pt head instead of fold selection.')
    parser.add_argument('--cuda-id', type=int)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run(args.config, head_checkpoint=args.head_checkpoint,
        output_dir=args.output_dir, cuda_id=args.cuda_id)


if __name__ == '__main__':
    main()
