#!/usr/bin/env python3
"""Combine the three Anthropic score sets into one comparison table.

Reads the raw per-design score CSVs written by ``train_exp_anthr.py``
(GraPPI and ESM) and ``score_original_head.py`` (published head), checks that
they describe the same designs with the same labels, and emits pooled and
per-stratum comparisons.

Runs on CPU in seconds; no GPU, no model loading.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _path in (str(REPO_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from anthr_eval import aggregate, binary_metrics, stratified_metrics  # noqa: E402

DEFAULT_SOURCES = {
    'GraPPI': ('2layers_10hdim_esm_exp_train_anthr_test', 'anthr_test_raw_scores.csv'),
    'GraPPI base': ('2layers_10hdim_exp_train_anthr_test',
                    'anthr_test_base_raw_scores.csv'),
    'ESM': ('esm1280_exp_train_anthr_test', 'anthr_test_esm_raw_scores.csv'),
    'Original head': ('2layers_10hdim_esm_exp_train_anthr_test',
                      'anthr_test_original_head_raw_scores.csv'),
}


RESULTS_ROOT = REPO_ROOT.parent / 'GraPPI_data' / 'trained_data_ssl_finetune'


def resolve_sources(results_root: Path = RESULTS_ROOT) -> Dict[str, Path]:
    """Map method name -> expected raw-score CSV under ``results_root``."""
    return {name: Path(results_root) / subdir / filename
            for name, (subdir, filename) in DEFAULT_SOURCES.items()}


def summary_path_for(score_path: Path) -> Path:
    """The run summary JSON that accompanies a raw-score CSV."""
    return Path(score_path).with_name(
        Path(score_path).name.replace('_raw_scores.csv', '_summary.json'))


def load_available(sources: Dict[str, Path], verbose: bool = True):
    """Load whichever score sets exist; return (frames, missing).

    Missing runs are reported rather than raised, so the notebook stays usable
    while a run is still training.
    """
    frames: Dict[str, pd.DataFrame] = {}
    missing: Dict[str, Path] = {}
    for name, path in sources.items():
        path = Path(path)
        if not path.is_file():
            missing[name] = path
            if verbose:
                print(f'[skip] {name}: not found at {path}')
            continue
        frames[name] = load_scores(path)
        if verbose:
            print(f'[load] {name}: {len(frames[name])} designs from {path}')
    return frames, missing


def load_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {'name', 'label', 'prob'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    if frame['name'].duplicated().any():
        raise ValueError(f'{path}: duplicate design ids')
    return frame


def align(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Verify the score sets cover the same designs with the same labels.

    Mismatches are raised rather than silently intersected: comparing methods
    over different cohorts would make the numbers incomparable, which is the
    one thing this table exists to avoid.
    """
    items = list(frames.items())
    ref_name, ref = items[0]
    ref_index = ref.set_index('name')['label'].sort_index()
    for name, frame in items[1:]:
        other = frame.set_index('name')['label'].sort_index()
        if list(other.index) != list(ref_index.index):
            only_ref = sorted(set(ref_index.index) - set(other.index))[:3]
            only_other = sorted(set(other.index) - set(ref_index.index))[:3]
            raise ValueError(
                f"'{name}' and '{ref_name}' cover different designs "
                f"({len(other)} vs {len(ref_index)}); "
                f"only in {ref_name}: {only_ref}; only in {name}: {only_other}")
        if not other.equals(ref_index):
            disagree = other.index[other.values != ref_index.values].tolist()[:3]
            raise ValueError(f"'{name}' labels disagree with '{ref_name}' "
                             f"(e.g. {disagree})")
    return frames


def pooled_table(frames: Dict[str, pd.DataFrame], strata: Sequence[str]) -> pd.DataFrame:
    """One row per method: pooled metrics plus across-stratum medians/IQRs."""
    rows: List[dict] = []
    for method, frame in frames.items():
        record = {'method': method}
        record.update({f'pooled_{k}': v for k, v in
                       binary_metrics(frame['label'], frame['prob']).items()})
        for key in strata:
            if key not in frame.columns:
                continue
            table = stratified_metrics(frame.to_dict('records'), key)
            for metric in ('ap', 'auroc'):
                agg = aggregate([t[metric] for t in table])
                record[f'{key}_{metric}_median'] = agg['median']
                record[f'{key}_{metric}_q25'] = agg['q25']
                record[f'{key}_{metric}_q75'] = agg['q75']
                record[f'{key}_{metric}_iqr'] = agg['iqr']
            record[f'{key}_n_groups'] = len(table)
        rows.append(record)
    return pd.DataFrame(rows).set_index('method')


def stratum_table(frames: Dict[str, pd.DataFrame], key: str) -> pd.DataFrame:
    """Per-group AP and AUROC for every method, side by side."""
    merged = None
    for method, frame in frames.items():
        if key not in frame.columns:
            continue
        table = pd.DataFrame(stratified_metrics(frame.to_dict('records'), key))
        counts = table[[key, 'n', 'n_pos', 'n_neg', 'prevalence']]
        scores = table[[key, 'ap', 'auroc']].rename(
            columns={'ap': f'{method} AP', 'auroc': f'{method} AUROC'})
        merged = counts.merge(scores, on=key) if merged is None \
            else merged.merge(scores, on=key)
    if merged is None:
        raise ValueError(f'No score set carries the stratification column {key!r}')
    return merged.sort_values('n', ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path,
                        default=REPO_ROOT.parent / 'GraPPI_data' / 'trained_data_ssl_finetune',
                        help='Parent of the per-run output directories.')
    parser.add_argument('--score-csv', action='append', default=[], metavar='NAME=PATH',
                        help='Add or override a score set, e.g. "GraPPI=/path/scores.csv". '
                             'Repeatable; replaces the defaults entirely when used.')
    parser.add_argument('--strata', nargs='*', default=['target', 'design_model', 'campaign'])
    parser.add_argument('--out-dir', type=Path,
                        help='Write comparison CSVs here (default: --results-root/anthr_comparison).')
    args = parser.parse_args()

    if args.score_csv:
        sources = {}
        for spec in args.score_csv:
            if '=' not in spec:
                parser.error(f'--score-csv expects NAME=PATH, got {spec!r}')
            name, path = spec.split('=', 1)
            sources[name] = Path(path)
    else:
        sources = resolve_sources(args.results_root)

    frames, _ = load_available(sources)
    if not frames:
        raise SystemExit('No score CSVs found; run the training/scoring scripts first.')
    align(frames)

    out_dir = args.out_dir or (args.results_root / 'anthr_comparison')
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled = pooled_table(frames, args.strata)
    pooled.T.to_csv(out_dir / 'anthr_comparison_summary.csv')
    print('\n=== Pooled and across-stratum summary (methods as columns) ===')
    with pd.option_context('display.width', 200, 'display.max_columns', 50):
        print(pooled.T.to_string(float_format=lambda v: f'{v:.4f}'))

    for key in args.strata:
        try:
            table = stratum_table(frames, key)
        except ValueError as exc:
            print(f'[skip] {key}: {exc}')
            continue
        table.to_csv(out_dir / f'anthr_comparison_per_{key}.csv', index=False)
        print(f'\n=== Per {key} ===')
        with pd.option_context('display.width', 200, 'display.max_columns', 50):
            print(table.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    print(f'\nWrote comparison tables -> {out_dir}')


if __name__ == '__main__':
    main()
