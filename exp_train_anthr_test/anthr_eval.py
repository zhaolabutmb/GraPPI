#!/usr/bin/env python3
"""Shared evaluation helpers for the exp-train / Anthropic-test comparison.

Imported by both ``train_exp_anthr.py`` (newly trained GraPPI and ESM heads)
and ``score_original_head.py`` (the published classifier), so every score set
goes through the same metrics, the same stratification, and the same output
schema. The pure functions here have no torch dependency and are unit-tested
in ``tests/``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Metrics (pure)
# ---------------------------------------------------------------------------
def binary_metrics(y_true: Sequence[float], y_prob: Sequence[float]) -> Dict[str, float]:
    """AUROC + AP, with counts and the positive-rate baseline.

    AUROC/AP are NaN for a single-class group (e.g. the Anthropic MBP target,
    which has zero positives) rather than an arbitrary 0.5 — a one-class group
    has no ranking metric, and silently substituting one would bias the
    across-target median.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = int((y_true == 1).sum())
    out = {
        'n': int(y_true.size),
        'n_pos': n_pos,
        'n_neg': int((y_true == 0).sum()),
        'prevalence': float(n_pos / y_true.size) if y_true.size else float('nan'),
    }
    if y_true.size and len(np.unique(y_true)) > 1:
        out['auroc'] = float(roc_auc_score(y_true, y_prob))
        out['ap'] = float(average_precision_score(y_true, y_prob))
    else:
        out['auroc'] = float('nan')
        out['ap'] = float('nan')
    # Lift over the random-ranking reference; NaN when AP is undefined.
    out['ap_lift'] = (out['ap'] / out['prevalence']
                      if out['prevalence'] and np.isfinite(out['ap'])
                      else float('nan'))
    return out


def threshold_metrics(y_true: Sequence[float], y_prob: Sequence[float],
                      threshold: float = 0.5) -> Dict[str, float]:
    """Confusion counts and precision/recall/F1 at a fixed probability cut."""
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_prob, dtype=float) >= threshold
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    f1 = (2 * precision * recall / (precision + recall)
          if np.isfinite(precision) and np.isfinite(recall) and (precision + recall)
          else float('nan'))
    return {'threshold': float(threshold), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'accuracy': (tp + tn) / y_true.size if y_true.size else float('nan')}


def aggregate(values: Sequence[float]) -> Dict[str, float]:
    """Median / IQR / mean / std over per-group metrics, ignoring NaN groups."""
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return {k: float('nan') for k in
                ('median', 'iqr', 'q25', 'q75', 'mean', 'std')} | {'n': 0}
    q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
    return {'median': float(np.median(vals)), 'iqr': q75 - q25, 'q25': q25,
            'q75': q75, 'mean': float(vals.mean()), 'std': float(vals.std()),
            'n': int(vals.size)}


def stratified_metrics(rows: List[dict], key: str) -> List[dict]:
    """Per-group metrics for ``key``, one row per group, sorted by group name."""
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, '')), []).append(row)
    out = []
    for name in sorted(groups):
        members = groups[name]
        m = binary_metrics([r['label'] for r in members],
                           [r['prob'] for r in members])
        out.append({key: name, **m})
    return out


def summarize(rows: List[dict], strata: Sequence[str],
              threshold: float = 0.5) -> Dict[str, object]:
    """Pooled metrics, thresholded metrics, and per-stratum tables + aggregates."""
    y = [r['label'] for r in rows]
    p = [r['prob'] for r in rows]
    summary: Dict[str, object] = {
        'pooled': binary_metrics(y, p),
        'thresholded': threshold_metrics(y, p, threshold),
        'per_stratum': {},
        'across_strata': {},
    }
    for key in strata:
        table = stratified_metrics(rows, key)
        summary['per_stratum'][key] = table
        summary['across_strata'][key] = {
            'auroc': aggregate([t['auroc'] for t in table]),
            'ap': aggregate([t['ap'] for t in table]),
            'n_groups': len(table),
        }
    return summary


# ---------------------------------------------------------------------------
# Label metadata (pure)
# ---------------------------------------------------------------------------
def load_test_metadata(labels_csv: Path, id_column: str = 'binder_id',
                       extra_columns: Sequence[str] = ()) -> Dict[str, dict]:
    """Map design id -> the stratification columns from the Anthropic labels."""
    import pandas as pd

    frame = pd.read_csv(labels_csv)
    missing = ({id_column} | set(extra_columns)) - set(frame.columns)
    if missing:
        raise ValueError(f'{labels_csv}: missing columns {sorted(missing)}')
    meta = {}
    for _, row in frame.iterrows():
        meta[str(row[id_column])] = {c: str(row[c]) for c in extra_columns}
    return meta


def attach_metadata(rows: List[dict], meta: Dict[str, dict],
                    columns: Sequence[str]) -> List[dict]:
    """Add the stratification columns to score rows, keyed by ``name``.

    Raises if a scored design is absent from the labels table: that would mean
    the cache and the labels file disagree, and silently emitting empty strata
    would quietly drop those designs from every per-group table.
    """
    unknown = [r['name'] for r in rows if r['name'] not in meta]
    if unknown:
        raise ValueError(
            f'{len(unknown)} scored design(s) are not in the labels table '
            f'(e.g. {unknown[:3]}); cache and labels are out of sync.')
    for row in rows:
        for column in columns:
            row[column] = meta[row['name']].get(column, '')
    return rows


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------
def write_scores(path: Path, rows: List[dict], columns: Sequence[str]) -> None:
    """Write per-design raw scores (label, logit, prob) plus strata columns."""
    fieldnames = ['name', *columns, 'label', 'logit', 'prob']
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_stratum_tables(out_dir: Path, prefix: str,
                         summary: Dict[str, object]) -> List[Path]:
    """One CSV per stratification key (target, design_model, campaign)."""
    written = []
    for key, table in summary['per_stratum'].items():  # type: ignore[union-attr]
        path = out_dir / f'{prefix}_per_{key}.csv'
        if not table:
            continue
        with open(path, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)
        written.append(path)
    return written


def write_json(path: Path, payload: object) -> None:
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Not JSON serializable: {type(value)}')


def print_summary(label: str, summary: Dict[str, object],
                  coverage: Optional[str] = None) -> None:
    pooled = summary['pooled']  # type: ignore[index]
    print(f"\n{'=' * 60}\n{label}")
    if coverage:
        print(f'  coverage: {coverage}')
    print(f"  pooled AUROC {pooled['auroc']:.4f} | AP {pooled['ap']:.4f} "
          f"(prevalence {pooled['prevalence']:.4f}, "
          f"{pooled['n']} designs, {pooled['n_pos']} pos)")
    for key, agg in summary['across_strata'].items():  # type: ignore[union-attr]
        ap, roc = agg['ap'], agg['auroc']
        print(f"  by {key} ({agg['n_groups']} groups, {ap['n']} scorable): "
              f"median AP {ap['median']:.4f} (IQR {ap['q25']:.4f}-{ap['q75']:.4f}) | "
              f"median AUROC {roc['median']:.4f}")
