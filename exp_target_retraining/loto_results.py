"""Read and compare completed experimental leave-one-target-out runs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score


RUN_FILES = {
    'summary': 'exp_target_loto_summary.json',
    'splits': 'exp_target_loto_splits.json',
    'scores': 'exp_target_loto_raw_scores.csv',
    'targets': 'exp_target_loto_per_target.csv',
}


@dataclass
class LotoResults:
    directory: Path
    metadata: dict
    scores: pd.DataFrame
    per_target: pd.DataFrame


@dataclass
class FeatureResults:
    source: Path
    scores: pd.DataFrame
    per_target: pd.DataFrame


def missing_run_files(directory: Path) -> list[str]:
    """List final exports that have not yet been written."""
    return [name for name in RUN_FILES.values() if not (directory / name).is_file()]


def _validate_ids_and_labels(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    missing = {'name', 'target_id', 'label'} - set(frame.columns)
    if missing:
        raise ValueError(f'{source}: missing columns {sorted(missing)}')
    frame = frame.copy()
    for column in ('name', 'target_id'):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq('').any():
            raise ValueError(f'{source}: blank values in {column}')
        frame[column] = frame[column].astype(str)
    if frame['name'].duplicated().any():
        duplicates = frame.loc[frame['name'].duplicated(), 'name'].head(5).tolist()
        raise ValueError(f'{source}: duplicate sample names: {duplicates}')
    frame['label'] = pd.to_numeric(frame['label'], errors='raise')
    if not frame['label'].isin([0, 1]).all():
        raise ValueError(f'{source}: labels must be 0 or 1')
    frame['label'] = frame['label'].astype(int)
    return frame


def read_dataset_labels(path: Path) -> pd.DataFrame:
    """Read the labeled reference set, using PDB stems as cached sample names."""
    frame = pd.read_csv(path, dtype={'pdb_file': str, 'target_id': str})
    if 'pdb_file' not in frame or frame['pdb_file'].isna().any():
        raise ValueError(f'{path}: pdb_file is missing or contains empty values')
    frame['name'] = frame['pdb_file'].map(lambda value: Path(value).stem)
    return _validate_ids_and_labels(frame, path)[['name', 'target_id', 'label']]


def reference_counts(labels: pd.DataFrame) -> pd.DataFrame:
    counts = labels.groupby('target_id', sort=True)['label'].agg(
        dataset_n='size', dataset_pos='sum').reset_index()
    counts['dataset_neg'] = counts['dataset_n'] - counts['dataset_pos']
    counts['dataset_prevalence'] = counts['dataset_pos'] / counts['dataset_n']
    return counts.sort_values('dataset_n', ascending=False).reset_index(drop=True)


def _metrics(labels: pd.Series, scores: pd.Series) -> tuple[float, float]:
    if labels.nunique() != 2:
        return float('nan'), float('nan')
    return (float(roc_auc_score(labels, scores)),
            float(average_precision_score(labels, scores)))


def _per_target_metrics(scores: pd.DataFrame, score_column: str = 'prob') -> pd.DataFrame:
    records = []
    for target, subset in scores.groupby('target_id', sort=True):
        n = len(subset)
        n_pos = int(subset['label'].sum())
        auroc, ap = _metrics(subset['label'], subset[score_column])
        prevalence = n_pos / n
        records.append({
            'target_id': target, 'n': n, 'n_pos': n_pos, 'n_neg': n - n_pos,
            'prevalence': prevalence, 'auroc': auroc, 'ap': ap,
            'lift': ap / prevalence if prevalence > 0 else float('nan'),
        })
    return pd.DataFrame.from_records(records)


def _check_against_labels(scores: pd.DataFrame, labels: pd.DataFrame, source: Path) -> None:
    joined = scores[['name', 'target_id', 'label']].merge(
        labels, on='name', how='left', suffixes=('_run', '_dataset'),
        indicator=True, validate='one_to_one')
    invalid = ((joined['_merge'] != 'both')
               | (joined['target_id_run'] != joined['target_id_dataset'])
               | (joined['label_run'] != joined['label_dataset']))
    if invalid.any():
        examples = joined.loc[invalid, 'name'].head(5).tolist()
        raise ValueError(f'{source}: sample names, targets, or labels disagree with '
                         f'the dataset; examples: {examples}')


def load_run(directory: Path, expected_mode: str, labels: pd.DataFrame) -> LotoResults:
    """Load final exports and recompute metrics from their saved sample scores."""
    missing = missing_run_files(directory)
    if missing:
        raise FileNotFoundError(f'{directory}: final exports missing: {", ".join(missing)}')
    with (directory / RUN_FILES['summary']).open() as handle:
        summary = json.load(handle)
    with (directory / RUN_FILES['splits']).open() as handle:
        splits = json.load(handle)
    if summary['protocol'] != 'leave_one_target_out' or splits['protocol'] != 'leave_one_target_out':
        raise ValueError(f'{directory}: expected leave-one-target-out results')
    if summary['encoder_mode'] != expected_mode:
        raise ValueError(f'{directory}: expected encoder mode {expected_mode!r}, '
                         f'found {summary["encoder_mode"]!r}')

    scores_path = directory / RUN_FILES['scores']
    scores = pd.read_csv(scores_path, dtype={'name': str, 'target_id': str})
    scores = _validate_ids_and_labels(scores, scores_path)
    if scores.empty:
        raise ValueError(f'{scores_path}: no sample scores')
    for column in ('prob', 'logit'):
        if column not in scores:
            raise ValueError(f'{scores_path}: missing {column!r} column')
        scores[column] = pd.to_numeric(scores[column], errors='raise')
        if not np.isfinite(scores[column]).all():
            raise ValueError(f'{scores_path}: non-finite {column} values')
    if not scores['prob'].between(0, 1).all():
        raise ValueError(f'{scores_path}: probabilities must be in [0, 1]')
    if not np.allclose(scores['prob'], expit(scores['logit']), rtol=0, atol=1e-6):
        raise ValueError(f'{scores_path}: probabilities do not match sigmoid(logit)')
    _check_against_labels(scores, labels, scores_path)

    calculated = _per_target_metrics(scores)
    target_path = directory / RUN_FILES['targets']
    reported = pd.read_csv(target_path, dtype={'target': str})
    required = {'target', 'n', 'n_pos', 'n_neg', 'auroc', 'auprc', 'best_epoch'}
    if required - set(reported.columns):
        raise ValueError(f'{target_path}: missing columns {sorted(required - set(reported.columns))}')
    if reported['target'].isna().any() or reported['target'].duplicated().any():
        raise ValueError(f'{target_path}: blank or duplicated target IDs')
    for column in ('n', 'n_pos', 'n_neg', 'auroc', 'auprc', 'best_epoch'):
        reported[column] = pd.to_numeric(reported[column], errors='raise')
    if (reported['best_epoch'].isna().any()
            or (reported['best_epoch'] < 1).any()
            or (reported['best_epoch'] % 1 != 0).any()):
        raise ValueError(f'{target_path}: best_epoch must be a positive integer')
    if set(calculated['target_id']) != set(reported['target']):
        raise ValueError(f'{target_path}: target IDs do not match the saved scores')
    per_target = calculated.merge(reported, left_on='target_id', right_on='target',
                                  validate='one_to_one', suffixes=('', '_reported'))
    for column in ('n', 'n_pos', 'n_neg'):
        if not per_target[column].equals(per_target[f'{column}_reported']):
            raise ValueError(f'{target_path}: {column} does not match the saved scores')
    for calculated_column, reported_column in (('auroc', 'auroc_reported'), ('ap', 'auprc')):
        if not np.allclose(per_target[calculated_column], per_target[reported_column],
                           rtol=0, atol=1e-6, equal_nan=True):
            raise ValueError(f'{target_path}: {reported_column} does not match the saved scores')

    targets = set(calculated['target_id'])
    if (summary['n_folds'] != len(targets) or summary['n_targets'] != len(targets)
            or set(splits['folds']) != targets):
        raise ValueError(f'{directory}: target/fold counts do not match saved scores')
    summary_rows = summary['per_target']
    summary_targets = [row['target'] for row in summary_rows]
    if len(summary_targets) != len(targets) or set(summary_targets) != targets:
        raise ValueError(f'{directory}: summary targets do not match saved scores')
    summary_by_target = {row['target']: row for row in summary_rows}
    if (summary['n_pos_total'] != int(scores['label'].sum())
            or summary['n_neg_total'] != int((scores['label'] == 0).sum())):
        raise ValueError(f'{directory}: class totals do not match saved scores')
    for row in per_target.itertuples(index=False):
        fold = splits['folds'][row.target_id]
        if (fold['n_val'] != row.n or fold['val_pos'] != row.n_pos
                or fold['val_neg'] != row.n_neg or fold['n_train'] != len(scores) - row.n):
            raise ValueError(f'{directory}: split counts disagree for {row.target_id}')
        summary_row = summary_by_target[row.target_id]
        summary_metrics = summary_row['metrics']
        if (summary_row['best_epoch'] != row.best_epoch
                or summary_metrics['n'] != row.n
                or summary_metrics['n_pos'] != row.n_pos
                or summary_metrics['n_neg'] != row.n_neg
                or not np.isclose(summary_metrics['auroc'], row.auroc, rtol=0,
                                  atol=1e-7, equal_nan=True)
                or not np.isclose(summary_metrics['auprc'], row.ap, rtol=0,
                                  atol=1e-7, equal_nan=True)):
            raise ValueError(f'{directory}: summary disagrees with saved scores for {row.target_id}')

    columns = ['target_id', 'n', 'n_pos', 'n_neg', 'prevalence',
               'auroc', 'ap', 'lift', 'best_epoch']
    return LotoResults(directory, summary, scores, per_target[columns])


def check_comparable(runs: dict[str, LotoResults]) -> None:
    """Require the same samples, labels, and key settings before comparison."""
    if len(runs) < 2:
        return
    first_name, first = next(iter(runs.items()))
    first_rows = first.scores[['name', 'target_id', 'label']].sort_values('name').reset_index(drop=True)
    settings = ('protocol', 'seed', 'pos_weight', 'select_metric', 'label_smoothing', 'head_config')
    for name, run in list(runs.items())[1:]:
        rows = run.scores[['name', 'target_id', 'label']].sort_values('name').reset_index(drop=True)
        if not first_rows.equals(rows):
            raise ValueError(f'{first_name} and {name} do not use identical samples, targets, '
                             'and labels; do not compare their metrics directly.')
        mismatched = [key for key in settings if first.metadata[key] != run.metadata[key]]
        if mismatched:
            raise ValueError(f'{first_name} and {name} have different settings: {mismatched}')


def load_af3_features(
    path: Path, labels: pd.DataFrame, runs: dict[str, LotoResults],
) -> dict[str, FeatureResults]:
    """Recompute the paper's Figure 3B AF3 scores on the model evaluation cohort."""
    check_comparable(runs)
    cohort = (next(iter(runs.values())).scores[['name', 'target_id', 'label']]
              if runs else labels[['name', 'target_id', 'label']])
    required = {'binder_id', 'target_id', 'binder', 'af3_ipSAE_min',
                'af3_rosetta_interface_dG_SASA_ratio'}
    source = pd.read_csv(path, usecols=lambda column: column in required,
                         dtype={'binder_id': str, 'target_id': str})
    if required - set(source.columns):
        raise ValueError(f'{path}: missing columns {sorted(required - set(source.columns))}')
    source = source.loc[source['binder'].notna()].copy()
    source['binder'] = source['binder'].astype(str).str.strip().str.lower().map(
        {'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0})
    if source['binder'].isna().any():
        raise ValueError(f'{path}: binder values must be True, False, 1, or 0')
    source = source.rename(columns={'binder_id': 'name', 'binder': 'label'})
    source = _validate_ids_and_labels(source, path)
    merged = cohort.merge(source, on='name', how='left', suffixes=('_run', '_source'),
                          indicator=True, validate='one_to_one')
    mismatch = ((merged['_merge'] != 'both')
                | (merged['target_id_run'] != merged['target_id_source'])
                | (merged['label_run'] != merged['label_source']))
    if mismatch.any():
        examples = merged.loc[mismatch, 'name'].head(5).tolist()
        raise ValueError(f'{path}: evaluated samples, targets, or labels do not match '
                         f'the paper dataset; examples: {examples}')

    feature_columns = ('af3_ipSAE_min', 'af3_rosetta_interface_dG_SASA_ratio')
    for column in feature_columns:
        try:
            merged[column] = pd.to_numeric(merged[column], errors='raise')
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{path}: {column} contains non-numeric feature values') from exc
        invalid = ~np.isfinite(merged[column])
        if invalid.any():
            examples = merged.loc[invalid, 'name'].head(5).tolist()
            raise ValueError(f'{path}: {column} is missing or non-finite for '
                             f'{int(invalid.sum())} evaluated samples; examples: {examples}')

    ipsae = merged['af3_ipSAE_min']
    feature_scores = {
        'AF3_ipSAE': ipsae,
        'AF3_interaction': -ipsae * merged['af3_rosetta_interface_dG_SASA_ratio'],
    }
    results = {}
    for name, values in feature_scores.items():
        scores = cohort.copy()
        scores['score'] = values.to_numpy()
        if not np.isfinite(scores['score']).all():
            raise ValueError(f'{path}: computed {name} contains non-finite scores')
        results[name] = FeatureResults(path, scores, _per_target_metrics(scores, 'score'))
    return results


def _check_feature_cohorts(
    runs: dict[str, LotoResults], features: dict[str, FeatureResults],
) -> None:
    check_comparable(runs)
    overlap = set(runs) & set(features)
    if overlap:
        raise ValueError(f'Model and feature names must be distinct: {sorted(overlap)}')
    cohorts = [(name, run.scores) for name, run in runs.items()]
    cohorts.extend((name, feature.scores) for name, feature in features.items())
    if len(cohorts) < 2:
        return
    first_name, first = cohorts[0]
    reference = first[['name', 'target_id', 'label']].sort_values('name').reset_index(drop=True)
    for name, scores in cohorts[1:]:
        current = scores[['name', 'target_id', 'label']].sort_values('name').reset_index(drop=True)
        if not reference.equals(current):
            raise ValueError(f'{first_name} and {name} do not use identical samples, targets, '
                             'and labels; do not compare their metrics directly.')


def comparison_table(
    labels: pd.DataFrame, runs: dict[str, LotoResults],
    features: dict[str, FeatureResults] | None = None,
) -> pd.DataFrame:
    """Show dataset counts, evaluated counts, and each available method's metrics."""
    features = {} if features is None else features
    reference = reference_counts(labels)
    if not runs and not features:
        return reference
    _check_feature_cohorts(runs, features)
    first = next(iter(runs.values())) if runs else next(iter(features.values()))
    base = first.per_target[['target_id', 'n', 'n_pos', 'n_neg', 'prevalence']]
    table = reference.merge(base, on='target_id', how='left', validate='one_to_one')
    for column in ('n', 'n_pos', 'n_neg'):
        table[column] = table[column].fillna(0).astype(int)
    table['n_excluded'] = table['dataset_n'] - table['n']
    for name, run in runs.items():
        columns = run.per_target[['target_id', 'auroc', 'ap', 'lift', 'best_epoch']].rename(
            columns={'auroc': f'{name}_AUROC', 'ap': f'{name}_AP',
                     'lift': f'{name}_lift', 'best_epoch': f'{name}_best_epoch'})
        table = table.merge(columns, on='target_id', how='left', validate='one_to_one')
    for name, feature in features.items():
        columns = feature.per_target[['target_id', 'auroc', 'ap', 'lift']].rename(
            columns={'auroc': f'{name}_AUROC', 'ap': f'{name}_AP', 'lift': f'{name}_lift'})
        table = table.merge(columns, on='target_id', how='left', validate='one_to_one')
    if {'GraPPI', 'ESM'} <= set(runs):
        table['AP_difference'] = table['GraPPI_AP'] - table['ESM_AP']
        table['AUROC_difference'] = table['GraPPI_AUROC'] - table['ESM_AUROC']
    return table


def aggregate_table(
    runs: dict[str, LotoResults], features: dict[str, FeatureResults] | None = None,
) -> pd.DataFrame:
    """Summarize target-macro median/IQR and pooled model and feature metrics."""
    features = {} if features is None else features
    _check_feature_cohorts(runs, features)
    methods = [(name, run.scores, run.per_target, 'prob') for name, run in runs.items()]
    methods.extend((name, feature.scores, feature.per_target, 'score')
                   for name, feature in features.items())
    records = []
    for name, scores, per_target, score_column in methods:
        row = {'model': name, 'n_samples': len(scores), 'n_targets': len(per_target)}
        for column, prefix in (('ap', 'AP'), ('auroc', 'AUROC')):
            values = per_target[column].dropna().to_numpy(dtype=float)
            row[f'{prefix}_n_targets'] = len(values)
            if len(values):
                q25, median, q75 = np.percentile(values, [25, 50, 75])
                row.update({f'{prefix}_median': median, f'{prefix}_q25': q25,
                            f'{prefix}_q75': q75, f'{prefix}_iqr': q75 - q25})
            else:
                row.update({f'{prefix}_{key}': float('nan')
                            for key in ('median', 'q25', 'q75', 'iqr')})
        row['pooled_AUROC'], row['pooled_AP'] = _metrics(scores['label'], scores[score_column])
        records.append(row)
    return pd.DataFrame.from_records(records)
