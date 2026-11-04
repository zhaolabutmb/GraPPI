"""Offline unit tests for the exp-train / Anthropic-test pipeline.

Covers the pure logic only — metrics, stratification, label parsing, cohort
alignment and the WBCE weight. No GPU, no network, no model checkpoints.

Run: python -m unittest discover -s exp_train_anthr_test/tests
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import anthr_eval  # noqa: E402
import compare_results  # noqa: E402
from train_exp_anthr import check_disjoint, resolve_pos_weight  # noqa: E402


def _rows():
    return [
        {'name': 'a', 'target': 'T1', 'label': 1, 'prob': 0.9},
        {'name': 'b', 'target': 'T1', 'label': 0, 'prob': 0.1},
        {'name': 'c', 'target': 'T2', 'label': 1, 'prob': 0.2},
        {'name': 'd', 'target': 'T2', 'label': 0, 'prob': 0.8},
    ]


def _score_frame(probs):
    return pd.DataFrame({'name': ['a', 'b', 'c', 'd'],
                         'target': ['T1', 'T1', 'T2', 'T2'],
                         'label': [1, 0, 1, 0],
                         'prob': probs})


class TestMetrics(unittest.TestCase):
    def test_perfect_ranking(self):
        m = anthr_eval.binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(m['auroc'], 1.0)
        self.assertAlmostEqual(m['ap'], 1.0)
        self.assertEqual((m['n'], m['n_pos'], m['n_neg']), (4, 2, 2))
        self.assertAlmostEqual(m['prevalence'], 0.5)
        self.assertAlmostEqual(m['ap_lift'], 2.0)

    def test_single_class_group_is_nan_not_half(self):
        # The Anthropic MBP target has zero positives; a one-class group has no
        # ranking metric and must not contribute a fabricated 0.5 to the median.
        m = anthr_eval.binary_metrics([0, 0, 0], [0.1, 0.5, 0.9])
        self.assertTrue(np.isnan(m['auroc']))
        self.assertTrue(np.isnan(m['ap']))
        self.assertEqual(m['n_pos'], 0)
        self.assertEqual(m['prevalence'], 0.0)

    def test_threshold_counts(self):
        t = anthr_eval.threshold_metrics([1, 1, 0, 0], [0.9, 0.2, 0.7, 0.1], 0.5)
        self.assertEqual((t['tp'], t['fn'], t['fp'], t['tn']), (1, 1, 1, 1))
        self.assertAlmostEqual(t['precision'], 0.5)
        self.assertAlmostEqual(t['recall'], 0.5)
        self.assertAlmostEqual(t['f1'], 0.5)

    def test_aggregate_skips_nan_groups(self):
        agg = anthr_eval.aggregate([0.2, 0.4, 0.6, float('nan')])
        self.assertEqual(agg['n'], 3)
        self.assertAlmostEqual(agg['median'], 0.4)
        self.assertAlmostEqual(agg['iqr'], 0.2)

    def test_aggregate_all_nan(self):
        agg = anthr_eval.aggregate([float('nan'), float('nan')])
        self.assertEqual(agg['n'], 0)
        self.assertTrue(np.isnan(agg['median']))


class TestStratification(unittest.TestCase):
    def test_splits_by_group(self):
        table = anthr_eval.stratified_metrics(_rows(), 'target')
        self.assertEqual([t['target'] for t in table], ['T1', 'T2'])
        self.assertAlmostEqual(table[0]['auroc'], 1.0)
        self.assertAlmostEqual(table[1]['auroc'], 0.0)

    def test_summarize_reports_pooled_and_across_strata(self):
        summary = anthr_eval.summarize(_rows(), ['target'])
        self.assertEqual(summary['pooled']['n'], 4)
        self.assertEqual(summary['across_strata']['target']['n_groups'], 2)
        self.assertAlmostEqual(
            summary['across_strata']['target']['auroc']['median'], 0.5)

    def test_attach_metadata_rejects_unknown_design(self):
        rows = [{'name': 'ghost', 'label': 1, 'prob': 0.5}]
        with self.assertRaisesRegex(ValueError, 'not in the labels table'):
            anthr_eval.attach_metadata(rows, {'a': {'target': 'T1'}}, ['target'])

    def test_attach_metadata_fills_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'labels.csv'
            path.write_text('binder_id,target,design_model\nx,T1,Opus\n')
            meta = anthr_eval.load_test_metadata(
                path, 'binder_id', ['target', 'design_model'])
        rows = [{'name': 'x', 'label': 1, 'prob': 0.7}]
        anthr_eval.attach_metadata(rows, meta, ['target', 'design_model'])
        self.assertEqual(rows[0]['target'], 'T1')
        self.assertEqual(rows[0]['design_model'], 'Opus')

    def test_write_scores_round_trip(self):
        rows = _rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scores.csv'
            for row in rows:
                row['logit'] = 0.0
            anthr_eval.write_scores(path, rows, ['target'])
            back = pd.read_csv(path)
        self.assertEqual(list(back.columns),
                         ['name', 'target', 'label', 'logit', 'prob'])
        self.assertEqual(len(back), 4)


class TestAnthropicLabelSchema(unittest.TestCase):
    def test_drops_unlabeled_rows(self):
        from cache_graphs_anthr import load_label_map

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'labels.csv'
            # Third row is unlabeled: excluded, never treated as a negative.
            path.write_text('binder_id,target,label\na,T1,1\nb,T1,0\nc,T2,\n')
            frame = load_label_map(path)
        self.assertEqual(list(frame.index), ['a', 'b'])
        self.assertEqual(frame.attrs['n_unlabeled_dropped'], 1)
        self.assertEqual(frame.loc['a', 'label'], 1)
        self.assertEqual(frame.loc['a', 'target_id'], 'T1')

    def test_rejects_duplicate_ids(self):
        from cache_graphs_anthr import load_label_map

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'labels.csv'
            path.write_text('binder_id,target,label\na,T1,1\na,T1,0\n')
            with self.assertRaisesRegex(ValueError, 'duplicate design ids'):
                load_label_map(path)

    def test_rejects_missing_columns(self):
        from cache_graphs_anthr import load_label_map

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'labels.csv'
            path.write_text('binder_id,label\na,1\n')
            with self.assertRaisesRegex(ValueError, 'missing columns'):
                load_label_map(path)


class TestTrainingHelpers(unittest.TestCase):
    def test_global_pos_weight_matches_exp_set(self):
        self.assertAlmostEqual(resolve_pos_weight('global', 394, 3275), 8.3122, places=4)

    def test_explicit_float(self):
        self.assertAlmostEqual(resolve_pos_weight(2.5, 394, 3275), 2.5)

    def test_rejects_zero_positives(self):
        with self.assertRaisesRegex(ValueError, 'zero positives'):
            resolve_pos_weight('global', 0, 100)

    def test_check_disjoint(self):
        self.assertEqual(check_disjoint(['a', 'b'], ['b', 'c']), ['b'])
        self.assertEqual(check_disjoint(['a'], ['b']), [])


class TestComparison(unittest.TestCase):
    def test_align_accepts_matching_cohorts(self):
        frames = {'A': _score_frame([0.9, 0.1, 0.2, 0.8]),
                  'B': _score_frame([0.6, 0.4, 0.3, 0.7])}
        self.assertIs(compare_results.align(frames), frames)

    def test_align_rejects_different_cohorts(self):
        frames = {'A': _score_frame([0.9, 0.1, 0.2, 0.8]),
                  'B': _score_frame([0.9, 0.1, 0.2, 0.8]).iloc[:3]}
        with self.assertRaisesRegex(ValueError, 'different designs'):
            compare_results.align(frames)

    def test_align_rejects_label_disagreement(self):
        other = _score_frame([0.9, 0.1, 0.2, 0.8])
        other.loc[0, 'label'] = 0
        with self.assertRaisesRegex(ValueError, 'labels disagree'):
            compare_results.align({'A': _score_frame([0.9, 0.1, 0.2, 0.8]),
                                   'B': other})

    def test_pooled_table_one_row_per_method(self):
        frames = {'A': _score_frame([0.9, 0.1, 0.2, 0.8]),
                  'B': _score_frame([0.6, 0.4, 0.3, 0.7])}
        table = compare_results.pooled_table(frames, ['target'])
        self.assertEqual(list(table.index), ['A', 'B'])
        self.assertEqual(table.loc['A', 'pooled_n'], 4)
        self.assertEqual(table.loc['A', 'target_n_groups'], 2)

    def test_stratum_table_has_one_column_pair_per_method(self):
        frames = {'A': _score_frame([0.9, 0.1, 0.2, 0.8]),
                  'B': _score_frame([0.6, 0.4, 0.3, 0.7])}
        table = compare_results.stratum_table(frames, 'target')
        self.assertTrue({'target', 'n', 'n_pos', 'prevalence',
                         'A AP', 'A AUROC', 'B AP', 'B AUROC'}
                        <= set(table.columns))
        self.assertEqual(len(table), 2)

    def test_load_scores_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 's.csv'
            path.write_text('name,label,prob\na,1,0.5\na,0,0.4\n')
            with self.assertRaisesRegex(ValueError, 'duplicate design ids'):
                compare_results.load_scores(path)


if __name__ == '__main__':
    unittest.main()
