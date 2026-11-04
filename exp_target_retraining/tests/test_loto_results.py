"""Tests for the LOTO results reader and notebook."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'exp_target_retraining'))
import loto_results  # noqa: E402


class LotoResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.labels_file = self.root / 'labels.csv'
        pd.DataFrame([
            {'pdb_file': 'A_pos.pdb', 'target_id': 'EGFR', 'label': 1},
            {'pdb_file': 'A_neg.pdb', 'target_id': 'EGFR', 'label': 0},
            {'pdb_file': 'A_extra.pdb', 'target_id': 'EGFR', 'label': 0},
            {'pdb_file': 'B_pos.pdb', 'target_id': 'TargetB', 'label': 1},
            {'pdb_file': 'B_neg.pdb', 'target_id': 'TargetB', 'label': 0},
        ]).to_csv(self.labels_file, index=False)
        self.labels = loto_results.read_dataset_labels(self.labels_file)
        self.paper_file = self.root / 'final_dataset.csv'
        self.paper_rows = pd.DataFrame([
            {'binder_id': 'A_pos', 'target_id': 'EGFR', 'binder': True,
             'af3_ipSAE_min': 0.8, 'af3_rosetta_interface_dG_SASA_ratio': -1.0},
            {'binder_id': 'A_neg', 'target_id': 'EGFR', 'binder': False,
             'af3_ipSAE_min': 0.2, 'af3_rosetta_interface_dG_SASA_ratio': -10.0},
            {'binder_id': 'A_extra', 'target_id': 'EGFR', 'binder': False,
             'af3_ipSAE_min': 0.5, 'af3_rosetta_interface_dG_SASA_ratio': -2.0},
            {'binder_id': 'B_pos', 'target_id': 'TargetB', 'binder': True,
             'af3_ipSAE_min': 0.2, 'af3_rosetta_interface_dG_SASA_ratio': -15.0},
            {'binder_id': 'B_neg', 'target_id': 'TargetB', 'binder': False,
             'af3_ipSAE_min': 0.7, 'af3_rosetta_interface_dG_SASA_ratio': -1.0},
            {'binder_id': 'unlabeled', 'target_id': None, 'binder': None,
             'af3_ipSAE_min': None, 'af3_rosetta_interface_dG_SASA_ratio': None},
            {'binder_id': 'unlabeled', 'target_id': None, 'binder': None,
             'af3_ipSAE_min': None, 'af3_rosetta_interface_dG_SASA_ratio': None},
        ])
        self.paper_rows.to_csv(self.paper_file, index=False)

    def write_run(self, mode, a_negative='A_neg'):
        directory = self.root / mode
        directory.mkdir()
        if mode == 'grappi':
            probabilities = [0.8, 0.2, 0.4, 0.7]
        else:
            probabilities = [0.4, 0.8, 0.9, 0.1]
        samples = [
            ('A_pos', 'EGFR', 1), (a_negative, 'EGFR', 0),
            ('B_pos', 'TargetB', 1), ('B_neg', 'TargetB', 0),
        ]
        scores = pd.DataFrame([
            {'name': name, 'target_id': target, 'label': label,
             'logit': np.log(prob / (1 - prob)), 'prob': prob}
            for (name, target, label), prob in zip(samples, probabilities)
        ])
        scores.to_csv(directory / loto_results.RUN_FILES['scores'], index=False)
        target_rows = []
        folds = {}
        for target, part in scores.groupby('target_id'):
            pos = int(part['label'].sum())
            target_rows.append({
                'target': target, 'n': len(part), 'n_pos': pos, 'n_neg': len(part) - pos,
                'auroc': roc_auc_score(part['label'], part['prob']),
                'auprc': average_precision_score(part['label'], part['prob']),
                'best_epoch': 3,
            })
            folds[target] = {'n_train': len(scores) - len(part), 'n_val': len(part),
                             'val_pos': pos, 'val_neg': len(part) - pos}
        pd.DataFrame(target_rows).to_csv(directory / loto_results.RUN_FILES['targets'], index=False)
        summary = {
            'protocol': 'leave_one_target_out', 'encoder_mode': mode,
            'n_folds': 2, 'n_targets': 2, 'n_pos_total': 2, 'n_neg_total': 2,
            'seed': 42, 'pos_weight': 1.0, 'select_metric': 'auprc',
            'label_smoothing': 0.0, 'head_config': {'pool_mode': 'cross_attn'},
            'per_target': [
                {'target': row['target'], 'best_epoch': row['best_epoch'],
                 'metrics': {key: row[key] for key in ('n', 'n_pos', 'n_neg', 'auroc', 'auprc')}}
                for row in target_rows
            ],
        }
        (directory / loto_results.RUN_FILES['summary']).write_text(json.dumps(summary))
        (directory / loto_results.RUN_FILES['splits']).write_text(
            json.dumps({'protocol': 'leave_one_target_out', 'folds': folds}))
        return directory

    def test_missing_exports_are_reported(self):
        directory = self.root / 'not_finished'
        self.assertEqual(len(loto_results.missing_run_files(directory)), 4)
        with self.assertRaisesRegex(FileNotFoundError, 'final exports missing'):
            loto_results.load_run(directory, 'grappi', self.labels)

    def test_metrics_and_lift_use_evaluated_not_dataset_prevalence(self):
        grappi = loto_results.load_run(self.write_run('grappi'), 'grappi', self.labels)
        esm = loto_results.load_run(self.write_run('esm'), 'esm', self.labels)
        runs = {'GraPPI': grappi, 'ESM': esm}
        table = loto_results.comparison_table(self.labels, runs).set_index('target_id')
        egfr = table.loc['EGFR']
        self.assertEqual((egfr['dataset_n'], egfr['n'], egfr['n_excluded']), (3, 2, 1))
        self.assertEqual((egfr['n_pos'], egfr['n_neg']), (1, 1))
        self.assertAlmostEqual(egfr['prevalence'], 0.5)
        self.assertAlmostEqual(egfr['GraPPI_AP'], 1.0)
        self.assertAlmostEqual(egfr['ESM_AP'], 0.5)
        self.assertAlmostEqual(egfr['GraPPI_lift'], 2.0)
        self.assertAlmostEqual(egfr['ESM_lift'], 1.0)
        self.assertAlmostEqual(egfr['AP_difference'], 0.5)
        self.assertAlmostEqual(egfr['AUROC_difference'], 1.0)
        aggregate = loto_results.aggregate_table(runs).set_index('model')
        self.assertEqual(aggregate.loc['GraPPI', 'AP_n_targets'], 2)
        self.assertAlmostEqual(aggregate.loc['GraPPI', 'AP_median'], 0.75)
        self.assertAlmostEqual(aggregate.loc['GraPPI', 'AP_q25'], 0.625)
        self.assertAlmostEqual(aggregate.loc['GraPPI', 'AP_q75'], 0.875)
        self.assertAlmostEqual(aggregate.loc['GraPPI', 'AP_iqr'], 0.25)
        self.assertAlmostEqual(
            aggregate.loc['GraPPI', 'pooled_AP'],
            average_precision_score(grappi.scores['label'], grappi.scores['prob']))

    def test_dataset_counts_available_before_runs_finish(self):
        table = loto_results.comparison_table(self.labels, {}).set_index('target_id')
        self.assertEqual(table.loc['EGFR', 'dataset_n'], 3)
        self.assertEqual(table.loc['EGFR', 'dataset_pos'], 1)
        self.assertEqual(table.loc['EGFR', 'dataset_neg'], 2)
        self.assertAlmostEqual(table.loc['EGFR', 'dataset_prevalence'], 1 / 3)
        self.assertTrue(loto_results.aggregate_table({}).empty)

    def test_single_class_target_does_not_contribute_to_aggregate(self):
        scores = pd.DataFrame([
            {'target_id': 'A', 'label': 0, 'prob': 0.1},
            {'target_id': 'A', 'label': 0, 'prob': 0.2},
            {'target_id': 'B', 'label': 0, 'prob': 0.2},
            {'target_id': 'B', 'label': 1, 'prob': 0.8},
        ])
        targets = loto_results._per_target_metrics(scores).set_index('target_id')
        self.assertTrue(np.isnan(targets.loc['A', 'ap']))
        self.assertTrue(np.isnan(targets.loc['A', 'auroc']))
        self.assertTrue(np.isnan(targets.loc['A', 'lift']))
        run = loto_results.LotoResults(self.root, {}, scores, targets.reset_index())
        aggregate = loto_results.aggregate_table({'GraPPI': run}).iloc[0]
        self.assertEqual(aggregate['n_targets'], 2)
        self.assertEqual(aggregate['AP_n_targets'], 1)
        self.assertEqual(aggregate['AUROC_n_targets'], 1)
        self.assertAlmostEqual(aggregate['AP_median'], 1.0)

    def test_inconsistent_samples_cannot_be_compared(self):
        grappi = loto_results.load_run(self.write_run('grappi'), 'grappi', self.labels)
        esm = loto_results.load_run(self.write_run('esm', a_negative='A_extra'), 'esm', self.labels)
        with self.assertRaisesRegex(ValueError, 'identical samples'):
            loto_results.comparison_table(self.labels, {'GraPPI': grappi, 'ESM': esm})

    def test_inconsistent_settings_cannot_be_compared(self):
        grappi = loto_results.load_run(self.write_run('grappi'), 'grappi', self.labels)
        esm = loto_results.load_run(self.write_run('esm'), 'esm', self.labels)
        esm.metadata['pos_weight'] = 3.0
        with self.assertRaisesRegex(ValueError, 'pos_weight'):
            loto_results.check_comparable({'GraPPI': grappi, 'ESM': esm})

    def test_af3_scores_match_evaluated_samples_and_reverse_product(self):
        run = loto_results.load_run(self.write_run('grappi'), 'grappi', self.labels)
        self.paper_rows.sample(frac=1, random_state=4).to_csv(self.paper_file, index=False)
        features = loto_results.load_af3_features(self.paper_file, self.labels, {'GraPPI': run})
        self.assertEqual(set(features), {'AF3_ipSAE', 'AF3_interaction'})
        self.assertEqual(len(features['AF3_ipSAE'].scores), 4)
        self.assertNotIn('A_extra', features['AF3_ipSAE'].scores['name'].tolist())
        ipsae = features['AF3_ipSAE'].scores.set_index('name')['score']
        interaction = features['AF3_interaction'].scores.set_index('name')['score']
        self.assertAlmostEqual(ipsae['A_pos'], 0.8)
        self.assertAlmostEqual(interaction['A_pos'], 0.8)
        self.assertAlmostEqual(interaction['A_neg'], 2.0)
        self.assertAlmostEqual(interaction['B_pos'], 3.0)
        table = loto_results.comparison_table(self.labels, {'GraPPI': run}, features)
        egfr = table.set_index('target_id').loc['EGFR']
        self.assertEqual((egfr['n'], egfr['n_excluded']), (2, 1))
        self.assertAlmostEqual(egfr['AF3_ipSAE_AP'], 1.0)
        self.assertAlmostEqual(egfr['AF3_interaction_AP'], 0.5)
        self.assertAlmostEqual(egfr['AF3_interaction_AUROC'], 0.0)
        aggregate = loto_results.aggregate_table({'GraPPI': run}, features).set_index('model')
        self.assertAlmostEqual(aggregate.loc['AF3_ipSAE', 'AP_median'], 0.75)
        self.assertAlmostEqual(aggregate.loc['AF3_interaction', 'AP_median'], 0.75)
        self.assertAlmostEqual(
            aggregate.loc['AF3_interaction', 'pooled_AP'],
            average_precision_score(features['AF3_interaction'].scores['label'],
                                    features['AF3_interaction'].scores['score']))

    def test_af3_uses_labeled_reference_if_no_run_is_ready(self):
        features = loto_results.load_af3_features(self.paper_file, self.labels, {})
        self.assertEqual(len(features['AF3_ipSAE'].scores), len(self.labels))
        table = loto_results.comparison_table(self.labels, {}, features).set_index('target_id')
        self.assertEqual((table.loc['EGFR', 'dataset_n'], table.loc['EGFR', 'n']), (3, 3))
        self.assertEqual(table['n_excluded'].sum(), 0)
        self.assertEqual(len(loto_results.aggregate_table({}, features)), 2)

    def test_af3_numeric_labels_with_unlabeled_rows_are_supported(self):
        rows = self.paper_rows.copy()
        rows['binder'] = rows['binder'].map({True: 1.0, False: 0.0})
        rows.to_csv(self.paper_file, index=False)
        features = loto_results.load_af3_features(self.paper_file, self.labels, {})
        self.assertEqual(len(features['AF3_ipSAE'].scores), 5)
        self.assertEqual(features['AF3_ipSAE'].scores['label'].sum(), 2)

    def test_af3_missing_mismatched_or_duplicate_labeled_samples_raise(self):
        run = loto_results.load_run(self.write_run('grappi'), 'grappi', self.labels)
        runs = {'GraPPI': run}
        cases = {}
        cases['missing'] = self.paper_rows.loc[self.paper_rows['binder_id'] != 'A_pos'].copy()
        cases['target'] = self.paper_rows.copy()
        cases['target'].loc[0, 'target_id'] = 'wrong'
        cases['label'] = self.paper_rows.copy()
        cases['label'].loc[0, 'binder'] = False
        cases['duplicate'] = pd.concat([self.paper_rows, self.paper_rows.iloc[[0]]], ignore_index=True)
        for case, rows in cases.items():
            with self.subTest(case=case):
                rows.to_csv(self.paper_file, index=False)
                message = 'duplicate sample' if case == 'duplicate' else 'do not match'
                with self.assertRaisesRegex(ValueError, message):
                    loto_results.load_af3_features(self.paper_file, self.labels, runs)

    def test_af3_missing_or_nonfinite_features_raise(self):
        run = loto_results.load_run(self.write_run('grappi'), 'grappi', self.labels)
        for column, value in (('af3_ipSAE_min', np.nan),
                              ('af3_rosetta_interface_dG_SASA_ratio', np.inf)):
            with self.subTest(column=column):
                rows = self.paper_rows.copy()
                rows.loc[0, column] = value
                rows.to_csv(self.paper_file, index=False)
                with self.assertRaisesRegex(ValueError, f'{column} is missing or non-finite'):
                    loto_results.load_af3_features(self.paper_file, self.labels, {'GraPPI': run})
        self.paper_rows.drop(columns=['af3_ipSAE_min']).to_csv(self.paper_file, index=False)
        with self.assertRaisesRegex(ValueError, 'missing columns.*af3_ipSAE_min'):
            loto_results.load_af3_features(self.paper_file, self.labels, {'GraPPI': run})
        rows = self.paper_rows.copy()
        rows['af3_ipSAE_min'] = rows['af3_ipSAE_min'].astype(object)
        rows.loc[0, 'af3_ipSAE_min'] = 'not-a-score'
        rows.to_csv(self.paper_file, index=False)
        with self.assertRaisesRegex(ValueError, 'af3_ipSAE_min contains non-numeric'):
            loto_results.load_af3_features(self.paper_file, self.labels, {'GraPPI': run})

    def test_af3_mismatched_cohorts_cannot_be_compared(self):
        run = loto_results.load_run(self.write_run('grappi'), 'grappi', self.labels)
        features = loto_results.load_af3_features(self.paper_file, self.labels, {'GraPPI': run})
        features['AF3_interaction'].scores = features['AF3_interaction'].scores.iloc[:-1]
        with self.assertRaisesRegex(ValueError, 'identical samples'):
            loto_results.comparison_table(self.labels, {'GraPPI': run}, features)
        with self.assertRaisesRegex(ValueError, 'identical samples'):
            loto_results.aggregate_table({'GraPPI': run}, features)

    def test_duplicate_scores_raise(self):
        directory = self.write_run('grappi')
        path = directory / loto_results.RUN_FILES['scores']
        scores = pd.read_csv(path)
        scores.loc[1, 'name'] = scores.loc[0, 'name']
        scores.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'duplicate sample'):
            loto_results.load_run(directory, 'grappi', self.labels)

    def test_disagreeing_labels_raise(self):
        directory = self.write_run('grappi')
        path = directory / loto_results.RUN_FILES['scores']
        scores = pd.read_csv(path)
        scores.loc[0, 'label'] = 0
        scores.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'disagree with the dataset'):
            loto_results.load_run(directory, 'grappi', self.labels)

    def test_disagreeing_probabilities_raise(self):
        directory = self.write_run('grappi')
        path = directory / loto_results.RUN_FILES['scores']
        scores = pd.read_csv(path)
        scores.loc[0, 'prob'] = 0.1
        scores.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'sigmoid'):
            loto_results.load_run(directory, 'grappi', self.labels)

    def test_disagreeing_reported_metric_raises(self):
        directory = self.write_run('grappi')
        path = directory / loto_results.RUN_FILES['targets']
        targets = pd.read_csv(path)
        targets.loc[0, 'auprc'] = 0.123
        targets.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'does not match the saved scores'):
            loto_results.load_run(directory, 'grappi', self.labels)

    def test_disagreeing_split_counts_raise(self):
        directory = self.write_run('grappi')
        path = directory / loto_results.RUN_FILES['splits']
        split = json.loads(path.read_text())
        split['folds']['EGFR']['val_pos'] = 2
        path.write_text(json.dumps(split))
        with self.assertRaisesRegex(ValueError, 'split counts disagree'):
            loto_results.load_run(directory, 'grappi', self.labels)

    def test_disagreeing_summary_raises(self):
        directory = self.write_run('grappi')
        path = directory / loto_results.RUN_FILES['summary']
        summary = json.loads(path.read_text())
        summary['per_target'][0]['metrics']['auprc'] = 0.123
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'summary disagrees with saved scores'):
            loto_results.load_run(directory, 'grappi', self.labels)

    def test_roundtrip_actual_training_exports(self):
        import pickle
        import torch
        import yaml
        from torch_geometric.data import HeteroData
        import train_exp_target_wbce as trainer

        graphs = {}
        for row in self.labels[self.labels['name'] != 'A_extra'].itertuples(index=False):
            graph = HeteroData()
            for node_type in ('receptor', 'ligand'):
                graph[node_type].x = torch.arange(970, dtype=torch.float32).reshape(2, 485) / 1000
            graph.name, graph.target_id = row.name, row.target_id
            graph.y = torch.tensor(float(row.label))
            graphs[row.name] = graph
        with (self.root / 'graphs.pkl').open('wb') as handle:
            pickle.dump(graphs, handle)
        config = {
            'system': {'seed': 42}, 'data': {'data_dir': str(self.root), 'graph_cache': 'graphs.pkl'},
            'encoder': {'dir': str(self.root), 'esm_dim': 480, 'embedding_type': 'esm'},
            'head': {'ft_hdim': 8, 'cross_attn_heads': 2, 'dropout': 0.0},
            'train': {'n_epochs': 1, 'batch_size': 2, 'pos_weight': 'global'},
            'output': {'dir': str(self.root / 'unused_default')},
        }
        config_path = self.root / 'training.yaml'
        config_path.write_text(yaml.safe_dump(config))

        def fake_precompute(encoder, names, sample_graphs, device, **kwargs):
            for graph in sample_graphs:
                for node_type in ('receptor', 'ligand'):
                    graph[node_type].x = graph[node_type].x[:, :8].contiguous()

        original_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with patch('torch.cuda.is_available', return_value=False), \
                    patch('utils.Training_modules.common_utils.load_ssl_config',
                          return_value={'hidden_dim_power': 3}), \
                    patch('model_collection.FineTuneModels.load_pretrained_encoder',
                          return_value=torch.nn.Identity()), \
                    patch('model_collection.FineTuneModels.precompute_embeddings', fake_precompute), \
                    contextlib.redirect_stdout(io.StringIO()):
                for mode in ('grappi', 'esm'):
                    trainer.run(config_path, mode=mode, output_dir=self.root / mode)
        finally:
            torch.set_num_threads(original_threads)
        runs = {
            'GraPPI': loto_results.load_run(self.root / 'grappi', 'grappi', self.labels),
            'ESM': loto_results.load_run(self.root / 'esm', 'esm', self.labels),
        }
        loto_results.check_comparable(runs)
        table = loto_results.comparison_table(self.labels, runs).set_index('target_id')
        self.assertEqual(table.loc['EGFR', 'n_excluded'], 1)
        self.assertEqual(len(loto_results.aggregate_table(runs)), 2)

    def execute_notebook(self, paths, working_dir=REPO_ROOT, paper_file=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        notebook_path = REPO_ROOT / 'exp_target_retraining' / 'loto_results_summary.ipynb'
        notebook = json.loads(notebook_path.read_text())
        self.assertEqual(notebook['nbformat'], 4)
        ids = [cell['id'] for cell in notebook['cells']]
        self.assertEqual(len(ids), len(set(ids)))
        namespace = {'__name__': '__main__'}
        try:
            with patch.object(Path, 'cwd', return_value=working_dir), \
                    patch.object(plt, 'show'), \
                    contextlib.redirect_stdout(io.StringIO()):
                for cell in notebook['cells']:
                    if cell['cell_type'] != 'code':
                        continue
                    exec(compile(''.join(cell['source']), str(notebook_path), 'exec'), namespace)
                    if cell['id'] == 'setup':
                        namespace['LABELS_FILE'] = self.labels_file
                        namespace['RUN_PATHS'] = paths
                        namespace['PAPER_DATA_FILE'] = paper_file or self.paper_file
            namespace['_figure_count'] = len(plt.get_fignums())
            return namespace
        finally:
            plt.close('all')

    def test_notebook_runs_with_no_finished_runs(self):
        namespace = self.execute_notebook({
            'GraPPI': (self.root / 'missing_grappi', 'grappi'),
            'ESM': (self.root / 'missing_esm', 'esm'),
        }, paper_file=self.root / 'missing_paper.csv')
        self.assertEqual(namespace['runs'], {})
        self.assertEqual(namespace['paper_features'], {})
        self.assertEqual(len(namespace['per_target']), 2)
        self.assertTrue(namespace['aggregates'].empty)
        self.assertEqual(namespace['_figure_count'], 0)
        self.assertTrue(all('waiting' in row['status'] for row in namespace['status_rows'][:2]))
        self.assertIn('source CSV missing', namespace['status_rows'][-1]['status'])

    def test_notebook_runs_with_paper_features_before_model_exports(self):
        namespace = self.execute_notebook({
            'GraPPI': (self.root / 'missing_grappi', 'grappi'),
            'ESM': (self.root / 'missing_esm', 'esm'),
        })
        self.assertEqual(namespace['runs'], {})
        self.assertEqual(set(namespace['paper_features']), {'AF3_ipSAE', 'AF3_interaction'})
        self.assertEqual(set(namespace['aggregates']['model']), {'AF3_ipSAE', 'AF3_interaction'})
        self.assertEqual(namespace['per_target']['n'].sum(), len(self.labels))
        self.assertEqual(namespace['_figure_count'], 2)
        self.assertIn('no model ready', namespace['status_rows'][-1]['status'])

    def test_notebook_finds_repository_from_notebook_directory(self):
        namespace = self.execute_notebook({
            'GraPPI': (self.root / 'missing_grappi', 'grappi'),
            'ESM': (self.root / 'missing_esm', 'esm'),
        }, working_dir=REPO_ROOT / 'exp_target_retraining')
        self.assertEqual(namespace['REPO_ROOT'], REPO_ROOT)

    def test_notebook_runs_with_one_finished_run(self):
        namespace = self.execute_notebook({
            'GraPPI': (self.write_run('grappi'), 'grappi'),
            'ESM': (self.root / 'missing_esm', 'esm'),
        })
        self.assertEqual(list(namespace['runs']), ['GraPPI'])
        self.assertIn('GraPPI_AP', namespace['per_target'])
        self.assertNotIn('ESM_AP', namespace['per_target'])
        self.assertIn('AF3_ipSAE_AP', namespace['per_target'])
        self.assertEqual(len(namespace['aggregates']), 3)
        self.assertEqual(namespace['_figure_count'], 2)

    def test_notebook_runs_with_two_finished_runs(self):
        namespace = self.execute_notebook({
            'GraPPI': (self.write_run('grappi'), 'grappi'),
            'ESM': (self.write_run('esm'), 'esm'),
        })
        self.assertEqual(set(namespace['runs']), {'GraPPI', 'ESM'})
        table = namespace['per_target'].set_index('target_id')
        self.assertAlmostEqual(table.loc['EGFR', 'AP_difference'], 0.5)
        self.assertAlmostEqual(table.loc['EGFR', 'AF3_ipSAE_AP'], 1.0)
        self.assertAlmostEqual(table.loc['EGFR', 'AF3_interaction_AP'], 0.5)
        self.assertEqual(len(namespace['aggregates']), 4)
        self.assertEqual(namespace['_figure_count'], 2)
        expected = set(namespace['METHOD_LABELS'].values())
        scatter_labels = {item.get_text() for item in namespace['axes'][0].get_legend().get_texts()}
        self.assertTrue(expected <= scatter_labels)
        curve_labels = {item.get_label() for item in namespace['ax'].lines}
        self.assertTrue(all(any(label in curve for curve in curve_labels) for label in expected))


if __name__ == '__main__':
    unittest.main()
