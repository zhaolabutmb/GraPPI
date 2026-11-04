"""Unit tests for the target-aware WBCE retraining split/metric logic.

These cover split and metric logic, a small synthetic CPU baseline run, and
cache worker behavior on a few real structures when the dataset is available.

Run: python -m unittest discover -s exp_target_retraining/tests
"""
import sys
import unittest
import contextlib
import io
import json
import pickle
import tempfile
import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_exp_target_wbce as trainer  # noqa: E402


def _toy_dataset():
    """15 targets with uneven sizes mimicking the real distribution shape."""
    sizes = {f'T{i:02d}': n for i, n in enumerate(
        [200, 50, 40, 30, 25, 20, 18, 16, 14, 12, 10, 9, 8, 7, 6])}
    target_ids, labels = [], []
    rng = np.random.RandomState(0)
    for target, n in sizes.items():
        target_ids += [target] * n
        # ~10% positives, but guarantee at least 2 positives per target
        pos = max(2, n // 10)
        lab = [1] * pos + [0] * (n - pos)
        rng.shuffle(lab)
        labels += lab
    return target_ids, labels


class TestMakeLotoFolds(unittest.TestCase):
    def setUp(self):
        self.target_ids, self.labels = _toy_dataset()
        self.n_targets = len(set(self.target_ids))

    def test_one_fold_per_target_sorted(self):
        folds = trainer.make_loto_folds(self.target_ids, self.labels)
        self.assertEqual(len(folds), self.n_targets)
        self.assertEqual([f['target'] for f in folds], sorted(set(self.target_ids)))

    def test_val_is_single_target_and_partitions(self):
        n = len(self.target_ids)
        for f in trainer.make_loto_folds(self.target_ids, self.labels):
            val = set(int(i) for i in f['val_idx'])
            tr = set(int(i) for i in f['train_idx'])
            self.assertTrue(all(self.target_ids[i] == f['target'] for i in val))
            self.assertTrue(all(self.target_ids[i] != f['target'] for i in tr))
            self.assertEqual(tr | val, set(range(n)))
            self.assertEqual(tr & val, set())

    def test_every_sample_validated_exactly_once(self):
        counts = {}
        for f in trainer.make_loto_folds(self.target_ids, self.labels):
            for i in f['val_idx']:
                counts[int(i)] = counts.get(int(i), 0) + 1
        self.assertEqual(sorted(counts), list(range(len(self.target_ids))))
        self.assertTrue(all(c == 1 for c in counts.values()))

    def test_deterministic(self):
        a = trainer.make_loto_folds(self.target_ids, self.labels)
        b = trainer.make_loto_folds(self.target_ids, self.labels)
        self.assertEqual([f['target'] for f in a], [f['target'] for f in b])

    def test_needs_two_targets(self):
        with self.assertRaises(ValueError):
            trainer.make_loto_folds(['A', 'A', 'A'], [0, 1, 0])

    def test_single_class_training_pool_raises(self):
        # All positives in target A, all negatives in B: holding out either leaves
        # a single-class training pool.
        with self.assertRaises(ValueError):
            trainer.make_loto_folds(['A', 'A', 'B', 'B'], [1, 1, 0, 0])


class TestPosWeight(unittest.TestCase):
    def test_global(self):
        self.assertAlmostEqual(
            trainer.resolve_pos_weight('global', 394, 3275), 3275 / 394, places=6)

    def test_explicit_float(self):
        self.assertEqual(trainer.resolve_pos_weight(9.4, 394, 3275), 9.4)

    def test_zero_positive_global_raises(self):
        with self.assertRaises(ValueError):
            trainer.resolve_pos_weight('global', 0, 10)


class TestBinaryMetrics(unittest.TestCase):
    def test_perfect_separation(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        m = trainer.binary_metrics(y, p)
        self.assertAlmostEqual(m['auroc'], 1.0)
        self.assertAlmostEqual(m['auprc'], 1.0)
        self.assertEqual((m['n'], m['n_pos'], m['n_neg']), (4, 2, 2))

    def test_single_class_is_nan(self):
        m = trainer.binary_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
        self.assertTrue(np.isnan(m['auroc']))
        self.assertTrue(np.isnan(m['auprc']))


class TestCudaDeviceSelection(unittest.TestCase):
    def test_explicit_id_overrides_config(self):
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.device_count', return_value=2):
            self.assertEqual(str(trainer.resolve_device(0, 1)), 'cuda:1')

    def test_config_id_used_when_no_override(self):
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.device_count', return_value=2):
            self.assertEqual(str(trainer.resolve_device(1)), 'cuda:1')

    def test_cpu_fallback_when_no_gpu_requested(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(str(trainer.resolve_device(0)), 'cpu')

    def test_explicit_gpu_requires_cuda(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA is not available'):
                trainer.resolve_device(0, 1)

    def test_invalid_gpu_ids_raise(self):
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.device_count', return_value=2):
            with self.assertRaisesRegex(ValueError, 'non-negative'):
                trainer.resolve_device(0, -1)
            with self.assertRaisesRegex(ValueError, '2 device\\(s\\) visible'):
                trainer.resolve_device(0, 2)

    def test_cli_passes_gpu_id_to_training(self):
        argv = ['train_exp_target_wbce.py', '--mode', 'esm', '--cuda-id', '1']
        with patch.object(sys, 'argv', argv), patch.object(trainer, 'run') as run:
            trainer.main()
        run.assert_called_once_with(
            Path(trainer.__file__).with_name('config_exp_target_wbce.yaml'),
            mode='esm', output_dir=None, cuda_id=1)

    def test_run_sets_requested_device_before_loading_cache(self):
        import torch
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                'system': {'cuda_id': 0},
                'data': {'data_dir': str(root), 'graph_cache': 'missing.pkl'},
                'encoder': {'esm_dim': 480}, 'head': {}, 'train': {},
                'output': {'dir': str(root / 'grappi')},
            }
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump(config))
            for mode, cuda_id in (('grappi', 0), ('esm', 1)):
                with self.subTest(mode=mode), \
                        patch('torch.cuda.is_available', return_value=True), \
                        patch('torch.cuda.device_count', return_value=2), \
                        patch('torch.cuda.set_device') as set_device, \
                        patch('utils.Training_modules.common_utils.set_seed'), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(FileNotFoundError):
                        trainer.run(config_path, mode=mode, cuda_id=cuda_id)
                    set_device.assert_called_once_with(torch.device('cuda', cuda_id))


class TestLoadLabelMap(unittest.TestCase):
    def test_real_labels_csv_15_targets(self):
        import cache_graphs
        labels_path = (REPO_ROOT.parent / 'GraPPI_data' / 'exp_binder_data' / 'labels.csv')
        if not labels_path.is_file():
            self.skipTest('experimental labels.csv not available')
        frame = cache_graphs.load_label_map(labels_path)
        self.assertEqual(len(frame), 3669)
        self.assertEqual(frame['target_id'].nunique(), 15)
        self.assertEqual(int((frame['label'] == 1).sum()), 394)
        self.assertEqual(int((frame['label'] == 0).sum()), 3275)


class TestEsmBaseline(unittest.TestCase):
    @staticmethod
    def graph(name, target, label, esm_dim=480):
        import torch
        from torch_geometric.data import HeteroData

        graph = HeteroData()
        for node_type in ('receptor', 'ligand'):
            graph[node_type].x = torch.arange(
                2 * (esm_dim + 5), dtype=torch.float32).reshape(2, esm_dim + 5) / 1000
        graph.name, graph.target_id = name, target
        graph.y = torch.tensor(float(label))
        return graph

    def test_uses_exact_esm_columns_and_preserves_labels(self):
        import torch
        for dim in (480, 1280):
            graph = self.graph('sample', 'target', 1, dim)
            expected = graph['receptor'].x[:, :dim].clone()
            self.assertEqual(trainer.prepare_esm_graphs([graph], dim), dim)
            torch.testing.assert_close(graph['receptor'].x, expected)
            self.assertEqual(tuple(graph['ligand'].x.shape), (2, dim))
            self.assertEqual((graph.name, graph.target_id, graph.y.item()),
                             ('sample', 'target', 1.0))

    def test_wrong_cache_width_is_rejected_before_mutation(self):
        graph = self.graph('good', 'target', 1)
        wrong = self.graph('wrong', 'target', 0, 1280)
        with self.assertRaisesRegex(ValueError, 'wrong'):
            trainer.prepare_esm_graphs([graph, wrong], 480)
        self.assertEqual(graph['receptor'].x.shape[1], 485)

    def test_cpu_loto_run_outputs_raw_scores_without_encoder(self):
        import torch
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = {
                f'{target}_{label}': self.graph(f'{target}_{label}', target, label)
                for target in ('A', 'B') for label in (0, 1)
            }
            cache_path = root / 'graphs.pkl'
            with cache_path.open('wb') as handle:
                pickle.dump(cache, handle)
            config = {
                'system': {'seed': 42},
                'data': {'data_dir': str(root), 'graph_cache': 'graphs.pkl'},
                'encoder': {'dir': '/nonexistent/encoder', 'esm_dim': 480},
                'head': {'ft_hdim': 8, 'cross_attn_heads': 2, 'dropout': 0.0},
                'train': {'n_epochs': 1, 'batch_size': 2, 'pos_weight': 'global'},
                'output': {'dir': str(root / 'grappi')},
            }
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump(config))
            previous_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                with patch('torch.cuda.is_available', return_value=False), \
                        patch('model_collection.FineTuneModels.load_pretrained_encoder',
                              side_effect=AssertionError('ESM baseline must bypass encoder')), \
                        patch('model_collection.FineTuneModels.precompute_embeddings',
                              side_effect=AssertionError('ESM baseline must bypass precompute')), \
                        contextlib.redirect_stdout(io.StringIO()):
                    summary = trainer.run(config_path, mode='esm')
            finally:
                torch.set_num_threads(previous_threads)
            output = root / 'esm480_exp_target_loto_wbce'
            self.assertFalse((root / 'grappi').exists())
            self.assertEqual(summary['encoder_mode'], 'esm')
            self.assertEqual(summary['device'], 'cpu')
            self.assertIsNone(summary['encoder_dir'])
            self.assertEqual(summary['pool_input_dim'], 480)
            self.assertEqual(summary['n_folds'], 2)
            self.assertEqual(summary['across_targets']['auprc']['n'], 2)
            with (output / 'exp_target_loto_raw_scores.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual({r['name'] for r in rows}, set(cache))
            for row in rows:
                self.assertAlmostEqual(
                    float(row['prob']), 1 / (1 + np.exp(-float(row['logit']))), places=6)
            with (output / 'exp_target_loto_summary.json').open() as handle:
                self.assertEqual(json.load(handle)['pooled']['n'], 4)
            self.assertEqual(len(list(output.glob('*_best.pt'))), 2)
            with cache_path.open('rb') as handle:
                unchanged = pickle.load(handle)
            self.assertEqual(unchanged['A_0']['receptor'].x.shape[1], 485)


class TestGraphWorkers(unittest.TestCase):
    def test_invalid_worker_count(self):
        import cache_graphs
        with self.assertRaises(ValueError):
            cache_graphs.build_graphs([], 8.0, 0)

    def test_spawn_builds_two_real_graphs(self):
        import cache_graphs
        import torch
        root = REPO_ROOT.parent / 'GraPPI_data' / 'exp_binder_data'
        if not (root / 'input_pdbs').is_dir():
            self.skipTest('experimental PDBs not available')
        entries = cache_graphs.resolve_inputs(
            str(root / 'input_pdbs'), str(root / 'chains.csv'), 'pdb')[:2]
        results = cache_graphs.build_graphs(entries, 8.0, workers=2)
        self.assertEqual([r['name'] for r in results], [e['name'] for e in entries])
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(result['status'], 'ok', result)
            self.assertEqual(result['graph']['receptor'].x.shape[1], 25)
            self.assertEqual(len(result['rec_seq']), result['graph']['receptor'].num_nodes)
            self.assertEqual(len(result['lig_seq']), result['graph']['ligand'].num_nodes)
        expected = cache_graphs._build_one(entries[0], 8.0)
        self.assertEqual(expected['status'], 'ok', expected)
        self.assertEqual(results[0]['rec_seq'], expected['rec_seq'])
        self.assertEqual(results[0]['lig_seq'], expected['lig_seq'])
        for actual_store, expected_store in zip(
                results[0]['graph'].stores, expected['graph'].stores):
            self.assertEqual(actual_store._key, expected_store._key)
            self.assertEqual(set(actual_store.keys()), set(expected_store.keys()))
            for key, value in expected_store.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(actual_store[key], value, rtol=0, atol=0)
                else:
                    self.assertEqual(actual_store[key], value)

    def test_many_real_graphs_under_low_fd_limit(self):
        import os
        import resource
        import torch
        import cache_graphs

        root = REPO_ROOT.parent / 'GraPPI_data' / 'exp_binder_data'
        if not (root / 'input_pdbs').is_dir() or not Path('/proc/self/fd').is_dir():
            self.skipTest('experimental PDBs or Linux descriptor statistics unavailable')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY and hard < 128:
            self.skipTest('hard file descriptor limit is below 128')
        entry = cache_graphs.resolve_inputs(
            str(root / 'input_pdbs'), str(root / 'chains.csv'), 'pdb')[0]
        open_before = len(os.listdir('/proc/self/fd'))
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (128, hard))
            results = cache_graphs.build_graphs([entry] * 16, 8.0, workers=2)
            self.assertEqual(len(results), 16)
            for result in results:
                self.assertEqual(result['status'], 'ok', result)
                self.assertEqual(result['name'], entry['name'])
                torch.testing.assert_close(
                    result['graph']['receptor'].x,
                    results[0]['graph']['receptor'].x,
                )
                torch.testing.assert_close(
                    result['graph']['receptor', 'receptor_ligand', 'ligand'].edge_index,
                    results[0]['graph']['receptor', 'receptor_ligand', 'ligand'].edge_index,
                )
            open_after = len(os.listdir('/proc/self/fd'))
            self.assertLessEqual(open_after - open_before, 12)
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


if __name__ == '__main__':
    unittest.main()
