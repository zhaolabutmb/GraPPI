#!/usr/bin/env python3
"""Build base-feature (non-ESM) graph caches for the GraPPI-base encoder.

Why this exists
---------------
``apply_esm_features`` builds ``cat(esm_emb, base[:, 20:])``: it **replaces**
the 20-dim one-hot amino-acid block with the 1,280-dim ESM embedding. The ESM
caches therefore hold 1,285-dim nodes and no longer contain the 25-dim base
features that the ``2layers_10hdim`` (``embedding_type: base``) encoder expects,
and ESM embeddings cannot be inverted back into one-hot. The base features must
be rebuilt from the PDBs.

The upside: this is stage 1 only. No ESM model, no GPU, no network — just the
multiprocessed CPU graph build, reusing the exact worker functions from
``cache_graphs_anthr.py`` so the two caches differ in features alone.

Both datasets are supported through ``--dataset``, which only presets that
dataset's label-column names; everything else is shared.

    exp   : labels.csv has pdb_file / target_id   (3,669 labeled designs)
    anthr : labels.csv has binder_id / target     (1,320 labeled, 1,189 built)

Output is a pickle of {name: HeteroData} with 25-dim node features, ``y``,
``target_id`` and ``name`` — the same structure as the ESM caches.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _path in (str(REPO_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.inference_utils import resolve_inputs  # noqa: E402

from cache_graphs_anthr import build_graphs, load_label_map  # noqa: E402

BASE_NODE_DIM = 25

# Per-dataset label-schema presets; see the module docstring.
DATASETS = {
    'exp': {
        'data_dir': REPO_ROOT.parent / 'GraPPI_data' / 'exp_binder_data',
        'id_column': 'pdb_file',
        'target_column': 'target_id',
        'expected': 3669,
    },
    'anthr': {
        'data_dir': REPO_ROOT.parent / 'GraPPI_data' / 'anthr_binder_data',
        'id_column': 'binder_id',
        'target_column': 'target',
        'expected': 1320,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=sorted(DATASETS), required=True,
                        help='Which dataset to build; presets its label columns.')
    parser.add_argument('--data-dir', type=Path,
                        help='Override the dataset root.')
    parser.add_argument('--chains-csv', default='chains.csv')
    parser.add_argument('--labels-csv', default='labels.csv')
    parser.add_argument('--input-pdbs', default='input_pdbs')
    parser.add_argument('--id-column', help='Override the design-id column.')
    parser.add_argument('--target-column', help='Override the target column.')
    parser.add_argument('--label-column', default='label')
    parser.add_argument('--out', default='grappi_graphs_base_8A/graphs_base_8A.pkl',
                        help='Output pickle path, relative to the data dir unless absolute.')
    parser.add_argument('--dist', type=float, default=8.0)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--limit', type=int, default=0,
                        help='Process only the first N entries (debugging).')
    args = parser.parse_args()

    preset = DATASETS[args.dataset]
    data_dir = (args.data_dir or preset['data_dir']).resolve()
    id_column = args.id_column or preset['id_column']
    target_column = args.target_column or preset['target_column']

    chains_path = data_dir / args.chains_csv
    labels_path = data_dir / args.labels_csv
    pdb_dir = data_dir / args.input_pdbs
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = data_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_df = load_label_map(labels_path, id_column, target_column, args.label_column)
    print(f'{args.dataset}: {len(label_df)} labeled designs '
          f'({int((label_df["label"] == 1).sum())} pos / '
          f'{int((label_df["label"] == 0).sum())} neg), '
          f'{label_df.attrs.get("n_unlabeled_dropped", 0)} unlabeled row(s) dropped.')

    entries = resolve_inputs(str(pdb_dir), str(chains_path), 'pdb')
    labeled = [e for e in entries if e['name'] in label_df.index]
    prepared = {e['name'] for e in entries}
    no_structure = sorted(set(label_df.index) - prepared)
    if no_structure:
        print(f'[info] {len(no_structure)} labeled design(s) have no prepared PDB; '
              f'skipped (e.g. {no_structure[:3]}).')
    if args.limit:
        labeled = labeled[: args.limit]
    print(f'Building BASE-feature graphs for {len(labeled)} complexes '
          f'(dist={args.dist} A, {args.workers} CPU workers, no ESM pass)...')

    results = build_graphs(labeled, args.dist, args.workers)
    ok = [r for r in results if r['status'] == 'ok']
    failed = [r for r in results if r['status'] != 'ok']
    print(f'  built {len(ok)}; failed/skipped {len(failed)}')

    cached: dict = {}
    for r in ok:
        graph = r['graph']
        for node_type in ('receptor', 'ligand'):
            width = graph[node_type].x.shape[1]
            if width != BASE_NODE_DIM:
                raise ValueError(
                    f"{r['name']}: expected {BASE_NODE_DIM}-dim base {node_type} "
                    f'features, got {width}. Do not run this script on graphs that '
                    'already carry ESM features.')
        row = label_df.loc[r['name']]
        graph.y = torch.tensor(float(int(row['label'])))
        graph.target_id = str(row['target_id'])
        graph.name = r['name']
        cached[r['name']] = graph

    with open(out_path, 'wb') as handle:
        pickle.dump(cached, handle)

    manifest = pd.DataFrame(
        [{'name': n, 'status': 'no_structure', 'error': 'no prepared PDB'}
         for n in no_structure]
        + [{'name': r['name'], 'status': 'graph_error', 'error': r['error']}
           for r in failed])
    manifest_path = out_path.with_name('cache_failures.csv')
    manifest.to_csv(manifest_path, index=False)

    n_pos = sum(1 for g in cached.values() if float(g.y) > 0)
    print(f'\nCached {len(cached)} base graphs ({n_pos} pos / {len(cached) - n_pos} neg) '
          f'-> {out_path}')
    print(f'Coverage: {len(cached)}/{len(label_df)} labeled designs')
    print(f'Unscored ({len(manifest)}) -> {manifest_path}')


if __name__ == '__main__':
    main()
