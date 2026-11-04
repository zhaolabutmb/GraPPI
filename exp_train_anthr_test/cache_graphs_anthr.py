#!/usr/bin/env python3
"""Build and cache GraPPI graph + ESM features for the Anthropic binder set.

This is the ``exp_target_retraining/cache_graphs.py`` pipeline pointed at
``GraPPI_data/anthr_binder_data`` with that dataset's label schema. The
experimental (Zenodo) cache already exists and is reused unchanged; only the
Anthropic test set needs building.

Schema difference handled here
------------------------------
``exp_binder_data/labels.csv``  : binder_id, pdb_file, target_id, source, binder, label
``anthr_binder_data/labels.csv``: binder_id, target, design_model, campaign,
                                  binder_length, label

So the PDB stem comes from ``binder_id`` (``<binder_id>.pdb`` in ``input_pdbs/``)
and the target column is ``target``. Column names are CLI-overridable.

The 131 labeled designs without a released ``designed.cif`` have no prepared
PDB; they are not in ``chains.csv`` and are skipped here, exactly as in the
original validation pipeline. They are reported in the cache manifest so
coverage is never silently lost.

Stage 1 (CPU, multiprocessed): PDB -> base-feature graph + chain sequences.
Stage 2 (GPU, single process): one ESM-2 load, embeddings spliced into graph.x.
Output pickle maps {name: HeteroData} with ``y`` (1 binder / 0 non-binder),
``target_id`` and ``name`` attached. No training happens here.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.inference_utils import (  # noqa: E402
    resolve_inputs,
    build_graph,
    get_chain_sequences,
    EsmEmbedder,
    apply_esm_features,
    NoInterfaceError,
)


# ---------------------------------------------------------------------------
# Label loading (pure, unit-testable)
# ---------------------------------------------------------------------------
def load_label_map(labels_path: Path, id_column: str = 'binder_id',
                   target_column: str = 'target',
                   label_column: str = 'label') -> pd.DataFrame:
    """Load the Anthropic labels table, indexed by PDB stem (= binder_id).

    Keeps only rows with a usable 0/1 label; the 120 designs whose
    ``binder_final`` outcome is blank are dropped (they are not negatives).
    """
    frame = pd.read_csv(labels_path)
    required = {id_column, target_column, label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'{labels_path}: missing columns {sorted(missing)}')

    # A blank cell makes pandas read the whole column as float, so '1' arrives
    # as '1.0'; parse numerically instead of matching strings.
    numeric = pd.to_numeric(frame[label_column], errors='coerce')
    keep = numeric.isin([0, 1])
    dropped = int((~keep).sum())
    frame = frame.loc[keep].copy()
    frame['label'] = numeric.loc[keep].astype(int)
    frame['target_id'] = frame[target_column].astype(str)
    frame['stem'] = frame[id_column].astype(str).map(
        lambda v: os.path.splitext(v)[0])

    if frame['stem'].duplicated().any():
        dup = frame.loc[frame['stem'].duplicated(), 'stem'].tolist()
        raise ValueError(f'{labels_path}: duplicate design ids {dup[:5]}')
    frame.attrs['n_unlabeled_dropped'] = dropped
    return frame.set_index('stem')


# ---------------------------------------------------------------------------
# Stage 1 workers
# ---------------------------------------------------------------------------
def _build_one(entry, dist=8.0):
    """PDB -> (name, base-feature graph, receptor/ligand sequences).

    Failures are returned as status dicts rather than raised, so one bad
    structure cannot abort a 1,189-structure build.
    """
    name = entry['name']
    rec, lig = entry['complexes'][0]
    try:
        graph = build_graph(entry['pdb_path'], rec, lig, name, float(dist))
        rec_seq, lig_seq = get_chain_sequences(entry['pdb_path'], rec, lig)
    except (NoInterfaceError, RuntimeError, ValueError, FileNotFoundError) as exc:
        return {'name': name, 'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
    return {
        'name': name,
        'status': 'ok',
        'graph': graph,
        'rec_seq': rec_seq,
        'lig_seq': lig_seq,
        'receptor_chains': '+'.join(rec),
        'ligand_chains': '+'.join(lig),
    }


def _build_one_serialized(entry, dist):
    # Serialize in the worker: returning live tensors leaks one shared-memory
    # file descriptor per tensor into the parent, which exhausted the 1024-fd
    # limit on the experimental run ("received 0 items of ancdata").
    return pickle.dumps(_build_one(entry, dist), protocol=pickle.HIGHEST_PROTOCOL)


def _init_worker() -> None:
    torch.set_num_threads(1)


def build_graphs(entries: list[dict], dist: float, workers: int) -> list[dict]:
    """Build CPU graphs with spawned workers and completion-order progress."""
    if workers < 1:
        raise ValueError('workers must be at least 1')
    thread_vars = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
    previous_env = {key: os.environ.get(key) for key in thread_vars}
    try:
        for key in thread_vars:
            os.environ[key] = '1'
        print(f'Starting {workers} spawn workers (1 CPU thread per worker)...',
              flush=True)
        results = {}
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context('spawn'),
            initializer=_init_worker,
        ) as executor:
            futures = {
                executor.submit(_build_one_serialized, entry, dist): i
                for i, entry in enumerate(entries)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc='graphs'):
                result = pickle.loads(future.result())
                results[futures[future]] = result
                if result['status'] != 'ok':
                    tqdm.write(f"[graphs] {result['name']}: {result['error']}")
        return [results[i] for i in range(len(entries))]
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=REPO_ROOT.parent / 'GraPPI_data' / 'anthr_binder_data')
    parser.add_argument('--chains-csv', default='chains.csv')
    parser.add_argument('--labels-csv', default='labels.csv')
    parser.add_argument('--input-pdbs', default='input_pdbs')
    parser.add_argument('--id-column', default='binder_id')
    parser.add_argument('--target-column', default='target')
    parser.add_argument('--label-column', default='label')
    parser.add_argument('--out', default='grappi_graphs_esm_8A/graphs_esm_8A.pkl',
                        help='Output pickle path, relative to --data-dir unless absolute.')
    parser.add_argument('--dist', type=float, default=8.0)
    parser.add_argument('--esm-dim', type=int, default=1280, choices=(1280, 480))
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--cuda-id', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0,
                        help='Process only the first N entries (debugging).')
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    chains_path = data_dir / args.chains_csv
    labels_path = data_dir / args.labels_csv
    pdb_dir = data_dir / args.input_pdbs
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = data_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_df = load_label_map(labels_path, args.id_column,
                              args.target_column, args.label_column)
    n_unlabeled = label_df.attrs.get('n_unlabeled_dropped', 0)
    print(f'{labels_path.name}: {len(label_df)} labeled designs '
          f'({int((label_df["label"] == 1).sum())} pos / '
          f'{int((label_df["label"] == 0).sum())} neg), '
          f'{n_unlabeled} unlabeled row(s) dropped.')

    entries = resolve_inputs(str(pdb_dir), str(chains_path), 'pdb')
    labeled = [e for e in entries if e['name'] in label_df.index]
    unlabeled_inputs = [e['name'] for e in entries if e['name'] not in label_df.index]
    prepared = {e['name'] for e in entries}
    no_structure = sorted(set(label_df.index) - prepared)
    if unlabeled_inputs:
        print(f'[info] {len(unlabeled_inputs)} prepared PDB(s) have no usable label; '
              f'excluded (e.g. {unlabeled_inputs[:3]}).')
    if no_structure:
        print(f'[info] {len(no_structure)} labeled design(s) have no prepared PDB '
              f'(no released designed.cif); skipped (e.g. {no_structure[:3]}).')
    if args.limit:
        labeled = labeled[: args.limit]
    print(f'Building graphs for {len(labeled)} labeled complexes '
          f'(dist={args.dist} A, {args.workers} CPU workers)...')

    # ---- Stage 1: multiprocessed CPU graph + sequence build ----
    results = build_graphs(labeled, args.dist, args.workers)
    ok = [r for r in results if r['status'] == 'ok']
    failed = [r for r in results if r['status'] != 'ok']
    print(f'  built {len(ok)}; failed/skipped {len(failed)}')

    # ---- Stage 2: single-GPU ESM pass spliced into graph.x ----
    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Loading ESM-2 (dim {args.esm_dim}) on {device}...')
    embedder = EsmEmbedder(args.esm_dim, device)

    cached: dict = {}
    esm_failed = []
    for r in tqdm(ok, desc='esm'):
        name = r['name']
        graph = r['graph']
        try:
            apply_esm_features(graph, r['rec_seq'], r['lig_seq'], embedder)
        except (RuntimeError, ValueError) as exc:
            esm_failed.append({'name': name, 'error': f'{type(exc).__name__}: {exc}'})
            continue
        row = label_df.loc[name]
        graph.y = torch.tensor(float(int(row['label'])))
        graph.target_id = str(row['target_id'])
        graph.name = name
        cached[name] = graph

    with open(out_path, 'wb') as handle:
        pickle.dump(cached, handle)

    manifest = pd.DataFrame(
        [{'name': n, 'status': 'no_structure', 'error': 'no prepared PDB'}
         for n in no_structure]
        + [{'name': r['name'], 'status': 'graph_error', 'error': r['error']} for r in failed]
        + [{'name': r['name'], 'status': 'esm_error', 'error': r['error']} for r in esm_failed]
    )
    manifest_path = out_path.with_name('cache_failures.csv')
    manifest.to_csv(manifest_path, index=False)

    n_pos = sum(1 for g in cached.values() if float(g.y) > 0)
    print(f'\nCached {len(cached)} graphs ({n_pos} pos / {len(cached) - n_pos} neg) '
          f'-> {out_path}')
    print(f'Coverage: {len(cached)}/{len(label_df)} labeled designs')
    print(f'Unscored ({len(manifest)}) -> {manifest_path}')


if __name__ == '__main__':
    main()
