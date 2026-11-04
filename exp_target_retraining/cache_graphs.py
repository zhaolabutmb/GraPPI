#!/usr/bin/env python3
"""Build and cache GraPPI graph + ESM features for the experimental binder set.

Mirrors the existing two-step pipeline (database_process/mp_pdb_process.py for
multi-CPU graph building, database_process/Include_esm.ipynb for ESM features)
but runs them for the experimental PDBs and merges both into a single cached
object, as requested. All heavy lifting reuses existing repository functions:

    utils.inference_utils.resolve_inputs      (read chains.csv -> entries)
    utils.inference_utils.build_graph         (PDB -> HeteroData, 8A interface)
    utils.inference_utils.get_chain_sequences (receptor/ligand sequences)
    utils.inference_utils.EsmEmbedder         (ESM-2 per-residue embeddings)
    utils.inference_utils.apply_esm_features  (splice ESM into graph.x)

Stage 1 (CPU, multiprocessed): parse each PDB into a base-feature graph plus the
receptor/ligand sequences. Stage 2 (GPU, single process): load ESM-2 once and
splice embeddings into every graph. The result is pickled as {name: graph},
where each graph carries ``y`` (1 binder / 0 non-binder) and ``target_id``.

No model training happens here. Run once before train_exp_target_wbce.py.
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

def _build_one(entry, dist=8.0):
    """Stage-1 worker: PDB -> (name, base-feature graph, rec_seq, lig_seq).

    Returns a dict with status so failures are logged rather than fatal.
    Uses only the first (receptor, ligand) complex of the entry (all
    experimental rows are a single B,A complex).
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
    # Avoid PyTorch's per-tensor shared-memory file descriptors accumulating
    # in the parent while it retains all completed graphs.
    return pickle.dumps(_build_one(entry, dist), protocol=pickle.HIGHEST_PROTOCOL)


def _init_worker() -> None:
    torch.set_num_threads(1)


def build_graphs(entries: list[dict], dist: float, workers: int) -> list[dict]:
    """Build CPU graphs with fresh workers and completion-order progress."""
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


def load_label_map(labels_path: Path) -> pd.DataFrame:
    """Load labels.csv and return a frame indexed by stem(pdb_file).

    Keeps only labeled rows (non-empty ``binder``) and attaches integer labels.
    """
    frame = pd.read_csv(labels_path)
    required = {'pdb_file', 'target_id', 'label', 'binder'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'{labels_path}: missing columns {sorted(missing)}')
    frame = frame[frame['binder'].astype(str).str.strip() != ''].copy()
    frame['label'] = frame['label'].astype(int)
    frame['stem'] = frame['pdb_file'].map(lambda p: os.path.splitext(str(p))[0])
    if frame['stem'].duplicated().any():
        dup = frame.loc[frame['stem'].duplicated(), 'stem'].tolist()
        raise ValueError(f'{labels_path}: duplicate pdb stems {dup[:5]}')
    return frame.set_index('stem')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=REPO_ROOT.parent / 'GraPPI_data' / 'exp_binder_data')
    parser.add_argument('--chains-csv', default='chains.csv')
    parser.add_argument('--labels-csv', default='labels.csv')
    parser.add_argument('--input-pdbs', default='input_pdbs')
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

    label_df = load_label_map(labels_path)
    entries = resolve_inputs(str(pdb_dir), str(chains_path), 'pdb')
    # Keep only entries that have a label; warn about the rest.
    labeled = [e for e in entries if e['name'] in label_df.index]
    skipped_unlabeled = [e['name'] for e in entries if e['name'] not in label_df.index]
    if skipped_unlabeled:
        print(f'[info] {len(skipped_unlabeled)} input PDB(s) have no label; excluded '
              f'(e.g. {skipped_unlabeled[:3]}).')
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
        [{'name': r['name'], 'status': 'error', 'error': r['error']} for r in failed]
        + [{'name': r['name'], 'status': 'esm_error', 'error': r['error']} for r in esm_failed]
    )
    manifest_path = out_path.with_name('cache_failures.csv')
    manifest.to_csv(manifest_path, index=False)

    print(f'\nCached {len(cached)} graphs -> {out_path}')
    print(f'Failures ({len(failed) + len(esm_failed)}) -> {manifest_path}')


if __name__ == '__main__':
    main()
