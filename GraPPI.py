#!/usr/bin/env python
"""
GraPPI.py — Inference CLI for trained GraPPI encoders / task heads.

Runs one of four tasks on one or more protein complexes (a PDB file with
chain annotations describing the two binding partners):

    embed       Extract per-residue encoder embeddings (no task head).
    bind_score  Predict a binder/non-binder probability (classification).
    dg          Predict binding free energy (ΔG) (regression).
    mut_dg      Predict the effect of mutation on binding (ΔΔG) — requires
                both a wild-type PDB (-pdb) and a mutant PDB (-mut_pdb).

Chain annotation syntax (per complex): "<side_A_chains>,<side_B_chains>"
    "A,B"    -> side A = chain A,           side B = chain B
    "AB,C"   -> side A = chains A and B,    side B = chain C
Use '+' to separate multi-character chain IDs within one side (e.g. "AA+BB,C").
Multiple complexes in one PDB file can be given separated by ';'
    e.g. "A,B;A,C" -> two complexes: (A vs B) and (A vs C).

Input modes:
    Single complex:  -pdb complex.pdb   -chains "A,B"
    Batch (folder):  -pdb pdb_folder/   -chains chains.csv
                      chains.csv columns: pdb_file,chains
                      (pdb_file is the filename relative to the -pdb folder)

Default models (used only when -encoder_config_path / -task_model_config_path
are not given):
    embed       esm-6-1024  (6-layer encoder, hidden dim 1024, ESM node feats)
    bind_score  esm-2-1024 encoder + MLP (PoolHeadClassifier) head
    dg          esm-6-512  encoder + SVR head
    mut_dg      esm-2-1024 encoder + SVR head

Downstream task heads (bind_score/dg/mut_dg) always use the frozen encoder's
Jumping-Knowledge-aggregated (mean) embeddings, matching how they were
trained; `embed` always returns the raw last-layer encoder output.

NOTE: No checkpoints are bundled with this repository. Default paths are
looked up under ../GraPPI_data/trained_data_ssl_finetune/ (sibling of this
repo), mirroring the directory layout produced by train_unified.py. If a
default checkpoint is missing, the script raises a clear error telling you
exactly which file is expected — either place a trained checkpoint there, or
point explicitly at your own via -encoder_config_path / -task_model_config_path.

Custom checkpoints:
    -encoder_config_path may point to:
        - a directory containing both ssl_edge_config.json and ssl_edge_best.pt
        - the ssl_edge_config.json file itself (checkpoint assumed to be
          ssl_edge_best.pt in the same directory)
        - the encoder .pt checkpoint itself (config assumed to be
          ssl_edge_config.json in the same directory)
    -task_model_config_path points DIRECTLY at a task-head checkpoint file:
        - a .pt file  -> a PyTorch pool+head model (architecture is assumed
          to match this repo's default config for the task, i.e.
          model_configs/config_disc_binder.yaml / config_dG.yaml /
          config_ddG.yaml — only the weights differ from the default).
        - a .pkl file -> a scikit-learn model (self-contained, no
          architecture assumptions needed).
    -task_model_config_path requires -encoder_config_path to also be given.

Examples:
    python GraPPI.py -pdb complex.pdb -chains A,B -task embed
    python GraPPI.py -pdb complex.pdb -chains A,B -task bind_score
    python GraPPI.py -pdb complex.pdb -chains A,B -task dg
    python GraPPI.py -pdb complex.pdb -chains A,B -task mut_dg -mut_pdb complex_mut.pdb
    python GraPPI.py -pdb pdb_folder/ -chains chains.csv -task dg -output_dir results/
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model_collection.FineTuneModels import precompute_embeddings, get_pool_input_dim
from utils.Training_modules.common_utils import load_ssl_config
from utils.inference_utils import (
    get_model_dirs, find_best_ml_fold, select_best_fold_by_key,
    resolve_inputs, resolve_mut_pdb_path, build_graph, get_chain_sequences,
    EsmEmbedder, apply_esm_features, load_encoder, build_poolhead_model,
    load_torch_head_state, load_pickle_model, extract_graph_features, extract_ddg_features,
    NoInterfaceError,
)

MODEL_CONFIGS_DIR = os.path.join(REPO_ROOT, 'model_configs')
DEFAULT_MODEL_ROOT = os.path.normpath(os.path.join(REPO_ROOT, '..', 'GraPPI_data', 'trained_data_ssl_finetune'))

# Downstream task heads are always trained on JK(mean)-aggregated embeddings.
USE_JK = True
JK_MODE = 'mean'

# Default encoder architecture per task (embed depends on -emb_type).
DEFAULT_ENCODERS = {
    ('embed', 'esm'): {'num_layers': 6, 'hidden_dim_power': 10, 'embedding_type': 'esm'},
    ('embed', 'base'): {'num_layers': 6, 'hidden_dim_power': 10, 'embedding_type': 'base'},
    'bind_score': {'num_layers': 2, 'hidden_dim_power': 10, 'embedding_type': 'esm'},
    'dg': {'num_layers': 6, 'hidden_dim_power': 9, 'embedding_type': 'esm'},
    'mut_dg': {'num_layers': 2, 'hidden_dim_power': 10, 'embedding_type': 'esm'},
}
DEFAULT_ENCODER_EXTRA = {'hgt_heads': 4, 'message_style': 'gated_src'}

# Default task-head checkpoint filename templates (relative to the
# JK-suffixed encoder directory), plus the per-fold results/summary file used
# to automatically pick the fold with the best held-out test performance.
# model_key is the sklearn model name inside the ml results json ('ml'-tasks
# only); None for the classifier, which is scored via its own summary file.
DEFAULT_HEAD_TEMPLATES = {
    'bind_score': ('disc_binder_jkmean_fold{fold}_best.pt',
                    'disc_binder_jkmean_5fold_summary.json', None),
    'dg': ('ml_dg_jkmean_fold_{fold}_SVR.pkl',
           'ml_dg_jkmean_5fold_results.json', 'SVR'),
    'mut_dg': ('ml_ddg_jkmean_fold_{fold}_SVR.pkl',
               'ml_ddg_jkmean_5fold_results.json', 'SVR'),
}

# Where to read the (fixed) default head architecture from, for the 'torch'
# (.pt) checkpoint case. {task: (yaml_file, top_level_key)}
HEAD_ARCH_YAML = {
    'bind_score': ('config_disc_binder.yaml', 'disc_binder'),
    'dg': ('config_dG.yaml', 'dg_reg'),
    'mut_dg': ('config_ddG.yaml', 'mutation'),
}


# ============================================================================
# Encoder / head resolution (CLI-specific; wraps utils/inference_utils.py)
# ============================================================================

def resolve_encoder_config(args, task):
    if args.encoder_config_path:
        cfg_path = args.encoder_config_path
        if os.path.isdir(cfg_path):
            json_path = os.path.join(cfg_path, 'ssl_edge_config.json')
            ckpt_path = os.path.join(cfg_path, 'ssl_edge_best.pt')
        elif cfg_path.endswith('.json'):
            json_path = cfg_path
            ckpt_path = os.path.join(os.path.dirname(cfg_path) or '.', 'ssl_edge_best.pt')
        else:
            ckpt_path = cfg_path
            json_path = os.path.join(os.path.dirname(cfg_path) or '.', 'ssl_edge_config.json')
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Encoder config JSON not found: {json_path}")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Encoder checkpoint not found: {ckpt_path}")
        with open(json_path, 'r') as f:
            enc_cfg = json.load(f)
        enc_cfg['checkpoint_path'] = ckpt_path
        return enc_cfg

    key = ('embed', args.emb_type) if task == 'embed' else task
    defaults = DEFAULT_ENCODERS[key]
    ssl_dir, _, _, _, _ = get_model_dirs(
        DEFAULT_MODEL_ROOT, defaults['num_layers'], defaults['hidden_dim_power'], defaults['embedding_type'],
    )
    ckpt_path = os.path.join(ssl_dir, 'ssl_edge_best.pt')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Default encoder checkpoint not found at: {ckpt_path}\n"
            f"Place a trained SSL checkpoint there, or supply -encoder_config_path explicitly."
        )
    enc_cfg = load_ssl_config(ckpt_path) or dict(defaults, **DEFAULT_ENCODER_EXTRA)
    enc_cfg['checkpoint_path'] = ckpt_path
    return enc_cfg


def resolve_head_path(args, task, enc_cfg):
    if args.task_model_config_path:
        if not args.encoder_config_path:
            raise ValueError("-task_model_config_path requires -encoder_config_path to also be specified.")
        path = args.task_model_config_path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Task model checkpoint not found: {path}")
        return path

    _, head_dir, _, _, _ = get_model_dirs(
        DEFAULT_MODEL_ROOT, enc_cfg['num_layers'], enc_cfg['hidden_dim_power'], enc_cfg['embedding_type'],
        use_jk=USE_JK, jk_mode=JK_MODE,
    )
    filename_template, results_filename, model_key = DEFAULT_HEAD_TEMPLATES[task]
    fold = 1
    try:
        results_path = os.path.join(head_dir, results_filename)
        if task == 'bind_score':
            fold, _, _ = select_best_fold_by_key(results_path, score_key='fold_val_score')
        else:
            fold, _, _ = find_best_ml_fold(results_path, model_key, metric='rp')
        print(f"Selected fold {fold} for default '{task}' head (best held-out test performance).")
    except FileNotFoundError:
        print(f"Could not find {os.path.join(head_dir, results_filename)}; defaulting to fold 1.")

    path = os.path.join(head_dir, filename_template.format(fold=fold))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Default '{task}' head checkpoint not found at: {path}\n"
            f"Place a trained checkpoint there, or supply -task_model_config_path "
            f"(together with -encoder_config_path)."
        )
    return path


def load_head_arch(task):
    yaml_file, key = HEAD_ARCH_YAML[task]
    with open(os.path.join(MODEL_CONFIGS_DIR, yaml_file), 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg[key]


# ============================================================================
# Per-task pipelines
# ============================================================================

def build_all_graphs(entries, dist_cutoff, embedder=None):
    """Build a flat list of {'tag', 'entry', 'receptor_chains', 'ligand_chains', 'graph'},
    skipping (with a printed warning) any complex that fails to build."""
    out = []
    for entry in entries:
        multi = len(entry['complexes']) > 1
        for cidx, (rec, lig) in enumerate(entry['complexes']):
            tag = f"{entry['name']}_complex{cidx + 1}" if multi else entry['name']
            try:
                graph = build_graph(entry['pdb_path'], rec, lig, entry['name'], dist_cutoff)
                if embedder is not None:
                    rec_seq, lig_seq = get_chain_sequences(entry['pdb_path'], rec, lig)
                    apply_esm_features(graph, rec_seq, lig_seq, embedder)
            except (NoInterfaceError, RuntimeError, ValueError) as e:
                print(f"  [skip] {tag}: {e}")
                continue
            out.append({'tag': tag, 'entry': entry, 'receptor_chains': rec, 'ligand_chains': lig, 'graph': graph})
    return out


def run_embed(args, entries, device):
    enc_cfg = resolve_encoder_config(args, 'embed')
    encoder = load_encoder(enc_cfg, device)
    embedder = EsmEmbedder(1280 if enc_cfg['embedding_type'] == 'esm' else 480, device) \
        if enc_cfg['embedding_type'] in ('esm', 'esm480') else None

    items = build_all_graphs(entries, args.dist, embedder=embedder)
    if not items:
        print("No complexes could be processed.")
        return

    manifest_rows = []
    encoder.eval()
    for item in items:
        loader = DataLoader([item['graph']], batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(device)
        with torch.no_grad():
            out = encoder(batch)
        rec_emb = out['receptor'].float().cpu()
        lig_emb = out['ligand'].float().cpu()
        save_path = os.path.join(args.output_dir, f"{item['tag']}_embeddings.pt")
        torch.save({
            'receptor': rec_emb, 'ligand': lig_emb,
            'receptor_chains': item['receptor_chains'], 'ligand_chains': item['ligand_chains'],
        }, save_path)
        manifest_rows.append({
            'name': item['tag'], 'pdb_file': item['entry']['pdb_path'],
            'receptor_chains': '+'.join(item['receptor_chains']),
            'ligand_chains': '+'.join(item['ligand_chains']),
            'n_receptor_nodes': rec_emb.shape[0], 'n_ligand_nodes': lig_emb.shape[0],
            'embedding_dim': rec_emb.shape[1], 'output_file': save_path,
        })
        print(f"  {item['tag']}: receptor {tuple(rec_emb.shape)}, ligand {tuple(lig_emb.shape)} -> {save_path}")

    manifest_path = os.path.join(args.output_dir, 'embed_manifest.csv')
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\nManifest written to {manifest_path}")


def run_single_graph_task(args, entries, device, task):
    enc_cfg = resolve_encoder_config(args, task)
    head_path = resolve_head_path(args, task, enc_cfg)
    is_torch_head = head_path.endswith('.pt') or head_path.endswith('.pth')

    encoder = load_encoder(enc_cfg, device)
    embedder = EsmEmbedder(1280 if enc_cfg['embedding_type'] == 'esm' else 480, device) \
        if enc_cfg['embedding_type'] in ('esm', 'esm480') else None

    items = build_all_graphs(entries, args.dist, embedder=embedder)
    if not items:
        print("No complexes could be processed.")
        return

    names = [item['tag'] for item in items]
    graphs = [item['graph'] for item in items]
    precompute_embeddings(
        encoder, names, graphs, device,
        batch_size=min(32, len(graphs)), embedding_type=enc_cfg['embedding_type'],
        esm_dict=None, use_amp=(device.type == 'cuda'), use_jk=USE_JK, jk_mode=JK_MODE,
    )
    del encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    hidden_dim = 2 ** enc_cfg['hidden_dim_power']
    pool_input_dim = get_pool_input_dim(
        hidden_dim, enc_cfg['embedding_type'], use_jk=USE_JK, jk_mode=JK_MODE,
        num_hgt_layers=enc_cfg['num_layers'],
    )

    rows = []
    if is_torch_head:
        arch_cfg = load_head_arch(task)
        model = build_poolhead_model(task, arch_cfg, pool_input_dim, device)
        load_torch_head_state(model, head_path, device)
        loader = DataLoader(graphs, batch_size=min(32, len(graphs)), shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch).squeeze(-1)
                preds.append(out.cpu())
        preds = torch.cat(preds).numpy()
        for item, pred in zip(items, preds):
            row = {
                'PDB': item['tag'],
                'receptor_chains': '+'.join(item['receptor_chains']),
                'ligand_chains': '+'.join(item['ligand_chains']),
                'pdb_file': item['entry']['pdb_path'],
            }
            if task == 'bind_score':
                prob = float(torch.sigmoid(torch.tensor(pred)))
                row['probability'] = prob
                row['predicted_label'] = int(prob >= 0.5)
            else:
                row['predicted_dG'] = float(pred)
                row = {'PDB': row.pop('PDB'), 'predicted_dG': row.pop('predicted_dG'), **row}
            rows.append(row)
    else:
        reg = load_pickle_model(head_path)
        X = extract_graph_features(graphs)
        preds = reg.predict(X)
        for item, pred in zip(items, preds):
            rows.append({
                'PDB': item['tag'], 'predicted_dG': float(pred),
                'receptor_chains': '+'.join(item['receptor_chains']),
                'ligand_chains': '+'.join(item['ligand_chains']),
                'pdb_file': item['entry']['pdb_path'],
            })

    out_path = os.path.join(args.output_dir, f"{task}_results.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nResults written to {out_path}")
    print(pd.DataFrame(rows).to_string(index=False))


def run_mut_dg(args, entries, device):
    task = 'mut_dg'
    enc_cfg = resolve_encoder_config(args, task)
    head_path = resolve_head_path(args, task, enc_cfg)
    is_torch_head = head_path.endswith('.pt') or head_path.endswith('.pth')

    encoder = load_encoder(enc_cfg, device)
    embedder = EsmEmbedder(1280 if enc_cfg['embedding_type'] == 'esm' else 480, device) \
        if enc_cfg['embedding_type'] in ('esm', 'esm480') else None

    # Build wt/mut graph pairs.
    pairs = []
    for entry in entries:
        multi = len(entry['complexes']) > 1
        mut_pdb_path = resolve_mut_pdb_path(args.mut_pdb, entry)
        for cidx, (rec, lig) in enumerate(entry['complexes']):
            tag = f"{entry['name']}_complex{cidx + 1}" if multi else entry['name']
            try:
                wt_graph = build_graph(entry['pdb_path'], rec, lig, entry['name'] + '_wt', args.dist)
                mut_graph = build_graph(mut_pdb_path, rec, lig, entry['name'] + '_mut', args.dist)
                if embedder is not None:
                    wt_rec_seq, wt_lig_seq = get_chain_sequences(entry['pdb_path'], rec, lig)
                    mut_rec_seq, mut_lig_seq = get_chain_sequences(mut_pdb_path, rec, lig)
                    apply_esm_features(wt_graph, wt_rec_seq, wt_lig_seq, embedder)
                    apply_esm_features(mut_graph, mut_rec_seq, mut_lig_seq, embedder)
            except (NoInterfaceError, RuntimeError, ValueError) as e:
                print(f"  [skip] {tag}: {e}")
                continue
            pairs.append({
                'tag': tag, 'entry': entry, 'receptor_chains': rec, 'ligand_chains': lig,
                'wt_graph': wt_graph, 'mut_graph': mut_graph, 'mut_pdb_path': mut_pdb_path,
            })

    if not pairs:
        print("No complexes could be processed.")
        return

    wt_graphs = [p['wt_graph'] for p in pairs]
    mut_graphs = [p['mut_graph'] for p in pairs]
    wt_names = [p['tag'] + '_wt' for p in pairs]
    mut_names = [p['tag'] + '_mut' for p in pairs]
    precompute_embeddings(
        encoder, wt_names, wt_graphs, device,
        batch_size=min(32, len(wt_graphs)), embedding_type=enc_cfg['embedding_type'],
        esm_dict=None, use_amp=(device.type == 'cuda'), use_jk=USE_JK, jk_mode=JK_MODE,
    )
    precompute_embeddings(
        encoder, mut_names, mut_graphs, device,
        batch_size=min(32, len(mut_graphs)), embedding_type=enc_cfg['embedding_type'],
        esm_dict=None, use_amp=(device.type == 'cuda'), use_jk=USE_JK, jk_mode=JK_MODE,
    )
    del encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    hidden_dim = 2 ** enc_cfg['hidden_dim_power']
    pool_input_dim = get_pool_input_dim(
        hidden_dim, enc_cfg['embedding_type'], use_jk=USE_JK, jk_mode=JK_MODE,
        num_hgt_layers=enc_cfg['num_layers'],
    )

    rows = []
    if is_torch_head:
        arch_cfg = load_head_arch(task)
        model = build_poolhead_model(task, arch_cfg, pool_input_dim, device)
        load_torch_head_state(model, head_path, device)
        mut_loader = DataLoader(mut_graphs, batch_size=min(32, len(mut_graphs)), shuffle=False)
        wt_loader = DataLoader(wt_graphs, batch_size=min(32, len(wt_graphs)), shuffle=False)
        preds = []
        with torch.no_grad():
            for mut_batch, wt_batch in zip(mut_loader, wt_loader):
                mut_batch = mut_batch.to(device)
                wt_batch = wt_batch.to(device)
                out = model(mut_batch, wt_batch).squeeze(-1)
                preds.append(out.cpu())
        preds = torch.cat(preds).numpy()
    else:
        reg = load_pickle_model(head_path)
        X = extract_ddg_features(mut_graphs, wt_graphs)
        preds = reg.predict(X)

    for pair, pred in zip(pairs, preds):
        rows.append({
            'name': pair['tag'], 'wt_pdb': pair['entry']['pdb_path'], 'mut_pdb': pair['mut_pdb_path'],
            'receptor_chains': '+'.join(pair['receptor_chains']),
            'ligand_chains': '+'.join(pair['ligand_chains']),
            'predicted_ddG': float(pred),
        })

    out_path = os.path.join(args.output_dir, 'mut_dg_results.csv')
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nResults written to {out_path}")
    print(pd.DataFrame(rows).to_string(index=False))


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='GraPPI inference CLI: embeddings / bind_score / dG / mutation ddG prediction.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('-pdb', required=True, help='PDB file or a folder of PDB files.')
    parser.add_argument('-chains', required=True,
                         help="Chain spec e.g. 'A,B' or 'AB,C', or a CSV file (pdb_file,chains) when -pdb is a folder.")
    parser.add_argument('-task', required=True, choices=['embed', 'bind_score', 'dg', 'mut_dg'])
    parser.add_argument('-emb_type', default='esm', choices=['esm', 'base'],
                         help="Only used when -task embed. Default: esm")
    parser.add_argument('-mut_pdb', default=None,
                         help='Mutant PDB file or folder. Required when -task mut_dg.')
    parser.add_argument('-output_dir', default=os.getcwd())
    parser.add_argument('-gpu_id', type=int, default=0)
    parser.add_argument('-encoder_config_path', default=None,
                         help='SSL encoder config/checkpoint (directory, .json, or .pt). Overrides defaults.')
    parser.add_argument('-task_model_config_path', default=None,
                         help='Task head checkpoint (.pt or .pkl), directly. Requires -encoder_config_path.')
    parser.add_argument('-dist', type=float, default=8.0,
                         help='Distance cutoff (Å) for interface/edge construction. Default: 8.0')
    args = parser.parse_args()

    if args.task == 'mut_dg' and not args.mut_pdb:
        parser.error("-mut_pdb is required when -task mut_dg")
    if args.task_model_config_path and not args.encoder_config_path:
        parser.error("-task_model_config_path requires -encoder_config_path to also be specified.")
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    entries = resolve_inputs(args.pdb, args.chains, 'pdb')
    print(f"Resolved {len(entries)} PDB entr{'y' if len(entries) == 1 else 'ies'}, "
          f"{sum(len(e['complexes']) for e in entries)} complex(es) total.")

    if args.task == 'embed':
        run_embed(args, entries, device)
    elif args.task in ('bind_score', 'dg'):
        run_single_graph_task(args, entries, device, args.task)
    elif args.task == 'mut_dg':
        run_mut_dg(args, entries, device)


if __name__ == '__main__':
    main()
