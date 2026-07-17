import os
import json
import pickle

import torch
import numpy as np
from torch_geometric.loader import DataLoader

from model_collection.FineTuneModels import (
    load_pretrained_encoder,
    precompute_embeddings,
    get_pool_input_dim,
    create_poolhead_model,
)
from utils.Training_modules.common_utils import (
    MODEL_INIT_DIM,
    EDGE_IN_DIM,
    METADATA,
    filter_samples_finetune,
)


# ---------------------------------------------------------------------------
# Path / config helpers
# ---------------------------------------------------------------------------

def get_model_dirs(ssl_trained_data_root, num_layer, hidden_dim, embedding_type,
                   use_jk=True, jk_mode='mean', if_estra_path=False):
    """Return standard directory paths and embedding tags derived from hyperparameters.

    Returns
    -------
    ssl_dir : str
        Root SSL checkpoint directory.
    ft_dir : str
        Fine-tuning checkpoint directory (with jk suffix, e.g. ``_jkmean``).
    hidden_dim_val : int
        Actual hidden dimension (2 ** hidden_dim).
    file_emb_tag : str
        Tag appended to checkpoint filenames ('_esm480', or '' for base/esm).
    esm_suffix : str
        Suffix for PDB data directories ('_esm', '_esm480', or '').
    """
    emb_suffix = f'_{embedding_type}' if embedding_type != 'base' else ''
    ssl_dir = f'{ssl_trained_data_root}/{num_layer}layers_{hidden_dim}hdim{emb_suffix}'
    jk_suffix = f'_jk{jk_mode}' if use_jk else ''
    ft_dir = f'{ssl_trained_data_root}/{num_layer}layers_{hidden_dim}hdim{emb_suffix}{jk_suffix}'
    if if_estra_path:
        ft_dir = f'{ssl_trained_data_root}/test_for_new_mutation_model/{num_layer}layers_{hidden_dim}hdim{emb_suffix}{jk_suffix}'
    hidden_dim_val = 2 ** hidden_dim
    file_emb_tag = f'_{embedding_type}' if embedding_type not in ('base', 'esm') else ''
    esm_suffix = '_esm' if embedding_type == 'esm' else ('_esm480' if embedding_type == 'esm480' else '')
    return ssl_dir, ft_dir, hidden_dim_val, file_emb_tag, esm_suffix


def load_ssl_encoder_config(ssl_dir):
    """Load SSL encoder config and return (hgt_heads, dropout, message_style)."""
    cfg = json.load(open(f'{ssl_dir}/ssl_edge_config.json'))
    return cfg['hgt_heads'], cfg['dropout'], cfg['message_style']


def load_best_fold(summary_path):
    """Load a fold-summary JSON and return (best_fold_num, stored_rp, summary_dict)."""
    summary = json.load(open(summary_path))
    best_key = max(summary['fold_test_rp'], key=summary['fold_test_rp'].get)
    best_fold_num = int(best_key.split('_')[1])
    stored_rp = summary['fold_test_rp'][best_key]
    return best_fold_num, stored_rp, summary


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_samples_from_dir(directory):
    """Load all .pkl files from *directory* and extract their protein_graph.

    Returns
    -------
    dict[str, Data]
        Mapping from filename stem to the protein_graph object.
    """
    samples = {}
    for fn in sorted(os.listdir(directory)):
        if fn.endswith('.pkl'):
            name = fn.replace('.pkl', '')
            with open(os.path.join(directory, fn), 'rb') as f:
                st = pickle.load(f)
            samples[name] = st.protein_graph
    return samples


def load_ddg_test_pairs(pdb_root, embedding_type, min_nodes=5):
    """Load, filter, and pair WT/mutant graphs from the standard test-set directory.

    Parameters
    ----------
    pdb_root : str
        Root directory that contains the ``testset_mut*`` folder.
    embedding_type : str
        One of ``'base'``, ``'esm'``, or ``'esm480'``.
    min_nodes : int
        Minimum graph size passed to ``filter_samples_finetune``.

    Returns
    -------
    mut_names : list[str]
    mut_graphs : list[Data]
    wt_graphs : list[Data]
        Paired lists – ``mut_graphs[i]`` corresponds to ``wt_graphs[i]``.
    y_true_mut_aff : np.ndarray
    y_true_wt_aff : np.ndarray
    y_true_ddg : np.ndarray
        True ΔΔG = wt_aff − mut_aff, consistent with training convention.
    unique_wt_names : list[str]
    unique_wt_graphs : list[Data]
        De-duplicated WT entries (needed for efficient embedding).
    """
    esm_suffix = '_esm' if embedding_type == 'esm' else ('_esm480' if embedding_type == 'esm480' else '')
    test_dir = f'{pdb_root}/testset_mut{esm_suffix}'

    all_sttgs = {}
    for fn in sorted(os.listdir(test_dir)):
        if fn.endswith('.pkl'):
            name = fn.replace('.pkl', '')
            with open(os.path.join(test_dir, fn), 'rb') as f:
                all_sttgs[name] = pickle.load(f)

    wt_sttgs  = {k: v for k, v in all_sttgs.items() if '_' not in k}
    mut_sttgs = {k: v for k, v in all_sttgs.items() if '_' in k}
    wt_sttgs  = filter_samples_finetune(wt_sttgs,  min_nodes=min_nodes)
    mut_sttgs = filter_samples_finetune(mut_sttgs, min_nodes=min_nodes)

    wt_graphs_dict  = {k: v.protein_graph for k, v in wt_sttgs.items()}
    mut_graphs_dict = {k: v.protein_graph for k, v in mut_sttgs.items()}
    print(f"After filtering: {len(mut_graphs_dict)} mutants, {len(wt_graphs_dict)} wild-types")

    pairs = []
    for mut_name, mut_g in mut_graphs_dict.items():
        pdb_id = mut_name.split('_')[0]
        if pdb_id in wt_graphs_dict:
            wt_g = wt_graphs_dict[pdb_id]
            pairs.append((mut_name, mut_g, wt_g, mut_g.y.item(), wt_g.y.item()))
        else:
            print(f"  Missing WT for {mut_name}")

    print(f"Matched pairs: {len(pairs)}")
    mut_names      = [p[0] for p in pairs]
    mut_graphs     = [p[1] for p in pairs]
    wt_graphs      = [p[2] for p in pairs]
    y_true_mut_aff = np.array([p[3] for p in pairs])
    y_true_wt_aff  = np.array([p[4] for p in pairs])
    y_true_ddg     = y_true_wt_aff - y_true_mut_aff

    # De-duplicate WT graphs so each is encoded only once
    seen, unique_wt_names, unique_wt_graphs = set(), [], []
    for p in pairs:
        pdb_id = p[0].split('_')[0]
        if pdb_id not in seen:
            seen.add(pdb_id)
            unique_wt_names.append(pdb_id)
            unique_wt_graphs.append(p[2])

    return (mut_names, mut_graphs, wt_graphs,
            y_true_mut_aff, y_true_wt_aff, y_true_ddg,
            unique_wt_names, unique_wt_graphs)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def load_encoder_and_embed(ssl_dir, names, graphs, embedding_type,
                            num_layer, hidden_dim_val, hgt_heads, message_style,
                            device, use_jk=True, jk_mode='mean', batch_size=32):
    """Load the pretrained SSL encoder, precompute embeddings, then free GPU memory.

    Embeddings are stored in-place inside each graph object (as done by
    ``precompute_embeddings``), so the caller does not need a return value.
    """
    encoder = load_pretrained_encoder(
        checkpoint_path=f'{ssl_dir}/ssl_edge_best.pt',
        node_in_dim=MODEL_INIT_DIM[embedding_type],
        edge_in_dim=EDGE_IN_DIM,
        metadata=METADATA,
        hidden_dim=hidden_dim_val,
        num_hgt_layers=num_layer,
        hgt_heads=hgt_heads,
        device=device,
        message_style=message_style,
    )
    precompute_embeddings(
        encoder, names, graphs, device,
        batch_size=batch_size,
        embedding_type=embedding_type,
        use_amp=(device.type == 'cuda'),
        use_jk=use_jk,
        jk_mode=jk_mode,
    )
    del encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model building helpers
# ---------------------------------------------------------------------------

def build_and_load_regressor(ft_dir, file_emb_tag, best_fold_num,
                              pool_input_dim, dropout, device):
    """Build a ΔG regressor and load the best-fold checkpoint.

    Architecture (hidden dim, number of linear layers) is inferred from the
    checkpoint state-dict, so no manual specification is needed.

    Returns
    -------
    torch.nn.Module
        Model in eval mode on *device*.
    """
    ckpt_path = f'{ft_dir}/wt_dg{file_emb_tag}_jkmean_fold{best_fold_num}_best.pt'
    ckpt_sd = torch.load(ckpt_path, map_location=device)
    ft_hdim = ckpt_sd['regressor.0.weight'].shape[0]
    n_linear = sum(
        1 for k in ckpt_sd
        if k.startswith('regressor.') and k.endswith('.weight') and len(ckpt_sd[k].shape) == 2
    )
    model = create_poolhead_model(
        model_type='regressor',
        pool_input_dim=pool_input_dim,
        hidden_dim=ft_hdim,
        metadata=METADATA,
        dropout=dropout,
        pool_mode='cross_attn',
        cross_attn_queries=2,
        cross_attn_heads=4,
        regressor_layers=n_linear,
    ).to(device)
    model.load_state_dict(ckpt_sd)
    model.eval()
    print(f"Loaded ΔG regressor: {ckpt_path}  (ft_hdim={ft_hdim}, n_linear={n_linear})")
    return model


def build_and_load_ddg_regressor(ft_dir, best_fold_num, ddg_input_mode,
                                  pool_input_dim, dropout, device):
    """Build a ΔΔG regressor and load the best-fold checkpoint.

    Architecture is inferred from the checkpoint state-dict.

    Returns
    -------
    torch.nn.Module
        Model in eval mode on *device*.
    """
    ckpt_path = f'{ft_dir}/mut_ddg_jkmean_fold{best_fold_num}_best.pt'
    ckpt_sd = torch.load(ckpt_path, map_location=device)
    ft_hdim = ckpt_sd['regressor.0.weight'].shape[0]
    n_linear = sum(
        1 for k in ckpt_sd
        if k.startswith('regressor.') and k.endswith('.weight') and len(ckpt_sd[k].shape) == 2
    )
    cross_attn_queries = ckpt_sd['cross_attn_agg.receptor_queries'].shape[1]
    model = create_poolhead_model(
        model_type='ddg_regressor',
        pool_input_dim=pool_input_dim,
        hidden_dim=ft_hdim,
        metadata=METADATA,
        dropout=dropout,
        pool_mode='cross_attn',
        cross_attn_queries=cross_attn_queries,
        cross_attn_heads=2,
        regressor_layers=n_linear,
        ddg_input_mode=ddg_input_mode,
    ).to(device)
    model.load_state_dict(ckpt_sd)
    model.eval()
    print(f"Loaded ΔΔG regressor: {ckpt_path}  "
          f"(ft_hdim={ft_hdim}, n_linear={n_linear}, cross_attn_queries={cross_attn_queries})")
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference(graphs, model, device, batch_size=32):
    """Run forward pass and return (y_true, y_pred) as numpy arrays."""
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    y_true_all, y_pred_all = [], []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).squeeze(-1)
            y_true_all.append(batch.y.cpu())
            y_pred_all.append(pred.cpu())
    return torch.cat(y_true_all).numpy(), torch.cat(y_pred_all).numpy()


def run_ddg_inference(mut_graphs, wt_graphs, model, device, batch_size=32):
    """Run paired forward pass and return predicted mut_aff as a numpy array."""
    mut_loader = DataLoader(mut_graphs, batch_size=batch_size, shuffle=False)
    wt_loader  = DataLoader(wt_graphs,  batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for mut_batch, wt_batch in zip(mut_loader, wt_loader):
            mut_batch = mut_batch.to(device)
            wt_batch  = wt_batch.to(device)
            pred = model(mut_batch, wt_batch).squeeze(-1)
            preds.append(pred.cpu())
    return torch.cat(preds).numpy()


# ---------------------------------------------------------------------------
# ML model helpers (sklearn-based inference)
# ---------------------------------------------------------------------------

# Valid ML model names that map to keys used in the results JSON / pkl files
ML_MODEL_NAMES = ('GradientBoosting', 'RandomForest', 'SVR', 'DecisionTree')


def _ml_model_pkl_path(ml_dir, task, jk_suffix, model_name, fold_num=None):
    """Build the expected pkl path for a saved sklearn model."""
    if fold_num is not None:
        return os.path.join(ml_dir, f'ml_{task}{jk_suffix}_fold_{fold_num}_{model_name}.pkl')
    # disc_binder has no fold dimension
    return os.path.join(ml_dir, f'ml_{task}{jk_suffix}_{model_name}.pkl')


def find_best_ml_fold(results_path, model_name, metric='rp'):
    """Find the best fold for a given ML model from a results JSON.

    Parameters
    ----------
    results_path : str
        Path to an ``ml_*_results.json`` file.
    model_name : str
        One of ``ML_MODEL_NAMES``.
    metric : str
        Metric key to maximise (``'rp'`` for regression, ``'auc'`` for classification).

    Returns
    -------
    best_fold_num : int
    best_score : float
    results : dict
        The full loaded results dict.
    """
    results = json.load(open(results_path))
    model_results = results['ml'][model_name]
    best_fold, best_score = None, -float('inf')
    for key, val in model_results.items():
        if not key.startswith('fold_'):
            continue
        score = val['test'].get(metric, val['test'].get('auc', 0))
        if score > best_score:
            best_score = score
            best_fold = key
    best_fold_num = int(best_fold.split('_')[1])
    return best_fold_num, best_score, results


def load_ml_model(ml_dir, task, jk_suffix, model_name, fold_num=None):
    """Load a saved sklearn model pickle.

    Raises ``FileNotFoundError`` with an actionable message if the pkl
    does not exist (training modules need to be re-run with model saving
    enabled).
    """
    path = _ml_model_pkl_path(ml_dir, task, jk_suffix, model_name, fold_num)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ML model pickle not found: {path}\n"
            "Re-run the corresponding ML training module to save model pickles."
        )
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"Loaded ML model: {path}")
    return model


def extract_graph_features(graphs):
    """Mean-pool precomputed node embeddings → feature matrix [N, dim].

    Each graph's receptor + ligand node features are concatenated and
    averaged to produce a single vector.
    """
    from utils.Training_modules.ml_utils import mean_pool_graph
    return np.stack([mean_pool_graph(g) for g in graphs])


def extract_ddg_features(mut_graphs, wt_graphs):
    """Build paired ΔΔG features: [mut_pooled || wt_pooled] → [N, 2*dim]."""
    mut_X = extract_graph_features(mut_graphs)
    wt_X  = extract_graph_features(wt_graphs)
    return np.hstack([mut_X, wt_X])


def build_and_load_ml_dg_model(ml_dir, model_name, jk_suffix='_jkconcat'):
    """Load the best-fold ΔG ML model from saved pickles.

    Returns
    -------
    model : sklearn estimator
    best_fold_num : int
    best_rp : float
    """
    results_path = os.path.join(ml_dir, f'ml_dg{jk_suffix}_5fold_results.json')
    best_fold_num, best_rp, _ = find_best_ml_fold(results_path, model_name, metric='rp')
    print(f"Best ΔG fold for {model_name}: fold_{best_fold_num} (test rp={best_rp:.4f})")
    model = load_ml_model(ml_dir, 'dg', jk_suffix, model_name, best_fold_num)
    return model, best_fold_num, best_rp


def build_and_load_ml_ddg_model(ml_dir, model_name, jk_suffix='_jkconcat'):
    """Load the best-fold ΔΔG ML model from saved pickles.

    Returns
    -------
    model : sklearn estimator
    best_fold_num : int
    best_rp : float
    """
    results_path = os.path.join(ml_dir, f'ml_ddg{jk_suffix}_5fold_results.json')
    best_fold_num, best_rp, _ = find_best_ml_fold(results_path, model_name, metric='rp')
    print(f"Best ΔΔG fold for {model_name}: fold_{best_fold_num} (test rp={best_rp:.4f})")
    model = load_ml_model(ml_dir, 'ddg', jk_suffix, model_name, best_fold_num)
    return model, best_fold_num, best_rp


def build_and_load_ml_disc_model(ml_dir, model_name, jk_suffix='_jkconcat'):
    """Load the binder-discrimination ML classifier from saved pickle.

    Returns
    -------
    model : sklearn estimator
    test_auc : float
    """
    results_path = os.path.join(ml_dir, f'ml_disc_binder{jk_suffix}_results.json')
    results = json.load(open(results_path))
    # Disc binder has no fold structure — SVM key is 'SVM' in classifiers
    cls_key = 'SVM' if model_name == 'SVR' else model_name
    test_auc = results['ml'][cls_key]['test']['auc']
    print(f"Disc binder {cls_key}: test AUC={test_auc:.4f}")
    model = load_ml_model(ml_dir, 'disc_binder', jk_suffix, cls_key)
    return model, test_auc


def run_ml_dg_inference(graphs, model):
    """Run sklearn ΔG regression on graphs with precomputed embeddings.

    Returns
    -------
    y_true : np.ndarray
    y_pred : np.ndarray
    """
    X = extract_graph_features(graphs)
    y_pred = model.predict(X)
    y_true = np.array([g.y.item() for g in graphs])
    return y_true, y_pred


def run_ml_ddg_inference(mut_graphs, wt_graphs, model):
    """Run sklearn ΔΔG regression on paired graphs with precomputed embeddings.

    Returns y_pred of predicted **mutant affinity**.  The training target
    is mut_graph.y (= mut_aff) directly.
    To recover ΔΔG predictions use:  ``pred_ΔΔG = y_true_wt_aff − y_pred``
    """
    X = extract_ddg_features(mut_graphs, wt_graphs)
    return model.predict(X)


# ---------------------------------------------------------------------------
# Binder discrimination helpers
# ---------------------------------------------------------------------------
def load_disc_binder_test_data(
    ssl_dir,
    pdb_root,
    embedding_type,
    dist='8',
    min_nodes=5,
    ft_dir=None,
):
    """Load binder-discrimination test data grouped by negative source.

    Uses classifier_split_info.json (saved during training) to reproduce
    the exact test split, then loads each sample from its source directory.

    Split-info lookup priority:
    1) ft_dir/classifier_split_info.json (if ft_dir is provided and file exists)
    2) ssl_dir/classifier_split_info.json

    Parameters
    ----------
    ssl_dir : str
        SSL checkpoint directory (fallback location for classifier_split_info.json).
    pdb_root : str
        Root directory containing the PDB data folders.
    embedding_type : str
        One of 'base', 'esm', 'esm480'.
    dist : str
        Distance cutoff used in folder names (default '8').
    min_nodes : int
        Minimum receptor/ligand nodes for filtering.
    ft_dir : str | None
        Fine-tuning directory; checked first for classifier_split_info.json.

    Returns
    -------
    test_graphs : list[Data]
    test_labels : np.ndarray
        Binary labels (0 or 1).
    test_names : list[str]
    subset_indices : dict[str, list[int]]
        Maps subset name ('positive', 'swapped_abag', 'random_mut', 'preppi', etc.)
        to index lists into test_graphs/test_labels.
    """
    from utils.Training_modules.classifier_data_loader import (
        _get_pos_dir, _get_neg_dir,
    )

    # Resolve split-info path: ft_dir first, then ssl_dir
    split_info_path = None
    if ft_dir is not None:
        ft_split = os.path.join(ft_dir, 'classifier_split_info.json')
        if os.path.exists(ft_split):
            split_info_path = ft_split
            print(f"Using classifier split info from ft_dir: {split_info_path}")

    if split_info_path is None:
        ssl_split = os.path.join(ssl_dir, 'classifier_split_info.json')
        if os.path.exists(ssl_split):
            split_info_path = ssl_split
            print(f"Using classifier split info from ssl_dir: {split_info_path}")

    if split_info_path is None:
        raise FileNotFoundError(
            "classifier_split_info.json not found in either location:\n"
            f"  ft_dir:  {os.path.join(ft_dir, 'classifier_split_info.json') if ft_dir else 'None'}\n"
            f"  ssl_dir: {os.path.join(ssl_dir, 'classifier_split_info.json')}"
        )

    with open(split_info_path) as f:
        split_info = json.load(f)

    test_pos_names = split_info['test_pos_names']
    test_neg_names = split_info['test_neg_names']
    test_neg_sources = split_info['test_neg_sources']

    pos_dir = _get_pos_dir(pdb_root, dist, embedding_type)
    source_dirs = {
        'swapped_abag': _get_neg_dir(pdb_root, 'swapped_abag_sthg', dist, embedding_type),
        'random_mut':   _get_neg_dir(pdb_root, 'random_mut_sthg',   dist, embedding_type),
        'preppi':       _get_neg_dir(pdb_root, 'preppi_sthg',       dist, embedding_type),
    }

    def _try_load(directory, name):
        for variant in [name, name.upper(), name.lower()]:
            path = os.path.join(directory, variant + '.pkl')
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    return pickle.load(f)
        return None

    def _passes_filter(graph):
        try:
            if graph['receptor'].x.size(0) < min_nodes:
                return False
            if graph['ligand'].x.size(0) < min_nodes:
                return False
            edge_key = ('receptor', 'receptor_ligand', 'ligand')
            if edge_key not in graph.edge_index_dict:
                return False
            if graph[edge_key].edge_index.size(1) < 3:
                return False
            return True
        except Exception:
            return False

    graphs, labels, names = [], [], []
    subset_indices = {}

    # Positive test samples
    pos_indices = []
    for name in sorted(test_pos_names):
        st = _try_load(pos_dir, name)
        if st is None:
            continue
        g = st.protein_graph
        if not _passes_filter(g):
            continue
        g.y = torch.tensor(1.0)
        pos_indices.append(len(graphs))
        graphs.append(g)
        labels.append(1)
        names.append(name)
    subset_indices['positive'] = pos_indices
    print(f"Positive test samples: {len(pos_indices)}")

    # Negative test samples grouped by source
    for name in sorted(test_neg_names):
        source = test_neg_sources.get(name, 'unknown')
        src_dir = source_dirs.get(source)
        if src_dir is None:
            continue

        st = _try_load(src_dir, name)
        if st is None:
            continue

        g = st.protein_graph

        # Remove source-specific extra fields to avoid PyG collate key mismatch
        if hasattr(g, 'mut_info'):
            del g.mut_info

        if not _passes_filter(g):
            continue

        g.y = torch.tensor(0.0)
        subset_indices.setdefault(source, []).append(len(graphs))
        graphs.append(g)
        labels.append(0)
        names.append(name)

    for src in sorted(subset_indices):
        if src != 'positive':
            print(f"Negative test samples ({src}): {len(subset_indices[src])}")

    return graphs, np.array(labels), names, subset_indices


def build_and_load_disc_classifier(ft_dir, file_emb_tag, pool_input_dim,
                                    dropout, device, jk_suffix='_jkmean'):
    """Build a binder-discrimination MLP classifier and load its checkpoint.

    Architecture is inferred from the checkpoint state-dict.

    Returns
    -------
    torch.nn.Module
        Model in eval mode on *device*.
    """
    ckpt_path = os.path.join(
        ft_dir, f'disc_binder{file_emb_tag}{jk_suffix}_best.pt'
    )
    ckpt_sd = torch.load(ckpt_path, map_location=device)
    ft_hdim = ckpt_sd['classifier.0.weight'].shape[0]
    n_cls_layers = sum(
        1 for k in ckpt_sd
        if k.startswith('classifier.') and k.endswith('.weight')
        and len(ckpt_sd[k].shape) == 2
    )
    model = create_poolhead_model(
        model_type='classifier',
        pool_input_dim=pool_input_dim,
        hidden_dim=ft_hdim,
        metadata=METADATA,
        dropout=dropout,
        pool_mode='cross_attn',
        cross_attn_queries=2,
        cross_attn_heads=4,
        classifier_layers=n_cls_layers,
    ).to(device)
    model.load_state_dict(ckpt_sd)
    model.eval()
    print(f"Loaded disc binder classifier: {ckpt_path}  "
          f"(ft_hdim={ft_hdim}, n_cls_layers={n_cls_layers})")
    return model


def run_disc_binder_mlp_inference(graphs, model, device, batch_size=32):
    """Run MLP classifier forward pass and return probabilities.

    Returns
    -------
    y_prob : np.ndarray
        Sigmoid of raw logits, shape ``[N]``.
    """
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch).squeeze(-1)
            probs.append(torch.sigmoid(logits).cpu())
    return torch.cat(probs).numpy()


def run_ml_disc_binder_inference(graphs, model):
    """Run sklearn classifier inference on graphs with precomputed embeddings.

    Returns
    -------
    y_prob : np.ndarray
        Predicted probability of class 1, shape ``[N]``.
    """
    X = extract_graph_features(graphs)
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_disc_metrics(y_true, y_prob, cutoff=0.5):
    """Compute AUROC, accuracy, F1, precision, and recall for binary classification.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth (0 or 1).
    y_prob : array-like
        Predicted probability for class 1.
    cutoff : float
        Threshold for binary predictions.

    Returns
    -------
    dict
        Keys: ``'auroc'``, ``'accuracy'``, ``'f1'``, ``'precision'``,
        ``'recall'``, ``'n_pos'``, ``'n_neg'``.
    """
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= cutoff).astype(int)

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    metrics = {
        'n_pos': n_pos,
        'n_neg': n_neg,
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if n_pos > 0 and n_neg > 0:
        metrics['auroc'] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics['auroc'] = float('nan')

    return metrics


# ---------------------------------------------------------------------------
# Baseline (no encoder) disc-binder inference helpers
# ---------------------------------------------------------------------------

def is_baseline_mode(input_type: str) -> bool:
    """Return True if *input_type* denotes a no-encoder baseline run.

    Recognised baseline prefixes: ``'only_base'``, ``'only_esm'``,
    ``'only_esm480'``.
    """
    return input_type.startswith('only_')


def get_baseline_config(ssl_trained_data_root: str, input_type: str):
    """Return ``(baseline_dir, raw_emb_type, esm_dim)`` for a baseline run.

    Parameters
    ----------
    ssl_trained_data_root : str
    input_type : str
        One of ``'only_base'``, ``'only_esm'``, ``'only_esm480'``.

    Returns
    -------
    baseline_dir : str
        Directory containing the saved baseline checkpoint / results.
    raw_emb_type : str
        Embedding type to pass to graph-loading helpers (``'base'``,
        ``'esm'``, or ``'esm480'``).
    esm_dim : int | None
        Feature-slice dimension for mean pooling.  ``None`` means use the
        full feature vector (structural-only baseline).
    """
    _cfg = {
        'only_base':   (os.path.join(ssl_trained_data_root, 'baseline'),     'base',   None),
        'only_esm':    (os.path.join(ssl_trained_data_root, 'esm_baseline'), 'esm',    1280),
        'only_esm480': (os.path.join(ssl_trained_data_root, 'esm480_baseline'), 'esm480',  480),
    }
    if input_type not in _cfg:
        raise ValueError(
            f"Unknown baseline input_type: {input_type!r}. "
            "Expected one of: 'only_base', 'only_esm', 'only_esm480'."
        )
    return _cfg[input_type]


def _baseline_dg_results_filename(input_type: str) -> str:
    """Return baseline ΔG results JSON filename for a baseline input_type."""
    _results_json = {
        'only_base': 'dG_baseline_results.json',
        'only_esm': 'dG_esm_baseline_results.json',
        'only_esm480': 'dG_esm480_baseline_results.json',
    }
    if input_type not in _results_json:
        raise ValueError(
            f"Unknown baseline input_type: {input_type!r}. "
            "Expected one of: 'only_base', 'only_esm', 'only_esm480'."
        )
    return _results_json[input_type]


def _best_fold_key_from_dict(fold_dict: dict, score_getter):
    """Select fold key (e.g. fold_1) that maximizes score_getter(fold_metrics)."""
    fold_keys = [k for k in fold_dict if k.startswith('fold_')]
    if not fold_keys:
        raise ValueError('No fold_* entries found in baseline results.')
    return max(fold_keys, key=lambda k: score_getter(fold_dict[k]))


def build_and_load_baseline_dg_regressor(baseline_dir: str, input_type: str, device):
    """Load best-fold baseline ΔG MLP regressor.

    Fold selection is based on max ``best_val_rp`` from baseline results JSON.

    Returns
    -------
    model : torch.nn.Module
    best_fold_num : int
    best_val_rp : float
    """
    from model_collection.FineTune_baselineModels import create_baseline_model

    results_path = os.path.join(baseline_dir, _baseline_dg_results_filename(input_type))
    results = json.load(open(results_path))
    nn_results = results['nn']

    best_fold_key = _best_fold_key_from_dict(
        nn_results,
        score_getter=lambda x: float(x.get('best_val_rp', -float('inf'))),
    )
    best_fold_num = int(best_fold_key.split('_')[1])
    best_val_rp = float(nn_results[best_fold_key].get('best_val_rp', float('nan')))

    ckpt_path = os.path.join(baseline_dir, f'dG_nn_{best_fold_key}_best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Baseline ΔG NN checkpoint not found: {ckpt_path}")

    ckpt_sd = torch.load(ckpt_path, map_location=device)
    node_in_dim = ckpt_sd['regressor.0.weight'].shape[1]
    hidden_dim = ckpt_sd['regressor.0.weight'].shape[0]
    regressor_layers = sum(
        1 for k in ckpt_sd
        if k.startswith('regressor.') and k.endswith('.weight')
        and len(ckpt_sd[k].shape) == 2
    )

    model = create_baseline_model(
        'regressor',
        node_in_dim=node_in_dim,
        hidden_dim=hidden_dim,
        regressor_layers=regressor_layers,
    ).to(device)
    model.load_state_dict(ckpt_sd)
    model.eval()

    print(
        f"Loaded baseline ΔG regressor: {ckpt_path}  "
        f"(node_in_dim={node_in_dim}, hidden_dim={hidden_dim}, "
        f"n_linear={regressor_layers}, best_val_rp={best_val_rp:.4f})"
    )
    return model, best_fold_num, best_val_rp


def run_baseline_dg_mlp_inference(graphs, model, device, esm_dim=None, batch_size=64):
    """Run baseline ΔG MLP regression on raw mean-pooled graph features.

    Parameters
    ----------
    graphs : list[Data]
    model : BaselineRegressor
    device : torch.device
    esm_dim : int | None
        Optional feature slicing (1280 for only_esm, 480 for only_esm480,
        None for only_base).
    batch_size : int

    Returns
    -------
    y_true : np.ndarray
    y_pred : np.ndarray
    """
    X = extract_graph_features(graphs)
    if esm_dim is not None:
        X = X[:, :esm_dim]
    X_t = torch.from_numpy(X).float()

    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            pred = model(X_t[i:i + batch_size].to(device)).squeeze(-1)
            preds.append(pred.cpu())

    y_pred = torch.cat(preds).numpy()
    y_true = np.array([g.y.item() for g in graphs])
    return y_true, y_pred


def build_and_load_ml_baseline_dg_model(
    baseline_dir: str,
    model_name: str,
    input_type: str,
):
    """Load best-fold baseline sklearn ΔG regressor.

    Returns
    -------
    model : sklearn estimator
    best_fold_num : int
    best_test_rp : float
    """
    results_path = os.path.join(baseline_dir, _baseline_dg_results_filename(input_type))
    results = json.load(open(results_path))
    ml_results = results['ml'][model_name]

    best_fold_key = _best_fold_key_from_dict(
        ml_results,
        score_getter=lambda x: float(x.get('test', {}).get('rp', -float('inf'))),
    )
    best_fold_num = int(best_fold_key.split('_')[1])
    best_test_rp = float(ml_results[best_fold_key]['test'].get('rp', float('nan')))

    candidates = [
        os.path.join(baseline_dir, f'dG_ml_{best_fold_key}_{model_name}.pkl'),
        os.path.join(baseline_dir, f'dG_ml_{best_fold_key}_{model_name.lower()}.pkl'),
        os.path.join(baseline_dir, f'ml_dg_{best_fold_key}_{model_name}.pkl'),
        os.path.join(baseline_dir, f'ml_dg_{best_fold_key}_{model_name.lower()}.pkl'),
        os.path.join(baseline_dir, f'ml_dg_fold_{best_fold_num}_{model_name}.pkl'),
        os.path.join(baseline_dir, f'ml_dg_fold_{best_fold_num}_{model_name.lower()}.pkl'),
    ]
    pkl_path = next((p for p in candidates if os.path.exists(p)), None)
    if pkl_path is None:
        tried = '\n'.join(f'  - {p}' for p in candidates)
        raise FileNotFoundError(
            'Baseline ΔG ML model pickle not found. Tried:\n'
            f'{tried}\n'
            'If baseline ML pickles were not saved during training, '
            're-run baseline training with model saving enabled.'
        )

    with open(pkl_path, 'rb') as fh:
        model = pickle.load(fh)
    print(
        f"Loaded baseline ΔG ML model: {pkl_path}  "
        f"(best_fold=fold_{best_fold_num}, test_rp={best_test_rp:.4f})"
    )
    return model, best_fold_num, best_test_rp


def run_ml_baseline_dg_inference(graphs, model, esm_dim=None):
    """Run baseline sklearn ΔG regression on raw mean-pooled graph features.

    Returns
    -------
    y_true : np.ndarray
    y_pred : np.ndarray
    """
    X = extract_graph_features(graphs)
    if esm_dim is not None:
        X = X[:, :esm_dim]
    y_pred = model.predict(X)
    y_true = np.array([g.y.item() for g in graphs])
    return y_true, y_pred


def build_and_load_baseline_disc_classifier(baseline_dir: str, device):
    """Load BaselineClassifier for disc-binder from its checkpoint.

    Architecture (node_in_dim, hidden_dim, n_cls_layers) is inferred from
    the checkpoint state-dict so the call-site needs no extra configuration.

    Returns
    -------
    torch.nn.Module
        Model in eval mode on *device*.
    """
    from model_collection.FineTune_baselineModels import create_baseline_model

    ckpt_path = os.path.join(baseline_dir, 'disc_binder_nn_best.pt')
    ckpt_sd = torch.load(ckpt_path, map_location=device)
    node_in_dim = ckpt_sd['classifier.0.weight'].shape[1]
    hidden_dim  = ckpt_sd['classifier.0.weight'].shape[0]
    n_cls_layers = sum(
        1 for k in ckpt_sd
        if k.startswith('classifier.') and k.endswith('.weight')
        and len(ckpt_sd[k].shape) == 2
    )
    model = create_baseline_model(
        'classifier',
        node_in_dim=node_in_dim,
        hidden_dim=hidden_dim,
        classifier_layers=n_cls_layers,
    ).to(device)
    model.load_state_dict(ckpt_sd)
    model.eval()
    print(
        f"Loaded baseline disc classifier: {ckpt_path}  "
        f"(node_in_dim={node_in_dim}, hidden_dim={hidden_dim}, "
        f"n_cls_layers={n_cls_layers})"
    )
    return model


def run_baseline_disc_mlp_inference(graphs, model, device, esm_dim=None, batch_size=64):
    """Mean-pool raw graph node features → BaselineClassifier → probabilities.

    Parameters
    ----------
    graphs : list[Data]
    model : BaselineClassifier (no encoder)
    device : torch.device
    esm_dim : int | None
        If set, slice the mean-pooled vector to ``[:esm_dim]`` before feeding
        the model.  Use 1280 for ``'only_esm'``, 480 for ``'only_esm480'``,
        None for ``'only_base'``.
    batch_size : int

    Returns
    -------
    y_prob : np.ndarray  shape [N]
    """
    from utils.Training_modules.ml_utils import mean_pool_graph

    X = np.stack([mean_pool_graph(g) for g in graphs])
    if esm_dim is not None:
        X = X[:, :esm_dim]
    X_t = torch.from_numpy(X).float()

    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            logits = model(X_t[i:i + batch_size].to(device)).squeeze(-1)
            probs.append(torch.sigmoid(logits).cpu())
    return torch.cat(probs).numpy()


def build_and_load_ml_baseline_disc_model(
    baseline_dir: str, model_name: str, input_type: str
):
    """Load a baseline sklearn binder-discrimination classifier.

    Checkpoint naming: ``disc_binder_ml_{model_name.lower()}.pkl``

    Results JSON naming (produced by *esm_baseline_modules.py* /
    *baseline_modules.py*):

    * ``'only_base'``    → ``disc_binder_baseline_results.json``
    * ``'only_esm'``     → ``disc_binder_esm_baseline_results.json``
    * ``'only_esm480'``  → ``disc_binder_esm480_baseline_results.json``

    Note: ``model_name='SVR'`` is mapped to ``'SVM'`` (disc-binder is a
    classification task so ``SVC`` was used during training).

    Returns
    -------
    model : sklearn estimator
    test_auc : float
    """
    _results_json = {
        'only_base':   'disc_binder_baseline_results.json',
        'only_esm':    'disc_binder_esm_baseline_results.json',
        'only_esm480': 'disc_binder_esm480_baseline_results.json',
    }
    if input_type not in _results_json:
        raise ValueError(
            f"Unknown baseline input_type: {input_type!r}. "
            "Expected one of: 'only_base', 'only_esm', 'only_esm480'."
        )

    cls_key = 'SVM' if model_name == 'SVR' else model_name
    results_path = os.path.join(baseline_dir, _results_json[input_type])
    results = json.load(open(results_path))
    # Baseline ML results are stored flat under 'ml.model_name' (no 'test' nesting)
    test_auc = results['ml'][cls_key]['auc']
    print(f"Baseline disc-binder {cls_key} ({input_type}): test AUC={test_auc:.4f}")

    pkl_path = os.path.join(baseline_dir, f'disc_binder_ml_{cls_key.lower()}.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Baseline ML model not found: {pkl_path}")
    with open(pkl_path, 'rb') as fh:
        model = pickle.load(fh)
    print(f"Loaded baseline ML model: {pkl_path}")
    return model, test_auc


def run_ml_baseline_disc_inference(graphs, model, esm_dim=None):
    """Baseline sklearn disc-binder inference with optional feature slicing.

    Baseline ML models are trained on raw mean-pooled node features,
    optionally sliced to ``[:esm_dim]``.  Use this instead of
    ``run_ml_disc_binder_inference`` for baseline-mode predictions.

    Returns
    -------
    y_prob : np.ndarray  shape [N]
    """
    X = extract_graph_features(graphs)
    if esm_dim is not None:
        X = X[:, :esm_dim]
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
