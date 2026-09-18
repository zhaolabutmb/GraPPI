"""
Fine-tuning Models for Pre-trained Protein Complex Encoders.

Provides two pipelines:

LEGACY Pipeline (FineTuneClassifier/Regressor/Gated):
  Encoder + Pool + Head in one model. Encoder can be frozen/unfrozen.

NEW Pipeline (PoolHeadClassifier/Regressor/Gated):
  1. Encoder loaded separately, frozen, used to precompute node embeddings
  2. For +esm/+esm480: ESM embeddings concatenated at node level
  3. Precomputed embeddings stored in graph.x
  4. Pool + Head model trained on precomputed embeddings (no encoder)

Use create_finetune_model() for legacy, create_poolhead_model() for new pipeline.
"""

import torch
import torch.nn as nn
from contextlib import nullcontext
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm
# Import the encoder and cross-attention aggregator
from model_collection.EMP_HGT import EdgeEnhancedHGT
from model_collection.EMP_HGT import CrossAttentionAggregator

# Model-specific kwargs that should NOT be passed to the encoder
_MODEL_SPECIFIC_KWARGS = {
    'regressor_layers',
    'classifier_layers', 
    'use_reg_adaptor',
    'pool_mode',
    'cls_grad_scale',
    'reg_adaptor_layers',
    'cross_attn_queries',
    'cross_attn_heads',
    'plus_esm',
}


def _filter_encoder_kwargs(kwargs: dict) -> dict:
    """
    Filter out model-specific kwargs that shouldn't be passed to the encoder.
    Returns a new dict with only encoder-compatible kwargs.
    """
    return {k: v for k, v in kwargs.items() if k not in _MODEL_SPECIFIC_KWARGS}


# ============================================================================
# NEW Pipeline: Precomputed Encoder Embeddings + Pool + Head
# ============================================================================

# ESM embedding dimensions for +esm and only_esm types
ESM_EMB_DIM = {'+esm': 1280, '+esm480': 480, 'only_esm': 1280, 'only_esm480': 480}

# Valid JK aggregation modes
JK_MODES = ('sum', 'mean', 'max', 'concat')


def get_jk_output_dim(hidden_dim: int, num_hgt_layers: int, jk_mode: str) -> int:
    """
    Return the per-node feature dimension after JK aggregation.
    
    For sum/mean/max: dimension stays hidden_dim (element-wise reduction).
    For concat: dimension = hidden_dim * (num_hgt_layers + 1)
        (initial projection + each HGT layer output).
    """
    assert jk_mode in JK_MODES, f"Unknown jk_mode '{jk_mode}'. Must be one of {JK_MODES}"
    if jk_mode == 'concat':
        return hidden_dim * (num_hgt_layers + 1)
    return hidden_dim


def get_pool_input_dim(
    hidden_dim: int,
    embedding_type: str,
    use_jk: bool = False,
    jk_mode: str = 'mean',
    num_hgt_layers: int = 0,
) -> int:
    """
    Get the input dimension for the pool layer based on embedding type and JK config.
    
    For +esm/+esm480: base_dim + esm_dim (concatenation at node level).
    For only_esm: esm_dim only (ESM replaces encoder output).
    For base/esm/esm480: just base_dim (encoder output only).
    
    When use_jk=True, base_dim accounts for JK aggregation (hidden_dim for
    sum/mean/max, or hidden_dim * (num_layers+1) for concat).
    """
    base_dim = get_jk_output_dim(hidden_dim, num_hgt_layers, jk_mode) if use_jk else hidden_dim
    if embedding_type in ['+esm', '+esm480']:
        return base_dim + ESM_EMB_DIM[embedding_type]
    elif embedding_type in ['only_esm', 'only_esm480']:
        return ESM_EMB_DIM[embedding_type]
    else:
        return base_dim


def load_pretrained_encoder(
    checkpoint_path: str,
    node_in_dim: int,
    edge_in_dim: int,
    metadata: Tuple,
    hidden_dim: int,
    num_hgt_layers: int,
    hgt_heads: int,
    device: torch.device,
    **encoder_kwargs,
) -> EdgeEnhancedHGT:
    """
    Load a pretrained encoder from an SSL checkpoint for embedding precomputation.
    
    Returns the encoder in eval mode with all parameters frozen.
    
    Args:
        checkpoint_path: Path to SSL checkpoint
        node_in_dim: Input node feature dimension (e.g. 25 for base)
        edge_in_dim: Input edge feature dimension
        metadata: Graph metadata (node_types, edge_types)
        hidden_dim: Encoder hidden dimension
        num_hgt_layers: Number of HGT layers
        hgt_heads: Number of attention heads
        device: Target device
        **encoder_kwargs: Additional encoder kwargs (e.g. message_style)
    
    Returns:
        Frozen encoder in eval mode
    """
    encoder = EdgeEnhancedHGT(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=hidden_dim,
        metadata=metadata,
        hgt_heads=hgt_heads,
        hgt_layers=num_hgt_layers,
        **encoder_kwargs,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'encoder_state_dict' in checkpoint:
        encoder_state = checkpoint['encoder_state_dict']
        print(f"Loaded encoder weights from {checkpoint_path}")
        if 'best_epoch' in checkpoint:
            print(f"  Pre-trained at epoch {checkpoint['best_epoch']+1} "
                  f"with loss {checkpoint.get('best_loss', 'N/A')}")
    else:
        encoder_state = {
            k.replace('encoder.', ''): v
            for k, v in checkpoint['full_model_state_dict'].items()
            if k.startswith('encoder.')
        }
        print(f"Loaded encoder weights from full model checkpoint")
    
    result = encoder.load_state_dict(encoder_state, strict=False)
    
    loaded_params = sum(p.numel() for p in encoder_state.values())
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder parameters loaded: {loaded_params:,} / {total_params:,} "
          f"({100*loaded_params/total_params:.1f}%)")
    
    if result.missing_keys:
        print(f"  Missing keys: {len(result.missing_keys)}")
    if result.unexpected_keys:
        print(f"  Unexpected keys: {len(result.unexpected_keys)}")
    
    encoder = encoder.to(device)
    encoder.eval()
    
    # Freeze all parameters
    for p in encoder.parameters():
        p.requires_grad = False
    
    return encoder


def _aggregate_jk_layers(all_layer_outputs, jk_mode: str):
    """
    Aggregate a list of per-layer node embedding dicts into a single dict.
    
    Args:
        all_layer_outputs: List of dicts, each {node_type: [N, D]}.
        jk_mode: One of 'sum', 'mean', 'max', 'concat'.
    
    Returns:
        Dict {node_type: [N, D']} where D' = D for sum/mean/max, or D*L for concat.
    """
    node_types = all_layer_outputs[0].keys()
    result = {}
    for nt in node_types:
        stacked = torch.stack([layer[nt] for layer in all_layer_outputs], dim=0)  # [L, N, D]
        if jk_mode == 'sum':
            result[nt] = stacked.sum(dim=0)
        elif jk_mode == 'mean':
            result[nt] = stacked.mean(dim=0)
        elif jk_mode == 'max':
            result[nt] = stacked.max(dim=0).values
        elif jk_mode == 'concat':
            result[nt] = torch.cat([layer[nt] for layer in all_layer_outputs], dim=-1)  # [N, D*L]
        else:
            raise ValueError(f"Unknown jk_mode '{jk_mode}'. Must be one of {JK_MODES}")
    return result


def precompute_embeddings(
    encoder: EdgeEnhancedHGT,
    sample_names: List[str],
    graphs: List,
    device: torch.device,
    batch_size: int = 32,
    embedding_type: str = 'base',
    esm_dict: Optional[Dict] = None,
    use_amp: bool = True,
    use_jk: bool = False,
    jk_mode: str = 'mean',
) -> List:
    """
    Precompute encoder embeddings for all graphs in batches, storing results
    back into each graph's node features (graph[node_type].x).
    
    For 'base', 'esm', 'esm480':
        graph.x = encoder_output  [n_nodes, hidden_dim]
    For '+esm', '+esm480':
        graph.x = cat(encoder_output, esm_emb)  [n_nodes, hidden_dim + esm_dim]
    
    When use_jk=True, intermediate HGT layer outputs are aggregated using
    jk_mode ('sum', 'mean', 'max', 'concat') before storing. For concat,
    the feature dimension becomes hidden_dim * (num_hgt_layers + 1).
    
    Batch pairing strategy:
        - DataLoader processes graphs in order (shuffle=False)
        - batch_data[nt].batch gives per-node graph indices within the batch
        - encoder output is split per-graph using these indices
        - ESM embeddings are matched by sample_names[graph_idx]
    
    Args:
        encoder: Pretrained encoder (frozen, eval mode)
        sample_names: List of sample names (same order as graphs)
        graphs: List of HeteroData graphs (modified in place)
        device: Device for encoder forward pass
        batch_size: Batch size for encoder processing
        embedding_type: One of 'base', 'esm', 'esm480', '+esm', '+esm480', 'only_esm'
        esm_dict: Dict of ESM embeddings per sample (required for +esm/only_esm types)
        use_amp: Whether to use automatic mixed precision
        use_jk: If True, aggregate intermediate HGT layers (Jumping Knowledge)
        jk_mode: JK aggregation mode: 'sum', 'mean', 'max', or 'concat'
    
    Returns:
        The same list of graphs with x replaced by precomputed embeddings
    """
    jk_str = f" (JK mode={jk_mode})" if use_jk else ""
    print(f"\nPrecomputing encoder embeddings for {len(graphs)} graphs...{jk_str}")
    
    if use_jk:
        assert jk_mode in JK_MODES, f"Unknown jk_mode '{jk_mode}'. Must be one of {JK_MODES}"
    
    plus_esm = embedding_type in ['+esm', '+esm480']
    only_esm = embedding_type == 'only_esm'
    if plus_esm or only_esm:
        assert esm_dict is not None, f"esm_dict required for embedding_type={embedding_type}"
        esm_dim = ESM_EMB_DIM[embedding_type]
    
    # Process in order (no shuffle) for correct name-based pairing
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    
    amp_ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
               if (use_amp and device.type == 'cuda') else nullcontext())
    
    graph_idx = 0
    
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Encoding"):
            batch_data = batch_data.to(device)
            
            with amp_ctx:
                if use_jk:
                    # Get all layer outputs and aggregate
                    all_layer_outputs = encoder(batch_data, return_all_layers=True)
                    x_hgt = _aggregate_jk_layers(all_layer_outputs, jk_mode)
                else:
                    x_hgt = encoder(batch_data)  # {'receptor': [N_r, D], 'ligand': [N_l, D]}
            
            num_graphs_in_batch = batch_data.num_graphs
            
            for nt in ['receptor', 'ligand']:
                batch_indices = batch_data[nt].batch.cpu()
                embs = x_hgt[nt].float().cpu()  # [N_total, D'], float32
                
                for i in range(num_graphs_in_batch):
                    mask = (batch_indices == i)
                    node_emb = embs[mask]  # [n_nodes, D']
                    
                    actual_idx = graph_idx + i
                    
                    if plus_esm or only_esm:
                        name = sample_names[actual_idx]
                        chain_key = 'receptor' if nt == 'receptor' else 'ligand'
                        esm_emb = esm_dict[name][chain_key]['seq_emb']  # [n_nodes, esm_dim]
                        
                        if node_emb.shape[0] != esm_emb.shape[0]:
                            raise ValueError(
                                f"Node count mismatch for {name}/{nt}: "
                                f"encoder={node_emb.shape[0]}, ESM={esm_emb.shape[0]}"
                            )
                        
                        if only_esm:
                            # Replace encoder output with ESM embeddings
                            graphs[actual_idx][nt].x = esm_emb.float()
                        else:
                            # Concatenate encoder + ESM embeddings at node level
                            combined = torch.cat([node_emb, esm_emb.float()], dim=-1)
                            graphs[actual_idx][nt].x = combined
                    else:
                        graphs[actual_idx][nt].x = node_emb
            
            graph_idx += num_graphs_in_batch
    
    # Report dimensions
    sample_graph = graphs[0]
    r_dim = sample_graph['receptor'].x.shape[-1]
    l_dim = sample_graph['ligand'].x.shape[-1]
    print(f"Precomputed embedding dims: receptor={r_dim}, ligand={l_dim}")
    
    return graphs


# ============================================================================
# Student encoder (sequence) — drop-in replacement for the GraPPI encoder
# ============================================================================
def load_student_encoder(checkpoint_path: str, device: torch.device):
    """
    Load a trained sequence-student encoder (frozen, eval mode).

    Expects `student_config.json` alongside the checkpoint describing the
    architecture. Returns (student, student_cfg).
    """
    import json
    import os
    from model_collection.StudentModels import create_student_model

    cfg_path = os.path.join(os.path.dirname(checkpoint_path), 'student_config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"student_config.json not found at {cfg_path}; required to rebuild the "
            "student architecture.")
    with open(cfg_path, 'r') as f:
        scfg = json.load(f)

    student = create_student_model(
        esm_dim=scfg['esm_dim'],
        hidden_dim=scfg['hidden_dim'],
        n_blocks=scfg['n_blocks'],
        n_heads=scfg['n_heads'],
        dropout=scfg.get('dropout', 0.0),
        ffn_mult=scfg.get('ffn_mult', 4),
    )
    state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    student.load_state_dict(state)
    student = student.to(device)
    student.eval()
    for p in student.parameters():
        p.requires_grad = False
    print(f"Loaded student encoder from {checkpoint_path} "
          f"(esm_dim={scfg['esm_dim']}, hidden_dim={scfg['hidden_dim']}, "
          f"n_blocks={scfg['n_blocks']})")
    return student, scfg


@torch.no_grad()
def precompute_embeddings_student(
    student,
    esm_dim: int,
    sample_names: List[str],
    graphs: List,
    device: torch.device,
    batch_size: int = 16,
    use_amp: bool = True,
) -> List:
    """
    Run the sequence student on each complex's ESM residue embeddings and store
    the resulting per-node embeddings into graph[nt].x — mirroring the output
    contract of precompute_embeddings so downstream pool+head code is unchanged.

    The student consumes only x[:, :esm_dim] per protein (no edges). No residue
    cap is applied here (inference must cover all lengths).
    """
    from utils.Training_modules.student_data_loader import pad_and_mask

    print(f"\nPrecomputing STUDENT embeddings for {len(graphs)} graphs...")
    if graphs and graphs[0]['receptor'].x.size(1) < esm_dim:
        raise ValueError(
            f"Graph node features have {graphs[0]['receptor'].x.size(1)} dims but the "
            f"student needs esm_dim={esm_dim}. Use an ESM graph set (embedding_type "
            "'esm'/'esm480') matching the student.")

    amp_ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
               if (use_amp and device.type == 'cuda') else nullcontext())
    student.eval()

    for start in tqdm(range(0, len(graphs), batch_size), desc="Student encoding"):
        chunk = graphs[start:start + batch_size]
        recs = [g['receptor'].x[:, :esm_dim].float() for g in chunk]
        ligs = [g['ligand'].x[:, :esm_dim].float() for g in chunk]
        rec_x, rec_mask = pad_and_mask(recs)
        lig_x, lig_mask = pad_and_mask(ligs)
        rec_x, lig_x = rec_x.to(device), lig_x.to(device)
        rec_mask, lig_mask = rec_mask.to(device), lig_mask.to(device)

        with amp_ctx:
            s_rec, s_lig = student(rec_x, lig_x, rec_mask, lig_mask)
        s_rec = s_rec.float().cpu()
        s_lig = s_lig.float().cpu()

        for i, g in enumerate(chunk):
            n_rec = recs[i].size(0)
            n_lig = ligs[i].size(0)
            g['receptor'].x = s_rec[i, :n_rec]
            g['ligand'].x = s_lig[i, :n_lig]

    sample_graph = graphs[0]
    print(f"Precomputed student embedding dims: "
          f"receptor={sample_graph['receptor'].x.shape[-1]}, "
          f"ligand={sample_graph['ligand'].x.shape[-1]}")
    return graphs


# ============================================================================
# Structure student encoder (individual monomer graphs) — drop-in replacement
# ============================================================================
def load_structure_student_encoder(checkpoint_path: str, device: torch.device):
    """
    Load a trained structure-student encoder (frozen, eval mode).

    Expects `structure_student_config.json` alongside the checkpoint. Returns
    (student, student_cfg).
    """
    import json
    import os
    from model_collection.StructureStudentModels import create_structure_student_model

    cfg_path = os.path.join(os.path.dirname(checkpoint_path), 'structure_student_config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"structure_student_config.json not found at {cfg_path}; required to "
            "rebuild the student architecture.")
    with open(cfg_path, 'r') as f:
        scfg = json.load(f)

    student = create_structure_student_model(
        node_in_dim=scfg['node_in_dim'],
        edge_in_dim=scfg['edge_in_dim'],
        hidden_dim=scfg['hidden_dim'],
        n_gnn_layers=scfg['n_gnn_layers'],
        gnn_heads=scfg.get('gnn_heads', 4),
        n_cross_blocks=scfg['n_cross_blocks'],
        n_heads=scfg['n_heads'],
        dropout=scfg.get('dropout', 0.0),
        ffn_mult=scfg.get('ffn_mult', 4),
    )
    state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    student.load_state_dict(state)
    student = student.to(device)
    student.eval()
    for p in student.parameters():
        p.requires_grad = False
    print(f"Loaded structure student encoder from {checkpoint_path} "
          f"(node_in_dim={scfg['node_in_dim']}, hidden_dim={scfg['hidden_dim']}, "
          f"n_gnn_layers={scfg['n_gnn_layers']}, n_cross_blocks={scfg['n_cross_blocks']})")
    return student, scfg


@torch.no_grad()
def precompute_embeddings_structure_student(
    student,
    sample_names: List[str],
    graphs: List,
    device: torch.device,
    batch_size: int = 16,
    use_amp: bool = True,
) -> List:
    """
    Run the structure student on each complex's two intra-chain homo graphs
    (sliced in-memory from the complex HeteroData) and store the resulting
    per-node embeddings into graph[nt].x — mirroring the output contract of
    precompute_embeddings so downstream pool+head code is unchanged.
    """
    from torch_geometric.data import Batch
    from utils.Training_modules.homo_graph_cache import hetero_to_homo_pair

    print(f"\nPrecomputing STRUCTURE STUDENT embeddings for {len(graphs)} graphs...")
    amp_ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
               if (use_amp and device.type == 'cuda') else nullcontext())
    student.eval()

    for start in tqdm(range(0, len(graphs), batch_size), desc="Structure student encoding"):
        chunk = graphs[start:start + batch_size]
        pairs = [hetero_to_homo_pair(g) for g in chunk]
        rec_batch = Batch.from_data_list([p['receptor'] for p in pairs]).to(device)
        lig_batch = Batch.from_data_list([p['ligand'] for p in pairs]).to(device)

        with amp_ctx:
            s_rec, s_lig, rec_mask, lig_mask = student(rec_batch, lig_batch)
        s_rec = s_rec.float().cpu()
        s_lig = s_lig.float().cpu()

        for i, g in enumerate(chunk):
            n_rec = pairs[i]['receptor'].num_nodes
            n_lig = pairs[i]['ligand'].num_nodes
            g['receptor'].x = s_rec[i, :n_rec]
            g['ligand'].x = s_lig[i, :n_lig]

    sample_graph = graphs[0]
    print(f"Precomputed structure student embedding dims: "
          f"receptor={sample_graph['receptor'].x.shape[-1]}, "
          f"ligand={sample_graph['ligand'].x.shape[-1]}")
    return graphs


def build_embedder(
    pretrained_path: str,
    device: torch.device,
    *,
    encoder_type: str,
    embedding_type: str,
    node_in_dim: int,
    edge_in_dim: int,
    metadata: Tuple,
    hidden_dim: int,
    num_hgt_layers: int,
    hgt_heads: int,
    message_style: str,
    use_jk: bool,
    jk_mode: str,
    batch_size: int,
    use_amp: bool,
):
    """
    Unified dispatcher for downstream modules. Loads either the GraPPI SSL
    encoder or the sequence student, and returns:

        (precompute_fn, resolved_hidden_dim, resolved_use_jk)

    where precompute_fn(sample_names, graphs, esm_dict=None) fills graph[nt].x
    in place with per-node embeddings. For the student, use_jk is always False
    and hidden_dim is taken from the student checkpoint.
    """
    if encoder_type == 'student':
        student, scfg = load_student_encoder(pretrained_path, device)
        esm_dim = scfg['esm_dim']
        h = scfg['hidden_dim']

        def precompute_fn(sample_names, graphs, esm_dict=None):
            return precompute_embeddings_student(
                student, esm_dim, sample_names, graphs, device,
                batch_size=batch_size, use_amp=use_amp,
            )

        return precompute_fn, h, False

    if encoder_type == 'structure_student':
        student, scfg = load_structure_student_encoder(pretrained_path, device)
        h = scfg['hidden_dim']

        def precompute_fn(sample_names, graphs, esm_dict=None):
            return precompute_embeddings_structure_student(
                student, sample_names, graphs, device,
                batch_size=batch_size, use_amp=use_amp,
            )

        return precompute_fn, h, False

    encoder = load_pretrained_encoder(
        checkpoint_path=pretrained_path,
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        metadata=metadata,
        hidden_dim=hidden_dim,
        num_hgt_layers=num_hgt_layers,
        hgt_heads=hgt_heads,
        device=device,
        message_style=message_style,
    )

    def precompute_fn(sample_names, graphs, esm_dict=None):
        return precompute_embeddings(
            encoder, sample_names, graphs, device,
            batch_size=batch_size, embedding_type=embedding_type,
            esm_dict=esm_dict, use_amp=use_amp, use_jk=use_jk, jk_mode=jk_mode,
        )

    return precompute_fn, hidden_dim, use_jk


# ============================================================================
# Pool + Head Base Class
# ============================================================================

class _PoolHeadBase(nn.Module):
    """
    Base class for pool+head models with precomputed encoder embeddings.
    
    Handles:
    - Input projection (for +esm types where pool_input_dim > hidden_dim)
    - Pooling (cross_attn, attn+mean, attn, mean+max, etc.)
    - No-op update_epoch() for compatibility with training loop
    
    Subclasses add task-specific heads (classifier, regressor, gated).
    """
    
    def __init__(
        self,
        pool_input_dim: int,
        hidden_dim: int = 1024,
        metadata: Tuple = None,
        dropout: float = 0.2,
        pool_mode: str = "attn+mean",
        cross_attn_queries: int = 8,
        cross_attn_heads: int = 4,
    ):
        super().__init__()
        self.pool_mode = pool_mode
        
        # Input projection (for +esm types where pool_input_dim > hidden_dim)
        if pool_input_dim != hidden_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(pool_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        else:
            self.input_proj = nn.Identity()
        
        # Pool setup (always operates on hidden_dim after projection)
        if pool_mode == "cross_attn":
            self.cross_attn_agg = CrossAttentionAggregator(
                hidden_dim=hidden_dim,
                num_queries=cross_attn_queries,
                num_heads=cross_attn_heads,
                dropout=dropout,
            )
            self._pooled_dim = self.cross_attn_agg.output_dim
            self.attn_pool = None
            self.post_pool_norm = None
        else:
            self.cross_attn_agg = None
            self.attn_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
            n_pools = sum(1 for p in ['attn', 'mean', 'max'] if p in pool_mode)
            node_types = metadata[0] if metadata else ['receptor', 'ligand']
            self._pooled_dim = hidden_dim * n_pools * max(1, len(node_types))
            self.post_pool_norm = nn.LayerNorm(self._pooled_dim)
    
    @property
    def pooled_dim(self) -> int:
        return self._pooled_dim
    
    def _project_and_pool(self, batch_data):
        """Project precomputed embeddings (if needed) and pool to graph level."""
        x_dict = batch_data.x_dict
        # Apply shared projection (Identity if pool_input_dim == hidden_dim)
        x_dict = {nt: self.input_proj(x) for nt, x in x_dict.items()}
        return self._pool(x_dict, batch_data)
    
    def _pool(self, x_dict, batch_data):
        """Aggregate node embeddings to graph-level."""
        if self.pool_mode == "cross_attn":
            batch_dict = {nt: batch_data[nt].batch for nt in x_dict}
            return self.cross_attn_agg(x_dict, batch_dict)
        
        outs_all_nt = []
        for nt, x_nt in x_dict.items():
            b_nt = batch_data[nt].batch
            outs = []
            if "attn" in self.pool_mode:
                outs.append(self.attn_pool(x_nt, b_nt))
            if "mean" in self.pool_mode:
                outs.append(global_mean_pool(x_nt, b_nt))
            if "max" in self.pool_mode:
                outs.append(global_max_pool(x_nt, b_nt))
            pooled_nt = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
            outs_all_nt.append(pooled_nt)
        pooled = torch.cat(outs_all_nt, dim=1) if len(outs_all_nt) > 1 else outs_all_nt[0]
        return self.post_pool_norm(pooled)
    
    def update_epoch(self, epoch: int):
        """No-op for compatibility with training loop (no encoder to freeze)."""
        pass


# ============================================================================
# Pool + Head: Classifier
# ============================================================================

class PoolHeadClassifier(_PoolHeadBase):
    """
    Pool + Classification head for precomputed encoder embeddings.
    
    Forward: batch_data.x_dict → project → pool → classifier → logits
    """
    
    def __init__(
        self,
        pool_input_dim: int,
        hidden_dim: int = 1024,
        metadata: Tuple = None,
        classifier_layers: int = 3,
        dropout: float = 0.2,
        pool_mode: str = "attn+mean",
        cross_attn_queries: int = 8,
        cross_attn_heads: int = 4,
    ):
        super().__init__(
            pool_input_dim=pool_input_dim,
            hidden_dim=hidden_dim,
            metadata=metadata,
            dropout=dropout,
            pool_mode=pool_mode,
            cross_attn_queries=cross_attn_queries,
            cross_attn_heads=cross_attn_heads,
        )
        
        # Classification head (progressive bottleneck)
        classifier_modules = []
        for i in range(classifier_layers):
            if i == 0:
                in_dim = self.pooled_dim
                out_dim = hidden_dim
            elif i == classifier_layers - 1:
                in_dim = hidden_dim // (2 ** (i - 1)) if i > 1 else hidden_dim
                out_dim = 1
            else:
                in_dim = hidden_dim // (2 ** (i - 1))
                out_dim = hidden_dim // (2 ** i)
            
            classifier_modules.append(nn.Linear(in_dim, out_dim))
            if i < classifier_layers - 1:
                classifier_modules.extend([
                    nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout)
                ])
        self.classifier = nn.Sequential(*classifier_modules)
    
    def forward(self, batch_data: Batch) -> torch.Tensor:
        pooled = self._project_and_pool(batch_data)
        return self.classifier(pooled)


# ============================================================================
# Pool + Head: Regressor
# ============================================================================

class PoolHeadRegressor(_PoolHeadBase):
    """
    Pool + Regression head for precomputed encoder embeddings.
    
    Forward: batch_data.x_dict → project → pool → regressor → prediction
    """
    
    def __init__(
        self,
        pool_input_dim: int,
        hidden_dim: int = 1024,
        metadata: Tuple = None,
        regressor_layers: int = 4,
        dropout: float = 0.2,
        pool_mode: str = "attn+mean",
        cross_attn_queries: int = 2,
        cross_attn_heads: int = 4,
    ):
        super().__init__(
            pool_input_dim=pool_input_dim,
            hidden_dim=hidden_dim,
            metadata=metadata,
            dropout=dropout,
            pool_mode=pool_mode,
            cross_attn_queries=cross_attn_queries,
            cross_attn_heads=cross_attn_heads,
        )
        
        # Regression head (progressive bottleneck)
        regressor_modules = []
        for i in range(regressor_layers):
            if i == 0:
                in_dim = self.pooled_dim
                out_dim = hidden_dim
            elif i == regressor_layers - 1:
                in_dim = hidden_dim // (2 ** (i - 1)) if i > 1 else hidden_dim
                out_dim = 1
            else:
                in_dim = hidden_dim // (2 ** (i - 1))
                out_dim = hidden_dim // (2 ** i)
            
            regressor_modules.append(nn.Linear(in_dim, out_dim))
            if i < regressor_layers - 1:
                regressor_modules.extend([
                    nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout)
                ])
        self.regressor = nn.Sequential(*regressor_modules)
    
    def forward(self, batch_data: Batch) -> torch.Tensor:
        pooled = self._project_and_pool(batch_data)
        return self.regressor(pooled)


# ============================================================================
# Pool + Head: ΔΔG Regressor (Dual-graph, mutant vs wild-type)
# ============================================================================

class PoolHeadDDGRegressor(_PoolHeadBase):
    """
    Pool + Regression head for ΔΔG prediction from paired mutant / wild-type
    precomputed embeddings.

    Forward accepts **two** batched graphs (mutant_batch, wt_batch).
    Each is independently projected and pooled, then combined into a single
    feature vector that is fed to an MLP regressor head.

    Combination modes (controlled by ``ddg_input_mode``):
        * ``'concat_diff'`` (default):
            feature = [emb_mut − emb_wt, emb_mut, emb_wt]
            → input dim to MLP = 3 × pooled_dim
        * ``'diff'``:
            feature = emb_mut − emb_wt
            → input dim to MLP = pooled_dim
    """

    def __init__(
        self,
        pool_input_dim: int,
        hidden_dim: int = 1024,
        metadata: Tuple = None,
        regressor_layers: int = 4,
        dropout: float = 0.2,
        pool_mode: str = "cross_attn",
        cross_attn_queries: int = 2,
        cross_attn_heads: int = 4,
        ddg_input_mode: str = "concat_diff",
    ):
        super().__init__(
            pool_input_dim=pool_input_dim,
            hidden_dim=hidden_dim,
            metadata=metadata,
            dropout=dropout,
            pool_mode=pool_mode,
            cross_attn_queries=cross_attn_queries,
            cross_attn_heads=cross_attn_heads,
        )

        self.ddg_input_mode = ddg_input_mode

        # Determine MLP input dimension based on combination mode
        if ddg_input_mode == "concat_diff":
            mlp_input_dim = self.pooled_dim * 3
        elif ddg_input_mode == "diff":
            mlp_input_dim = self.pooled_dim
        else:
            raise ValueError(f"Unknown ddg_input_mode: {ddg_input_mode}. "
                             f"Must be 'concat_diff' or 'diff'.")

        # Regression head (progressive bottleneck, same pattern as PoolHeadRegressor)
        regressor_modules = []
        for i in range(regressor_layers):
            if i == 0:
                in_dim = mlp_input_dim
                out_dim = hidden_dim
            elif i == regressor_layers - 1:
                in_dim = hidden_dim // (2 ** (i - 1)) if i > 1 else hidden_dim
                out_dim = 1
            else:
                in_dim = hidden_dim // (2 ** (i - 1))
                out_dim = hidden_dim // (2 ** i)

            regressor_modules.append(nn.Linear(in_dim, out_dim))
            if i < regressor_layers - 1:
                regressor_modules.extend([
                    nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout)
                ])
        self.regressor = nn.Sequential(*regressor_modules)

    def forward(
        self,
        mut_batch: Batch,
        wt_batch: Batch,
    ) -> torch.Tensor:
        """
        Args:
            mut_batch: Batched mutant graphs with precomputed embeddings.
            wt_batch:  Batched wild-type graphs with precomputed embeddings.

        Returns:
            Predicted ΔΔG values, shape (batch_size, 1).
        """
        emb_mut = self._project_and_pool(mut_batch)
        emb_wt = self._project_and_pool(wt_batch)

        diff = emb_mut - emb_wt

        if self.ddg_input_mode == "concat_diff":
            combined = torch.cat([diff, emb_mut, emb_wt], dim=-1)
        else:  # 'diff'
            combined = diff

        return self.regressor(combined)


# ============================================================================
# Factory for Pool + Head Models
# ============================================================================

def create_poolhead_model(
    model_type: str,
    pool_input_dim: int,
    hidden_dim: int = 1024,
    metadata: Tuple = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create pool+head fine-tuning models (new pipeline).
    
    These models operate on precomputed encoder embeddings stored in graph.x.
    Use load_pretrained_encoder() and precompute_embeddings() first.
    
    Args:
        model_type: 'classifier', 'regressor', 'gated', or 'ddg_regressor'
        pool_input_dim: Dimension of precomputed node embeddings
                        (hidden_dim for base/esm, hidden_dim + esm_dim for +esm)
        hidden_dim: Hidden dimension for head layers
        metadata: Graph metadata (node_types, edge_types)
        **kwargs: Model-specific and pool arguments
    
    Returns:
        Pool+Head model instance
    """
    pool_mode = kwargs.get('pool_mode', 'attn+mean')
    cross_attn_queries = kwargs.get('cross_attn_queries', 2)
    cross_attn_heads = kwargs.get('cross_attn_heads', 4)
    dropout = kwargs.get('dropout', 0.2)
    
    if model_type == 'classifier':
        model = PoolHeadClassifier(
            pool_input_dim=pool_input_dim,
            hidden_dim=hidden_dim,
            metadata=metadata,
            classifier_layers=kwargs.get('classifier_layers', 3),
            dropout=dropout,
            pool_mode=pool_mode,
            cross_attn_queries=cross_attn_queries,
            cross_attn_heads=cross_attn_heads,
        )
    elif model_type == 'regressor':
        model = PoolHeadRegressor(
            pool_input_dim=pool_input_dim,
            hidden_dim=hidden_dim,
            metadata=metadata,
            regressor_layers=kwargs.get('regressor_layers', 3),
            dropout=dropout,
            pool_mode=pool_mode,
            cross_attn_queries=cross_attn_queries,
            cross_attn_heads=cross_attn_heads,
        )
    elif model_type == 'gated':
        model = PoolHeadGated(
            pool_input_dim=pool_input_dim,
            hidden_dim=hidden_dim,
            metadata=metadata,
            classifier_layers=kwargs.get('classifier_layers', 3),
            regressor_layers=kwargs.get('regressor_layers', 3),
            use_reg_adaptor=kwargs.get('use_reg_adaptor', True),
            reg_adaptor_layers=kwargs.get('reg_adaptor_layers', 2),
            dropout=dropout,
            pool_mode=pool_mode,
            cross_attn_queries=cross_attn_queries,
            cross_attn_heads=cross_attn_heads,
            cls_grad_scale=kwargs.get('cls_grad_scale', 1.0),
        )
    elif model_type == 'ddg_regressor':
        model = PoolHeadDDGRegressor(
            pool_input_dim=pool_input_dim,
            hidden_dim=hidden_dim,
            metadata=metadata,
            regressor_layers=kwargs.get('regressor_layers', 4),
            dropout=dropout,
            pool_mode=pool_mode,
            cross_attn_queries=cross_attn_queries,
            cross_attn_heads=cross_attn_heads,
            ddg_input_mode=kwargs.get('ddg_input_mode', 'concat_diff'),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                         f"Must be 'classifier', 'regressor', 'gated', or 'ddg_regressor'")
    
    return model
