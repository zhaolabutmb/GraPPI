import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch_scatter import scatter
from torch_geometric.utils import dropout_edge
from .HGTConv_fixed import HGTConv
from torch.utils.checkpoint import checkpoint


class CrossAttentionAggregator(nn.Module):
    """
    Cross-attention aggregation for combining node embeddings from receptor and ligand.
    
    Uses learnable queries to attend over node embeddings, then applies
    cross-chain attention so each chain's summary is informed by the partner chain's
    node embeddings. This captures inter-molecular context before concatenation.
    
    Architecture:
        1. Independent query aggregation: queries attend to own chain's nodes
        2. Cross-chain interaction: each chain's summary attends to partner's nodes
        3. Normalize, flatten, and concatenate
    
    Args:
        hidden_dim: Dimension of node embeddings
        num_queries: Number of learnable query vectors
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(
        self,
        hidden_dim: int,
        num_queries: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        
        # Learnable queries - one set for receptor, one for ligand
        self.receptor_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        self.ligand_queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        
        # Stage 1: Independent query-to-node cross-attention
        self.receptor_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ligand_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Post stage-1 norms
        self.receptor_norm = nn.LayerNorm(hidden_dim)
        self.ligand_norm = nn.LayerNorm(hidden_dim)
        
        # Stage 2: Cross-chain interaction
        # Receptor summary attends to ligand nodes (and vice versa)
        self.receptor_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ligand_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Post stage-2 norms (after residual)
        self.receptor_cross_norm = nn.LayerNorm(hidden_dim)
        self.ligand_cross_norm = nn.LayerNorm(hidden_dim)
        
        # Final projection (combines receptor and ligand attended features)
        # Output: num_queries * hidden_dim * 2 (receptor + ligand)
        self.output_dim = num_queries * hidden_dim * 2
        self.output_norm = nn.LayerNorm(self.output_dim)
        
        # Initialize queries
        nn.init.xavier_uniform_(self.receptor_queries)
        nn.init.xavier_uniform_(self.ligand_queries)
    
    def forward(self, x_dict: dict, batch_dict: dict) -> torch.Tensor:
        """
        Forward pass for cross-attention aggregation with cross-chain interaction.
        
        Stage 1: Independent query-to-node attention (each chain summarized separately)
        Stage 2: Cross-chain attention (each summary attends to partner's nodes)
        Stage 3: Normalize, flatten, concatenate
        
        Args:
            x_dict: Dict of node embeddings {'receptor': [N_r, D], 'ligand': [N_l, D]}
            batch_dict: Dict of batch indices {'receptor': [N_r], 'ligand': [N_l]}
        
        Returns:
            Aggregated graph-level embedding [batch_size, output_dim]
        """
        device = next(iter(x_dict.values())).device
        
        # Get unique batch indices to determine number of graphs
        batch_indices = torch.unique(batch_dict['receptor'])
        batch_size = len(batch_indices)
        
        # Expand queries for batch
        receptor_q = self.receptor_queries.expand(batch_size, -1, -1)  # [B, Q, D]
        ligand_q = self.ligand_queries.expand(batch_size, -1, -1)      # [B, Q, D]
        
        # Build padded node sequences (reused by both stages)
        receptor_padded, receptor_mask = self._build_padded_batch(
            x_dict['receptor'], batch_dict['receptor'], batch_size
        )  # [B, max_Nr, D], [B, max_Nr]
        ligand_padded, ligand_mask = self._build_padded_batch(
            x_dict['ligand'], batch_dict['ligand'], batch_size
        )  # [B, max_Nl, D], [B, max_Nl]
        
        # --- Stage 1: Independent query-to-node cross-attention ---
        receptor_attended, _ = self.receptor_attn(
            receptor_q, receptor_padded, receptor_padded,
            key_padding_mask=receptor_mask
        )  # [B, Q, D]
        
        ligand_attended, _ = self.ligand_attn(
            ligand_q, ligand_padded, ligand_padded,
            key_padding_mask=ligand_mask
        )  # [B, Q, D]
        
        # Normalize after stage 1
        receptor_attended = self.receptor_norm(receptor_attended)
        ligand_attended = self.ligand_norm(ligand_attended)
        
        # --- Stage 2: Cross-chain interaction ---
        # Receptor summary attends to ligand nodes
        receptor_cross, _ = self.receptor_cross_attn(
            receptor_attended, ligand_padded, ligand_padded,
            key_padding_mask=ligand_mask
        )  # [B, Q, D]
        
        # Ligand summary attends to receptor nodes
        ligand_cross, _ = self.ligand_cross_attn(
            ligand_attended, receptor_padded, receptor_padded,
            key_padding_mask=receptor_mask
        )  # [B, Q, D]
        
        # Residual + norm
        receptor_attended = self.receptor_cross_norm(receptor_attended + receptor_cross)
        ligand_attended = self.ligand_cross_norm(ligand_attended + ligand_cross)
        
        # --- Stage 3: Flatten and concatenate ---
        receptor_flat = receptor_attended.reshape(batch_size, -1)  # [B, Q*D]
        ligand_flat = ligand_attended.reshape(batch_size, -1)      # [B, Q*D]
        
        output = torch.cat([receptor_flat, ligand_flat], dim=1)    # [B, 2*Q*D]
        output = self.output_norm(output)
        
        return output
    
    def _build_padded_batch(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        batch_size: int,
    ) -> tuple:
        """
        Build padded batch tensor and attention mask from variable-length node sequences.
        
        Args:
            x: Node embeddings [N, D]
            batch: Batch indices [N]
            batch_size: Number of graphs in batch
        
        Returns:
            Tuple of (padded_x [B, max_nodes, D], attn_mask [B, max_nodes])
            where attn_mask is True for padded positions (to be ignored)
        """
        device = x.device
        D = x.size(1)
        
        node_counts = torch.bincount(batch, minlength=batch_size)
        max_nodes = node_counts.max().item()
        
        if max_nodes == 0:
            return (torch.zeros(batch_size, 1, D, device=device),
                    torch.ones(batch_size, 1, dtype=torch.bool, device=device))
        
        padded_x = torch.zeros(batch_size, max_nodes, D, device=device)
        attn_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=device)
        
        for b in range(batch_size):
            mask = (batch == b)
            x_b = x[mask]
            n_b = x_b.size(0)
            if n_b > 0:
                padded_x[b, :n_b] = x_b
                attn_mask[b, :n_b] = False
        
        return padded_x, attn_mask


class EdgeEnhancedHGT(nn.Module):
    """
    Edge-Enhanced Heterogeneous Graph Transformer encoder.
    
    Pure encoder: returns node-level embeddings (dict of {node_type: [N, hidden_dim]}).
    Aggregation/pooling is the responsibility of downstream models (FineTune, Gated, etc.).
    
    Architecture:
        1. Node projection: node_in_dim -> hidden_dim
        2. Edge message passing (1 layer): gated or additive edge features
        3. HGT stack: multi-layer heterogeneous graph transformer
    """
    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        hidden_dim,
        metadata,
        hgt_heads=4,
        hgt_layers=4,
        use_edge_attr=True,
        message_style="gated_src", #additive
        edge_bottleneck=16,
        emb_drp_rate=0.15,
        nn_drp_rate=0.2,
        edge_drp_rate=0.15,
        use_checkpoint=False,
    ):
        super().__init__()
        node_types, edge_types = metadata
        self.hidden_dim = hidden_dim
        self.use_edge_attr = use_edge_attr
        self.message_style = message_style
        self.dropout = nn.Dropout(p=emb_drp_rate)
        self.edge_dropout = edge_drp_rate
        self.use_checkpoint = use_checkpoint

        # --- Stage 1: Edge modules ---
        self.edge_norm_layers = nn.ModuleDict()
        for _, rel, _ in edge_types:
            self.edge_norm_layers[rel] = nn.LayerNorm(hidden_dim)

        if message_style == "additive":
            # Edge embedding with bottleneck -> hidden_dim
            self.edge_encoders = nn.ModuleDict()
            for _, rel, _ in edge_types:
                self.edge_encoders[rel] = nn.Sequential(
                    nn.Linear(edge_in_dim, edge_bottleneck),
                    nn.LayerNorm(edge_bottleneck),
                    nn.GELU(),
                    nn.Dropout(nn_drp_rate),
                    nn.Linear(edge_bottleneck, hidden_dim),
                )
            # Optional small scalar per relation to temper edge influence
            self.rel_alpha = nn.ParameterDict({
                rel: nn.Parameter(torch.tensor(0.0)) for _, rel, _ in edge_types
            })
        else:
            # Gated scalar per edge (0..1) to modulate source node features
            self.edge_gates = nn.ModuleDict()
            for _, rel, _ in edge_types:
                self.edge_gates[rel] = nn.Sequential(
                    nn.Linear(edge_in_dim, edge_bottleneck),
                    nn.LayerNorm(edge_bottleneck),
                    nn.GELU(),
                    nn.Dropout(nn_drp_rate),
                    nn.Linear(edge_bottleneck, 1),
                    nn.Sigmoid(),
                )
            # Start with very small influence; learn to use edges
            self.rel_alpha = nn.ParameterDict({
                rel: nn.Parameter(torch.tensor(-4.0)) for _, rel, _ in edge_types  # sigmoid(-4) ~ 0.018
            })

        self.node_proj = Linear(node_in_dim, hidden_dim)

        # --- HGT stack ---
        self.hgt_layers = nn.ModuleList()
        self.hgt_norm_layers = nn.ModuleList()
        for _ in range(hgt_layers):
            self.hgt_layers.append(
                HGTConv(hidden_dim, hidden_dim, metadata, heads=hgt_heads)
            )
            self.hgt_norm_layers.append(nn.LayerNorm(hidden_dim))


    def forward(self, batch_data, return_all_layers: bool = False):
        """
        Forward pass: returns node-level embeddings.
        
        Args:
            batch_data: Batched HeteroData
            return_all_layers: If True, return a list of per-layer output dicts
                for Jumping Knowledge aggregation. The list has length
                (num_hgt_layers + 1): the initial projection+edge output
                followed by each HGT layer output.
                If False (default), return only the final layer output.
        
        Returns:
            If return_all_layers=False:
                Dict mapping node type to node embeddings {str: [N, hidden_dim]}
            If return_all_layers=True:
                List of such dicts, one per layer (length = num_hgt_layers + 1)
        """
        x_dict = batch_data.x_dict
        edge_index_dict = batch_data.edge_index_dict
        edge_attr_dict = batch_data.edge_attr_dict
        dev = next(self.parameters()).device
        #------------------------------
        # Project nodes once
        x_dict_proj = {nt: self.node_proj(x) for nt, x in x_dict.items()}

        if self.use_edge_attr:
            # Accumulate messages per destination type
            edge_messages = {nt: torch.zeros_like(x) for nt, x in x_dict_proj.items()}
            
            for edge_type, edge_index in edge_index_dict.items():
                src_type, rel_type, dst_type = edge_type
                # Ensure tensors are on the same device and correct dtype
                edge_index = edge_index.to(dev)
                edge_attr = edge_attr_dict[edge_type].to(dev) # [E, edge_in_dim]  

                if self.message_style == "additive":
                    msg = self.edge_encoders[rel_type](edge_attr)            # [E, hidden_dim]
                    msg = self.edge_norm_layers[rel_type](msg)
                else:
                    # Gated source features: msg = gate(edge) * x_src
                    gate = self.edge_gates[rel_type](edge_attr)              # [E, 1] in (0,1)
                    x_src = x_dict_proj[src_type][edge_index[0]]             # [E, hidden_dim]
                    msg = gate * x_src                                       # [E, hidden_dim]
                
                # Per-relation learnable scale, start small; Safer aggregation: specify dim_size and avoid writing via out=
                num_dst = x_dict_proj[dst_type].size(0)
                # Scatter to dst nodes
                agg = scatter(src=msg, index=edge_index[1], dim=0, dim_size=num_dst, reduce='sum')
                
                # Apply per-relation learnable scaling factor
                agg = agg * torch.sigmoid(self.rel_alpha[rel_type])

                edge_messages[dst_type] = edge_messages[dst_type] + agg
                
            x_hgt = {nt: x_dict_proj[nt] + edge_messages[nt] for nt in x_dict_proj}
        else:
            x_hgt = x_dict_proj

        # Collect all layer outputs for JK-Net if requested
        if return_all_layers:
            all_layer_outputs = [{nt: x.clone() for nt, x in x_hgt.items()}]

        # HGT layers with residual, norm, and dropout (+ optional edge dropout before non-last layers)
        for i, (hgt_layer, norm_layer) in enumerate(zip(self.hgt_layers, self.hgt_norm_layers)):
            if i < len(self.hgt_layers) - 1:
                edge_index_dict_dr = {et: dropout_edge(ei, p=self.edge_dropout, training=self.training)[0]
                                        for et, ei in edge_index_dict.items()}
                if self.use_checkpoint and self.training:
                    x_new = checkpoint(self._hgt_forward, hgt_layer, x_hgt, edge_index_dict_dr, use_reentrant=False)
                    x_new = {k: self.dropout(v) for k, v in x_new.items()}
                else:
                    x_new = hgt_layer(x_hgt, edge_index_dict_dr)
                    x_new = {k: self.dropout(v) for k, v in x_new.items()}
            else:
                edge_index_dict_dev = {et: ei.to(dev) for et, ei in edge_index_dict.items()}
                if self.use_checkpoint and self.training:
                    x_new = checkpoint(self._hgt_forward, hgt_layer, x_hgt, edge_index_dict_dev, use_reentrant=False)
                else:
                    x_new = hgt_layer(x_hgt, edge_index_dict_dev)

            x_hgt = {k: norm_layer(x_new[k] + x_hgt[k]) for k in x_hgt}

            if return_all_layers:
                all_layer_outputs.append({nt: x.clone() for nt, x in x_hgt.items()})

        if return_all_layers:
            return all_layer_outputs  # list of dicts, length = num_hgt_layers + 1

        return x_hgt
    
    def _hgt_forward(self, hgt_layer, x_dict, edge_index_dict):
        """Helper function for checkpointing HGT layers"""
        return hgt_layer(x_dict, edge_index_dict)