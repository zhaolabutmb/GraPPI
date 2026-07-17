"""
Self-Supervised Learning Models for Protein Complex Pre-training.

Implements SSL pre-training strategies:
Masked Edge Prediction (MEP): Predict existence of inter-molecular edges

The models wrap existing HGT encoders without modification, adding task-specific
prediction heads for SSL objectives.

Node features (no ESM):
- [0:20]: One-hot amino acid type (20 classes) - MASKED for node prediction
- [20:22]: Receptor/ligand indicator
- [22:25]: Amino acid properties (hydrophobicity, polarity, charge)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Dict, Tuple

# Import the encoder
from model_collection.EMP_HGT import EdgeEnhancedHGT

class SSLEdgePredictor(nn.Module):
    """
    Self-supervised model for masked edge prediction (link prediction).
    
    Predicts whether an inter-molecular edge should exist between
    pairs of receptor and ligand nodes.
    
    Args:
        node_in_dim: Input node feature dimension
        edge_in_dim: Input edge feature dimension
        metadata: Graph metadata (node_types, edge_types)
        hidden_dim: Hidden dimension for encoder
        num_hgt_layers: Number of HGT layers
        hgt_heads: Number of attention heads
        predictor_layers: Number of layers in edge predictor
        dropout: Dropout rate
        use_checkpoint: Whether to use gradient checkpointing
    """
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        metadata: Tuple,
        hidden_dim: int = 1024,
        num_hgt_layers: int = 4,
        hgt_heads: int = 4,
        predictor_layers: int = 2,
        dropout: float = 0.2,
        use_checkpoint: bool = False,
        message_style: str = "gated_src", # Optional: additive
        **encoder_kwargs
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Initialize encoder (pure node-level, no pooling)
        self.encoder = EdgeEnhancedHGT(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            metadata=metadata,
            hgt_heads=hgt_heads,
            hgt_layers=num_hgt_layers,
            use_checkpoint=use_checkpoint,
            nn_drp_rate=dropout,
            message_style=message_style,
        )
        
        # Edge prediction head: takes concatenated node embeddings
        # Input: [receptor_emb || ligand_emb || |receptor_emb - ligand_emb| || receptor_emb * ligand_emb]
        edge_input_dim = hidden_dim * 4
        
        predictor_modules = []
        for i in range(predictor_layers):
            in_dim = edge_input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < predictor_layers - 1 else 1
            predictor_modules.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < predictor_layers - 1 else nn.Identity(),
                nn.GELU() if i < predictor_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < predictor_layers - 1 else nn.Identity(),
            ])
        self.edge_predictor = nn.Sequential(*predictor_modules)
    
    def forward_encoder(self, batch_data: Batch) -> Dict[str, torch.Tensor]:
        """
        Run encoder and return node-level embeddings.
        
        Returns:
            Dict mapping node type to node embeddings
        """
        return self.encoder(batch_data)
    
    def forward(self, batch_data: Batch, mask_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for edge prediction.
        
        Args:
            batch_data: Batched HeteroData with masked edges
            mask_info: Dict containing edge pairs and labels
        
        Returns:
            Tuple of (edge_logits, edge_labels)
        """
        x_hgt = self.forward_encoder(batch_data)
        
        dev = next(self.parameters()).device
        
        all_edges = mask_info['all_edges'].to(dev)  # [2, num_edges]: (receptor_idx, ligand_idx)
        edge_labels = mask_info['edge_labels'].to(dev)
        
        if all_edges.size(1) == 0:
            return torch.empty(0, 1, device=dev), edge_labels
        
        # Get node embeddings for edge endpoints
        receptor_emb = x_hgt['receptor'][all_edges[0]]  # [num_edges, hidden_dim]
        ligand_emb = x_hgt['ligand'][all_edges[1]]      # [num_edges, hidden_dim]
        
        # Combine embeddings for edge prediction
        edge_features = torch.cat([
            receptor_emb,
            ligand_emb,
            torch.abs(receptor_emb - ligand_emb),
            receptor_emb * ligand_emb,
        ], dim=1)
        
        edge_logits = self.edge_predictor(edge_features)
        
        return edge_logits.squeeze(-1), edge_labels
    
    def compute_loss(
        self,
        edge_logits: torch.Tensor,
        edge_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss for edge prediction."""
        if len(edge_labels) == 0:
            return torch.tensor(0.0, device=edge_logits.device)
        
        return F.binary_cross_entropy_with_logits(edge_logits, edge_labels)


class SSLInterIntraEdgePredictor(nn.Module):
    """
    Self-supervised model for both inter- and intra-molecular edge prediction.
    
    Uses a shared encoder and two separate prediction heads:
    - Inter-edge head: predicts receptor-ligand contacts
    - Intra-edge head: predicts same-chain contacts near the interface
    
    Loss: L = beta * L_inter + (1 - beta) * L_intra
    
    Args:
        node_in_dim: Input node feature dimension
        edge_in_dim: Input edge feature dimension
        metadata: Graph metadata (node_types, edge_types)
        hidden_dim: Hidden dimension for encoder
        num_hgt_layers: Number of HGT layers
        hgt_heads: Number of attention heads
        predictor_layers: Number of layers in each edge predictor head
        dropout: Dropout rate
        use_checkpoint: Whether to use gradient checkpointing
        beta: Loss weight — L = beta * L_inter + (1 - beta) * L_intra
        message_style: HGT message style
    """
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        metadata: Tuple,
        hidden_dim: int = 1024,
        num_hgt_layers: int = 4,
        hgt_heads: int = 4,
        predictor_layers: int = 2,
        dropout: float = 0.2,
        use_checkpoint: bool = False,
        beta: float = 0.7,
        message_style: str = "gated_src",
        **encoder_kwargs
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.beta = beta
        
        # Shared encoder
        self.encoder = EdgeEnhancedHGT(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            metadata=metadata,
            hgt_heads=hgt_heads,
            hgt_layers=num_hgt_layers,
            use_checkpoint=use_checkpoint,
            nn_drp_rate=dropout,
            message_style=message_style,
        )
        
        # Edge prediction input: [emb_A || emb_B || |emb_A - emb_B| || emb_A * emb_B]
        edge_input_dim = hidden_dim * 4
        
        # Inter-edge prediction head (receptor ↔ ligand)
        inter_modules = []
        for i in range(predictor_layers):
            in_dim = edge_input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < predictor_layers - 1 else 1
            inter_modules.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < predictor_layers - 1 else nn.Identity(),
                nn.GELU() if i < predictor_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < predictor_layers - 1 else nn.Identity(),
            ])
        self.inter_edge_predictor = nn.Sequential(*inter_modules)
        
        # Intra-edge prediction head (same-chain, near interface)
        intra_modules = []
        for i in range(predictor_layers):
            in_dim = edge_input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < predictor_layers - 1 else 1
            intra_modules.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < predictor_layers - 1 else nn.Identity(),
                nn.GELU() if i < predictor_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < predictor_layers - 1 else nn.Identity(),
            ])
        self.intra_edge_predictor = nn.Sequential(*intra_modules)
    
    def forward_encoder(self, batch_data: Batch) -> Dict[str, torch.Tensor]:
        """Run shared encoder and return node-level embeddings."""
        return self.encoder(batch_data)
    
    def forward(self, batch_data: Batch, mask_info: Dict) -> Dict:
        """
        Forward pass for inter + intra edge prediction.
        
        Args:
            batch_data: Batched HeteroData with masked edges
            mask_info: Dict with 'inter_edge_mask_info' and 'intra_edge_mask_info'
        
        Returns:
            Dict with inter/intra logits and labels
        """
        x_hgt = self.forward_encoder(batch_data)
        dev = next(self.parameters()).device
        
        outputs = {}
        
        # ---- Inter-edge prediction (receptor ↔ ligand) ----
        inter_info = mask_info['inter_edge_mask_info']
        inter_edges = inter_info['all_edges'].to(dev)
        
        if inter_edges.size(1) > 0:
            receptor_emb = x_hgt['receptor'][inter_edges[0]]
            ligand_emb = x_hgt['ligand'][inter_edges[1]]
            
            inter_features = torch.cat([
                receptor_emb,
                ligand_emb,
                torch.abs(receptor_emb - ligand_emb),
                receptor_emb * ligand_emb,
            ], dim=1)
            
            outputs['inter_edge_logits'] = self.inter_edge_predictor(inter_features).squeeze(-1)
        else:
            outputs['inter_edge_logits'] = torch.empty(0, device=dev)
        
        outputs['inter_edge_labels'] = inter_info['edge_labels'].to(dev)
        
        # ---- Intra-edge prediction (same-chain near interface) ----
        intra_info = mask_info['intra_edge_mask_info']
        intra_edges = intra_info['all_edges'].to(dev)
        intra_node_type = intra_info['edge_node_type'].to(dev)  # 0=receptor, 1=ligand
        
        if intra_edges.size(1) > 0:
            # Get embeddings based on node type
            # receptor edges: both indices into x_hgt['receptor']
            # ligand edges: both indices into x_hgt['ligand']
            receptor_mask = (intra_node_type == 0)
            ligand_mask = (intra_node_type == 1)
            
            intra_logits = torch.empty(intra_edges.size(1), device=dev)
            
            if receptor_mask.any():
                r_edges = intra_edges[:, receptor_mask]
                emb_a = x_hgt['receptor'][r_edges[0]]
                emb_b = x_hgt['receptor'][r_edges[1]]
                features = torch.cat([
                    emb_a, emb_b,
                    torch.abs(emb_a - emb_b),
                    emb_a * emb_b,
                ], dim=1)
                intra_logits[receptor_mask] = self.intra_edge_predictor(features).squeeze(-1).float()
            
            if ligand_mask.any():
                l_edges = intra_edges[:, ligand_mask]
                emb_a = x_hgt['ligand'][l_edges[0]]
                emb_b = x_hgt['ligand'][l_edges[1]]
                features = torch.cat([
                    emb_a, emb_b,
                    torch.abs(emb_a - emb_b),
                    emb_a * emb_b,
                ], dim=1)
                intra_logits[ligand_mask] = self.intra_edge_predictor(features).squeeze(-1).float()
            
            outputs['intra_edge_logits'] = intra_logits
        else:
            outputs['intra_edge_logits'] = torch.empty(0, device=dev)
        
        outputs['intra_edge_labels'] = intra_info['edge_labels'].to(dev)
        
        return outputs
    
    def compute_loss(self, outputs: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss: L = beta * L_inter + (1 - beta) * L_intra
        
        Returns:
            Tuple of (total_loss, loss_dict with individual losses)
        """
        dev = next(self.parameters()).device
        losses = {}
        
        # Inter-edge loss
        if len(outputs['inter_edge_labels']) > 0:
            losses['inter_edge_loss'] = F.binary_cross_entropy_with_logits(
                outputs['inter_edge_logits'], outputs['inter_edge_labels']
            )
        else:
            losses['inter_edge_loss'] = torch.tensor(0.0, device=dev)
        
        # Intra-edge loss
        if len(outputs['intra_edge_labels']) > 0:
            losses['intra_edge_loss'] = F.binary_cross_entropy_with_logits(
                outputs['intra_edge_logits'], outputs['intra_edge_labels']
            )
        else:
            losses['intra_edge_loss'] = torch.tensor(0.0, device=dev)
        
        # Combined: L = beta * L_inter + (1 - beta) * L_intra
        total_loss = self.beta * losses['inter_edge_loss'] + (1 - self.beta) * losses['intra_edge_loss']
        losses['total_loss'] = total_loss
        
        return total_loss, losses


def create_ssl_model(
    node_in_dim: int,
    edge_in_dim: int,
    metadata: Tuple,
    hidden_dim: int = 1024,
    num_hgt_layers: int = 4,
    hgt_heads: int = 4,
    predictor_layers: int = 2,
    dropout: float = 0.2,
    use_checkpoint: bool = False,
    mask_type: str = 'edge',
    beta: float = 0.7,
    **kwargs  # Accept but ignore extra kwargs for compatibility
) -> nn.Module:
    """
    Factory function to create SSL edge prediction model.
    
    Args:
        node_in_dim: Input node feature dimension
        edge_in_dim: Input edge feature dimension
        metadata: Graph metadata
        hidden_dim: Encoder hidden dimension
        num_hgt_layers: Number of HGT layers
        hgt_heads: Number of attention heads
        predictor_layers: Number of layers in edge predictor
        dropout: Dropout rate
        use_checkpoint: Whether to use gradient checkpointing
        mask_type: 'edge' for inter-only, 'inter_intra_edge' for both
        beta: Loss weight for inter_intra_edge mode (L = beta*L_inter + (1-beta)*L_intra)
        **kwargs: Additional arguments (e.g. message_style)
    
    Returns:
        SSLEdgePredictor or SSLInterIntraEdgePredictor model instance
    """
    if mask_type == 'inter_intra_edge':
        return SSLInterIntraEdgePredictor(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            metadata=metadata,
            hidden_dim=hidden_dim,
            num_hgt_layers=num_hgt_layers,
            hgt_heads=hgt_heads,
            predictor_layers=predictor_layers,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            beta=beta,
            **kwargs
        )
    else:
        return SSLEdgePredictor(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            metadata=metadata,
            hidden_dim=hidden_dim,
            num_hgt_layers=num_hgt_layers,
            hgt_heads=hgt_heads,
            predictor_layers=predictor_layers,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
            **kwargs
        )
