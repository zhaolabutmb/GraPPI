"""
HGTConv with Attention Extraction Capability
This module provides a way to extract attention weights from HGTConv_fixed layers
without modifying the original model files.

Usage:
    from model_collection.HGTConv_with_attention_v2 import extract_attention_from_model
    
    attention_dict = extract_attention_from_model(model, data)
"""

import torch
import math
from typing import Dict, Optional
from torch import Tensor
from torch_geometric.utils import softmax
import copy

# Global dictionary to store attention weights during forward pass
_attention_storage = {}


def clear_attention_storage():
    """Clear the global attention storage."""
    global _attention_storage
    _attention_storage = {}


def get_attention_storage():
    """Get the current attention storage."""
    return _attention_storage


def monkey_patch_message(hgtconv_layer, edge_type_dict):
    """
    Monkey-patch the message() method of an HGTConv layer to capture attention.
    
    Args:
        hgtconv_layer: The HGTConv layer to patch
        edge_type_dict: Dictionary mapping this layer to its edge types
    """
    # Store original message method
    original_message = hgtconv_layer.message
    layer_id = id(hgtconv_layer)
    
    def patched_message(k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                       index: Tensor, ptr: Optional[Tensor],
                       size_i: Optional[int]) -> Tensor:
        
        # Compute attention scores (same as original HGTConv_fixed)
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha_normalized = softmax(alpha, index, ptr, size_i)
        
        # STORE ATTENTION WEIGHTS
        # We need to figure out which edge type this is for
        # This is tricky because propagate() is called once with all edges combined
        # For now, store all attention weights together
        if layer_id not in _attention_storage:
            _attention_storage[layer_id] = {
                'all_attention': []
            }
        
        _attention_storage[layer_id]['all_attention'].append({
            'alpha': alpha_normalized.detach().cpu(),
            'edge_index_j': index.detach().cpu(),
            'num_edges': alpha_normalized.size(0)
        })
        
        # Apply attention to values (same as original)
        out = v_j * alpha_normalized.view(-1, hgtconv_layer.heads, 1)
        return out.view(-1, hgtconv_layer.out_channels)
    
    # Replace the message method
    hgtconv_layer.message = patched_message
    
    return hgtconv_layer


def extract_attention_from_model(model, data, device='cpu'):
    """
    Extract attention weights from HGT layers by monkey-patching their message() methods.
    
    Args:
        model: HeteroGraphAffinityRegressor model (will be modified, so pass a copy!)
        data: Input graph data
        device: Device to run on
        
    Returns:
        Dictionary with attention information
    """
    from .HGTConv_fixed import HGTConv as FixedHGTConv
    
    # Clear previous storage
    clear_attention_storage()
    
    # Find all HGTConv layers and patch them
    hgtconv_layers = []
    hgtconv_names = []
    
    for name, module in model.named_modules():
        if isinstance(module, FixedHGTConv):
            hgtconv_layers.append(module)
            hgtconv_names.append(name)
            print(f"Found HGTConv layer: {name}")
    
    print(f"\nTotal HGTConv layers found: {len(hgtconv_layers)}")
    
    if len(hgtconv_layers) == 0:
        raise ValueError("No HGTConv layers found in model!")
    
    # Patch all layers
    for layer, name in zip(hgtconv_layers, hgtconv_names):
        monkey_patch_message(layer, None)
        print(f"Patched layer: {name}")
    
    # Run forward pass
    model.eval()
    print("\nRunning forward pass to capture attention...")
    with torch.no_grad():
        _ = model(data)
    
    # Get stored attention
    attention_storage = get_attention_storage()
    
    print(f"\nAttention storage keys (layer IDs): {list(attention_storage.keys())}")
    print(f"Number of layers that captured attention: {len(attention_storage)}")
    
    if not attention_storage:
        raise ValueError("No attention weights captured!")
    
    # Get attention from the last HGT layer
    last_layer_id = list(attention_storage.keys())[-1]
    last_layer_attention = attention_storage[last_layer_id]
    
    print(f"\nExtracted attention from layer {last_layer_id}")
    
    # Now we need to split the combined attention back into edge types
    # This requires knowing the edge_index_dict structure
    edge_index_dict = data.edge_index_dict
    
    # The construct_bipartite_edge_index combines edges in order of edge_types
    # We need to split them back
    result = {}
    
    # Get all attention data
    all_attn_data = last_layer_attention['all_attention']
    
    # For simplicity, let's just return the combined attention
    # and let the user split it if needed
    # Better approach: reconstruct per edge type based on edge counts
    
    edge_counts = {
        '__'.join(et): edge_index_dict[et].size(1) 
        for et in edge_index_dict.keys()
    }
    
    print(f"\nEdge counts per type: {edge_counts}")
    
    # Concatenate all attention from this layer
    if len(all_attn_data) > 0:
        all_alpha = []
        for attn_data in all_attn_data:
            all_alpha.append(attn_data['alpha'])
        
        combined_attention = torch.cat(all_alpha, dim=0)
        
        # Try to split back into edge types based on counts
        # This assumes edges are processed in the order they appear in edge_index_dict
        current_idx = 0
        for edge_type_tuple in edge_index_dict.keys():
            edge_type_str = '__'.join(edge_type_tuple)
            num_edges = edge_counts[edge_type_str]
            
            # Extract attention for this edge type
            edge_attn = combined_attention[current_idx:current_idx + num_edges]
            
            result[edge_type_str] = [{
                'alpha': edge_attn,
                'num_edges': num_edges
            }]
            
            current_idx += num_edges
            
            print(f"Split {edge_type_str}: {num_edges} edges, attention shape: {edge_attn.shape}")
    
    return result


def _split_layer_attention(layer_attention, edge_index_dict):
    """Split a single layer's combined attention tensor back into per-edge-type dicts."""
    all_attn_data = layer_attention['all_attention']
    if not all_attn_data:
        return {}

    combined_attention = torch.cat([a['alpha'] for a in all_attn_data], dim=0)

    result = {}
    current_idx = 0
    for edge_type_tuple in edge_index_dict.keys():
        edge_type_str = '__'.join(edge_type_tuple)
        num_edges = edge_index_dict[edge_type_tuple].size(1)
        edge_attn = combined_attention[current_idx:current_idx + num_edges]
        result[edge_type_str] = [{'alpha': edge_attn, 'num_edges': num_edges}]
        current_idx += num_edges
    return result


def extract_attention_all_layers(model, data, device='cpu'):
    """Extract attention weights from ALL HGT layers.

    Like extract_attention_from_model but returns a list of per-edge-type
    attention dicts, one per layer (index 0 = first/shallowest layer).

    Args:
        model: EdgeEnhancedHGT model (will be modified — pass a deep copy)
        data: Input graph data (on device)
        device: Device string or torch.device

    Returns:
        List of attention dicts, each in the same format as
        extract_attention_from_model: {edge_type_str: [{'alpha', 'num_edges'}]}
        Length equals the number of HGTConv layers in the model.
    """
    from .HGTConv_fixed import HGTConv as FixedHGTConv

    clear_attention_storage()

    hgtconv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, FixedHGTConv):
            hgtconv_layers.append(module)
            monkey_patch_message(module, None)

    if not hgtconv_layers:
        raise ValueError("No HGTConv layers found in model!")

    model.eval()
    with torch.no_grad():
        _ = model(data)

    attention_storage = get_attention_storage()
    edge_index_dict = data.edge_index_dict

    # attention_storage keys are object ids in forward-pass order
    layer_results = []
    for layer_id in attention_storage.keys():
        layer_attn_dict = _split_layer_attention(
            attention_storage[layer_id], edge_index_dict)
        layer_results.append(layer_attn_dict)

    return layer_results


def aggregate_attention_to_nodes(attention_dict, batch_data):
    """
    Aggregate edge-level attention to node-level attention scores.
    
    Returns:
        node_attention: Dict with keys 'receptor' and 'ligand', each containing:
            - 'total': total attention received by each node
            - 'incoming': attention from incoming edges  
            - 'outgoing': attention from outgoing edges
        edge_type_attention: Dict with attention statistics per edge type
    """
    # Initialize node attention storage
    num_receptor = batch_data['receptor'].x.size(0)
    num_ligand = batch_data['ligand'].x.size(0)
    
    node_attention = {
        'receptor': {
            'total': torch.zeros(num_receptor),
            'incoming': torch.zeros(num_receptor),
            'outgoing': torch.zeros(num_receptor),
        },
        'ligand': {
            'total': torch.zeros(num_ligand),
            'incoming': torch.zeros(num_ligand),
            'outgoing': torch.zeros(num_ligand),
        }
    }
    
    edge_type_attention = {}
    
    # Process each edge type
    for edge_type_str, attn_list in attention_dict.items():
        src_type, rel_type, dst_type = edge_type_str.split('__')
        
        # Get edge indices for this edge type
        edge_type_tuple = (src_type, rel_type, dst_type)
        edge_index = batch_data[edge_type_tuple].edge_index
        
        # Concatenate all attention weights (average across heads)
        all_alpha = []
        for attn_data in attn_list:
            alpha = attn_data['alpha']  # [num_edges, heads]
            alpha_mean = alpha.mean(dim=1)  # Average across heads
            all_alpha.append(alpha_mean)
        
        attention_weights = torch.cat(all_alpha, dim=0)  # [total_edges]
        
        # Store edge type statistics
        edge_type_attention[edge_type_str] = {
            'mean': attention_weights.mean().item(),
            'std': attention_weights.std().item(),
            'max': attention_weights.max().item(),
            'min': attention_weights.min().item(),
            'num_edges': attention_weights.size(0)
        }
        
        # Aggregate to nodes
        src_nodes = edge_index[0].cpu()
        dst_nodes = edge_index[1].cpu()
        
        # Source nodes (outgoing attention)
        for i, attn in zip(src_nodes, attention_weights):
            node_attention[src_type]['outgoing'][i] += attn
            node_attention[src_type]['total'][i] += attn
        
        # Destination nodes (incoming attention)
        for i, attn in zip(dst_nodes, attention_weights):
            node_attention[dst_type]['incoming'][i] += attn
            node_attention[dst_type]['total'][i] += attn
    
    attn_list_AB = attention_dict['receptor__receptor_ligand__ligand']
    attn_list_BA = attention_dict['ligand__ligand_receptor__receptor']
    all_alpha = []
    for attn_data_AB, attn_data_BA in zip(attn_list_AB, attn_list_BA):
        alpha_AB = attn_data_AB['alpha']
        alpha_mean_AB = alpha_AB.mean(dim=1)
        alpha_BA = attn_data_BA['alpha']
        alpha_mean_BA = alpha_BA.mean(dim=1)
        all_alpha.append(torch.cat((alpha_mean_AB, alpha_mean_BA)))
    attention_weights = torch.cat(all_alpha, dim=0)
    
    edge_type_attention['intermolecule'] = {
        'mean': attention_weights.mean().item(),
            'std': attention_weights.std().item(),
            'max': attention_weights.max().item(),
            'min': attention_weights.min().item(),
            'num_edges': attention_weights.size(0)
    }
    
    
    return node_attention, edge_type_attention
