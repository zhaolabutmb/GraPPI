"""
SSL Pre-training functions.
This module contains training and validation functions for SSL edge prediction.
Supports both inter-only edge prediction (SSLEdgePredictor) and
inter+intra edge prediction (SSLInterIntraEdgePredictor).
"""
import time
import torch
from contextlib import nullcontext
from collections import defaultdict
from .common_utils import compute_edge_metrics


def _is_inter_intra_model(model) -> bool:
    """Check if model is the inter+intra edge predictor."""
    return hasattr(model, 'intra_edge_predictor')


def train_epoch_ssl(
    model,
    loader,
    optimizer,
    scaler,
    device,
    clip_max_norm: float = 5.0,
    use_amp: bool = True,
    debug: bool = False,
):
    """
    Train one epoch for SSL pre-training (edge prediction).
    Handles both inter-only and inter+intra models transparently.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    grad_norm_sum = 0.0
    total_batches = len(loader)
    epoch_start = time.time()
    
    all_metrics = defaultdict(list)
    is_inter_intra = _is_inter_intra_model(model)
    
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
    
    batch_end_time = time.time()  # track time between batches (data loading)
    
    for batch_data, mask_info, pdb_names in loader:
        data_time = time.time() - batch_end_time  # time spent in DataLoader
        compute_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        batch_data = batch_data.to(device)
        
        with amp_ctx:
            if is_inter_intra:
                outputs = model(batch_data, mask_info)
                loss, loss_dict = model.compute_loss(outputs)
                
                with torch.no_grad():
                    # Inter-edge metrics
                    if len(outputs['inter_edge_labels']) > 0:
                        inter_metrics = compute_edge_metrics(
                            outputs['inter_edge_logits'], outputs['inter_edge_labels']
                        )
                        all_metrics['inter_edge_acc'].append(inter_metrics['accuracy'])
                        all_metrics['inter_edge_auc'].append(inter_metrics['auc'])
                    # Intra-edge metrics
                    if len(outputs['intra_edge_labels']) > 0:
                        intra_metrics = compute_edge_metrics(
                            outputs['intra_edge_logits'], outputs['intra_edge_labels']
                        )
                        all_metrics['intra_edge_acc'].append(intra_metrics['accuracy'])
                        all_metrics['intra_edge_auc'].append(intra_metrics['auc'])
                    # Track individual losses
                    all_metrics['inter_edge_loss'].append(loss_dict['inter_edge_loss'].item())
                    all_metrics['intra_edge_loss'].append(loss_dict['intra_edge_loss'].item())
            else:
                edge_logits, edge_labels = model(batch_data, mask_info)
                loss = model.compute_loss(edge_logits, edge_labels)
                
                with torch.no_grad():
                    if len(edge_labels) > 0:
                        metrics = compute_edge_metrics(edge_logits, edge_labels)
                        all_metrics['edge_acc'].append(metrics['accuracy'])
                        all_metrics['edge_auc'].append(metrics['auc'])
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
        
        # Track gradient norm
        post_sq = sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None)
        grad_norm_sum += post_sq ** 0.5
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        n_batches += 1
        
        # Batch progress logging (enabled with debug=True)
        if debug:
            compute_time = time.time() - compute_start
            elapsed = time.time() - epoch_start
            avg_batch_time = elapsed / n_batches
            eta = avg_batch_time * (total_batches - n_batches)
            if n_batches % 10 == 1 or n_batches == total_batches:
                print(f"  Batch {n_batches}/{total_batches} | "
                      f"loss={loss.item():.4f} | "
                      f"data={data_time:.1f}s | "
                      f"compute={compute_time:.1f}s | "
                      f"elapsed={elapsed:.0f}s | "
                      f"ETA={eta:.0f}s", flush=True)
        batch_end_time = time.time()
    
    avg_loss = total_loss / max(1, n_batches)
    avg_grad_norm = grad_norm_sum / max(1, n_batches)
    
    # Average metrics
    import numpy as np
    avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in all_metrics.items()}
    
    # For backward compatibility: if inter_intra model, also expose 'edge_auc' as inter_edge_auc
    if is_inter_intra and 'inter_edge_auc' in avg_metrics:
        avg_metrics['edge_auc'] = avg_metrics['inter_edge_auc']
    
    return avg_loss, avg_grad_norm, avg_metrics


@torch.no_grad()
def validate_ssl(
    model,
    loader,
    device,
    use_amp: bool = True,
):
    """
    Validate SSL pre-training (edge prediction).
    Handles both inter-only and inter+intra models transparently.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_metrics = defaultdict(list)
    is_inter_intra = _is_inter_intra_model(model)
    
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
    
    for batch_data, mask_info, pdb_names in loader:
        batch_data = batch_data.to(device)
        
        with amp_ctx:
            if is_inter_intra:
                outputs = model(batch_data, mask_info)
                loss, loss_dict = model.compute_loss(outputs)
                
                if len(outputs['inter_edge_labels']) > 0:
                    inter_metrics = compute_edge_metrics(
                        outputs['inter_edge_logits'], outputs['inter_edge_labels']
                    )
                    all_metrics['inter_edge_acc'].append(inter_metrics['accuracy'])
                    all_metrics['inter_edge_auc'].append(inter_metrics['auc'])
                if len(outputs['intra_edge_labels']) > 0:
                    intra_metrics = compute_edge_metrics(
                        outputs['intra_edge_logits'], outputs['intra_edge_labels']
                    )
                    all_metrics['intra_edge_acc'].append(intra_metrics['accuracy'])
                    all_metrics['intra_edge_auc'].append(intra_metrics['auc'])
                all_metrics['inter_edge_loss'].append(loss_dict['inter_edge_loss'].item())
                all_metrics['intra_edge_loss'].append(loss_dict['intra_edge_loss'].item())
            else:
                edge_logits, edge_labels = model(batch_data, mask_info)
                loss = model.compute_loss(edge_logits, edge_labels)
                
                if len(edge_labels) > 0:
                    metrics = compute_edge_metrics(edge_logits, edge_labels)
                    all_metrics['edge_acc'].append(metrics['accuracy'])
                    all_metrics['edge_auc'].append(metrics['auc'])
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / max(1, n_batches)
    
    import numpy as np
    avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in all_metrics.items()}
    
    # Backward compatibility
    if is_inter_intra and 'inter_edge_auc' in avg_metrics:
        avg_metrics['edge_auc'] = avg_metrics['inter_edge_auc']
    
    return avg_loss, avg_metrics


class SSLEarlyStopping:
    """Early stopping for SSL pre-training."""
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        save_path: str = None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, model, epoch: int):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.save_path:
                self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def save_checkpoint(self, model):
        """Save encoder weights."""
        torch.save({
            'encoder_state_dict': model.encoder.state_dict(),
            'full_model_state_dict': model.state_dict(),
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
        }, self.save_path)
