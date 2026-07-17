import torch
from contextlib import nullcontext
from .common_utils import compute_classification_metrics, compute_regression_metrics

# -----------------------------
# Classifier Functions
# -----------------------------
def train_epoch_classifier(
    model, loader, optimizer, scaler, criterion, device,
    clip_max_norm: float = 5.0, use_amp: bool = True,
    label_smoothing: float = 0.0,
):
    """Train one epoch for classifier."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    grad_norm_sum = 0.0
    
    all_true = []
    all_pred = []
    
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
    
    for batch_data in loader:
        optimizer.zero_grad(set_to_none=True)
        batch_data = batch_data.to(device)
        
        y_affinity = batch_data.y
        y_cls_hard = (y_affinity > 0).float().unsqueeze(-1)
        y_cls = y_cls_hard
        if label_smoothing > 0:
            y_cls = y_cls_hard * (1.0 - label_smoothing) + 0.5 * label_smoothing
        
        with amp_ctx:
            logits = model(batch_data)
            loss = criterion(logits, y_cls)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
        
        post_sq = sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None)
        grad_norm_sum += post_sq ** 0.5
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        n_batches += 1
        
        all_true.extend(y_cls_hard.squeeze(-1).cpu().tolist())
        all_pred.extend(torch.sigmoid(logits).squeeze(-1).detach().cpu().tolist())
    
    avg_loss = total_loss / max(1, n_batches)
    avg_grad_norm = grad_norm_sum / max(1, n_batches)
    metrics = compute_classification_metrics(all_true, all_pred)
    
    return avg_loss, avg_grad_norm, metrics


@torch.no_grad()
def validate_classifier(model, loader, criterion, device, use_amp: bool = True,
                        label_smoothing: float = 0.0):
    """Validate classifier."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_true = []
    all_pred = []
    
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
    
    for batch_data in loader:
        batch_data = batch_data.to(device)
        
        y_affinity = batch_data.y
        y_cls_hard = (y_affinity > 0).float().unsqueeze(-1)
        y_cls = y_cls_hard
        if label_smoothing > 0:
            y_cls = y_cls_hard * (1.0 - label_smoothing) + 0.5 * label_smoothing
        
        with amp_ctx:
            logits = model(batch_data)
            loss = criterion(logits, y_cls)
        
        total_loss += loss.item()
        n_batches += 1
        
        all_true.extend(y_cls_hard.squeeze(-1).cpu().tolist())
        all_pred.extend(torch.sigmoid(logits).squeeze(-1).cpu().tolist())
    
    avg_loss = total_loss / max(1, n_batches)
    metrics = compute_classification_metrics(all_true, all_pred)
    
    return avg_loss, metrics


# -----------------------------
# Regressor Functions
# -----------------------------
def train_epoch_regressor(
    model, loader, optimizer, scaler, criterion, device,
    clip_max_norm: float = 5.0, use_amp: bool = True,
):
    """Train one epoch for regressor (on premium samples only)."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    grad_norm_sum = 0.0
    
    all_true = []
    all_pred = []
    
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
    
    for batch_data in loader:
        optimizer.zero_grad(set_to_none=True)
        batch_data = batch_data.to(device)
        
        y = batch_data.y.unsqueeze(-1)
        
        with amp_ctx:
            pred = model(batch_data)
            loss = criterion(pred, y)
            
            #y_premium = y_affinity[premium_mask].unsqueeze(-1)
            #output_premium = output[premium_mask]
            #loss = criterion(output_premium, y_premium)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
        
        post_sq = sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None)
        grad_norm_sum += post_sq ** 0.5
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        n_batches += 1

        all_true.extend(y.squeeze(1).cpu().tolist())
        all_pred.extend(pred.squeeze(1).detach().cpu().tolist())
    
    avg_loss = total_loss / max(1, n_batches)
    avg_grad_norm = grad_norm_sum / max(1, n_batches)
    metrics = compute_regression_metrics(all_true, all_pred)
    
    return avg_loss, avg_grad_norm, metrics


@torch.no_grad()
def validate_regressor(
    model, loader, criterion, device, use_amp: bool = True):
    """Validate regressor (on premium samples only)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_true = []
    all_pred = []
    
    amp_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
    
    for batch_data in loader:
        batch_data = batch_data.to(device)
        
        y = batch_data.y.unsqueeze(-1)
        with amp_ctx:
            pred = model(batch_data)
            loss = criterion(pred, y)

        
        total_loss += loss.item()
        n_batches += 1
        
        all_true.extend(y.squeeze(1).cpu().tolist())
        all_pred.extend(pred.squeeze(1).cpu().tolist())
    
    avg_loss = total_loss / max(1, n_batches)
    metrics = compute_regression_metrics(all_true, all_pred)
    
    return avg_loss, metrics


# -----------------------------
# ΔΔG Dual-Graph Regressor Functions
# -----------------------------
def train_epoch_ddg(
    model, mut_loader, wt_loader, optimizer, scaler, criterion, device,
    clip_max_norm: float = 5.0, use_amp: bool = True,
):
    """
    Train one epoch for ΔΔG dual-graph regressor.

    The model takes paired mutant and wild-type batches and predicts
    mutant binding affinity (mut_aff).  mut_batch.y already contains
    the original mut_aff loaded from the graph database.
    Loaders must iterate in the same order and have the same length.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    grad_norm_sum = 0.0

    all_true = []
    all_pred = []

    amp_ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
               if use_amp else nullcontext())

    for mut_batch, wt_batch in zip(mut_loader, wt_loader):
        optimizer.zero_grad(set_to_none=True)
        mut_batch = mut_batch.to(device)
        wt_batch = wt_batch.to(device)

        y = mut_batch.y.unsqueeze(-1)

        with amp_ctx:
            pred = model(mut_batch, wt_batch)
            loss = criterion(pred, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)

        post_sq = sum(
            p.grad.pow(2).sum().item()
            for p in model.parameters() if p.grad is not None
        )
        grad_norm_sum += post_sq ** 0.5

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

        all_true.extend(y.squeeze(1).cpu().tolist())
        all_pred.extend(pred.squeeze(1).detach().cpu().tolist())

    avg_loss = total_loss / max(1, n_batches)
    avg_grad_norm = grad_norm_sum / max(1, n_batches)
    metrics = compute_regression_metrics(all_true, all_pred)

    return avg_loss, avg_grad_norm, metrics


@torch.no_grad()
def validate_ddg(
    model, mut_loader, wt_loader, criterion, device, use_amp: bool = True,
):
    """
    Validate ΔΔG dual-graph regressor.
    Target is mut_batch.y (= mut_aff).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_true = []
    all_pred = []

    amp_ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
               if use_amp else nullcontext())

    for mut_batch, wt_batch in zip(mut_loader, wt_loader):
        mut_batch = mut_batch.to(device)
        wt_batch = wt_batch.to(device)

        y = mut_batch.y.unsqueeze(-1)

        with amp_ctx:
            pred = model(mut_batch, wt_batch)
            loss = criterion(pred, y)

        total_loss += loss.item()
        n_batches += 1

        all_true.extend(y.squeeze(1).cpu().tolist())
        all_pred.extend(pred.squeeze(1).cpu().tolist())

    avg_loss = total_loss / max(1, n_batches)
    metrics = compute_regression_metrics(all_true, all_pred)

    return avg_loss, metrics