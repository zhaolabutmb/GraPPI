import torch
from torch import nn

EPS = 1e-8
class ExpNegAbsLoss(nn.Module):
    """
    Training-friendly loss: loss = 1 - exp(-gamma * |pred - target|) in [0,1).
    Minimizing this is equivalent to maximizing exp(-|.|).
    Perfect match -> 0. Large error -> approaches 1.
    """
    def __init__(self, gamma: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        # similarity = exp(-gamma * diff); loss = 1 - similarity
        loss = -torch.expm1(-self.gamma * diff)  # = 1 - exp(-gamma*diff) with better numeric stability
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

class HuberLoss(nn.Module):
    """Smooth L1 with tunable delta."""
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta, self.reduction = delta, reduction
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        quad = 0.5 * (diff ** 2) / self.delta
        lin = diff - 0.5 * self.delta
        loss = torch.where(diff <= self.delta, quad, lin)
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss

class LogCoshLoss(nn.Module):
    """log(cosh(error)) ~ L2 near 0, L1 for large errors (smooth)."""
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    def forward(self, pred, target):
        x = pred - target
        # stable logcosh
        loss = x + torch.nn.functional.softplus(-2 * x) - torch.log(torch.tensor(2.0, device=x.device))
        loss = loss.abs()  # symmetric
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss
# consider add some perturbation (a consistent noise, according to training curve) to gradient, avoiding local minima
class CharbonnierLoss(nn.Module):
    """sqrt((pred - target)^2 + eps^2), robust L1."""
    def __init__(self, eps: float = 1e-3, reduction: str = 'mean'):
        super().__init__()
        self.eps, self.reduction = eps, reduction
    def forward(self, pred, target):
        loss = torch.sqrt((pred - target) ** 2 + self.eps ** 2)
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss

class CauchyLoss(nn.Module):
    """Robust to outliers; c controls tail heaviness."""
    def __init__(self, c: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.c, self.reduction = c, reduction
    def forward(self, pred, target):
        r = (pred - target) / self.c
        loss = torch.log1p(r * r)
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss

class TukeyBiweightLoss(nn.Module):
    """Zero influence beyond c; very robust."""
    def __init__(self, c: float = 4.685, reduction: str = 'mean'):
        super().__init__()
        self.c, self.reduction = c, reduction
    def forward(self, pred, target):
        r = (pred - target) / (self.c + EPS)
        abs_r = torch.abs(r)
        inl = (1 - (1 - (abs_r ** 2)) ** 3) * (abs_r <= 1).float()
        loss = (self.c ** 2 / 6.0) * inl  # 0 outside
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss

class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression; tau in (0,1)."""
    def __init__(self, tau: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        assert 0 < tau < 1
        self.tau, self.reduction = tau, reduction
    def forward(self, pred, target):
        e = target - pred
        loss = torch.maximum(self.tau * e, (self.tau - 1) * e)
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss

class RMSLELoss(nn.Module):
    """Root mean squared log error; for positive targets/preds."""
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    def forward(self, pred, target):
        pred = torch.clamp(pred, min=0)
        target = torch.clamp(target, min=0)
        diff = torch.log1p(pred) - torch.log1p(target)
        loss = torch.sqrt(torch.clamp(diff ** 2, min=0))
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss

class MAPELoss(nn.Module):
    """Mean absolute percentage error; sensitive near zero."""
    def __init__(self, eps: float = 1e-3, reduction: str = 'mean'):
        super().__init__()
        self.eps, self.reduction = eps, reduction
    def forward(self, pred, target):
        loss = torch.abs((pred - target) / (torch.abs(target) + self.eps))
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss

class SMAPELoss(nn.Module):
    """Symmetric MAPE: 2|p-y|/(|p|+|y|+eps)."""
    def __init__(self, eps: float = 1e-3, reduction: str = 'mean'):
        super().__init__()
        self.eps, self.reduction = eps, reduction
    def forward(self, pred, target):
        num = 2.0 * torch.abs(pred - target)
        den = torch.abs(pred) + torch.abs(target) + self.eps
        loss = num / den
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss

class PearsonCorrLoss(nn.Module):
    """1 - Pearson r; scale-invariant. Good when correlation matters.
    
    Computes correlation across the entire batch (not per-sample).
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        # Flatten to 1D for batch-level correlation
        x = pred.view(-1)
        y = target.view(-1)
        
        # Need at least 2 samples
        if x.size(0) < 2:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        
        num = (x_centered * y_centered).sum()
        den = torch.sqrt((x_centered.pow(2).sum() + self.eps) * (y_centered.pow(2).sum() + self.eps))
        r = num / den
        return 1 - r

class ConcordanceCorrLoss(nn.Module):
    """1 - CCC; aligns both correlation and bias/scale."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        x = pred.view(pred.size(0), -1)
        y = target.view(target.size(0), -1)
        mean_x, mean_y = x.mean(dim=1, keepdim=True), y.mean(dim=1, keepdim=True)
        vx = ((x - mean_x) ** 2).mean(dim=1)
        vy = ((y - mean_y) ** 2).mean(dim=1)
        cov = ((x - mean_x) * (y - mean_y)).mean(dim=1)
        ccc = (2 * cov) / (vx + vy + (mean_x - mean_y).pow(2).squeeze(1) + self.eps)
        return (1 - ccc).mean()

class GaussianNLLHetero(nn.Module):
    """
    Heteroscedastic Gaussian NLL.
    Model must output (mu, log_var) or pass log_var separately.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    def forward(self, pred_mu, target, pred_log_var=None):
        if pred_log_var is None:
            # pred is a tuple (mu, log_var)
            pred_mu, pred_log_var = pred_mu
        inv_var = torch.exp(-pred_log_var)
        loss = 0.5 * (pred_log_var + (pred_mu - target) ** 2 * inv_var)
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss

class L1CorrCombinedLoss(nn.Module):
    """alpha * L1 + (1 - alpha) * (1 - Pearson r)."""
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mae = nn.L1Loss(reduction='mean')
        self.corr = PearsonCorrLoss()
    def forward(self, pred, target):
        return self.alpha * self.mae(pred, target) + (1 - self.alpha) * self.corr(pred, target)

def get_loss(name: str, *, gamma: float = 1.0, delta: float = 1.0, alpha: float = 0.5, reduction: str = 'mean'):
    """
    Factory returning a loss by string.
    Supported: 'l1','l2','mse','smoothl1','huber','logcosh','expnegabs','smape','pearson','l1corr'
    """
    key = (name or '').lower()
    if key in ('l1','mae'):
        return nn.L1Loss(reduction=reduction)
    if key in ('l2','mse'):
        return nn.MSELoss(reduction=reduction)
    if key in ('smoothl1',):
        return nn.SmoothL1Loss(reduction=reduction, beta=delta)  # delta ~ beta
    if key in ('huber',):
        return HuberLoss(delta=delta, reduction=reduction)
    if key in ('logcosh','log_cosh'):
        return LogCoshLoss(reduction=reduction)
    if key in ('expnegabs','exp_neg_abs'):
        return ExpNegAbsLoss(gamma=gamma, reduction=reduction)
    if key in ('smape',):
        return SMAPELoss(reduction=reduction)
    if key in ('pearson','pearson_corr'):
        return PearsonCorrLoss()
    if key in ('l1corr','mae_corr'):
        return L1CorrCombinedLoss(alpha=alpha)
    raise ValueError(f"Unknown loss_fun: {name}")