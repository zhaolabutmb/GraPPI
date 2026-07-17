import torch
# -----------------------------
# Early Stopping Class
# -----------------------------
class FineTuneEarlyStopping:
    """Early stopping for fine-tuning."""
    def __init__(
        self,
        patience: int = 35,
        min_delta: float = 1e-5,
        save_path: str = None,
        mode: str = 'min',  # 'min' for loss, 'max' for metrics like AUC
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.mode = mode
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.best_state = None
    
    def __call__(self, score: float, model, epoch: int):
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        # Check for improvement
        if improved:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            self.best_epoch = epoch
            if self.save_path is not None:
                torch.save(self.best_state, self.save_path)
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def get_best_state(self):
        """Return the best model state dict."""
        return self.best_state
