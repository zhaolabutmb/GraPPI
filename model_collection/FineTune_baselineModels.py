import torch
import torch.nn as nn

class BaselineClassifier(nn.Module):
    """
    Baseline MLP for binary classification without pretrained encoder.

    Accepts pre-pooled graph features [batch_size, node_in_dim].

    Args:
        node_in_dim: Pooled feature dimension
        hidden_dim: Hidden dimension for MLP layers
        classifier_layers: Number of MLP layers
        dropout: Dropout rate
    """
    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 512,
        classifier_layers: int = 3,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()

        input_dim = node_in_dim

        classifier_modules = []
        for i in range(classifier_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < classifier_layers - 1 else 1
            classifier_modules.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < classifier_layers - 1 else nn.Identity(),
                nn.ReLU() if i < classifier_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < classifier_layers - 1 else nn.Identity(),
            ])
        self.classifier = nn.Sequential(*classifier_modules)
        self.current_epoch = 0

    def update_epoch(self, epoch: int):
        self.current_epoch = epoch

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: Pre-pooled features [batch_size, node_in_dim]
        Returns:
            Logits [batch_size, 1]
        """
        return self.classifier(pooled)


class BaselineRegressor(nn.Module):
    """
    Baseline MLP for binding affinity regression without pretrained encoder.

    Accepts pre-pooled graph features [batch_size, node_in_dim].

    Args:
        node_in_dim: Pooled feature dimension
        hidden_dim: Hidden dimension for MLP layers
        regressor_layers: Number of MLP layers
        dropout: Dropout rate
    """
    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 512,
        regressor_layers: int = 4,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()

        input_dim = node_in_dim

        regressor_modules = []
        for i in range(regressor_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < regressor_layers - 1 else 1
            regressor_modules.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < regressor_layers - 1 else nn.Identity(),
                nn.GELU() if i < regressor_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < regressor_layers - 1 else nn.Identity(),
            ])
        self.regressor = nn.Sequential(*regressor_modules)
        self.current_epoch = 0

    def update_epoch(self, epoch: int):
        self.current_epoch = epoch

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: Pre-pooled features [batch_size, node_in_dim]
        Returns:
            Predicted binding affinity [batch_size, 1]
        """
        return self.regressor(pooled)


class BaselineDDGRegressor(nn.Module):
    """
    Baseline MLP for ΔΔG / mutant-ΔG prediction without pretrained encoder.

    Accepts two pre-pooled vectors (mutant, wild-type) each [B, node_in_dim],
    concatenates them → [B, 2*node_in_dim], then feeds through an MLP.

    Args:
        node_in_dim: Pooled feature dimension per graph
        hidden_dim: Hidden dimension for MLP layers
        regressor_layers: Number of MLP layers
        dropout: Dropout rate
    """
    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 512,
        regressor_layers: int = 4,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__()

        input_dim = node_in_dim * 2  # [mut_pooled ∥ wt_pooled]

        regressor_modules = []
        for i in range(regressor_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < regressor_layers - 1 else 1
            regressor_modules.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < regressor_layers - 1 else nn.Identity(),
                nn.GELU() if i < regressor_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < regressor_layers - 1 else nn.Identity(),
            ])
        self.regressor = nn.Sequential(*regressor_modules)
        self.current_epoch = 0

    def update_epoch(self, epoch: int):
        self.current_epoch = epoch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Pre-concatenated [mut_pooled ∥ wt_pooled] features [B, 2*node_in_dim]
        Returns:
            Predicted scalar per sample [B, 1]
        """
        return self.regressor(x)


def create_baseline_model(
    model_type: str,
    node_in_dim: int,
    hidden_dim: int = 512,
    dropout: float = 0.2,
    **kwargs
) -> nn.Module:
    """
    Factory function to create baseline models.

    Args:
        model_type: One of 'classifier', 'regressor', 'gated', 'ddg'
        node_in_dim: Input node feature dimension
        hidden_dim: Hidden dimension for MLP layers
        dropout: Dropout rate
        **kwargs: Additional arguments passed to model constructors

    Returns:
        Initialized baseline model
    """
    if model_type == 'classifier':
        model = BaselineClassifier(
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            **kwargs
        )
    elif model_type == 'regressor':
        model = BaselineRegressor(
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            **kwargs
        )
    elif model_type == 'ddg':
        model = BaselineDDGRegressor(
            node_in_dim=node_in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of: classifier, regressor, gated, ddg")

    return model
