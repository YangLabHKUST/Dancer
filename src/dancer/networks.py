"""Label-conditioned encoders used by Dancer."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        *,
        n_hidden: int,
        n_layers: int,
        dropout: float,
        layer_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        current = n_input
        for _ in range(n_layers):
            layers.append(nn.Linear(current, n_hidden))
            if layer_norm:
                layers.append(nn.LayerNorm(n_hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = n_hidden
        layers.append(nn.Linear(current, n_output))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def one_hot(labels: torch.Tensor, n_types: int) -> torch.Tensor:
    return F.one_hot(labels.long(), n_types).to(dtype=torch.float32)

def normalize_encoder_input(counts: torch.Tensor) -> torch.Tensor:
    mean = counts.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.log1p(counts / mean)

class ConditionalGaussianEncoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_types: int,
        n_latent: int,
        *,
        n_hidden: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.n_types = n_types
        self.body = MLP(
            n_genes + n_types,
            2 * n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout=dropout,
        )

    def forward(self, counts: torch.Tensor, labels: torch.Tensor):
        inputs = torch.cat([normalize_encoder_input(counts), one_hot(labels, self.n_types)], dim=-1)
        mean, logvar = self.body(inputs).chunk(2, dim=-1)
        return mean, logvar.clamp(-10.0, 6.0)

def inverse_softplus(value: torch.Tensor | float) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    return torch.log(torch.expm1(tensor))
