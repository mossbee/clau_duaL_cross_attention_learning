"""
Stochastic Depth (DropPath) utility.

Implements per-sample DropPath as used in Vision Transformers and ConvNets.
Apply to residual branches: output = input + drop_path(residual).
"""

import torch
import torch.nn as nn


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample.

    Args:
        x: Input tensor of shape [batch, ...]
        drop_prob: Probability of dropping the path
        training: Whether in training mode
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # Work with shape [batch_size, 1, 1, ...] to broadcast across remaining dims
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor = random_tensor.div(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """DropPath module wrapping drop_path function."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


