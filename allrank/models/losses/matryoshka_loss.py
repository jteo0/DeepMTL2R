"""
Matryoshka Ranking Loss for Matryoshka Feature Projection.

Wraps any standard LTR loss function (e.g. lambdaLoss) to support nested
supervision across multiple embedding granularities. For each nesting dimension,
the same loss function is applied independently, then summed with uniform weights.

Adapted from Matryoshka_CE_Loss in:
  Kusupati et al. (2022). Matryoshka Representation Learning. NeurIPS.
  Reference implementation: ../MRL/MRL.py
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple


class MatryoshkaRankingLoss(nn.Module):
    """
    Matryoshka Ranking Loss: applies a base LTR loss at each nesting granularity
    and returns their weighted sum (uniform weights = 1.0 by default).

    Usage in training loop:
        outputs = model(x, mask, indices)   # tuple of tensors, one per nesting dim
        loss = MatryoshkaRankingLoss(base_loss_func)(outputs, y_true)
    """

    def __init__(self, base_loss_func: Callable, relative_importance: Optional[list] = None):
        """
        :param base_loss_func: a callable LTR loss (e.g. partial(lambdaLoss, ...))
                               with signature f(y_pred, y_true) -> scalar tensor.
        :param relative_importance: list of floats, one weight per nesting dimension.
                                    If None, all weights default to 1.0 (uniform).
        """
        super(MatryoshkaRankingLoss, self).__init__()
        self.base_loss_func = base_loss_func
        self.relative_importance = relative_importance

    def forward(self, outputs: Tuple[torch.Tensor, ...], y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted sum of losses across all nesting dimensions.

        :param outputs: tuple of score tensors from MatryoshkaOutputLayer,
                        each of shape [batch_size, slate_length].
                        Length = number of nesting dimensions.
        :param y_true: ground truth relevance labels, shape [batch_size, slate_length].
        :return: scalar loss tensor (sum of per-granularity losses).
        """
        num_granularities = len(outputs)

        # Compute loss for each nesting granularity independently
        losses = torch.stack([
            self.base_loss_func(output_i, y_true)
            for output_i in outputs
        ])  # shape: [num_granularities]

        # Apply relative importance weights (uniform 1.0 by default)
        if self.relative_importance is None:
            weights = torch.ones(num_granularities, device=losses.device)
        else:
            weights = torch.tensor(
                self.relative_importance,
                dtype=losses.dtype,
                device=losses.device
            )

        # Weighted sum across granularities
        return (weights * losses).sum()
