from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pithtrain.dualpipe.utils import WeightGradStore


class GroupLinearFunc(torch.autograd.Function):
    """
    Custom autograd Function for BF16 grouped linear layer (MoE experts).

    Forward:  output      = grouped_mm(input, weight.T)   [2D-3D, jagged on M]
    Dgrad:    grad_input  = grouped_mm(grad_output, weight)  [2D-3D, jagged on M]
    Wgrad:    weight_grad = grouped_mm(grad_output.T, input) [2D-2D, jagged on K
              of the shared M dim; output is (G, N, K) matching weight.shape]

    The wgrad is split from dgrad so DualPipeV's zero-bubble W-phase can defer
    it via WeightGradStore, freeing the critical path during stage3_b.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
    ) -> torch.Tensor:
        output = F.grouped_mm(input, weight.transpose(1, 2), offs=grouped_mm_offs)
        ctx.save_for_backward(input, grouped_mm_offs)
        ctx.weight_ref = weight
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grouped_mm_offs = ctx.saved_tensors
        weight = ctx.weight_ref

        grad_input = F.grouped_mm(grad_output, weight, offs=grouped_mm_offs)

        def grad_weight_fn(gy, x, offs):
            dW = F.grouped_mm(gy.transpose(0, 1), x, offs=offs)
            if weight.grad is None:
                weight.grad = dW
            else:
                weight.grad += dW

        if WeightGradStore.enabled:
            WeightGradStore.put(
                partial(
                    grad_weight_fn,
                    grad_output.detach(),
                    input.detach(),
                    grouped_mm_offs.detach(),
                )
            )
        else:
            grad_weight_fn(grad_output, input, grouped_mm_offs)

        return grad_input, None, None


class GroupLinear(nn.Module):
    """
    Grouped linear layer that partitions input data and applies a distinct
    linear transformation per group. This is useful for the MLP layers in
    the mixture-of-experts models.
    """

    def __init__(self, num_groups: int, in_features: int, out_features: int):
        super().__init__()
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((num_groups, out_features, in_features)))

    def forward(
        self,
        input: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: Optional[list] = None,
        ks_tensor: Optional[torch.Tensor] = None,
        group_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input.shape[0] == 0:
            # Use a matmul instead of new_empty to preserve the autograd graph.
            # With 0 tokens the result is (0, out_features) and gradients are zero,
            # but the grad_fn must exist so that run_backward does not crash.
            return input @ self.weight[0].T
        return GroupLinearFunc.apply(input, self.weight, grouped_mm_offs)
