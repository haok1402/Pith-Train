"""
Expert-parallel dispatch preparation.

For ``ep_size > 1`` the routing is handed to DeepEP's dispatch kernel
(``pithtrain.operators.deepep``), which deduplicates tokens per destination
rank and computes the receiver-side layout itself inside stage 2 of the
DualPipeV pipeline. For ``ep_size == 1`` each token is replicated once per
selected expert for the local grouped GEMM.
"""

import torch

from pithtrain.models.interface import RoutingInfo


def prepare_dispatch(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    num_experts: int,
    ep_size: int,
) -> tuple[torch.Tensor, RoutingInfo]:
    """Prepare the expert-parallel dispatch for one MoE layer."""
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    if ep_size > 1:
        # DeepEP owns deduplication and the all-to-all inside its dispatch kernel;
        # stage 2 consumes ``topk_idx``/``num_experts`` from the routing info.
        return hidden_states, RoutingInfo(
            topk_weight, topk_idx=topk_ids.to(torch.int64), num_experts=num_experts
        )

    k = topk_ids.shape[1]
    dispatch_tokens = (
        hidden_states.unsqueeze(1).expand(-1, k, -1).reshape(-1, hidden_states.shape[-1])
    )
    return dispatch_tokens, RoutingInfo(topk_weight, topk_ids.view(-1))
