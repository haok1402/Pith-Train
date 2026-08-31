"""Expert-parallel dispatch/combine over the DeepEP legacy ``Buffer``.

The four ``*_forward``/``*_backward`` helpers follow the autograd pattern from
DeepEP's ``docs/legacy.md``: dispatch's backward is a combine, and combine's
backward is a dispatch reusing the forward handle.  ``recv_dispatch_layout``
turns the received per-token expert ids into the expand indices consumed by
the grouped-GEMM expert path, and ``weighted_combine_input`` applies the router
weights expert-side so the unweighted combine sums each rank's partial output.
"""

from typing import Optional, Tuple

import deep_ep
import torch
import torch.distributed as dist

from pithtrain.models.interface import RoutingInfo

_buffer: Optional[deep_ep.Buffer] = None

deep_ep.Buffer.set_num_sms(24)


def get_hidden_bytes(x: torch.Tensor) -> int:
    t = x[0] if isinstance(x, tuple) else x
    return t.size(1) * max(t.element_size(), 2)


def get_buffer(group: dist.ProcessGroup, hidden_bytes: int) -> deep_ep.Buffer:
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        deep_ep.Buffer.get_dispatch_config(group.size()),
        deep_ep.Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = deep_ep.Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


class DeepEPWork:
    """Adapts DeepEP's ``EventOverlap`` (or a ``torch.cuda.Event``) to the pipeline's ``Work``-like ``wait()``."""

    def __init__(self, event):
        self.event = event

    def wait(self) -> None:
        if hasattr(self.event, "current_stream_wait"):
            self.event.current_stream_wait()
        else:
            self.event.wait()


def dispatch_forward(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    buffer: deep_ep.Buffer,
    async_finish: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple, Optional[deep_ep.EventOverlap]]:
    """
    Dispatch and return received tokens, expert ids, and weights.

    The kernels run on the buffer's internal comm stream and the CPU only waits
    for the received counts, so the current stream is ordered against the
    finished data movement before returning — the receive-side layout derived
    from ``recv_topk_idx`` is read immediately by the caller.
    """
    comm_stream = buffer.get_comm_stream()
    for t in (x, topk_idx, topk_weights):
        t.record_stream(comm_stream)
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        event,
    ) = buffer.get_dispatch_layout(
        topk_idx,
        num_experts,
        previous_event=buffer.capture(),
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    recv_x, recv_topk_idx, recv_topk_weights, num_recv_per_expert, handle, event = buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=event,
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    if not async_finish:
        event.current_stream_wait()
        return recv_x, recv_topk_idx, recv_topk_weights, num_recv_per_expert, handle, None
    return recv_x, recv_topk_idx, recv_topk_weights, num_recv_per_expert, handle, event


def dispatch_backward(
    grad_combined_x: torch.Tensor, handle: Tuple, buffer: deep_ep.Buffer, async_finish: bool = True
) -> Tuple[torch.Tensor, Optional[deep_ep.EventOverlap]]:
    grad_combined_x.record_stream(buffer.get_comm_stream())
    grad_recv_x, _, _, _, _, event = buffer.dispatch(
        grad_combined_x,
        handle=handle,
        previous_event=buffer.capture(),
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    if not async_finish:
        event.current_stream_wait()
        return grad_recv_x, None
    return grad_recv_x, event


def combine_forward(
    x: torch.Tensor, handle: Tuple, buffer: deep_ep.Buffer, async_finish: bool = True
) -> Tuple[torch.Tensor, Optional[deep_ep.EventOverlap]]:
    x.record_stream(buffer.get_comm_stream())
    combined_x, _, event = buffer.combine(
        x, handle, previous_event=buffer.capture(), async_finish=True, allocate_on_comm_stream=True
    )
    if not async_finish:
        event.current_stream_wait()
        return combined_x, None
    return combined_x, event


def combine_backward(
    grad_recv_x: torch.Tensor,
    grad_recv_topk_weights: torch.Tensor,
    handle: Tuple,
    buffer: deep_ep.Buffer,
    async_finish: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[deep_ep.EventOverlap]]:
    comm_stream = buffer.get_comm_stream()
    for t in (grad_recv_x, grad_recv_topk_weights):
        t.record_stream(comm_stream)
    grad_x, grad_topk_weights, event = buffer.combine(
        grad_recv_x,
        handle,
        topk_weights=grad_recv_topk_weights,
        previous_event=buffer.capture(),
        async_finish=True,
        allocate_on_comm_stream=True,
    )
    if not async_finish:
        event.current_stream_wait()
        return grad_x, grad_topk_weights, None
    return grad_x, grad_topk_weights, event


def recv_dispatch_layout(
    recv_topk_idx: torch.Tensor,
    num_recv_pairs: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expand indices for the tokens received by this rank.

    ``recv_topk_idx`` is ``[num_recv, topk]`` holding local expert ids with -1
    for the experts hosted elsewhere.  Returns the flat expert id, index into
    the deduplicated ``recv_x`` rows, and position in the flattened
    ``[num_recv, topk]`` weight matrix for every received token-expert pair.
    ``num_recv_pairs`` (from DeepEP's ``num_recv_tokens_per_expert_list``, already
    on the host) sizes the compaction so no device-to-host sync is needed.
    """
    valid = (recv_topk_idx >= 0).view(-1).to(torch.int8)
    moe_local_idxs = torch.argsort(valid, stable=True, descending=True)[:num_recv_pairs]
    expert_idxs = recv_topk_idx.view(-1)[moe_local_idxs]
    expand_idx = moe_local_idxs // recv_topk_idx.shape[1]
    return expert_idxs, expand_idx, moe_local_idxs


def weighted_combine_input(
    moe_outs: torch.Tensor,
    moe_local_idxs: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    num_recv_tokens: int,
) -> torch.Tensor:
    """Sum each received token's local expert outputs, weighted by the router weights."""
    permuted_probs = recv_topk_weights.view(-1)[moe_local_idxs]
    token_indices = moe_local_idxs // recv_topk_weights.shape[1]
    weighted = (moe_outs.float() * permuted_probs.unsqueeze(-1)).to(moe_outs.dtype)
    aggregated = moe_outs.new_zeros(num_recv_tokens, moe_outs.shape[-1])
    aggregated.scatter_add_(0, token_indices[:, None].expand_as(weighted), weighted)
    return aggregated


def is_deepep_routing(routing: Optional[RoutingInfo]) -> bool:
    return routing is not None and routing.topk_idx is not None
