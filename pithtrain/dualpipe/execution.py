"""
Execution for each stage in the schedule.

Each decoder layer is split into five stages so the pipeline scheduler can interleave different
micro-batches and overlap the compute of one with the communication of another.

- Stage 1: pre-dispatch compute.
- Stage 2: dispatch all-to-all.
- Stage 3: expert compute.
- Stage 4: combine all-to-all.
- Stage 5: post-combine compute.
"""

from dataclasses import dataclass, fields
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed

from pithtrain.contexts import distributed
from pithtrain.dualpipe.utils import WeightGradStore, run_backward
from pithtrain.models.interface import LayerProtocol, ModelProtocol, RoutingInfo
from pithtrain.operators.deepep import (
    DeepEPWork,
    combine_backward,
    combine_forward,
    dispatch_backward,
    dispatch_forward,
    get_buffer,
    get_hidden_bytes,
    is_deepep_routing,
    recv_dispatch_layout,
)

# fmt: off

@dataclass(init=False, slots=True)
class ExecutionCtx:
    """Shared context for the overlapped forward-backward execution loop."""

    comp_stream: torch.cuda.Stream
    """Main compute stream for forward/backward kernels."""
    comm_stream: torch.cuda.Stream
    """Separate stream for asynchronous all-to-all communication."""
    fwd_event: torch.cuda.Event
    """Event recorded after forward compute; comm_stream waits on it before dispatch."""
    bwd_event: torch.cuda.Event
    """Event recorded after backward compute; comm_stream waits on it before combine."""
    fwd_comm_work: Optional[DeepEPWork]
    """Wait handle for the in-flight forward DeepEP kernel (dispatch or combine)."""
    bwd_comm_work: Optional[DeepEPWork]
    """Wait handle for the in-flight backward DeepEP kernel."""
    fwd_deepep_handle: Optional[Tuple]
    """DeepEP handle from the most recent stage-2 dispatch, consumed by stage-4 combine."""


# ------------------------------------------------------------
# STAGE1(F/B)
# ------------------------------------------------------------


class Stage1Args(NamedTuple):
    prev_hidden_states: torch.Tensor
    next_hidden_states: torch.Tensor


class Stage1Outs(NamedTuple):
    dispatch_tokens: torch.Tensor
    residual: torch.Tensor
    topk_weight: Optional[torch.Tensor] = None


@dataclass(init=False, slots=True)
class Stage1Record:
    args: Stage1Args
    outs: Stage1Outs


def stage1_f(ctx: ExecutionCtx, layer: LayerProtocol, hidden_states: torch.Tensor, rotary_posemb: Tuple[torch.Tensor, torch.Tensor], cu_seqlens: Optional[torch.Tensor] = None):
    """Stage1 forward."""
    nvtx.range_push("layer%02d.stage1_f" % layer.idx)
    record = Stage1Record()

    prev_hidden_states = hidden_states
    next_hidden_states = hidden_states.detach().requires_grad_()
    record.args = Stage1Args(prev_hidden_states, next_hidden_states)

    dispatch_tokens, residual, routing = layer.forward_stage1(next_hidden_states, rotary_posemb, cu_seqlens)
    ctx.comp_stream.record_event(ctx.fwd_event)

    topk_weight = routing.topk_weight if routing is not None else None
    record.outs = Stage1Outs(dispatch_tokens, residual, topk_weight)

    nvtx.range_pop()
    return record, dispatch_tokens, residual, routing


def stage1_b(ctx: ExecutionCtx, layer: LayerProtocol, record: Stage1Record, grad_tensors: tuple):
    """Stage1 backward."""
    nvtx.range_push("layer%02d.stage1_b" % layer.idx)

    if ctx.bwd_comm_work is not None:
        ctx.bwd_comm_work.wait()

    run_backward(record.outs, grad_tensors)

    hidden_states_grad = record.args.next_hidden_states.grad
    record.args.prev_hidden_states.grad = hidden_states_grad

    nvtx.range_pop()
    return hidden_states_grad


# ------------------------------------------------------------
# STAGE2(F/B)
# ------------------------------------------------------------


@dataclass(init=False, slots=True)
class Stage2Record:
    ctx: Optional[tuple]


def stage2_f(ctx: ExecutionCtx, layer: LayerProtocol, dispatch_tokens: torch.Tensor, routing: Optional[RoutingInfo], ep_group: Optional[torch.distributed.ProcessGroup] = None):
    """Stage2 forward: DeepEP expert dispatch (passthrough at ep_size == 1 or for dense layers)."""
    nvtx.range_push("layer%02d.stage2_f" % layer.idx)
    record = Stage2Record()

    ctx.comm_stream.wait_event(ctx.fwd_event)

    dispatch_tokens = dispatch_tokens.detach()
    if is_deepep_routing(routing):
        buffer = get_buffer(ep_group, get_hidden_bytes(dispatch_tokens))
        with torch.cuda.stream(ctx.comm_stream):
            recv_x, recv_topk_idx, recv_topk_weights, num_recv_per_expert, handle, event = dispatch_forward(
                dispatch_tokens, routing.topk_idx, routing.topk_weight, routing.num_experts, buffer)
        for t in (recv_x, recv_topk_idx, recv_topk_weights):
            t.record_stream(ctx.comp_stream)
        ctx.fwd_comm_work = DeepEPWork(event)
        ctx.fwd_deepep_handle = handle
        record.ctx = ("deepep", handle)
        # The receive-side layout is derived in stage 3, after the compute stream has
        # waited on the dispatch event, so stage 2 stays fully asynchronous.
        recv_routing = RoutingInfo(topk_weight=recv_topk_weights, topk_idx=recv_topk_idx,
                                   num_recv_pairs=sum(num_recv_per_expert))
        nvtx.range_pop()
        return record, recv_x, recv_routing
    record.ctx = None
    ctx.fwd_comm_work = None
    nvtx.range_pop()
    return record, dispatch_tokens, routing


def stage2_b(ctx: ExecutionCtx, layer: LayerProtocol, record: Stage2Record, grad_tensors: tuple):
    """Stage2 backward: DeepEP combine carrying the token and router-weight grads back."""
    nvtx.range_push("layer%02d.stage2_b" % layer.idx)

    ctx.comm_stream.wait_event(ctx.bwd_event)

    if record.ctx is not None:
        with torch.cuda.stream(ctx.comm_stream):
            dispatch_tokens_grad, topk_weight_grad, event = combine_backward(
                grad_tensors[0], grad_tensors[1], record.ctx[1],
                get_buffer(distributed.ep_group, get_hidden_bytes(grad_tensors[0])))
        ctx.bwd_comm_work = DeepEPWork(event)
    else:
        dispatch_tokens_grad = grad_tensors[0]
        ctx.bwd_comm_work = None
        topk_weight_grad = None

    nvtx.range_pop()
    return dispatch_tokens_grad, topk_weight_grad


# ------------------------------------------------------------
# STAGE3(F/B/W)
# ------------------------------------------------------------


class Stage3Args(NamedTuple):
    gathered_tokens: torch.Tensor
    recv_topk_weights: Optional[torch.Tensor] = None


class Stage3Outs(NamedTuple):
    moe_outs: torch.Tensor


@dataclass(init=False, slots=True)
class Stage3Record:
    args: Stage3Args
    outs: Stage3Outs


def stage3_f(ctx: ExecutionCtx, layer: LayerProtocol, gathered_tokens: torch.Tensor, routing: Optional[RoutingInfo]):
    """Stage3 forward."""
    nvtx.range_push("layer%02d.stage3_f" % layer.idx)
    record = Stage3Record()

    gathered_tokens = gathered_tokens.detach().requires_grad_()
    recv_topk_weights = routing.topk_weight if is_deepep_routing(routing) else None
    if recv_topk_weights is not None:
        recv_topk_weights = recv_topk_weights.detach().requires_grad_()
    record.args = Stage3Args(gathered_tokens, recv_topk_weights)

    if ctx.fwd_comm_work is not None:
        ctx.fwd_comm_work.wait()
        gathered_tokens.record_stream(ctx.comp_stream)

    if recv_topk_weights is not None:
        expert_idxs, expand_idx, moe_local_idxs = recv_dispatch_layout(
            routing.topk_idx, routing.num_recv_pairs)
    else:
        expert_idxs = routing.expert_idxs if routing is not None else None
        expand_idx = routing.expand_idx if routing is not None else None
        moe_local_idxs = routing.moe_local_idxs if routing is not None else None
    moe_outs = layer.forward_stage3(gathered_tokens, expert_idxs, expand_idx, moe_local_idxs, recv_topk_weights)
    record.outs = Stage3Outs(moe_outs)

    ctx.comp_stream.record_event(ctx.fwd_event)

    nvtx.range_pop()
    return record, moe_outs


def stage3_b(ctx: ExecutionCtx, layer: LayerProtocol, record: Stage3Record, grad_tensors: Stage3Outs):
    """Stage3 backward for input."""
    nvtx.range_push("layer%02d.stage3_b" % layer.idx)

    if ctx.bwd_comm_work is not None:
        ctx.bwd_comm_work.wait()

    WeightGradStore.enabled = True
    run_backward(record.outs, grad_tensors)
    WeightGradStore.enabled = False

    ctx.comp_stream.record_event(ctx.bwd_event)

    gathered_tokens_grad = record.args.gathered_tokens.grad
    recv_topk_weights_grad = record.args.recv_topk_weights.grad if record.args.recv_topk_weights is not None else None

    nvtx.range_pop()
    return gathered_tokens_grad, recv_topk_weights_grad


def stage3_w(ctx: ExecutionCtx, layer: LayerProtocol):
    """Stage3 backward for weight."""
    nvtx.range_push("layer%02d.stage3_w" % layer.idx)

    WeightGradStore.flush()
    WeightGradStore.pop()

    nvtx.range_pop()


# ------------------------------------------------------------
# STAGE4(F/B)
# ------------------------------------------------------------


@dataclass(init=False, slots=True)
class Stage4Record:
    ctx: Optional[tuple]


def stage4_f(ctx: ExecutionCtx, layer: LayerProtocol, moe_outs: torch.Tensor, ep_group: Optional[torch.distributed.ProcessGroup] = None):
    """Stage4 forward: DeepEP expert combine (passthrough at ep_size == 1 or for dense layers)."""
    nvtx.range_push("layer%02d.stage4_f" % layer.idx)
    record = Stage4Record()

    moe_outs = moe_outs.detach()
    ctx.comm_stream.wait_event(ctx.fwd_event)

    if ctx.fwd_deepep_handle is not None:
        with torch.cuda.stream(ctx.comm_stream):
            combined, event = combine_forward(
                moe_outs, ctx.fwd_deepep_handle,
                get_buffer(ep_group, get_hidden_bytes(moe_outs)))
        record.ctx = ("deepep", ctx.fwd_deepep_handle)
        ctx.fwd_deepep_handle = None
        ctx.fwd_comm_work = DeepEPWork(event)
        nvtx.range_pop()
        return record, combined
    record.ctx = None
    ctx.fwd_comm_work = None
    nvtx.range_pop()
    return record, moe_outs


def stage4_b(ctx: ExecutionCtx, layer: LayerProtocol, record: Stage4Record, grad_tensors: tuple):
    """Stage4 backward: DeepEP dispatch reusing the forward handle."""
    nvtx.range_push("layer%02d.stage4_b" % layer.idx)

    ctx.comm_stream.wait_event(ctx.bwd_event)

    if record.ctx is not None:
        with torch.cuda.stream(ctx.comm_stream):
            moe_outs_grad, event = dispatch_backward(
                grad_tensors[0], record.ctx[1],
                get_buffer(distributed.ep_group, get_hidden_bytes(grad_tensors[0])))
        ctx.bwd_comm_work = DeepEPWork(event)
    else:
        moe_outs_grad = grad_tensors[0]
        ctx.bwd_comm_work = None

    nvtx.range_pop()
    return moe_outs_grad


# ------------------------------------------------------------
# STAGE5(F/B)
# ------------------------------------------------------------


class Stage5Args(NamedTuple):
    moe_outs: torch.Tensor
    topk_weight: torch.Tensor
    residual: torch.Tensor


class Stage5Outs(NamedTuple):
    hidden_states: torch.Tensor


@dataclass(init=False, slots=True)
class Stage5Record:
    args: Stage5Args
    outs: Stage5Outs


def stage5_f(ctx: ExecutionCtx, layer: LayerProtocol, moe_outs: torch.Tensor, routing: Optional[RoutingInfo], residual: torch.Tensor):
    """Stage5 forward."""
    nvtx.range_push("layer%02d.stage5_f" % layer.idx)
    record = Stage5Record()

    moe_outs = moe_outs.detach().requires_grad_()
    deepep = is_deepep_routing(routing)
    topk_weight = routing.topk_weight if routing is not None and not deepep else None
    topk_weight = topk_weight.detach().requires_grad_() if topk_weight is not None else None
    residual = residual.detach().requires_grad_()
    record.args = Stage5Args(moe_outs, topk_weight, residual)

    if ctx.fwd_comm_work is not None:
        ctx.fwd_comm_work.wait()
        moe_outs.record_stream(ctx.comp_stream)

    moe_local_idxs = routing.moe_local_idxs if routing is not None and not deepep else None
    hidden_states = layer.forward_stage5(moe_outs, moe_local_idxs, topk_weight, residual)
    record.outs = Stage5Outs(hidden_states)

    nvtx.range_pop()
    return record, hidden_states


def stage5_b(ctx: ExecutionCtx, layer: LayerProtocol, record: Stage5Record, grad_tensors: Stage5Outs):
    """Stage5 backward."""
    nvtx.range_push("layer%02d.stage5_b" % layer.idx)

    run_backward(record.outs, grad_tensors)

    ctx.comp_stream.record_event(ctx.bwd_event)

    moe_outs_grad, topk_weight_grad, residual_grad = [t.grad if t is not None else None for t in record.args]

    nvtx.range_pop()
    return moe_outs_grad, topk_weight_grad, residual_grad


# ------------------------------------------------------------
# STAGE5_AND_STAGE1(F/B) - Merged stage 5 + stage 1
# ------------------------------------------------------------


def stage5_and_stage1_f(ctx: ExecutionCtx, prev_layer: LayerProtocol, next_layer: LayerProtocol, moe_outs: torch.Tensor, routing: Optional[RoutingInfo], residual: torch.Tensor, rotary_posemb: Tuple[torch.Tensor, torch.Tensor], cu_seqlens: Optional[torch.Tensor] = None):
    """
    Merged Stage5 and Stage1 forward.
    Returns (stage5_args, stage1_outs, dispatch_tokens, residual, routing) for the next layer.
    """
    nvtx.range_push("layer%02d_stage5_f_layer%02d_stage1_f" % (prev_layer.idx, next_layer.idx))

    moe_outs = moe_outs.detach().requires_grad_()
    deepep = is_deepep_routing(routing)
    topk_weight = routing.topk_weight if routing is not None and not deepep else None
    topk_weight = topk_weight.detach().requires_grad_() if topk_weight is not None else None
    residual = residual.detach().requires_grad_()
    stage5_args = Stage5Args(moe_outs, topk_weight, residual)

    if ctx.fwd_comm_work is not None:
        ctx.fwd_comm_work.wait()
        moe_outs.record_stream(ctx.comp_stream)

    moe_local_idxs = routing.moe_local_idxs if routing is not None and not deepep else None
    hidden_states = prev_layer.forward_stage5(moe_outs, moe_local_idxs, topk_weight, residual)

    dispatch_tokens, next_residual, next_routing = next_layer.forward_stage1(hidden_states, rotary_posemb, cu_seqlens)
    ctx.comp_stream.record_event(ctx.fwd_event)

    next_topk_weight = next_routing.topk_weight if next_routing is not None else None
    stage1_outs = Stage1Outs(dispatch_tokens, next_residual, next_topk_weight)

    nvtx.range_pop()
    return stage5_args, stage1_outs, dispatch_tokens, next_residual, next_routing


def stage5_and_stage1_b(ctx: ExecutionCtx, next_layer: LayerProtocol, prev_layer: LayerProtocol, stage1_outs: Stage1Outs, stage5_args: Stage5Args, grad_tensors: tuple):
    """
    Merged Stage5 and Stage1 backward.
    Takes stage1_outs (from next layer) and stage5_args (from prev layer) separately.
    """
    nvtx.range_push("layer%02d_stage5_b_layer%02d_stage1_b" % (prev_layer.idx, next_layer.idx))

    if ctx.bwd_comm_work is not None:
        ctx.bwd_comm_work.wait()

    run_backward(stage1_outs, grad_tensors)

    ctx.comp_stream.record_event(ctx.bwd_event)

    moe_outs_grad, topk_weight_grad, residual_grad = [t.grad if t is not None else None for t in stage5_args]

    nvtx.range_pop()
    return moe_outs_grad, topk_weight_grad, residual_grad


# ------------------------------------------------------------
# PROLOG(F/B)
# ------------------------------------------------------------


class PrologArgs(NamedTuple):
    pass


class PrologOuts(NamedTuple):
    hidden_states: torch.Tensor


@dataclass(init=False, slots=True)
class PrologRecord:
    args: PrologArgs
    outs: PrologOuts


def prolog_f(module: ModelProtocol, hidden_states: torch.Tensor, record: PrologRecord) -> torch.Tensor:
    """Prolog forward: embed the input tokens, recording into ``record`` for the backward."""
    nvtx.range_push("prolog_f")
    record.args = PrologArgs()
    hidden_states = module.forward_prolog(hidden_states)
    record.outs = PrologOuts(hidden_states)
    nvtx.range_pop()
    return hidden_states


def prolog_b(module: ModelProtocol, record: PrologRecord, grad_tensors: PrologOuts):
    """Prolog backward."""
    nvtx.range_push("prolog_b")

    run_backward(record.outs, grad_tensors)

    nvtx.range_pop()
    return


# ------------------------------------------------------------
# EPILOG(F/B)
# ------------------------------------------------------------


class EpilogArgs(NamedTuple):
    hidden_states: torch.Tensor


@dataclass(init=False, slots=True)
class EpilogRecord:
    args: EpilogArgs


def epilog_f(module: ModelProtocol, hidden_states: torch.Tensor, record: EpilogRecord) -> torch.Tensor:
    """
    Epilog forward: norm + lm_head, recording its input activation into ``record``.

    The backward is handled by ``loss.backward()`` which traverses the autograd
    graph through norm -> lm_head -> criterion.  The only thing the caller needs
    from the record is ``args.hidden_states.grad`` (populated by autograd).
    """
    nvtx.range_push("epilog_f")
    hidden_states = hidden_states.detach().requires_grad_()
    record.args = EpilogArgs(hidden_states)
    logits = module.forward_epilog(hidden_states)
    nvtx.range_pop()
    return logits


# ------------------------------------------------------------
# INTERMEDIATE TENSORS
# ------------------------------------------------------------


@dataclass(init=False, slots=True)
class LayerRecord:
    stage1: Stage1Record
    stage2: Stage2Record
    stage3: Stage3Record
    stage4: Stage4Record
    stage5: Stage5Record


@dataclass(init=False, slots=True)
class ChunkRecord:
    prolog: Optional[PrologRecord]
    epilog: Optional[EpilogRecord]
    layers: List[LayerRecord]


def create_layer_record() -> LayerRecord:
    """Create a pre-allocated LayerRecord with all records."""
    layer = LayerRecord()
    layer.stage1 = Stage1Record()
    layer.stage2 = Stage2Record()
    layer.stage2.ctx = None
    layer.stage3 = Stage3Record()
    layer.stage4 = Stage4Record()
    layer.stage4.ctx = None
    layer.stage5 = Stage5Record()
    return layer


def create_chunk_record(num_layers: int, has_prolog: bool, has_epilog: bool) -> ChunkRecord:
    """Create a pre-allocated ChunkRecord structure for reuse across iterations."""
    tensors = ChunkRecord()
    tensors.prolog = PrologRecord() if has_prolog else None
    tensors.epilog = EpilogRecord() if has_epilog else None
    tensors.layers = [create_layer_record() for _ in range(num_layers)]
    return tensors


# ------------------------------------------------------------
# SEQUENTIAL (NON-OVERLAPPED) LAYER + CHUNK PASSES
# ------------------------------------------------------------


def layer_forward(
    layer: LayerProtocol,
    hidden_states: torch.Tensor,
    rotary_posemb: Tuple[torch.Tensor, torch.Tensor],
    layer_record: LayerRecord,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """Forward pass for a DualPipeV decoder layer, recording each stage's tensors into ``layer_record`` for the pipeline backward."""

    # Stage 1.
    nvtx.range_push("layer%02d.stage1_f" % layer.idx)
    record = Stage1Record()
    prev_hidden_states = hidden_states
    next_hidden_states = hidden_states.detach().requires_grad_()
    record.args = Stage1Args(prev_hidden_states, next_hidden_states)

    dispatch_tokens, residual, routing = layer.forward_stage1(next_hidden_states, rotary_posemb, cu_seqlens)

    has_experts = routing is not None
    ep_group = distributed.ep_group if has_experts else None

    record.outs = Stage1Outs(dispatch_tokens, residual, routing.topk_weight if has_experts else None)
    layer_record.stage1 = record
    nvtx.range_pop()

    # Stage 2.
    nvtx.range_push("layer%02d.stage2_f" % layer.idx)
    record = Stage2Record()
    deepep = is_deepep_routing(routing if has_experts else None)
    dispatch_tokens = dispatch_tokens.detach()
    if deepep:
        buffer = get_buffer(ep_group, get_hidden_bytes(dispatch_tokens))
        recv_x, recv_topk_idx, recv_topk_weights, num_recv_per_expert, handle, _ = dispatch_forward(
            dispatch_tokens, routing.topk_idx, routing.topk_weight, routing.num_experts, buffer,
            async_finish=False)
        expert_idxs, expand_idx, moe_local_idxs = recv_dispatch_layout(
            recv_topk_idx, sum(num_recv_per_expert))
        gathered_tokens, stage3_routing = recv_x, RoutingInfo(
            topk_weight=recv_topk_weights, expert_idxs=expert_idxs,
            moe_local_idxs=moe_local_idxs, expand_idx=expand_idx, topk_idx=recv_topk_idx)
        record.ctx = ("deepep", handle)
    else:
        gathered_tokens, stage3_routing = dispatch_tokens, (routing if has_experts else None)
        record.ctx = None
    layer_record.stage2 = record
    nvtx.range_pop()

    # Stage 3.
    nvtx.range_push("layer%02d.stage3_f" % layer.idx)
    record = Stage3Record()
    gathered_tokens = gathered_tokens.detach().requires_grad_()
    recv_topk_weights = stage3_routing.topk_weight if deepep else None
    if recv_topk_weights is not None:
        recv_topk_weights = recv_topk_weights.detach().requires_grad_()
    record.args = Stage3Args(gathered_tokens, recv_topk_weights)

    moe_outs = layer.forward_stage3(gathered_tokens, stage3_routing.expert_idxs if has_experts else None,
                                    stage3_routing.expand_idx if has_experts else None,
                                    stage3_routing.moe_local_idxs if has_experts else None, recv_topk_weights)
    record.outs = Stage3Outs(moe_outs)
    layer_record.stage3 = record
    nvtx.range_pop()

    # Stage 4.
    nvtx.range_push("layer%02d.stage4_f" % layer.idx)
    record = Stage4Record()
    if deepep:
        moe_outs, _ = combine_forward(moe_outs.detach(), handle, buffer, async_finish=False)
        record.ctx = ("deepep", handle)
    else:
        moe_outs = moe_outs.detach()
        record.ctx = None
    layer_record.stage4 = record
    nvtx.range_pop()

    # Stage 5.
    nvtx.range_push("layer%02d.stage5_f" % layer.idx)
    record = Stage5Record()
    moe_outs = moe_outs.detach().requires_grad_()
    topk_weight = routing.topk_weight if has_experts and not deepep else None
    topk_weight = topk_weight.detach().requires_grad_() if topk_weight is not None else None
    residual = residual.detach().requires_grad_()
    record.args = Stage5Args(moe_outs, topk_weight, residual)

    moe_local_idxs = routing.moe_local_idxs if has_experts and not deepep else None
    hidden_states = layer.forward_stage5(moe_outs, moe_local_idxs, topk_weight, residual)

    record.outs = Stage5Outs(hidden_states)
    layer_record.stage5 = record
    nvtx.range_pop()

    return hidden_states


def layer_backward(
    layer: LayerProtocol,
    dy: Optional[List[torch.Tensor]],
    loss: Optional[torch.Tensor],
    layer_record: LayerRecord,
):
    """
    Backward pass for a DualPipeV decoder layer.

    Handles both normal and merged cases using asymmetric None pattern:
    - Merged stage1: stage1.outs is set, stage1.args is None
      -> Run backward on stage1.outs, grads flow to prev layer's stage5.args
      -> Return None to signal prev layer to get grads from stage5.args
    - Merged stage5: stage5.args is set, stage5.outs is None
      -> Get grads from stage5.args.*.grad (already computed by next layer)
    """

    # Check if this layer's stage5 was merged with the NEXT layer's stage1.
    # Detection: stage5.args is set, stage5.outs is None
    stage5_record = layer_record.stage5
    stage5_was_merged = (
        hasattr(stage5_record, "args")
        and stage5_record.args is not None
        and not (hasattr(stage5_record, "outs") and stage5_record.outs is not None)
    )

    # Check if this layer's stage1 is merged with the PREVIOUS layer's stage5.
    # Detection: stage1.outs is set, stage1.args is None
    stage1_record = layer_record.stage1
    stage1_is_merged = (
        hasattr(stage1_record, "outs")
        and stage1_record.outs is not None
        and not (hasattr(stage1_record, "args") and stage1_record.args is not None)
    )

    # Stage 5.
    if loss is not None:
        assert False, "loss should not be provided"
        loss.backward()
        loss.detach_()
    elif stage5_was_merged:
        nvtx.range_push("layer%02d.stage5_merged_skip" % layer.idx)
        moe_outs_grad, topk_weight_grad, residual_grad = [t.grad if t is not None else None for t in stage5_record.args]
        nvtx.range_pop()
    else:
        nvtx.range_push("layer%02d.stage5_b" % layer.idx)
        record = stage5_record
        run_backward(record.outs, dy)
        moe_outs_grad, topk_weight_grad, residual_grad = [t.grad if t is not None else None for t in record.args]
        nvtx.range_pop()

    # Stage 4.
    nvtx.range_push("layer%02d.stage4_b" % layer.idx)
    record = layer_record.stage4
    if record.ctx is not None:
        moe_outs_grad, _ = dispatch_backward(
            moe_outs_grad, record.ctx[1], get_buffer(distributed.ep_group, get_hidden_bytes(moe_outs_grad)),
            async_finish=False)
    nvtx.range_pop()

    # Stage 3.
    nvtx.range_push("layer%02d.stage3_b" % layer.idx)
    record = layer_record.stage3

    run_backward(record.outs, (moe_outs_grad,))
    gathered_tokens_grad = record.args.gathered_tokens.grad
    recv_topk_weights_grad = record.args.recv_topk_weights.grad if record.args.recv_topk_weights is not None else None
    nvtx.range_pop()

    # Stage 2.
    nvtx.range_push("layer%02d.stage2_b" % layer.idx)
    record = layer_record.stage2
    if record.ctx is not None:
        dispatch_tokens_grad, topk_weight_grad, _ = combine_backward(
            gathered_tokens_grad, recv_topk_weights_grad, record.ctx[1],
            get_buffer(distributed.ep_group, get_hidden_bytes(gathered_tokens_grad)), async_finish=False)
    else:
        dispatch_tokens_grad = gathered_tokens_grad
    nvtx.range_pop()

    # Stage 1.
    nvtx.range_push("layer%02d.stage1_b" % layer.idx)

    grad_tensors = (dispatch_tokens_grad, residual_grad, topk_weight_grad)

    if stage1_is_merged:
        # Merged case: this layer's stage1 + previous layer's stage5
        # Run backward through stage1.outs. Grads flow to prev layer's stage5.args.
        run_backward(stage1_record.outs, grad_tensors)
        nvtx.range_pop()

        # Clear tensor refs but keep pre-allocated records
        for field in fields(layer_record):
            record = getattr(layer_record, field.name)
            for rf in fields(record):
                setattr(record, rf.name, None)

        # Return None to signal prev layer to get grads from its stage5.args
        return None
    else:
        # Normal case: run stage1 backward
        record = stage1_record
        run_backward(record.outs, grad_tensors)
        hidden_states_grad = record.args.next_hidden_states.grad
        record.args.prev_hidden_states.grad = hidden_states_grad
        nvtx.range_pop()

        # Clear tensor refs but keep pre-allocated records
        for field in fields(layer_record):
            record = getattr(layer_record, field.name)
            for rf in fields(record):
                setattr(record, rf.name, None)

        return hidden_states_grad


def model_forward(
    module: ModelProtocol,
    hidden_states: torch.Tensor,
    chunk_record: ChunkRecord,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sequential (non-overlapped) forward for one pipeline chunk: prolog -> layers -> epilog.

    Records each stage's tensors into ``chunk_record`` for the pipeline backward.
    """
    if module.stage_index == 0:
        hidden_states = prolog_f(module, hidden_states, chunk_record.prolog)

    rotary_posemb = module.forward_posemb(hidden_states.shape[1], cu_seqlens)
    for (_, layer), layer_record in zip(module.layers.items(), chunk_record.layers):
        hidden_states = layer_forward(layer, hidden_states, rotary_posemb, layer_record, cu_seqlens)

    if module.stage_index == module.stage_count - 1:
        hidden_states = epilog_f(module, hidden_states, chunk_record.epilog)

    return hidden_states


def model_backward(
    module: ModelProtocol,
    dy: Optional[List[torch.Tensor]],
    loss: Optional[torch.Tensor],
    chunk_record: ChunkRecord,
):
    """
    Sequential (non-overlapped) backward for one pipeline chunk: epilog -> layers -> prolog.

    Backprops through the tensors ``model_forward`` saved in ``chunk_record`` and
    returns the input gradients to hand back to the previous pipeline stage.
    """
    if loss is not None:
        loss.backward()
        loss.detach_()
        dy = (chunk_record.epilog.args.hidden_states.grad,)
        chunk_record.epilog.args = None
        loss = None

    dx = dy
    layers = [layer for _, layer in module.layers.items()]
    for layer, layer_record in zip(reversed(layers), reversed(chunk_record.layers)):
        dx = (layer_backward(layer, dx, loss, layer_record),)

    final_grads = dx
    if module.stage_index == 0:
        record = chunk_record.prolog
        run_backward(record.outs, dx)
        record.args = None
        record.outs = None
        final_grads = (None,)
    return final_grads
