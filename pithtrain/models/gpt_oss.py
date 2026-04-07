"""
openai/gpt-oss-120b and openai/gpt-oss-20b.

GPT-OSS is a Mixture-of-Experts transformer with:
- YaRN RoPE for 131K context length
- Attention sinks (learned per-head bias in softmax denominator)
- Alternating sliding-window / full causal attention
- Modified SwiGLU activation with clamping and residual (up + 1) * glu
- Post-softmax expert routing (top-k then softmax on selected logits)
- All attention and expert projections carry bias terms
"""

import math
from dataclasses import fields
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from pithtrain.dualpipe.execution import (
    EpilogArgs,
    EpilogOuts,
    IntermediateTensors,
    PrologArgs,
    PrologOuts,
)
from pithtrain.dualpipe.modeling import decoder_layer_backward, decoder_layer_forward
from pithtrain.dualpipe.utils import run_backward
from pithtrain.layers.factory import (
    ModelImplMode,
    get_group_linear_cls,
    get_linear_cls,
)
from pithtrain.models.interface import ForwardAttnOutput
from pithtrain.modules.load_balance import MoELoadBalanceLossInjector, MoELoadBalanceLossTracker
from pithtrain.operators.ep_dispatch import moe_ep_prepare_dispatch
from pithtrain.operators.token_scatter import precompute_group_indices, scatter_for_grouped_gemm

torch._dynamo.allow_in_graph(MoELoadBalanceLossInjector)

# Pre-compile flex_attention for Triton kernel generation.
_flex_attention = torch.compile(flex_attention, dynamic=False)

# ---------------------------------------------------------------------------
# YaRN Rotary Position Embedding
# ---------------------------------------------------------------------------

SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int
) -> Tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(min_val: int, max_val: int, dim: int) -> torch.Tensor:
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


class GptOssYarnRotaryEmbedding(nn.Module):
    """
    YaRN-scaled rotary position embedding for GPT-OSS.

    Parameters match OpenAI's RotaryEmbedding: base=150000, factor=32,
    original_max_position_embeddings=4096, beta_fast=32, beta_slow=1.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 150000.0,
        scaling_factor: float = 32.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self._set_cos_sin_cache(max_position_embeddings, device, torch.get_default_dtype())

    def _set_cos_sin_cache(
        self, seq_len: int, device: Optional[torch.device], dtype: torch.dtype
    ):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # YaRN concentration factor (mscale).
        concentration = 0.1 * math.log(self.scaling_factor) + 1.0

        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * concentration).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * concentration).to(dtype), persistent=False
        )

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# ---------------------------------------------------------------------------
# Rotary embedding helpers
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# MoE Experts (SwiGLU with clamping + residual)
# ---------------------------------------------------------------------------


class GptOssExperts(nn.Module):
    """
    Expert FFN layers for GPT-OSS.

    Uses separate gate_proj / up_proj / down_proj (same layout as Qwen3/DeepSeek)
    with GPT-OSS-specific SwiGLU activation::

        gate = clamp(gate_proj(x) + gate_bias, max=7.0)
        up   = clamp(up_proj(x) + up_bias, -7.0, 7.0)
        glu  = gate * sigmoid(1.702 * gate)
        out  = down_proj((up + 1) * glu) + down_bias

    HF stores gate and up interleaved in a single ``gate_up_proj``; the
    checkpoint conversion un-interleaves into separate projections.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        GroupLinearCls = get_group_linear_cls()
        self.gate_proj = GroupLinearCls(num_experts, hidden_size, intermediate_size)
        self.up_proj = GroupLinearCls(num_experts, hidden_size, intermediate_size)
        self.down_proj = GroupLinearCls(num_experts, intermediate_size, hidden_size)
        self.gate_proj_bias = nn.Parameter(torch.zeros(num_experts, intermediate_size))
        self.up_proj_bias = nn.Parameter(torch.zeros(num_experts, intermediate_size))
        self.down_proj_bias = nn.Parameter(torch.zeros(num_experts, hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list | None = None,
        ks_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gi = precompute_group_indices(grouped_mm_offs, x.shape[0])
        kwargs = dict(
            grouped_mm_offs=grouped_mm_offs, ks=ks, ks_tensor=ks_tensor, group_indices=gi
        )

        # Compute group IDs for bias indexing.
        group_ids = torch.searchsorted(
            grouped_mm_offs.to(torch.int64),
            torch.arange(x.shape[0], device=x.device, dtype=torch.int64),
            right=True,
        )

        gate = self.gate_proj(x, **kwargs) + self.gate_proj_bias[group_ids]
        up = self.up_proj(x, **kwargs) + self.up_proj_bias[group_ids]

        # GPT-OSS SwiGLU activation with clamping and residual.
        gate = gate.clamp(max=SWIGLU_LIMIT)
        up = up.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        glu = gate * torch.sigmoid(SWIGLU_ALPHA * gate)
        activated = (up + 1) * glu

        out = self.down_proj(activated, **kwargs) + self.down_proj_bias[group_ids]
        return out


# ---------------------------------------------------------------------------
# MoE Gate (post-softmax top-k router)
# ---------------------------------------------------------------------------


class GptOssGate(nn.Module):
    """
    Top-K routing gate with post-softmax normalization.

    Unlike Qwen3/Mixtral which apply softmax to all experts then select top-k,
    GPT-OSS selects top-k logits first then softmaxes over the selected subset.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.load_balance_loss_fn = None
        self.weight = nn.Parameter(torch.empty((num_experts, hidden_size)), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_experts))

    @torch.compile(fullgraph=True)
    def compute(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        logits = F.linear(hidden_states, self.weight, self.bias)

        # Post-softmax: select top-k first, then softmax over selected.
        topk_logits, topk_idx = torch.topk(
            logits, k=self.num_experts_per_tok, dim=-1, sorted=True
        )
        topk_weight = F.softmax(topk_logits, dim=-1, dtype=torch.float32)

        if self.training and self.load_balance_loss_fn is not None:
            # Full softmax scores for load balance loss computation.
            scores = logits.softmax(dim=-1, dtype=torch.float32)
            lb_loss = self.load_balance_loss_fn(
                scores, topk_idx, self.num_experts, self.num_experts_per_tok
            )
            topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss)
        else:
            lb_loss = None

        return topk_idx, topk_weight, lb_loss

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_idx, topk_weight, lb_loss = self.compute(hidden_states)
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        return topk_idx, topk_weight


# ---------------------------------------------------------------------------
# MoE block
# ---------------------------------------------------------------------------


class GptOssMoE(nn.Module):
    """
    Mixture of Experts block with expert parallelism support.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        intermediate_size: int,
        ep_size: int = 1,
        ep_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.ep_size = ep_size
        self.ep_group = ep_group
        self.ep_rank = ep_group.rank() if ep_group is not None else 0
        self.experts_per_rank = num_experts // ep_size

        self.experts = GptOssExperts(self.experts_per_rank, hidden_size, intermediate_size)
        self.gate = GptOssGate(hidden_size, num_experts, num_experts_per_tok)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        return y

    def moe_infer(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        assert self.ep_size == 1, "Reference implementation only supports ep_size=1"
        expert_idxs = topk_ids.view(-1)
        sorted_tokens = (
            x.unsqueeze(1)
            .expand(-1, self.num_experts_per_tok, -1)
            .reshape(-1, x.shape[-1])
        )
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = (
            scatter_for_grouped_gemm(sorted_tokens, expert_idxs, self.experts_per_rank)
        )
        outs = self.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = outs[reverse_shuffle_idxs]

        final_out = (
            (outs.view(*topk_ids.shape, -1) * topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .to(outs.dtype)
        )
        return final_out


# ---------------------------------------------------------------------------
# Attention (GQA + sinks via Flex Attention)
# ---------------------------------------------------------------------------


class GptOssAttention(nn.Module):
    """
    Grouped Query Attention with attention sinks and optional sliding window.

    Attention sinks are implemented via a virtual KV token: K and V are extended
    by one zero vector, and the score for that position is replaced with a
    learned per-head bias.  This allows each head to "dump" attention to the
    sink, effectively paying zero attention to real tokens when the sink bias
    is large relative to Q·K scores.

    Uses PyTorch Flex Attention for fused score_mod + block_mask execution.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5

        LinearCls = get_linear_cls()
        self.q_proj = LinearCls(
            hidden_size, num_attention_heads * head_dim, bias=attention_bias
        )
        self.k_proj = LinearCls(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias
        )
        self.v_proj = LinearCls(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias
        )
        self.o_proj = LinearCls(
            num_attention_heads * head_dim, hidden_size, bias=attention_bias
        )

        # Attention sink: one learned scalar per query head.
        self.sinks = nn.Parameter(torch.zeros(num_attention_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        block_mask: BlockMask,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Extend KV with a virtual sink token (zeros — RoPE not applied).
        k_sink = key_states.new_zeros(bsz, 1, self.num_kv_heads, self.head_dim)
        v_sink = value_states.new_zeros(bsz, 1, self.num_kv_heads, self.head_dim)
        key_states = torch.cat([key_states, k_sink], dim=1)
        value_states = torch.cat([value_states, v_sink], dim=1)

        # Transpose to BHSD layout for flex_attention.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # score_mod: replace the virtual sink position's score with the
        # learned per-head bias.  For real positions the score is unchanged.
        sinks = self.sinks

        def score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(kv_idx == seq_len, sinks[h], score)

        attn_output = _flex_attention(
            query_states,
            key_states,
            value_states,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=self.scaling,
            enable_gqa=True,
        )

        # Transpose back to BSHD and project.
        attn_output = attn_output.transpose(1, 2).reshape(
            bsz, seq_len, self.num_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)
        return attn_output


# ---------------------------------------------------------------------------
# Decoder Layer (DualPipeV protocol)
# ---------------------------------------------------------------------------


class GptOssDecoderLayer(nn.Module):
    """
    GPT-OSS transformer decoder layer.

    Implements the DualPipeV protocol methods: forward_attn, forward_mlp,
    forward_aggregate, and reference_forward.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        rms_norm_eps: float,
        attention_bias: bool,
        layer_idx: int,
        ep_size: int = 1,
        ep_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.idx = layer_idx
        self.hidden_size = hidden_size

        self.self_attn = GptOssAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
        )

        self.mlp = GptOssMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            intermediate_size=intermediate_size,
            ep_size=ep_size,
            ep_group=ep_group,
        )

        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

        # Flex attention is not compatible with torch.compile(fullgraph=True)
        # via the same mechanism as ring attention — unwrap the compiled wrapper.
        self._forward_attn_compute = self._forward_attn_compute.__wrapped__.__get__(
            self, type(self)
        )

    @torch.compile(fullgraph=True)
    def _forward_attn_compute(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        position_embeddings = getattr(self, "_position_embeddings", None)
        if position_embeddings is None:
            raise RuntimeError(
                "Position embeddings must be set before calling forward_attn"
            )
        block_mask = getattr(self, "_block_mask", None)
        if block_mask is None:
            raise RuntimeError("Block mask must be set before calling forward_attn")

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            block_mask=block_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        return hidden_states, residual

    def forward_attn(self, hidden_states: torch.Tensor) -> ForwardAttnOutput:
        """LN + Attn + LN + Expert selection."""
        hidden_states, residual = self._forward_attn_compute(hidden_states)

        topk_ids, topk_weight = self.mlp.gate(hidden_states)
        (
            sorted_tokens,
            idxs,
            expert_idxs,
            expand_idx,
            dedup_input_splits,
            dedup_output_splits,
            input_splits,
            output_splits,
        ) = moe_ep_prepare_dispatch(
            hidden_states,
            topk_ids,
            self.mlp.num_experts,
            self.mlp.ep_size,
            self.mlp.experts_per_rank,
            self.mlp.ep_group,
        )

        return ForwardAttnOutput(
            sorted_tokens,
            idxs,
            topk_weight,
            output_splits,
            input_splits,
            expert_idxs,
            residual,
            expand_idx,
            dedup_input_splits,
            dedup_output_splits,
        )

    def forward_mlp(
        self,
        gathered_tokens: torch.Tensor,
        expert_idxs: Optional[torch.Tensor] = None,
        expand_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Expert forward."""
        assert expert_idxs is not None
        if expand_idx is not None:
            gathered_tokens = gathered_tokens[expand_idx]
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = (
            scatter_for_grouped_gemm(
                gathered_tokens, expert_idxs, self.mlp.experts_per_rank
            )
        )
        outs = self.mlp.experts(
            output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor
        )
        outs = outs[reverse_shuffle_idxs]
        return outs

    @torch.compile(fullgraph=True)
    def forward_aggregate(
        self,
        moe_outs: torch.Tensor,
        moe_local_idxs: Optional[torch.Tensor],
        topk_weight: Optional[torch.Tensor],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted expert output + residual connection."""
        if self.mlp.ep_size > 1:
            assert moe_local_idxs is not None
            seq_len, topk = topk_weight.shape
            permuted_probs = topk_weight.view(-1)[moe_local_idxs]
            token_indices = moe_local_idxs // topk
            weighted = (moe_outs.float() * permuted_probs.unsqueeze(-1)).to(moe_outs.dtype)
            hidden_states = moe_outs.new_zeros(seq_len, moe_outs.shape[-1])
            hidden_states.scatter_add_(
                0, token_indices[:, None].expand_as(weighted), weighted
            )
            hidden_states = hidden_states.view(*residual.shape)
        else:
            assert moe_local_idxs is None
            new_x = moe_outs
            final_out = new_x.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(
                dim=-1
            )
            final_out = final_out.sum(dim=1).to(new_x.dtype)
            hidden_states = final_out.view(*residual.shape)

        hidden_states = residual + hidden_states
        return hidden_states

    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reference forward for correctness validation."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        position_embeddings = getattr(self, "_position_embeddings", None)
        if position_embeddings is None:
            raise RuntimeError(
                "Position embeddings must be set before calling reference_forward"
            )
        block_mask = getattr(self, "_block_mask", None)
        if block_mask is None:
            raise RuntimeError(
                "Block mask must be set before calling reference_forward"
            )

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            block_mask=block_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


def _make_causal_sink_mask(seq_len: int):
    """Causal mask that also allows attending to the virtual sink at kv_idx == seq_len."""

    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx <= q_idx) | (kv_idx == seq_len)

    return mask_mod


def _make_sliding_sink_mask(seq_len: int, window_size: int):
    """Sliding-window mask (causal) that also allows the virtual sink."""

    def mask_mod(b, h, q_idx, kv_idx):
        causal_ok = (q_idx >= kv_idx) & (q_idx - kv_idx < window_size)
        return causal_ok | (kv_idx == seq_len)

    return mask_mod


class GptOssModel(nn.Module):
    """
    GPT-OSS model for DualPipeV pipeline parallelism.

    Supports stage partitioning for pipeline parallelism and expert
    parallelism for MoE layers.
    """

    def __init__(
        self,
        config,
        num_stages: int,
        stage_id: int,
        cp_group: Optional[dist.ProcessGroup] = None,
        ep_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.config = config
        self.stage_id = stage_id
        self.num_stages = num_stages

        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        intermediate_size = config.intermediate_size
        num_experts = getattr(config, "num_local_experts", 128)
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 4)
        rms_norm_eps = config.rms_norm_eps
        attention_bias = getattr(config, "attention_bias", True)
        vocab_size = config.vocab_size
        max_position_embeddings = config.max_position_embeddings
        sliding_window = getattr(config, "sliding_window", 128)
        rope_theta = getattr(config, "rope_theta", 150000.0)
        rope_scaling = getattr(config, "rope_scaling", None) or {}

        ep_size = getattr(config, "ep_size", 1)

        # Layer types: alternating sliding/full.
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            layer_types = [
                "sliding_attention" if i % 2 == 0 else "full_attention"
                for i in range(config.num_hidden_layers)
            ]
        self.layer_types = layer_types
        self.sliding_window = sliding_window

        self.embed_tokens = (
            nn.Embedding(vocab_size, hidden_size) if stage_id == 0 else None
        )

        # Distribute layers across pipeline stages.
        num_local_layers = [
            config.num_hidden_layers // num_stages for _ in range(num_stages)
        ]
        layers_per_stage_residual = config.num_hidden_layers % num_stages
        for i in range(layers_per_stage_residual):
            num_local_layers[(1 - (i % 2) * 2) * (i // 2) - (i % 2)] += 1
        layer_id_begin = sum(num_local_layers[:stage_id])
        layer_id_end = layer_id_begin + num_local_layers[stage_id]

        self.layers = nn.ModuleDict(
            {
                str(i): GptOssDecoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    rms_norm_eps=rms_norm_eps,
                    attention_bias=attention_bias,
                    layer_idx=i,
                    ep_size=ep_size,
                    ep_group=ep_group,
                )
                for i in range(layer_id_begin, layer_id_end)
            }
        )

        if stage_id == num_stages - 1:
            self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        else:
            self.norm = None
            self.lm_head = None

        # YaRN RoPE.
        self.rotary_emb = GptOssYarnRotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_factor=float(rope_scaling.get("factor", 32.0)),
            original_max_position_embeddings=int(
                rope_scaling.get("original_max_position_embeddings", 4096)
            ),
            beta_fast=float(rope_scaling.get("beta_fast", 32.0)),
            beta_slow=float(rope_scaling.get("beta_slow", 1.0)),
        )

        # Block mask cache: {seq_len: (causal_mask, sliding_mask)}.
        self._block_mask_cache: Dict[int, Tuple[BlockMask, BlockMask]] = {}

    def _get_block_masks(
        self, seq_len: int, device: torch.device
    ) -> Tuple[BlockMask, BlockMask]:
        """Return (causal_block_mask, sliding_block_mask) for given seq_len."""
        if seq_len not in self._block_mask_cache:
            kv_len = seq_len + 1  # +1 for virtual sink token
            causal_mask = create_block_mask(
                _make_causal_sink_mask(seq_len),
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=kv_len,
                device=device,
            )
            sliding_mask = create_block_mask(
                _make_sliding_sink_mask(seq_len, self.sliding_window),
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=kv_len,
                device=device,
            )
            self._block_mask_cache[seq_len] = (causal_mask, sliding_mask)
        return self._block_mask_cache[seq_len]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        intermediate_tensors: Optional[IntermediateTensors] = getattr(
            self, "_intermediate_tensors", None
        )

        if self.embed_tokens is not None:
            input_ids = hidden_states
            hidden_states = self.embed_tokens(input_ids)

        bsz, seq_len, _ = hidden_states.shape

        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len)
        position_embeddings = (
            cos[:seq_len].unsqueeze(0),
            sin[:seq_len].unsqueeze(0),
        )

        # Create block masks (cached by seq_len).
        causal_mask, sliding_mask = self._get_block_masks(seq_len, hidden_states.device)

        for layer_idx_str, layer in self.layers.items():
            layer._position_embeddings = position_embeddings
            layer_type = self.layer_types[int(layer_idx_str)]
            layer._block_mask = (
                sliding_mask
                if layer_type == "sliding_attention"
                else causal_mask
            )

        if intermediate_tensors is None:
            for _, layer in self.layers.items():
                ret = decoder_layer_forward(layer, hidden_states)
                hidden_states = ret[0] if isinstance(ret, tuple) else ret
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
                hidden_states = self.lm_head(hidden_states)
            return hidden_states

        layer_idx = 0
        if self.embed_tokens is not None:
            intermediate_tensors.prolog.args = PrologArgs()
            intermediate_tensors.prolog.outs = PrologOuts(hidden_states)

        for _, layer in self.layers.items():
            ret = decoder_layer_forward(layer, hidden_states)
            if len(ret) == 2:
                hidden_states, layer_record = ret
                dst = intermediate_tensors.layers[layer_idx]
                for field in fields(layer_record):
                    src_rec = getattr(layer_record, field.name)
                    if not hasattr(src_rec, "args"):
                        continue
                    dst_rec = getattr(dst, field.name)
                    for rf in fields(src_rec):
                        setattr(dst_rec, rf.name, getattr(src_rec, rf.name))
            else:
                hidden_states = ret[0]
                dst = intermediate_tensors.layers[layer_idx]
                for field in fields(dst):
                    record = getattr(dst, field.name)
                    for rf in fields(record):
                        setattr(record, rf.name, None)
            layer_idx += 1

        if self.norm is not None:
            assert self.lm_head is not None
            if not ModelImplMode.use_reference_fwd:
                hidden_states = hidden_states.detach().requires_grad_()
            intermediate_tensors.epilog.args = EpilogArgs(hidden_states)
            hidden_states = self.norm(hidden_states)
            hidden_states = self.lm_head(hidden_states)
            intermediate_tensors.epilog.outs = EpilogOuts(hidden_states)

        return hidden_states

    @staticmethod
    def backward(
        module: "GptOssModel",
        dy: Optional[List[torch.Tensor]],
        loss: Optional[torch.Tensor],
        intermediate_tensors: IntermediateTensors,
    ):
        assert (dy is None) != (loss is None), "Either dy or loss should be provided"

        if loss is not None:
            assert module.norm is not None
            assert module.lm_head is not None
            loss.backward()
            loss.detach_()
            dy = (intermediate_tensors.epilog.args.hidden_states.grad,)
            intermediate_tensors.epilog.args = None
            intermediate_tensors.epilog.outs = None
            loss = None
        else:
            assert module.norm is None
            assert module.lm_head is None

        dx = dy
        layers_list = [layer for _, layer in module.layers.items()]
        for layer, intermediate_tensors_layer in zip(
            reversed(layers_list), reversed(intermediate_tensors.layers)
        ):
            dx = (decoder_layer_backward(layer, dx, loss, intermediate_tensors_layer),)

        final_grads = dx
        if module.embed_tokens is not None:
            record = intermediate_tensors.prolog
            run_backward(record.outs, dx)
            for rf in fields(record):
                setattr(record, rf.name, None)
            final_grads = (None,)

        return final_grads
