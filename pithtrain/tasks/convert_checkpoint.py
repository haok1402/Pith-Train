"""
Checkpoint conversion from HuggingFace to DCP and vice versa.

Supports MXFP4 dequantization for GPT-OSS model weights.
"""

import json
import math
from contextlib import ExitStack
from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader

from pithtrain.config import SlottedDefault
from pithtrain.modules.logging import LoggingCfg, LoggingCtx, logging_context


@dataclass(init=False, slots=True)
class ConvertCheckpointCfg(SlottedDefault):
    """
    Configuration for checkpoint conversion.
    """

    operation: Literal["hf2dcp", "dcp2hf"]
    """
    Conversion operation: "hf2dcp" or "dcp2hf".
    """

    load_path: Path
    """
    Source checkpoint directory to load from.
    """

    save_path: Path
    """
    Destination checkpoint directory to save to.
    """

    max_shard_size: int = 8 * 1024**3
    """
    Maximum shard size in bytes for dcp2hf (default 8GB).
    """

    logging: LoggingCfg = field(default_factory=LoggingCfg)
    """
    Logging configuration.
    """


@dataclass(init=False, slots=True)
class ConvertCheckpointCtx(SlottedDefault):
    """
    Context for checkpoint conversion.
    """

    logging: LoggingCtx = field(default_factory=LoggingCtx)
    """
    Active logging context.
    """


def _dequantize_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    """
    Dequantize an MXFP4-packed tensor to full precision.

    Each byte in *blocks* encodes two 4-bit floats (low nibble first).
    Each row of *scales* provides a shared 8-bit exponent (biased by 127)
    for the corresponding row of blocks.

    Algorithm adapted from Megatron-Bridge ``gpt_oss_bridge._dequantize_mxfp4``.
    """
    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"
    FP4_VALUES = [
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    scales = scales.to(torch.int32) - 127
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)
        blk = blocks[r0:r1]
        exp = scales[r0:r1]
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)
        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]
        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp

    return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)


def _is_gpt_oss(load_path: Path) -> bool:
    """Check if checkpoint is a GPT-OSS model."""
    config_path = Path(load_path, "config.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return config.get("model_type") == "gpt_oss"
    return False


def _hf2dcp_gpt_oss(
    load_path: Path, save_path: Path, stdout: Logger
) -> None:
    """
    Convert a GPT-OSS HuggingFace checkpoint to DCP format.

    Handles MXFP4 dequantization, expert weight transposition, and key
    renaming (``mlp.router`` -> ``mlp.gate``, expert weight keys gain
    ``.weight`` suffix and are split into individual expert entries).
    """
    with open(Path(load_path, "config.json")) as f:
        model_config = json.load(f)
    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config["intermediate_size"]
    num_experts = model_config["num_local_experts"]

    with open(Path(load_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    shard_files = set(weight_map.values())
    stdout.info(
        "Converting GPT-OSS HF checkpoint from %s (%d shards)" % (load_path, len(shard_files))
    )

    # Load all raw tensors.
    raw: Dict[str, torch.Tensor] = dict()
    for i, shard_file in enumerate(sorted(shard_files), start=1):
        stdout.info("Reading shard %d/%d: %s" % (i, len(shard_files), shard_file))
        with safe_open(str(Path(load_path, shard_file)), framework="pt", device="cpu") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key)

    # Dequantize MXFP4 pairs and collect plain tensors.
    # MXFP4 dequantization flattens the last two dims, so we reshape
    # expert projections back to 3D.  HF GPT-OSS stores expert weights
    # transposed on disk (``is_transposed=True`` in HF), so the on-disk
    # layout for gate_up_proj is [E, 2*inter, hidden] and for down_proj
    # is [E, hidden, inter].  This matches GroupLinear's [E, out, in].
    dequantized: Dict[str, torch.Tensor] = dict()
    seen_blocks = set()
    for key in sorted(raw.keys()):
        if key.endswith("_blocks"):
            base = key.removesuffix("_blocks")
            scales_key = base + "_scales"
            if scales_key in raw:
                stdout.info("Dequantizing MXFP4: %s" % base)
                flat = _dequantize_mxfp4(raw[key], raw[scales_key])
                # Reshape to transposed on-disk layout [E, out, in].
                if "gate_up_proj" in base:
                    flat = flat.view(num_experts, 2 * intermediate_size, hidden_size)
                elif "down_proj" in base:
                    flat = flat.view(num_experts, hidden_size, intermediate_size)
                dequantized[base] = flat
                seen_blocks.add(key)
                seen_blocks.add(scales_key)
        elif key.endswith("_scales"):
            pass  # handled above
        # plain tensors added below

    for key, tensor in raw.items():
        if key not in seen_blocks and key not in dequantized:
            dequantized[key] = tensor

    # Build canonical DCP state dict: strip model. prefix, rename keys,
    # transpose expert weights, and split into individual expert entries.
    model_state_dict: Dict[str, torch.Tensor] = dict()
    for key, tensor in dequantized.items():
        canon = key.removeprefix("model.")

        # Rename router -> gate.
        canon = canon.replace(".mlp.router.", ".mlp.gate.")

        if canon.endswith(".mlp.experts.gate_up_proj"):
            # On-disk (transposed): [E, 2*inter, hidden] already in GroupLinear
            # [E, out, in] layout.  Interleaving is on dim 1 (the output dim):
            # even rows = gate, odd rows = up.
            E = tensor.shape[0]
            gate = tensor[:, ::2, :]    # [E, intermediate, hidden]
            up = tensor[:, 1::2, :]     # [E, intermediate, hidden]
            for idx in range(E):
                prefix = canon.replace(".experts.", ".experts.%d." % idx)
                model_state_dict[prefix.replace("gate_up_proj", "gate_proj") + ".weight"] = gate[idx]
                model_state_dict[prefix.replace("gate_up_proj", "up_proj") + ".weight"] = up[idx]
        elif canon.endswith(".mlp.experts.gate_up_proj_bias"):
            # [E, 2*intermediate] with interleaved gate/up.
            E = tensor.shape[0]
            gate_bias = tensor[:, ::2]    # [E, intermediate]
            up_bias = tensor[:, 1::2]     # [E, intermediate]
            for idx in range(E):
                prefix = canon.replace(".experts.", ".experts.%d." % idx)
                model_state_dict[prefix.replace("gate_up_proj_bias", "gate_proj_bias")] = gate_bias[idx]
                model_state_dict[prefix.replace("gate_up_proj_bias", "up_proj_bias")] = up_bias[idx]
        elif canon.endswith(".mlp.experts.down_proj"):
            # On-disk (transposed): [E, hidden, inter] already in GroupLinear
            # [E, out, in] layout.  No transpose needed.
            E = tensor.shape[0]
            for idx in range(E):
                expert_key = canon.replace(".experts.", ".experts.%d." % idx) + ".weight"
                model_state_dict[expert_key] = tensor[idx]
        elif canon.endswith(".mlp.experts.down_proj_bias"):
            E = tensor.shape[0]
            for idx in range(E):
                expert_key = canon.replace(".experts.", ".experts.%d." % idx)
                model_state_dict[expert_key] = tensor[idx]
        else:
            model_state_dict[canon] = tensor

    save_path.mkdir(parents=True, exist_ok=True)
    dcp.save({"app": {"model": model_state_dict}}, checkpoint_id=save_path, no_dist=True)
    stdout.info(
        "Saved DCP checkpoint to %s (%d weights)" % (save_path, len(model_state_dict))
    )


def hf2dcp(cfg: ConvertCheckpointCfg, stdout: Logger) -> None:
    """
    Convert HuggingFace checkpoint to DCP format.
    """
    load_path, save_path = Path(cfg.load_path), Path(cfg.save_path)

    # GPT-OSS requires MXFP4 dequantization and key remapping.
    if _is_gpt_oss(load_path):
        _hf2dcp_gpt_oss(load_path, save_path, stdout)
        return

    with open(Path(load_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    shard_files = set(weight_map.values())
    stdout.info("Converting HF checkpoint from %s (%d shards)" % (load_path, len(shard_files)))

    model_state_dict: Dict[str, torch.Tensor] = dict()
    for i, shard_file in enumerate(sorted(shard_files), start=1):
        stdout.info("Reading shard %d/%d: %s" % (i, len(shard_files), shard_file))
        with safe_open(str(Path(load_path, shard_file)), framework="pt", device="cpu") as f:
            for key in f.keys():
                model_state_dict[key.removeprefix("model.")] = f.get_tensor(key)

    save_path.mkdir(parents=True, exist_ok=True)
    dcp.save({"app": {"model": model_state_dict}}, checkpoint_id=save_path, no_dist=True)
    stdout.info("Saved DCP checkpoint to %s (%d weights)" % (save_path, len(model_state_dict)))


def dcp2hf(cfg: ConvertCheckpointCfg, stdout: Logger) -> None:
    """
    Convert DCP checkpoint to HuggingFace format.
    """
    load_path, save_path = Path(cfg.load_path), Path(cfg.save_path)
    max_shard_size = cfg.max_shard_size
    stdout.info("Converting DCP checkpoint from %s" % load_path)

    model_prefix = "app.model."
    state_dict, metadata = dict(), FileSystemReader(load_path).read_metadata()
    for key, tensor_meta in metadata.state_dict_metadata.items():
        if key.startswith(model_prefix):
            state_dict[key] = torch.empty(tensor_meta.size, dtype=tensor_meta.properties.dtype)
    dcp.load(state_dict, checkpoint_id=load_path, no_dist=True)
    stdout.info("Loaded %d model weights from DCP" % len(state_dict))

    hf_state_dict = dict()
    for key, tensor in state_dict.items():
        canon = key.removeprefix(model_prefix)
        hf_state_dict[canon if canon.startswith("lm_head.") else "model." + canon] = tensor

    shards: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    current_shard: Dict[str, torch.Tensor] = dict()
    current_size, shard_idx = 0, 0

    for key, tensor in hf_state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(("model-%05d.safetensors" % shard_idx, current_shard))
            current_shard, current_size, shard_idx = dict(), 0, shard_idx + 1
        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(("model-%05d.safetensors" % shard_idx, current_shard))

    weight_map, total_size = dict(), 0
    save_path.mkdir(parents=True, exist_ok=True)
    for i, (_, shard_tensors) in enumerate(shards):
        shard_name = "model-%05d-of-%05d.safetensors" % (i, len(shards))
        stdout.info("Writing shard %d/%d: %s" % (i + 1, len(shards), shard_name))
        save_file(shard_tensors, str(Path(save_path, shard_name)))
        for key in shard_tensors:
            weight_map[key] = shard_name
        total_size += sum(t.numel() * t.element_size() for t in shard_tensors.values())

    with open(Path(save_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)
    stdout.info(
        "Saved HF checkpoint to %s (%d weights, %d shards)"
        % (save_path, len(weight_map), len(shards))
    )


def launch(cfg: ConvertCheckpointCfg) -> None:
    """
    Launch checkpoint conversion.
    """
    with ExitStack() as stack:
        ctx = ConvertCheckpointCtx()
        stack.enter_context(logging_context(cfg, ctx))
        ctx.logging.stdout.info("launch(cfg=%s)" % cfg)
        match cfg.operation:
            case "hf2dcp":
                hf2dcp(cfg, ctx.logging.stdout)
            case "dcp2hf":
                dcp2hf(cfg, ctx.logging.stdout)
