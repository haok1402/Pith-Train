"""
Distributed runtime state.
"""

import torch

rank: int
"""
Global rank of this process, in [0, world_size).
"""

world_size: int
"""
Total number of processes across all nodes.
"""

local_rank: int
"""
Rank within this node, and the CUDA device index for this process.
"""

local_world_size: int
"""
Number of processes on this node, i.e. GPUs per node.
"""

device: torch.device
"""
The CUDA device for this process. Prefer it over plain cuda when allocating.
"""

attn_mesh: torch.distributed.DeviceMesh
"""
Attention view of the rank space: (pp, dp, cp).

CP sits innermost, so every CP group is a contiguous block of ranks and the ring K/V exchange
stays on NVLink whenever cp_size fits within a node. The flattened dp x cp submesh is both the
FSDP mesh for the attn parameters and the group the load-balance statistics reduce over.
"""

expt_mesh: torch.distributed.DeviceMesh
"""
Expert view of the same rank space: (pp, dp, ep).

MoE parallel folding (https://arxiv.org/abs/2504.14960 section 3.2): the experts factor the
world_size // pp_size ranks of one pipeline stage as dp x ep while attention factors them as
dp x cp, so dp * cp == ep * expt_dp. PP is the one axis the two views share and sits outermost
in each, which keeps every rank in agreement with itself about the stage it holds.

Two properties follow. First, cp_size and ep_size each need only divide the stage size, not each
other, so EP may span CP. Second, EP sits innermost here, so every EP group is also a contiguous
rank block. The dp axis of this view is the FSDP mesh for the expert weights.
"""

pp_group: torch.distributed.ProcessGroup
"""
Pipeline-parallel group. DualPipeV sends activations and gradients over it point to point.
"""

cp_group: torch.distributed.ProcessGroup
"""
Context-parallel group. Ring attention exchanges K/V over it, and the logged loss reduces over it.
"""

ep_group: torch.distributed.ProcessGroup
"""
Expert-parallel group. The MoE dispatch and combine all-to-alls run over it.
"""

pp_rank: int
"""
Index within the pipeline. Under DualPipeV, rank r holds chunks r and 2 * pp_size - 1 - r.
"""

pp_size: int
"""
Pipeline-parallel degree, set by DistributedCfg.pipeline_parallel_size.
"""

dp_rank: int
"""
Index within the data-parallel axis of the attention view, and the single authority on which
data this rank loads. See get_global_batch in tasks/pretrain_lm.py.

Expert parallelism shards experts across ranks and leaves the batch untouched, so the sample
offsets follow dp_rank and never ep_rank. Holding those two apart is what lets ep_size take any
divisor of the stage size.
"""

dp_size: int
"""
Data-parallel degree of the attention view, world_size // (pp_size * cp_size). Derived rather
than configured, and counts the slices the global batch is cut into, so global_batch_size
divides evenly by micro_batch_size * dp_size.
"""

cp_rank: int
"""
Position of this rank in the zigzag sequence layout, rather than in a contiguous slice.

The sequence is cut into 2 * cp_size blocks, and this rank holds the pair
(cp_rank, 2 * cp_size - cp_rank - 1), which balances the causal attention workload. Code that
slices or builds positions along the sequence reproduces that same pairing; see get_global_batch
in tasks/pretrain_lm.py and forward_posemb in each model.
"""

cp_size: int
"""
Context-parallel degree, set by DistributedCfg.context_parallel_size. sequence_length divides
evenly by 2 * cp_size, so the zigzag split is even.
"""

ep_rank: int
"""
Index within the expert-parallel group, naming the contiguous block of experts this rank hosts,
[ep_rank * experts_per_rank, (ep_rank + 1) * experts_per_rank). Expert placement only: which
data a rank loads comes from dp_rank. Read for the expert range in modules/checkpoint.py and as
the all-to-all destination during dispatch.
"""

ep_size: int
"""
Expert-parallel degree, set by DistributedCfg.expert_parallel_size. Divides both the expert count
of the model and the stage size, world_size // pp_size.
"""
