import torch
from typing import Optional

from sglang.srt.distributed.parallel_state import (
    get_world_group,
    init_model_parallel_group,
    get_bool_env_var,
    GroupCoordinator,
)
import sglang.srt.distributed.parallel_state as parallel_state

import sglang.srt.layers.dp_attention as dp_attention

_PARAS_EP: Optional[GroupCoordinator] = None

_PARAS_TP: Optional[GroupCoordinator] = None

def get_paras_tp_group() -> GroupCoordinator:
    assert _PARAS_TP is not None, "ParaS tensor parallel group is not initialized"
    return _PARAS_TP

_PARAS_DP: Optional[GroupCoordinator] = None

def get_paras_dp_group() -> GroupCoordinator:
    assert _PARAS_DP is not None, "ParaS data parallel group is not initialized"
    return _PARAS_DP

# TODO(shaoyuw): refactor code
# The parallel size and rank can be grouped together.
# There are 2 stages to consider:
# stage 1. EP 
# stage 2. 2D parallel (DTP?)

_PARAS_TP_SIZE: int = None
_PARAS_TP_RANK: int = None
_PARAS_DP_SIZE: int = None
_PARAS_DP_RANK: int = None
_PARAS_EP_SIZE: int = None
_PARAS_EP_RANK: int = None

def initialize_paras_parallel(
    dp_size: int = 1,
    tp_size: int = 1,
    global_rank: int = 0,
    backend: Optional[str] = None,
) -> None:
    """
    Initialize ParaS parallel groups.

    Arguments:
        dp_size: number of GPUs used for data parallelism.
        tp_size: number of GPUs used for tensor parallelism.
    """

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    if world_size != dp_size * tp_size:
        raise RuntimeError(
            f"ParaS: world_size ({world_size}) is not equal to "
            f"dp_size ({dp_size}) x tp_size ({tp_size})"
        )

    # get paras parallel size and rank
    global _PARAS_EP
    _PARAS_EP = parallel_state._TP

    global _PARAS_TP_SIZE, _PARAS_DP_SIZE, _PARAS_EP_SIZE, _PARAS_TP_RANK, _PARAS_DP_RANK, _PARAS_EP_RANK
    _PARAS_TP_SIZE = tp_size
    _PARAS_DP_SIZE = dp_size
    _PARAS_EP_SIZE = dp_size * tp_size

    _PARAS_TP_RANK = global_rank % tp_size
    _PARAS_DP_RANK = global_rank // tp_size
    _PARAS_EP_RANK = global_rank

    # Build the ParaS tensor model-parallel groups.
    num_paras_tensor_model_parallel_groups: int = world_size // tp_size
    global _PARAS_TP
    assert _PARAS_TP is None, "ParaS tensor parallel group is already initialized"
    group_ranks = []
    for i in range(num_paras_tensor_model_parallel_groups):
        ranks = list(
            range(i * tp_size, (i + 1) * tp_size)
        )
        group_ranks.append(ranks)

    _PARAS_TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=get_bool_env_var(
            "SGLANG_USE_MESSAGE_QUEUE_BROADCASTER", "true"
        ),
        group_name="paras_tp",
    )

    # Build the ParaS data model-parallel groups.
    num_paras_data_model_parallel_groups: int = world_size // dp_size
    global _PARAS_DP
    assert _PARAS_DP is None, "ParaS data parallel group is already initialized"
    group_ranks = []
    for i in range(num_paras_data_model_parallel_groups):
        ranks = list(range(i, world_size, num_paras_data_model_parallel_groups))
        group_ranks.append(ranks)
    
    _PARAS_DP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_custom_allreduce=False,
        group_name="paras_dp",
    )

    
def paras_comm_configure_tp():
    # global _TP
    parallel_state._TP = _PARAS_TP

    # global _ATTN_TP_RANK, _ATTN_TP_SIZE, _ATTN_DP_RANK, _ATTN_DP_SIZE
    dp_attention._ATTN_TP_RANK = _PARAS_TP_RANK
    dp_attention._ATTN_TP_SIZE = _PARAS_TP_SIZE
    dp_attention._ATTN_DP_RANK = 0
    dp_attention._ATTN_DP_SIZE = 1

    # global _LOCAL_ATTN_DP_RANK, _LOCAL_ATTN_DP_SIZE
    dp_attention._LOCAL_ATTN_DP_SIZE = 1
    dp_attention._LOCAL_ATTN_DP_RANK = 0

def paras_comm_configure_ep():
    # TODO(shaoyuw): adapt for moe dense tp
    # global _TP
    parallel_state._TP = _PARAS_EP

    # global _ATTN_TP_RANK, _ATTN_TP_SIZE, _ATTN_DP_RANK, _ATTN_DP_SIZE
    dp_attention._ATTN_TP_RANK = 0
    dp_attention._ATTN_TP_SIZE = 1
    dp_attention._ATTN_DP_RANK = _PARAS_EP_RANK
    dp_attention._ATTN_DP_SIZE = _PARAS_EP_SIZE

    # global _LOCAL_ATTN_DP_RANK, _LOCAL_ATTN_DP_SIZE
    dp_attention._LOCAL_ATTN_DP_SIZE = _PARAS_EP_SIZE
    dp_attention._LOCAL_ATTN_DP_RANK = _PARAS_EP_RANK
