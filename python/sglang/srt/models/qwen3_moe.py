# Adapted from qwen2_moe.py

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
import torch.distributed as dist

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    parallel_state,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather,
    attn_tp_reduce_scatter,
    dp_gather_partial,
    dp_scatter,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_local_attention_dp_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.ep_moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.managers.expert_location import ModelConfigForExpertLocation
from sglang.srt.managers.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP as Qwen3MoeMLP
from sglang.srt.models.qwen2_moe import Qwen2MoeModel
from sglang.srt.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.utils import DeepEPMode, add_prefix, is_non_idle_and_non_empty
from sglang.srt.paras.utils import paras_func, paras_weight_buffer
from sglang.srt.paras.paras_parallel_state import (
    get_paras_tp_size,
    get_paras_tp_rank,
    get_paras_dp_size,
    get_paras_dp_rank,
    get_paras_tp_group,
    get_paras_dp_group,
)
    

Qwen3MoeConfig = None

logger = logging.getLogger(__name__)

class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: Qwen3MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.experts = get_moe_impl_class()(
            num_experts=config.num_experts
            + global_server_args_dict["ep_num_redundant_experts"],
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            **(
                dict(deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]])
                if global_server_args_dict["enable_deepep_moe"]
                else {}
            ),
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        if global_server_args_dict["enable_deepep_moe"]:
            # TODO: we will support tp < ep in the future
            self.ep_size = get_tensor_model_parallel_world_size()
            self.num_experts = (
                config.num_experts + global_server_args_dict["ep_num_redundant_experts"]
            )
            self.top_k = config.num_experts_per_tok
            self.renormalize = config.norm_topk_prob

            self.deepep_dispatcher = MaybeTboDeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=self.num_experts,
                num_local_experts=config.num_experts // self.tp_size,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]],
                async_finish=True,  # TODO
                return_recv_hook=True,
            )

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: Optional[ForwardBatch] = None
    ) -> torch.Tensor:

        if global_server_args_dict["enable_deepep_moe"]:
            return self.forward_deepep(hidden_states, forward_batch)
        elif global_server_args_dict["enable_torch_a2a_moe"]:
            return self.forward_torch_a2a(hidden_states, forward_batch)
        else:
            return self.forward_normal(hidden_states)

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)
    
    def forward_torch_a2a(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        return final_hidden_states.view(num_tokens, hidden_dim)

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        forward_mode = forward_batch.forward_mode
        if is_non_idle_and_non_empty(forward_mode, hidden_states):
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)

            topk_weights, topk_idx = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=False,
                renormalize=self.renormalize,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_idx = torch.full(
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            topk_weights = torch.empty(
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )
        if self.ep_size > 1:
            # TODO(ch-wan): allow users to set num_max_dispatch_tokens_per_rank value
            (
                hidden_states,
                topk_idx,
                topk_weights,
                reorder_topk_ids,
                num_recv_tokens_per_expert,
                seg_indptr,
                masked_m,
                expected_m,
            ) = self.deepep_dispatcher.dispatch(
                hidden_states=hidden_states,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                forward_mode=forward_mode,
            )
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            reorder_topk_ids=reorder_topk_ids,
            seg_indptr=seg_indptr,
            masked_m=masked_m,
            expected_m=expected_m,
            num_recv_tokens_per_expert=num_recv_tokens_per_expert,
            forward_mode=forward_mode,
        )
        if self.ep_size > 1:
            final_hidden_states = self.deepep_dispatcher.combine(
                hidden_states=final_hidden_states,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                forward_mode=forward_mode,
            )
        return final_hidden_states

    def op_gate(self, state):
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):
            # router_logits: (num_tokens, n_experts)
            state.router_logits, _ = self.gate(state.hidden_states_mlp_input)
        else:
            state.router_logits = None

    def op_select_experts(self, state):
        router_logits = state.pop("router_logits")
        hidden_states = state.hidden_states_mlp_input
        if router_logits is not None:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.topk_weights_local, state.topk_idx_local = select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    top_k=self.top_k,
                    use_grouped_topk=False,
                    renormalize=self.renormalize,
                    num_token_non_padded=state.forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    ),
                )
        else:
            state.topk_idx_local = torch.full(
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            state.topk_weights_local = torch.empty(
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )

    def op_dispatch_a(self, state):
        if self.ep_size > 1:
            # TODO(ch-wan): allow users to set num_max_dispatch_tokens_per_rank value
            self.deepep_dispatcher.dispatch_a(
                hidden_states=state.pop("hidden_states_mlp_input"),
                topk_idx=state.pop("topk_idx_local"),
                topk_weights=state.pop("topk_weights_local"),
                forward_mode=state.forward_batch.forward_mode,
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_dispatch_b(self, state):
        if self.ep_size > 1:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                (
                    state.hidden_states_experts_input,
                    state.topk_idx_dispatched,
                    state.topk_weights_dispatched,
                    state.reorder_topk_ids,
                    state.num_recv_tokens_per_expert,
                    state.seg_indptr,
                    state.masked_m,
                    state.expected_m,
                ) = self.deepep_dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )

    def op_experts(self, state):
        state.hidden_states_experts_output = self.experts(
            hidden_states=state.pop("hidden_states_experts_input"),
            topk_idx=state.topk_idx_dispatched,
            topk_weights=state.topk_weights_dispatched,
            reorder_topk_ids=state.pop("reorder_topk_ids"),
            seg_indptr=state.pop("seg_indptr"),
            masked_m=state.pop("masked_m"),
            expected_m=state.pop("expected_m"),
            num_recv_tokens_per_expert=state.pop("num_recv_tokens_per_expert"),
            forward_mode=state.forward_batch.forward_mode,
        )

    def op_combine_a(self, state):
        if self.ep_size > 1:
            self.deepep_dispatcher.combine_a(
                hidden_states=state.pop("hidden_states_experts_output"),
                topk_idx=state.pop("topk_idx_dispatched"),
                topk_weights=state.pop("topk_weights_dispatched"),
                forward_mode=state.forward_batch.forward_mode,
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_combine_b(self, state):
        if self.ep_size > 1:
            state.hidden_states_after_combine = self.deepep_dispatcher.combine_b(
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_output(self, state):
        state.hidden_states_mlp_output = state.pop("hidden_states_after_combine")

class Qwen3MoeSparseMoeBlockParaS(Qwen3MoeSparseMoeBlock):
    def __init__(
        self,
        layer_id: int,
        config: Qwen3MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(layer_id, config, quant_config, prefix)

        # For ParaS, we start with torch A2A EP and switch to TP at a certain point
        assert global_server_args_dict["enable_torch_a2a_moe"] or global_server_args_dict["enable_deepep_moe"]
        self.num_global_experts = config.num_experts
        self.num_local_experts = self.num_global_experts // self.tp_size
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size

        self.ep_experts = self.experts
        self.tp_experts = FusedMoE(
            num_experts=config.num_experts
            + global_server_args_dict["ep_num_redundant_experts"],
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            tp_size=get_paras_tp_size(),
            tp_rank=get_paras_tp_rank(),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            skip_weights_init=True,
            # no deepep config
        )

        self.paralleism_config = "ep"

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: Optional[ForwardBatch] = None
    ) -> torch.Tensor:
        if self.paralleism_config == "ep":
            if global_server_args_dict["enable_deepep_moe"]:
                return self.forward_deepep(hidden_states, forward_batch)
            elif global_server_args_dict["enable_torch_a2a_moe"]:
                return self.forward_torch_a2a(hidden_states, forward_batch)
        else:
            return self.forward_normal(hidden_states)

    def paras_configure_helper(self):
        pass

    def paras_configure_tp_all_gather(self, stream=None, handles=[], async_op=False):
        paras_dp_group = get_paras_dp_group().device_group
        paras_dp_size = get_paras_dp_size()

        all_gather_handles = []
        # EP to DPxEP:
        with torch.cuda.stream(stream):
            for handle in handles:
                handle.wait()
            if paras_dp_size > 1:
                w13_ep = self.ep_experts.w13_weight.data.view(self.num_local_experts, 2 * self.moe_intermediate_size, self.hidden_size)
                self.w13_ep_gathered = paras_weight_buffer.get_buffer((self.num_local_experts * paras_dp_size, 2 * self.moe_intermediate_size, self.hidden_size), dtype=w13_ep.dtype, device=w13_ep.device)
                all_gather_handles.append(dist.all_gather_into_tensor(self.w13_ep_gathered, w13_ep, group=paras_dp_group, async_op=True))
                self.ep_experts.paras_drop_params("w13_weight")

                w2_ep = self.ep_experts.w2_weight.data.view(self.num_local_experts, self.hidden_size, self.moe_intermediate_size)
                self.w2_ep_gathered = paras_weight_buffer.get_buffer((self.num_local_experts * paras_dp_size, self.hidden_size, self.moe_intermediate_size), dtype=w2_ep.dtype, device=w2_ep.device)
                all_gather_handles.append(dist.all_gather_into_tensor(self.w2_ep_gathered, w2_ep, group=paras_dp_group, async_op=True))
                self.ep_experts.paras_drop_params("w2_weight")

                self.num_local_experts *= paras_dp_size
            else:
                w13_ep_gathered = self.ep_experts.w13_weight.data.view(self.num_local_experts, 2 * self.moe_intermediate_size, self.hidden_size)
                paras_weight_buffer.put(w13_ep_gathered)
                self.ep_experts.paras_drop_params("w13_weight")

                w2_ep_gathered = self.ep_experts.w2_weight.data.view(self.num_local_experts, self.hidden_size, self.moe_intermediate_size)
                paras_weight_buffer.put(w2_ep_gathered)
                self.ep_experts.paras_drop_params("w2_weight")
        
        if async_op:
            return all_gather_handles
        else:
            for handle in all_gather_handles:
                handle.wait()
        
    def paras_configure_tp_all_to_all(self, stream=None, handles=[]):
        paras_tp_size = get_paras_tp_size()
        paras_dp_size = get_paras_dp_size()
        paras_tp_group = get_paras_tp_group().device_group
        moe_intermediate_size_after_tp = self.moe_intermediate_size // paras_tp_size

        # DPxEP to DPxTP:
        with torch.cuda.stream(stream):
            for handle in handles:
                handle.wait()
            w13_ep = self.w13_ep_gathered.view(self.num_local_experts, 2, paras_tp_size, moe_intermediate_size_after_tp * self.hidden_size)
            # w13_ep_permuted = paras_weight_buffer.get_buffer_like(w13_ep).view(paras_tp_size, self.num_local_experts, 2, moe_intermediate_size_after_tp * self.hidden_size)
            # w13_ep_permuted.copy_(w13_ep.permute(2, 0, 1, 3))
            w13_ep_permuted = w13_ep.permute(2, 0, 1, 3).contiguous()
            w13_tp = w13_ep # reuse memory
            w13_handle = dist.all_to_all_single(output=w13_tp, input=w13_ep_permuted, group=paras_tp_group, async_op=True)

            w2_ep = self.w2_ep_gathered.data.view(self.num_local_experts, self.hidden_size, paras_tp_size, moe_intermediate_size_after_tp)
            # w2_ep_permuted = paras_weight_buffer.get_buffer_like(w2_ep).view(paras_tp_size, self.num_local_experts, self.hidden_size, moe_intermediate_size_after_tp)
            # w2_ep_permuted.copy_(w2_ep.permute(2, 0, 1, 3))
            w2_ep_permuted = w2_ep.permute(2, 0, 1, 3).contiguous()
            w2_tp = w2_ep # reuse memory
            w2_handle = dist.all_to_all_single(output=w2_tp, input=w2_ep_permuted, group=paras_tp_group, async_op=True)

            w13_handle.wait()
            if paras_dp_size > 1:
                w13_tp_permuted = w13_ep_permuted.view(paras_dp_size, paras_tp_size, -1)
                w13_tp_permuted.copy_(w13_tp.view(paras_tp_size, paras_dp_size, -1).transpose(0, 1))
                w13_tp_weight = w13_tp_permuted
                paras_weight_buffer.put(w13_tp)
            else:
                w13_tp_weight = w13_tp
                paras_weight_buffer.put(w13_ep_permuted)
            self.tp_experts.paras_load_params(w13_tp_weight.view(self.num_global_experts, 2 * moe_intermediate_size_after_tp, self.hidden_size), "w13_weight")
                
            w2_handle.wait()
            if paras_dp_size > 1:
                w2_tp_permuted = w2_ep_permuted.view(paras_dp_size, paras_tp_size, -1)
                w2_tp_permuted.copy_(w2_tp.view(paras_tp_size, paras_dp_size, -1).transpose(0, 1))
                w2_tp_weight = w2_tp_permuted
                paras_weight_buffer.put(w2_tp)
            else:
                w2_tp_weight = w2_tp
                paras_weight_buffer.put(w2_ep_permuted)
            self.tp_experts.paras_load_params(w2_tp_weight.view(self.num_global_experts, self.hidden_size, moe_intermediate_size_after_tp), "w2_weight")

    @paras_func
    def paras_configure_tp(self, paras_tp_size: int, paras_tp_rank: int):
        self.paralleism_config = "tp"
        self.tp_size = paras_tp_size
        self.experts = self.tp_experts

    @paras_func
    def paras_configure_ep(self):
        assert False, "Qwen3MoeSparseMoeBlockParaS does not support configure back to EP at this moment"
        self.paralleism_config = "ep"
        self.tp_size = 1
        self.experts = self.ep_experts
        
class Qwen3MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_by_head = q.reshape(-1, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        return q, k

    def op_prepare(self, state):
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
        )

    def op_core(self, state):
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)

        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state

    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        attn_output = self.attn(*inner_state)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        return self.forward_core(s)
    
    def paras_configure_helper(self):
        self.q_size = self.qkv_proj.q_proj_shard_size
        self.kv_size = self.qkv_proj.kv_proj_shard_size
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.num_kv_heads = self.total_num_kv_heads // self.attn_tp_size

    @paras_func
    def paras_configure_tp(self, paras_tp_size, paras_tp_rank):
        self.attn_tp_rank = paras_tp_rank
        self.attn_tp_size = paras_tp_size
        self.qkv_proj.paras_configure_tp(paras_tp_size, paras_tp_rank)
        self.attn.paras_configure_tp(paras_tp_size, paras_tp_rank)
        self.o_proj.paras_configure_tp(paras_tp_size, paras_tp_rank)

    @paras_func
    def paras_configure_ep(self):
        self.attn_tp_size = 1
        self.attn_tp_rank = 0
        self.qkv_proj.paras_configure_ep()
        self.attn.paras_configure_ep()
        self.o_proj.paras_configure_ep()

class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rms_norm_eps = config.rms_norm_eps
        attention_bias = config.attention_bias
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        self.layer_id = layer_id

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.local_dp_size = get_local_attention_dp_size()

        # Qwen3MoE all layers are sparse and have no nextn now
        self.is_layer_sparse = True
        is_previous_layer_sparse = True

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
        )

        if self.is_layer_sparse:
            qwen3_moe_impl_class = Qwen3MoeSparseMoeBlockParaS if global_server_args_dict["enable_paras_moe"] else Qwen3MoeSparseMoeBlock
            self.mlp = qwen3_moe_impl_class(
                layer_id=self.layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )

        if global_server_args_dict["enable_paras_moe"]:
            self.paras_ep_layer_communicator = self.layer_communicator
            self.paras_ep_layer_scatter_modes = self.layer_scatter_modes
            self.paras_tp_layer_communicator = None
            self.paras_tp_layer_scatter_modes = None

    def paras_configure_helper(self):
        pass

    def paras_configure_tp_attn(self, paras_tp_size: int, paras_tp_rank: int):
        self.self_attn.paras_configure_tp(paras_tp_size, paras_tp_rank)

    def paras_configure_tp_mlp(self, paras_tp_size: int, paras_tp_rank: int):
        self.mlp.paras_configure_tp_all_gather()
        self.mlp.paras_configure_tp_all_to_all()

    def paras_configure_tp_mlp_all_gather(self, stream, handles, async_op=False):
        return self.mlp.paras_configure_tp_all_gather(stream, handles, async_op)
    
    def paras_configure_tp_mlp_all_to_all(self, stream, handles):
        return self.mlp.paras_configure_tp_all_to_all(stream, handles)

    @paras_func
    def paras_configure_tp(self, paras_tp_size: int, paras_tp_rank: int):
        # Switch from EP to TP
        assert global_server_args_dict["enable_paras_moe"]

        # save previous ep context
        self.paras_ep_layer_scatter_modes = self.layer_scatter_modes
        self.paras_ep_layer_communicator = self.layer_communicator

        # config only
        self.mlp.paras_configure_tp(paras_tp_size, paras_tp_rank)

        # build new tp context
        if not self.paras_tp_layer_scatter_modes:
            self.paras_tp_layer_scatter_modes = LayerScatterModes.init_new(
                layer_id=self.layer_id,
                num_layers=self.config.num_hidden_layers,
                is_layer_sparse=self.is_layer_sparse,
                is_previous_layer_sparse=True, # all layers are sparse for Qwen3-MoE
            )
            assert not self.paras_tp_layer_communicator
            self.paras_tp_layer_communicator = LayerCommunicator(
                layer_scatter_modes=self.paras_tp_layer_scatter_modes,
                input_layernorm=self.input_layernorm,
                post_attention_layernorm=self.post_attention_layernorm,
            )

        # hack the layer scatter modes and communicator
        assert self.paras_tp_layer_scatter_modes
        assert self.paras_tp_layer_communicator
        self.layer_scatter_modes = self.paras_tp_layer_scatter_modes
        self.layer_communicator = self.paras_tp_layer_communicator
        self.attn_tp_size = paras_tp_size
        self.attn_tp_rank = paras_tp_rank
        self.local_dp_size = 1
        
    @paras_func
    def paras_configure_ep(self):
        # Switch from TP to EP
        assert global_server_args_dict["enable_paras_moe"]

        self.self_attn.paras_configure_ep()
        self.mlp.paras_configure_ep()

        # revert to ep context
        assert self.paras_ep_layer_scatter_modes is not None, "EP scatter modes are not initialized"
        assert self.paras_ep_layer_communicator is not None, "EP communication context is not initialized"
        self.layer_scatter_modes = self.paras_ep_layer_scatter_modes
        self.layer_communicator = self.paras_ep_layer_communicator

        self.attn_tp_size = 1
        self.attn_tp_rank = 0
        self.local_dp_size = get_local_attention_dp_size()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        hidden_states = self.mlp(hidden_states, forward_batch)
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tbo_subbatch_index: Optional[int] = None,
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_mlp(self, state):
        hidden_states = state.pop("hidden_states_mlp_input")
        state.hidden_states_mlp_output = self.mlp(
            hidden_states, state.forward_batch.forward_mode
        )

    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "tbo_subbatch_index",
            }
        )
        return output


class Qwen3MoeModel(Qwen2MoeModel):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen3MoeDecoderLayer,
        )

    def paras_configure_tp_naive(self, paras_tp_size: int, paras_tp_rank: int):
        for layer in self.layers:
            assert isinstance(layer, Qwen3MoeDecoderLayer), "Layer is not Qwen3MoeDecoderLayer"
            layer.paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
            layer.paras_configure_tp_mlp(paras_tp_size, paras_tp_rank)
            layer.paras_configure_tp(paras_tp_size, paras_tp_rank)

    def paras_configure_tp_overlap(self, paras_tp_size: int, paras_tp_rank: int):
        """
        Configure the model for tensor parallelism (TP).
        Note(shaoyuw): the embedding layer is set to DP, but it works for TP as well. 
                       There is no need to modify it.
        """
        stream_1 = torch.cuda.Stream()
        stream_2 = torch.cuda.Stream()
        
        self.layers[0].paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
        last_layer_handles = self.layers[0].paras_configure_tp_mlp_all_gather(stream_1, [], async_op=True)
        nlayers = len(self.layers)
        for i, layer in enumerate(self.layers):
            assert isinstance(layer, Qwen3MoeDecoderLayer), "Layer is not Qwen3MoeDecoderLayer"
            not_last_layer = i < nlayers - 1
            if not_last_layer:
                next_layer = self.layers[i+1]
                next_layer.paras_configure_tp_attn(paras_tp_size, paras_tp_rank)
                new_handles = next_layer.paras_configure_tp_mlp_all_gather(stream_2, last_layer_handles, async_op=True)

            layer.paras_configure_tp_mlp_all_to_all(stream_1, last_layer_handles)
            layer.paras_configure_tp(paras_tp_size, paras_tp_rank)

            if not_last_layer:
                last_layer_handles = new_handles
                stream_1, stream_2 = stream_2, stream_1

    @paras_func
    def paras_configure_tp(self, paras_tp_size: int, paras_tp_rank: int, overlap: bool = False):
        """
        Configure the model for tensor parallelism (TP).
        Note(shaoyuw): the embedding layer is set to DP, but it works for TP as well. 
                       There is no need to modify it.
        """
        if overlap:
            self.paras_configure_tp_overlap(paras_tp_size, paras_tp_rank)
        else:
            self.paras_configure_tp_naive(paras_tp_size, paras_tp_rank)

    @paras_func
    def paras_configure_ep(self):
        for layer in self.layers:
            assert isinstance(layer, Qwen3MoeDecoderLayer), "Layer is not Qwen3MoeDecoderLayer"
            layer.paras_configure_ep()


class Qwen3MoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Qwen3MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3MoeModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"],
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return hidden_states

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        import time
        torch.cuda.synchronize()
        start_loading = time.time()

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = get_moe_impl_class().make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        
        params_dict = dict(self.named_parameters())
        
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # if layer_id == 0:
                    #     print_on_rank_0(f"Loading expert param: {name} with shard_id: {shard_id} and expert_id: {expert_id}")
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    
                    if global_server_args_dict["enable_paras_moe"]:
                        name = name.replace("experts", "tp_experts")
                        if name in params_dict:
                            param = params_dict[name]
                            weight_loader = param.weight_loader
                            weight_loader(
                                param,
                                loaded_weight,
                                name,
                                shard_id=shard_id,
                                expert_id=expert_id,
                            )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

        # TODO mimic deepseek
        if not global_server_args_dict["enable_paras_moe"]:
            self.routed_experts_weights_of_layer = {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.start_layer, self.end_layer)
                if isinstance(self.model.layers[layer_id].mlp, Qwen3MoeSparseMoeBlock)
            }
        end_loading = time.time()
        torch.cuda.synchronize()
        logger.info(
            f"Qwen3MoeForCausalLM loaded weights in {end_loading - start_loading:.2f} seconds"
        )

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )
    
    def paras_configure_helper(self):
        torch.cuda.synchronize()
        paras_weight_buffer.release_all()

    @paras_func
    def paras_configure_tp(self, paras_tp_size: int, paras_tp_rank: int):
        """
        Configure the model for tensor parallelism (TP).
        Note(shaoyuw): the LMHead and logit processor are set to DP with enable_dp_lm_head=True,
                       but they work for TP as well. There is no need to modify them.
        """
        self.model.paras_configure_tp(paras_tp_size, paras_tp_rank)
        # self.lm_head.paras_configure_tp(paras_tp_size, paras_tp_rank)

    @paras_func
    def paras_configure_ep(self):
        self.model.paras_configure_ep()
        # self.lm_head.paras_configure_ep()


EntryClass = Qwen3MoeForCausalLM
