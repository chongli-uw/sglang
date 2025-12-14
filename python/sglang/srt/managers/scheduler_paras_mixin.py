from types import SimpleNamespace
from typing import List, Any, Optional
import torch
import logging
import torch

from sglang.srt.managers.io_struct import ParaSConfigureReq, ParaSConfigureReqOutput
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    global_server_args_dict,
)
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, MHATokenToKVPool, TokenToKVPoolAllocator

from sglang.srt.paras.utils import paras_func, paras_profile_func
from sglang.srt.paras.gather_manager import ParaSReqGatherManager

logger = logging.getLogger(__name__)

class SchedulerParasMixin:
    """
    This class implements the parallel configuration logic for Scheduler.
    """
    
    tp_worker: TpModelWorker
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: MHATokenToKVPool
    token_to_kv_pool_allocator: TokenToKVPoolAllocator
    
    def init_paras_config(self):
        if not self.server_args.enable_paras_moe:
            return
        
        # ParaS config
        self.paras_tp_size = self.server_args.paras_tp_size
        self.paras_tp_rank = self.tp_rank % self.paras_tp_size
        self.paras_dp_size = self.tp_size // self.paras_tp_size
        self.paras_dp_rank = self.tp_rank // self.paras_tp_size
        self.paras_tp_group = self.tp_worker.get_paras_tp_group()
        self.paras_tp_cpu_group = self.paras_tp_group.cpu_group

        if self.paras_tp_rank == 0:
            self.tp_recv_from_tokenizer = self.recv_from_tokenizer
            self.tp_send_to_tokenizer = self.send_to_tokenizer
            self.tp_send_to_detokenizer = self.send_to_detokenizer
            self.tp_recv_from_rpc = self.recv_from_rpc
        else:
            self.tp_recv_from_tokenizer = None
            self.tp_recv_from_rpc = None
            self.tp_send_to_tokenizer = SimpleNamespace(send_pyobj=lambda x: None)
            self.tp_send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)

        self.paras_ep_size = self.tp_size
        self.paras_ep_rank = self.tp_rank
        self.paras_ep_group = self.tp_group
        self.paras_ep_cpu_group = self.tp_cpu_group

        self.ep_recv_from_tokenizer = self.recv_from_tokenizer
        self.ep_recv_from_rpc = self.recv_from_rpc
        self.ep_send_to_tokenizer = self.send_to_tokenizer
        self.ep_send_to_detokenizer = self.send_to_detokenizer

        self.paras_parallelism_config = "EP"

    def paras_configure_helper(self):
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()

        self.tree_cache.reset()
        
    def paras_check(self):
        if len(self.waiting_queue) > 0:
            logger.warning("Waiting queue is not empty, parallelism switch is not allowed.")
            return False
        
        return True
    
    def paras_get_req_seqlens(self, reqs: List[Req]):
        seqlens = []
        for req in reqs:
            seqlens.append(req.seqlen)
        return seqlens
    
    def paras_get_local_reqs(self):
        # Merge the last batch into the running batch, now every request is in the decode status
        self.merge_last_batch()
        return self.running_batch.reqs
    
    @paras_func
    def paras_configure_tp(self):
        if not self.paras_check():
            return
        
        assert self.server_args.enable_paras_moe, "ParaS parallelism is not enabled."
        assert not self.enable_overlap, "Overlap schedule is not supported currently in ParaS."
        # switch from EP to DP x TP
        self.paras_parallelism_config = "TP"
        self.server_args.enable_dp_attention = False
        self.server_args.enable_torch_a2a_moe = False
        global_server_args_dict["enable_dp_attention"] = False
        global_server_args_dict["enable_torch_a2a_moe"] = False
        
        self.tree_cache.reset()
        local_reqs = self.paras_get_local_reqs()
        paras_gather_manager = ParaSReqGatherManager(
            local_reqs,
            self.paras_tp_group,
            self.req_to_token_pool, 
            self.token_to_kv_pool_allocator
        )
        # TODO: clean radix attention cache
        paras_gather_manager.gather_global_reqs()
        paras_gather_manager.reorchestrate_cache()
        paras_gather_manager.gather_cache()
        self.running_batch = paras_gather_manager.get_new_running_batch(
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor
        )
        # paras_gather_manager.update_running_batch_inplace(self.running_batch)
    
        self.tp_worker.paras_configure_tp(self.paras_tp_size, self.paras_tp_rank)

        # drop-in replacement for scheduler tp configs 
        self.tp_size = self.paras_tp_size
        self.tp_rank = self.paras_tp_rank
        self.tp_group = self.paras_tp_group
        self.tp_cpu_group = self.paras_tp_cpu_group

        # NOTE(shaoyuw): attn_dp_rank should be dealt with more carefully. 
        #                But now it seems to be used only a few times.
        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            self.paras_tp_rank,
            self.paras_tp_size,
            self.paras_dp_rank,
        )

        self.recv_from_tokenizer = self.tp_recv_from_tokenizer
        self.send_to_tokenizer = self.tp_send_to_tokenizer
        self.send_to_detokenizer = self.tp_send_to_detokenizer
        self.recv_from_rpc = self.tp_recv_from_rpc

    @paras_func
    def paras_configure_ep(self):
        if not self.paras_check():
            return
        
        assert self.server_args.enable_paras_moe, "ParaS parallelism is not enabled."
        # switch from TP to EP
        self.paras_parallelism_config = "EP"
        self.server_args.enable_dp_attention = True
        self.server_args.enable_torch_a2a_moe = True
        global_server_args_dict["enable_dp_attention"] = True
        global_server_args_dict["enable_torch_a2a_moe"] = True

        self.tp_worker.paras_configure_ep()

        # drop-in replacement for scheduler ep configs
        self.tp_size = self.paras_ep_size
        self.tp_rank = self.paras_ep_rank
        self.tp_group = self.paras_ep_group
        self.tp_cpu_group = self.paras_ep_cpu_group

        # equals to:
        # self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = 0, 1, self.tp_rank
        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            compute_dp_attention_world_info(
                self.server_args.enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
            )
        )

        self.recv_from_tokenizer = self.ep_recv_from_tokenizer
        self.send_to_tokenizer = self.ep_send_to_tokenizer
        self.send_to_detokenizer = self.ep_send_to_detokenizer
        self.recv_from_rpc = self.ep_recv_from_rpc

    def paras_configure_handle(self, recv_req: ParaSConfigureReq):
        if recv_req == ParaSConfigureReq.CONFIGURE_TP:
            self.paras_configure_tp()
        elif recv_req == ParaSConfigureReq.CONFIGURE_EP:
            self.paras_configure_ep()
        else:
            raise ValueError("Unrecognized ParaSConfigureReq value")
        return ParaSConfigureReqOutput()
    
    def paras_start_profile(self, op_name: str):
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                op_name,
                worker_name=f"rank{self.tp_rank}",
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self.profiler.start()
        
    def paras_stop_profile(self):
        self.profiler.stop()
        self.profiler = None