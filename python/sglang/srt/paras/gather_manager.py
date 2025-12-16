from typing import List, Optional, Tuple
import torch
import pickle
import numpy as np
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.managers.schedule_batch import (
    Req, 
    ScheduleBatch, 
    global_server_args_dict,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.mem_cache.memory_pool import (
    ReqToTokenPool, 
    TokenToKVPoolAllocator, 
    MHATokenToKVPool,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.distributed.parallel_state import GroupCoordinator

def paras_tp_group_all_gather_reqs(
    reqs: List[Req],
    group: GroupCoordinator,
) -> Tuple[List[Req], List[int]]:
    device = torch.device("cuda")
    
    num_ranks = group.world_size
    
    serialized_data = pickle.dumps(reqs)
    size = len(serialized_data)
    tensor_data = torch.ByteTensor(
        np.frombuffer(serialized_data, dtype=np.uint8)
    )
    tensor_size = torch.tensor([size], dtype=torch.long, device=device)
    
    gathered_size = torch.empty(num_ranks, dtype=torch.long, device=device)
    group.all_gather_into_tensor(gathered_size, tensor_size)
    gathered_size_list = gathered_size.tolist()
    max_size = gathered_size.max().item()
    if max_size == 0:
        return None, None
    
    padded_tensor_data = torch.empty((max_size,), dtype=torch.uint8, device=device)
    padded_tensor_data[:tensor_data.size(0)].copy_(tensor_data)

    gathered_data: torch.Tensor = torch.empty((max_size * num_ranks), dtype=torch.uint8, device=device)
    group.all_gather_into_tensor(gathered_data, padded_tensor_data)
    
    serialized_data_per_rank = np.split(gathered_data.cpu().numpy(), num_ranks, axis=0)
    
    gathered_reqs = []
    split_sizes = []
    for i in range(num_ranks):
        data = serialized_data_per_rank[i]
        effective_size = gathered_size_list[i]
        remote_reqs = pickle.loads(data[:effective_size]) if effective_size > 0 else []
        gathered_reqs.extend(remote_reqs)
        split_sizes.append(len(remote_reqs))
        
    print(f"num_gathered_reqs: {len(gathered_reqs)}, split_sizes: {split_sizes}")
    
    return gathered_reqs, split_sizes

# from EP to TP, requests are all-gathered from all ranks
class ParaSReqGatherManager:
    
    gather_group: GroupCoordinator
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: TokenToKVPoolAllocator
    
    local_reqs: List[Req]
    local_seqlens_list: List[int]
    local_token_indices: torch.Tensor
    
    global_reqs: List[Req]
    global_seqlens_list: List[int]
    global_token_indices: torch.Tensor
    
    def __init__(
        self, 
        local_reqs: List[Req],
        gather_group: GroupCoordinator,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
    ):
        self.local_reqs = local_reqs
        self.gather_group = gather_group
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.group_size = gather_group.world_size
        
        self.local_no_reqs = len(local_reqs) == 0
        self.local_seqlens_list = [req.seqlen for req in local_reqs]
        self.num_local_tokens = sum(self.local_seqlens_list) - len(local_reqs) # the last output token is not stored in kv cache
        
        if self.local_no_reqs:
            self.local_token_indices = None
        else:
            req_to_token_indices = []
            for req in local_reqs:
                indices = self.req_to_token_pool.req_to_token[req.req_pool_idx][ : req.seqlen - 1]
                req_to_token_indices.append(indices)

            self.local_token_indices = torch.cat(req_to_token_indices, dim=0)
            assert self.local_token_indices.shape[0] == self.num_local_tokens, \
                f"local tokens {self.num_local_tokens}, local token indices {self.local_token_indices.shape}"
    
    def gather_global_reqs(self):
        self.global_reqs, self.global_reqs_split_sizes = paras_tp_group_all_gather_reqs(self.local_reqs, self.gather_group)
        self.global_seqlens_list = [req.seqlen for req in self.global_reqs]
        
        start_index = 0
        self.global_num_tokens = []
        for split_size in self.global_reqs_split_sizes:
            end_index = start_index + split_size
            self.global_num_tokens.append(sum(self.global_seqlens_list[start_index:end_index]) - split_size)
            start_index = end_index
            
        self.num_global_tokens = sum(self.global_num_tokens)
        
    def reorchestrate_cache(
        self, 
        new_req_pool_size: Optional[int] = None,
        new_cache_size: Optional[int] = None
    ):
        '''
        Heads are sharded by group size, 
        so the size of the request to token pool and the cache need to be multiplied by the group size.
        '''
        if new_req_pool_size is None:
            new_req_pool_size = self.req_to_token_pool.size * self.group_size
        if new_cache_size is None:
            new_cache_size = self.token_to_kv_pool_allocator.size * self.group_size
        
        assert self.num_global_tokens <= new_cache_size, "The total size of the requests to reorchestrate is greater than the new size of the cache."
        
        self.new_req_pool_size = new_req_pool_size
        self.new_cache_size = new_cache_size
        
        num_reqs = len(self.global_reqs)
        assert num_reqs <= new_req_pool_size, "The number of requests to reorchestrate is greater than the new size of the request to token pool."
        self.req_to_token_pool.paras_resize_and_clear(new_req_pool_size)
        req_pool_indices = self.req_to_token_pool.alloc(num_reqs)
        
        self.token_to_kv_pool_allocator.paras_resize_and_clear(new_cache_size)

        if self.num_global_tokens > 0:        
            new_token_indices = []
            global_token_indices = self.token_to_kv_pool_allocator.alloc(self.num_global_tokens)
            start_index = 0
            # TODO: optimize writing to req_to_token_pool and token_to_kv_pool_allocator
            for req, req_pool_idx in zip(self.global_reqs, req_pool_indices):
                end_index = start_index + req.seqlen - 1
                req.req_pool_idx = req_pool_idx
                token_indices = global_token_indices[start_index:end_index]
                self.req_to_token_pool.write((req_pool_idx, slice(0, req.seqlen - 1)), token_indices)
                new_token_indices.extend(token_indices)
                start_index = end_index
                
            self.global_token_indices = global_token_indices
            assert self.global_token_indices.shape[0] == self.num_global_tokens, "The number of global tokens is not equal to the number of tokens in the global requests."
        else:
            self.global_token_indices = None

    def gather_cache(self) -> torch.Tensor:
        torch.cuda.empty_cache()
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        assert isinstance(kv_cache, MHATokenToKVPool), "Only MHATokenToKVPool is supported for now."
        
        num_layers = kv_cache.layer_num
        
        num_heads = kv_cache.head_num
        head_dim = kv_cache.head_dim
        sharded_num_heads = num_heads // self.group_size
        size_per_token = kv_cache.head_num * kv_cache.head_dim
        splited_size_per_token = size_per_token // self.group_size
        
        input_split_sizes = [2 * splited_size_per_token * self.num_local_tokens] * self.group_size
        output_split_sizes = [2 * (splited_size_per_token * num_tokens_of_rank) for num_tokens_of_rank in self.global_num_tokens]
        
        def gather_one_layer(layer_id: int) -> torch.Tensor:

            if self.num_local_tokens > 0:
                k_buffer = kv_cache.get_key_buffer(layer_id)
                v_buffer = kv_cache.get_value_buffer(layer_id)
                local_kcache = k_buffer[self.local_token_indices]
                local_vcache = v_buffer[self.local_token_indices]
                
                local_kvcache = torch.stack([local_kcache, local_vcache], dim=0).view(2, -1, kv_cache.head_num, kv_cache.head_dim)
                permuted_local_kvcache = local_kvcache.permute(2, 0, 1, 3).contiguous().flatten() # [num_heads, 2, num_tokens, head_dim]
            else:
                permuted_local_kvcache = torch.empty((0, ), dtype=kv_cache.store_dtype, device=kv_cache.device)
                
            kv_cache.paras_resize_cache(layer_id, self.new_cache_size, sharded_num_heads)
                
            if self.num_global_tokens > 0:
                gathered_kvcache = torch.empty(2 * self.num_global_tokens * splited_size_per_token, dtype=permuted_local_kvcache.dtype, device=permuted_local_kvcache.device)
                torch.distributed.all_to_all_single(gathered_kvcache, permuted_local_kvcache, output_split_sizes, input_split_sizes, group=self.gather_group.device_group)
                gathered_kvcache = gathered_kvcache.view(self.num_global_tokens, 2, sharded_num_heads, head_dim)
                permuted_gathered_kvcache = gathered_kvcache.permute(1, 0, 2, 3).contiguous() # [2, num_global_tokens, sharded_num_heads, head_dim]
                
                k_buffer = kv_cache.get_key_buffer(layer_id)
                v_buffer = kv_cache.get_value_buffer(layer_id)
                k_buffer[self.global_token_indices].copy_(permuted_gathered_kvcache[0])
                v_buffer[self.global_token_indices].copy_(permuted_gathered_kvcache[1])
                
        
        for layer_id in range(num_layers):
            gather_one_layer(layer_id)

    def get_new_running_batch(
        self,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_overlap: bool,
        spec_algorithm: SpeculativeAlgorithm,
        enable_custom_logit_processor: bool,
    ) -> ScheduleBatch:
        """
        Create a new ScheduleBatch from the global requests for decode mode.
        All requests are assumed to be in decode status (already have output_ids).
        """
        # Create a ScheduleBatch using init_new
        # tree_cache should be reset to empty state before calling this function

        batch = ScheduleBatch.init_new(
            self.global_reqs,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            tree_cache,
            model_config,
            enable_overlap,
            spec_algorithm,
            enable_custom_logit_processor,
        )
        
        for req in batch.reqs:
            req.last_node = batch.tree_cache.root_node
            
        # Set up decode batch fields
        bs = len(self.global_reqs)
        device = self.req_to_token_pool.device
        
        # Get the last token from each request (for decode input)
        # For decode, input_ids should be the last output token, or last input token if no output yet
        last_token_list = []
        for req in self.global_reqs:
            if len(req.output_ids) > 0:
                last_token_list.append(req.output_ids[-1])
            else:
                # No output yet, use last input token
                last_token_list.append(req.origin_input_ids[-1])
        
        # Get req_pool_indices and seq_lens
        req_pool_indices_list = [req.req_pool_idx for req in self.global_reqs]
        seq_lens_list = [req.seqlen for req in self.global_reqs]
        
        # Convert to tensors
        batch.output_ids = torch.tensor(last_token_list, dtype=torch.int64, device=device)
        batch.req_pool_indices = torch.tensor(req_pool_indices_list, dtype=torch.int64, device=device)
        batch.seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64, device=device)
        batch.seq_lens_sum = sum(seq_lens_list)
        
        # Create sampling_info before prepare_for_decode (it's required by prepare_for_decode)
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch,
            model_config.vocab_size,
        )
        return batch

    def update_running_batch_inplace(
        self,
        running_batch: ScheduleBatch,
    ) -> None:
        """
        Directly update the existing running batch with global requests (in-place modification).
        This is simpler than creating a new batch as it preserves existing batch state.
        All requests are assumed to be in decode status (already have output_ids).
        """
        model_config = running_batch.model_config
        device = self.req_to_token_pool.device
        
        # Update requests and metadata (matching init_new behavior)
        running_batch.reqs = self.global_reqs
        running_batch.return_logprob = any(req.return_logprob for req in self.global_reqs)
        running_batch.has_stream = any(req.stream for req in self.global_reqs)
        running_batch.has_grammar = any(req.grammar for req in self.global_reqs)
        running_batch.return_hidden_states = any(req.return_hidden_states for req in self.global_reqs)
        running_batch.chunked_req = None
        
        for req in running_batch.reqs:
            req.last_node = running_batch.tree_cache.root_node
        
        # Get the last token from each request (for decode input)
        input_ids_list = []
        for req in self.global_reqs:
            if len(req.output_ids) > 0:
                input_ids_list.append(req.output_ids[-1])
            else:
                # No output yet, use last input token
                input_ids_list.append(req.origin_input_ids[-1])
        
        # Get req_pool_indices and seq_lens
        req_pool_indices_list = [req.req_pool_idx for req in self.global_reqs]
        seq_lens_list = [req.seqlen for req in self.global_reqs]
        
        # Update batch tensors
        running_batch.input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device=device)
        running_batch.req_pool_indices = torch.tensor(req_pool_indices_list, dtype=torch.int64, device=device)
        running_batch.seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64, device=device)
        running_batch.seq_lens_sum = sum(seq_lens_list)
        
        # Set output_ids to input_ids (will be used by prepare_for_decode)
        running_batch.output_ids = running_batch.input_ids.clone()
        
        # Recreate sampling_info with new requests
        running_batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            running_batch,
            model_config.vocab_size,
        )