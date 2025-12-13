from typing import List, Optional, Tuple
import torch

from sglang.srt.managers.schedule_batch import (
    Req, 
    ScheduleBatch, 
    global_server_args_dict,
)
from sglang.srt.mem_cache.memory_pool import (
    ReqToTokenPool, 
    TokenToKVPoolAllocator, 
    MHATokenToKVPool,
)
from sglang.srt.paras.utils import paras_tp_group_all_gather_reqs
# from EP to TP, requests are all-gathered from all ranks
class ParaSReqGatherManager:
    
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
        gather_group: torch.distributed.ProcessGroup,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
    ):
        self.local_reqs = local_reqs
        self.gather_group = gather_group
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.group_size = gather_group.size()
        
        req_to_token_indices = []
        
        for req in local_reqs:
            indices = self.req_to_token_pool.req_to_token[req.req_pool_idx][ : req.seqlen - 1]
            req_to_token_indices.append(indices)

        self.local_seqlens_list = [req.seqlen for req in local_reqs]
        self.local_token_indices = torch.cat(req_to_token_indices, dim=0)
    
    def gather_global_reqs(self):
        self.global_reqs, self.global_reqs_split_sizes = paras_tp_group_all_gather_reqs(self.local_reqs, self.gather_group)
        self.global_seqlens_list = [req.seqlen for req in self.global_reqs]
        
        start_index = 0
        self.global_num_tokens = []
        for split_size in self.global_reqs_split_sizes:
            end_index = start_index + split_size
            self.global_num_tokens.append(sum(self.global_seqlens_list[start_index:end_index]))
            start_index = end_index
        
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
        
        self.new_req_pool_size = new_req_pool_size
        self.new_cache_size = new_cache_size
        
        num_reqs = len(self.global_reqs)
        assert num_reqs <= new_req_pool_size, "The number of requests to reorchestrate is greater than the new size of the request to token pool."
        self.req_to_token_pool.paras_resize_and_clear(new_req_pool_size)
        req_pool_indices = self.req_to_token_pool.alloc(num_reqs)
        
        total_cache_size = sum(req.seqlen for req in self.global_reqs)
        assert total_cache_size <= new_cache_size, "The total size of the requests to reorchestrate is greater than the new size of the cache."
        self.token_to_kv_pool_allocator.paras_resize_and_clear(new_cache_size)
        
        new_token_indices = []
        num_global_tokens = sum(self.global_num_tokens)
        global_token_indices = self.token_to_kv_pool_allocator.alloc(num_global_tokens)
        start_index = 0
        # TODO: optimize writing to req_to_token_pool and token_to_kv_pool_allocator
        for req, req_pool_idx in zip(self.global_reqs, req_pool_indices):
            end_index = start_index + req.seqlen
            req.req_pool_idx = req_pool_idx
            token_indices = global_token_indices[start_index:end_index]
            self.req_to_token_pool.write((req_pool_idx, slice(0, req.seqlen)), token_indices)
            new_token_indices.extend(token_indices)
            start_index = end_index
            
        self.global_token_indices = global_token_indices
    
    def gather_cache(self) -> torch.Tensor:
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        assert isinstance(kv_cache, MHATokenToKVPool), "Only MHATokenToKVPool is supported for now."
        
        num_layers = kv_cache.layer_num
        
        num_ranks = self.group_size
        num_heads = kv_cache.head_num
        head_dim = kv_cache.head_dim
        sharded_num_heads = num_heads // num_ranks
        size_per_token = kv_cache.head_num * kv_cache.head_dim * kv_cache.dtype.itemsize
        splited_size_per_token = size_per_token // num_ranks
        input_split_sizes = [splited_size_per_token * num_tokens for num_tokens in self.global_num_tokens]
        
        output_size = sum(self.local_seqlens_list) * size_per_token
        output_split_sizes = [output_size // num_ranks] * num_ranks
        num_global_tokens = sum(self.global_num_tokens)
        
        assert num_global_tokens * splited_size_per_token == sum(input_split_sizes)
        
        def gather_one_layer(layer_id: int) -> torch.Tensor:
            k_buffer = kv_cache.get_key_buffer(layer_id)
            v_buffer = kv_cache.get_value_buffer(layer_id)
            local_kcache = k_buffer[self.local_token_indices]
            local_vcache = v_buffer[self.local_token_indices]
            del k_buffer, v_buffer
            
            local_kvcache = torch.stack([local_kcache, local_vcache], dim=0).view(2, -1, kv_cache.head_num, kv_cache.head_dim)
            permuted_local_kvcache = local_kvcache.permute(2, 0, 1, 3).contiguous() # [num_heads, 2, num_tokens, head_dim]
            assert output_size == permuted_local_kvcache.numel() * permuted_local_kvcache.dtype.itemsize
            
            gathered_kvcache = torch.empty(2 * num_global_tokens * splited_size_per_token, dtype=permuted_local_kvcache.dtype, device=permuted_local_kvcache.device)
            torch.distributed.all_to_all_single(gathered_kvcache, permuted_local_kvcache, input_split_sizes, output_split_sizes, group=self.gather_group)
            gathered_kvcache = gathered_kvcache.view(2, num_global_tokens, sharded_num_heads, head_dim)
            
            kv_cache.paras_resize_cache(layer_id, self.new_cache_size, sharded_num_heads)
            k_buffer = kv_cache.get_key_buffer(layer_id)
            v_buffer = kv_cache.get_value_buffer(layer_id)
            k_buffer[self.global_token_indices].copy_(gathered_kvcache[0])
            v_buffer[self.global_token_indices].copy_(gathered_kvcache[1])
            del k_buffer, v_buffer
        
        for layer_id in range(num_layers):
            gather_one_layer(layer_id)

    def get_global_schedule_batch(self) -> ScheduleBatch:
        pass