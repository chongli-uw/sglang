"""
Triton kernels for ParaS operations.
"""

import torch
import triton
import triton.language as tl

def get_device_num_sms(device: torch.device = None) -> int:
    """Get the number of SMs (Streaming Multiprocessors) for the given CUDA device."""
    if device is None:
        device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).multi_processor_count


DEFAULT_NUM_BLOCKS = None  # Will be lazily initialized


def get_default_num_blocks() -> int:
    """Get the default number of blocks, lazily initialized to the device's SM count."""
    global DEFAULT_NUM_BLOCKS
    if DEFAULT_NUM_BLOCKS is None:
        DEFAULT_NUM_BLOCKS = get_device_num_sms()
    return DEFAULT_NUM_BLOCKS

@triton.jit
def _gather_kv_and_permute_kernel(
    k_buffer_ptr,  # [pool_size, num_heads, head_dim]
    v_buffer_ptr,  # [pool_size, num_heads, head_dim]
    indices_ptr,   # [num_tokens]
    output_ptr,    # [num_heads * 2 * num_tokens * head_dim] flattened
    num_tokens,
    num_heads,
    head_dim,
    kv_stride_token,  # stride for token dimension in k/v buffer (= num_heads * head_dim)
    total_work_items,  # num_tokens * 2 (k and v for each token)
    BLOCK_HD: tl.constexpr,
):
    """
    Fused kernel to gather K and V from cache buffers and permute to 
    [num_heads, 2, num_tokens, head_dim] layout (flattened).
    
    Uses grid-stride loop with fixed number of blocks for better SM utilization.
    Each block processes multiple tokens, handling all heads for each token.
    
    Input layout:
        k_buffer, v_buffer: [pool_size, num_heads, head_dim]
        indices: [num_tokens] - token indices to gather
    
    Output layout (flattened):
        [num_heads, 2, num_tokens, head_dim] where dim 1 is [k, v]
        
    Output flat index calculation:
        For output[h, kv, t, d]:
        flat_idx = h * (2 * num_tokens * head_dim) + kv * (num_tokens * head_dim) + t * head_dim + d
    """
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)
    
    # Each work item is (token_idx, kv_idx) - we process all heads for each
    # Grid-stride loop over work items
    for work_idx in range(pid, total_work_items, num_blocks):
        token_idx = work_idx // 2
        kv_idx = work_idx % 2
        
        # Load the actual token index from the indices array
        actual_token_idx = tl.load(indices_ptr + token_idx)
        
        # Source base in k/v buffer: [pool_size, num_heads, head_dim]
        src_token_base = actual_token_idx * kv_stride_token
        
        # Process all heads for this token
        for head_idx in range(num_heads):
            # Source for this head
            src_base = src_token_base + head_idx * head_dim
            
            # Output layout: [num_heads, 2, num_tokens, head_dim]
            # Strides: head_stride = 2 * num_tokens * head_dim
            #          kv_stride = num_tokens * head_dim
            #          token_stride = head_dim
            dst_base = head_idx * (2 * num_tokens * head_dim) + kv_idx * (num_tokens * head_dim) + token_idx * head_dim
            
            # Process head_dim in blocks
            for d_start in range(0, head_dim, BLOCK_HD):
                d_offsets = d_start + tl.arange(0, BLOCK_HD)
                mask = d_offsets < head_dim
                
                src_offsets = src_base + d_offsets
                dst_offsets = dst_base + d_offsets
                
                # Load from k or v buffer based on kv_idx
                if kv_idx == 0:
                    values = tl.load(k_buffer_ptr + src_offsets, mask=mask)
                else:
                    values = tl.load(v_buffer_ptr + src_offsets, mask=mask)
                
                tl.store(output_ptr + dst_offsets, values, mask=mask)

def gather_kv_and_permute_triton(
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor = None,
    num_blocks: int = None,
) -> torch.Tensor:
    """
    Gather K and V from cache buffers using indices and permute to 
    [num_heads, 2, num_tokens, head_dim] layout (returned flattened).
    
    This fuses the following operations:
        local_kcache = k_buffer[indices]
        local_vcache = v_buffer[indices]
        local_kvcache = torch.stack([local_kcache, local_vcache], dim=0)
        local_kvcache = local_kvcache.view(2, -1, num_heads, head_dim)
        permuted = local_kvcache.permute(2, 0, 1, 3).contiguous().flatten()
    
    Args:
        k_buffer: Key cache buffer with shape [pool_size, num_heads, head_dim]
        v_buffer: Value cache buffer with shape [pool_size, num_heads, head_dim]
        indices: Token indices to gather, shape [num_tokens]
        output: Optional pre-allocated output tensor
        num_blocks: Number of CUDA blocks to use (default: DEFAULT_NUM_BLOCKS)
    
    Returns:
        Flattened tensor with shape [num_heads * 2 * num_tokens * head_dim],
        logically [num_heads, 2, num_tokens, head_dim]
    """
    assert k_buffer.dim() == 3, f"k_buffer must be 3D, got {k_buffer.dim()}"
    assert v_buffer.dim() == 3, f"v_buffer must be 3D, got {v_buffer.dim()}"
    assert k_buffer.shape == v_buffer.shape, "k_buffer and v_buffer must have same shape"
    assert indices.dim() == 1, f"indices must be 1D, got {indices.dim()}"
    
    pool_size, num_heads, head_dim = k_buffer.shape
    num_tokens = indices.shape[0]
    
    # Ensure contiguous for proper stride calculation
    k_buffer = k_buffer.contiguous()
    v_buffer = v_buffer.contiguous()
    indices = indices.contiguous()
    
    if output is None:
        output = torch.empty(
            num_heads * 2 * num_tokens * head_dim,
            dtype=k_buffer.dtype,
            device=k_buffer.device,
        )
        
    # Calculate strides
    kv_stride_token = num_heads * head_dim
    
    # Use fixed number of blocks with grid-stride loop
    if num_blocks is None:
        num_blocks = get_default_num_blocks()
    total_work_items = num_tokens * 2  # Each work item handles all heads for one (token, kv) pair
    grid = (min(num_blocks, total_work_items),)
    
    BLOCK_HD = triton.next_power_of_2(min(head_dim, 128))
    
    _gather_kv_and_permute_kernel[grid](
        k_buffer,
        v_buffer,
        indices,
        output,
        num_tokens,
        num_heads,
        head_dim,
        kv_stride_token,
        total_work_items,
        BLOCK_HD,
    )
    
    return output


@triton.jit
def _permute_and_scatter_kv_kernel(
    input_ptr,      # [num_tokens * 2 * num_heads * head_dim] flattened, layout [num_tokens, 2, num_heads, head_dim]
    k_buffer_ptr,   # [pool_size, num_heads, head_dim]
    v_buffer_ptr,   # [pool_size, num_heads, head_dim]
    indices_ptr,    # [num_tokens]
    num_tokens,
    num_heads,
    head_dim,
    kv_stride_token,
    total_work_items,  # num_tokens * 2
    BLOCK_HD: tl.constexpr,
):
    """
    Scatter K and V from permuted layout back to cache buffers.
    
    Uses grid-stride loop with fixed number of blocks for better SM utilization.
    
    Input layout (flattened):
        [num_tokens, 2, num_heads, head_dim] where dim 1 is [k, v]
        
    Input flat index calculation:
        For input[t, kv, h, d]:
        flat_idx = t * (2 * num_heads * head_dim) + kv * (num_heads * head_dim) + h * head_dim + d
    """
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)
    
    # Each work item is (token_idx, kv_idx) - we process all heads for each
    # Grid-stride loop over work items
    for work_idx in range(pid, total_work_items, num_blocks):
        token_idx = work_idx // 2
        kv_idx = work_idx % 2
        
        # Load the actual token index from the indices array
        actual_token_idx = tl.load(indices_ptr + token_idx)
        
        # Destination base in k/v buffer
        dst_token_base = actual_token_idx * kv_stride_token
        
        # Source base for this token in input
        src_token_kv_base = token_idx * (2 * num_heads * head_dim) + kv_idx * (num_heads * head_dim)
        
        # Process all heads for this token
        for head_idx in range(num_heads):
            src_base = src_token_kv_base + head_idx * head_dim
            dst_base = dst_token_base + head_idx * head_dim
            
            # Process head_dim in blocks
            for d_start in range(0, head_dim, BLOCK_HD):
                d_offsets = d_start + tl.arange(0, BLOCK_HD)
                mask = d_offsets < head_dim
                
                src_offsets = src_base + d_offsets
                dst_offsets = dst_base + d_offsets
                
                values = tl.load(input_ptr + src_offsets, mask=mask)
                
                # Select destination buffer based on kv_idx (0 = k, 1 = v)
                if kv_idx == 0:
                    tl.store(k_buffer_ptr + dst_offsets, values, mask=mask)
                else:
                    tl.store(v_buffer_ptr + dst_offsets, values, mask=mask)


def permute_and_scatter_kv_triton(
    permuted_kvcache: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    indices: torch.Tensor,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    num_blocks: int = None,
) -> None:
    """
    Scatter K and V from permuted layout back to cache buffers.
    
    This fuses the following operations:
        gathered_kvcache = gathered_kvcache.view(num_tokens, 2, num_heads, head_dim)
        permuted_gathered_kvcache = gathered_kvcache.permute(1, 0, 2, 3).contiguous()
        k_buffer[indices] = permuted_gathered_kvcache[0]
        v_buffer[indices] = permuted_gathered_kvcache[1]
    
    Args:
        permuted_kvcache: Input tensor with shape [num_tokens, 2, num_heads, head_dim] (can be flat)
        k_buffer: Key cache buffer with shape [pool_size, num_heads, head_dim]
        v_buffer: Value cache buffer with shape [pool_size, num_heads, head_dim]
        indices: Token indices to scatter to, shape [num_tokens]
        num_tokens: Number of tokens
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_blocks: Number of CUDA blocks to use (default: DEFAULT_NUM_BLOCKS)
    """
    # Ensure contiguous
    permuted_kvcache = permuted_kvcache.contiguous()
    indices = indices.contiguous()
    
    # Calculate strides
    kv_stride_token = num_heads * head_dim
    
    BLOCK_HD = triton.next_power_of_2(min(head_dim, 128))
    
    # Use fixed number of blocks with grid-stride loop
    if num_blocks is None:
        num_blocks = get_default_num_blocks()
    total_work_items = num_tokens * 2
    grid = (min(num_blocks, total_work_items),)
    
    _permute_and_scatter_kv_kernel[grid](
        permuted_kvcache,
        k_buffer,
        v_buffer,
        indices,
        num_tokens,
        num_heads,
        head_dim,
        kv_stride_token,
        total_work_items,
        BLOCK_HD,
    )


def test_gather_kv_and_permute():
    """Test the gather_kv_and_permute kernel against PyTorch reference implementation."""
    torch.manual_seed(42)
    
    pool_size = 1024
    num_heads = 32
    head_dim = 128
    num_tokens = 64
    
    # Create test data
    k_buffer = torch.randn(pool_size, num_heads, head_dim, dtype=torch.float16, device="cuda")
    v_buffer = torch.randn(pool_size, num_heads, head_dim, dtype=torch.float16, device="cuda")
    indices = torch.randperm(pool_size, device="cuda")[:num_tokens]
    
    # Reference implementation (original PyTorch code)
    local_kcache = k_buffer[indices]
    local_vcache = v_buffer[indices]
    local_kvcache = torch.stack([local_kcache, local_vcache], dim=0).view(2, -1, num_heads, head_dim)
    ref_output = local_kvcache.permute(2, 0, 1, 3).contiguous().flatten()
    
    # Triton kernel
    triton_output = gather_kv_and_permute_triton(k_buffer, v_buffer, indices)
    
    # Compare
    if torch.allclose(ref_output, triton_output, rtol=1e-3, atol=1e-3):
        print("✓ gather_kv_and_permute: PASSED")
    else:
        print("✗ gather_kv_and_permute: FAILED")
        print(f"  Max diff: {(ref_output - triton_output).abs().max().item()}")
        return False
    
    return True


def test_permute_and_scatter_kv():
    """Test the permute_and_scatter_kv kernel against PyTorch reference implementation."""
    torch.manual_seed(42)
    
    pool_size = 1024
    num_heads = 32
    head_dim = 128
    num_tokens = 64
    
    # Create test data - simulating all_to_all output
    permuted_kvcache = torch.randn(num_tokens * 2 * num_heads * head_dim, dtype=torch.float16, device="cuda")
    indices = torch.randperm(pool_size, device="cuda")[:num_tokens]
    
    # Reference implementation (original PyTorch code)
    ref_k_buffer = torch.zeros(pool_size, num_heads, head_dim, dtype=torch.float16, device="cuda")
    ref_v_buffer = torch.zeros(pool_size, num_heads, head_dim, dtype=torch.float16, device="cuda")
    
    gathered_kvcache = permuted_kvcache.view(num_tokens, 2, num_heads, head_dim)
    permuted_gathered_kvcache = gathered_kvcache.permute(1, 0, 2, 3).contiguous()
    ref_k_buffer[indices] = permuted_gathered_kvcache[0]
    ref_v_buffer[indices] = permuted_gathered_kvcache[1]
    
    # Triton kernel
    triton_k_buffer = torch.zeros(pool_size, num_heads, head_dim, dtype=torch.float16, device="cuda")
    triton_v_buffer = torch.zeros(pool_size, num_heads, head_dim, dtype=torch.float16, device="cuda")
    permute_and_scatter_kv_triton(
        permuted_kvcache, triton_k_buffer, triton_v_buffer, indices,
        num_tokens, num_heads, head_dim
    )
    
    # Compare
    k_passed = torch.allclose(ref_k_buffer, triton_k_buffer, rtol=1e-3, atol=1e-3)
    v_passed = torch.allclose(ref_v_buffer, triton_v_buffer, rtol=1e-3, atol=1e-3)
    
    if k_passed and v_passed:
        print("✓ permute_and_scatter_kv: PASSED")
    else:
        print("✗ permute_and_scatter_kv: FAILED")
        if not k_passed:
            print(f"  K buffer max diff: {(ref_k_buffer - triton_k_buffer).abs().max().item()}")
        if not v_passed:
            print(f"  V buffer max diff: {(ref_v_buffer - triton_v_buffer).abs().max().item()}")
        return False
    
    return True


if __name__ == "__main__":
    print("Testing Triton KV cache operations...")
    all_passed = True
    all_passed &= test_gather_kv_and_permute()
    all_passed &= test_permute_and_scatter_kv()
    
    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
