import torch

from sglang.srt.paras.triton_ops import gather_kv_and_permute_triton, permute_and_scatter_kv_triton

def gather_kv_and_permute_torch(
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    local_kcache = k_buffer[indices]
    local_vcache = v_buffer[indices]
    local_kvcache = torch.stack([local_kcache, local_vcache], dim=0).view(2, -1, k_buffer.shape[1], k_buffer.shape[2])
    permuted_local_kvcache = local_kvcache.permute(2, 0, 1, 3).contiguous().flatten()
    return permuted_local_kvcache

def permute_and_scatter_kv_torch(
    permuted_kvcache: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    indices: torch.Tensor,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
) -> None:
    permuted_gathered_kvcache = permuted_kvcache.view(num_tokens, 2, num_heads, head_dim)
    permuted_gathered_kvcache = permuted_gathered_kvcache.permute(1, 0, 2, 3).contiguous()
    k_buffer[indices] = permuted_gathered_kvcache[0]
    v_buffer[indices] = permuted_gathered_kvcache[1]

# NOTE: current triton kernel is much slower than naive torch implementatio
def gather_kv_and_permute(
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor = None,
    num_blocks: int = None,
) -> torch.Tensor:
    return gather_kv_and_permute_torch(k_buffer, v_buffer, indices)

# NOTE: current triton kernel is much slower than naive torch implementation
def permute_and_scatter_kv(
    permuted_kvcache: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    indices: torch.Tensor,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    num_blocks: int = None,
) -> None:
    return permute_and_scatter_kv_torch(permuted_kvcache, k_buffer, v_buffer, indices, num_tokens, num_heads, head_dim)

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
    op_output = gather_kv_and_permute(k_buffer, v_buffer, indices)
    
    # Compare
    if torch.allclose(ref_output, op_output, rtol=1e-3, atol=1e-3):
        print("✓ gather_kv_and_permute: PASSED")
    else:
        print("✗ gather_kv_and_permute: FAILED")
        print(f"  Max diff: {(ref_output - op_output).abs().max().item()}")
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
    op_k_buffer = torch.zeros(pool_size, num_heads, head_dim, dtype=torch.float16, device="cuda")
    op_v_buffer = torch.zeros(pool_size, num_heads, head_dim, dtype=torch.float16, device="cuda")
    permute_and_scatter_kv(
        permuted_kvcache, op_k_buffer, op_v_buffer, indices,
        num_tokens, num_heads, head_dim
    )
    
    # Compare
    k_passed = torch.allclose(ref_k_buffer, op_k_buffer, rtol=1e-3, atol=1e-3)
    v_passed = torch.allclose(ref_v_buffer, op_v_buffer, rtol=1e-3, atol=1e-3)
    
    if k_passed and v_passed:
        print("✓ permute_and_scatter_kv: PASSED")
    else:
        print("✗ scatter_kv_from_permuted: FAILED")
        if not k_passed:
            print(f"  K buffer max diff: {(ref_k_buffer - op_k_buffer).abs().max().item()}")
        if not v_passed:
            print(f"  V buffer max diff: {(ref_v_buffer - op_v_buffer).abs().max().item()}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing Triton KV cache operations...")
    all_passed = True
    all_passed &= test_gather_kv_and_permute()
    all_passed &= test_scatter_kv_from_permuted()
    
    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
