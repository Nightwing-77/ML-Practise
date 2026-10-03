import torch
import triton
import triton.language as tl


@triton.jit
def fused_matmul_bias_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
    b_ptrs = b_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(K, BLOCK_K)):
        offsets_k = i * BLOCK_K + offset_k

        mask_a = (offset_m[:, None] < M) & (offsets_k[None, :] < K)
        mask_b = (offsets_k[:, None] < K) & (offset_n[None, :] < N)

        A = tl.load(a_ptrs, mask=mask_a, other=0.0)
        B = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc = tl.dot(A, B, acc, allow_tf32=False)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias_mask = offs_n < N
    bias_tile = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)

    acc = acc + bias_tile
    acc = tl.maximum(acc, 0.0)

    c_ptrs = c_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    mask_c = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def solve(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor, out: torch.Tensor) -> None:
    """Launch fused kernel: out = relu(A @ B + bias)."""
    M, K = A.shape
    K2, N = B.shape

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    fused_matmul_bias_relu_kernel[grid](
        A, B, bias, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )