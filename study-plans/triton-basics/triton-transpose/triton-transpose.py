import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    a_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m=tl.program_id(0)
    pid_n=tl.program_id(1)

    offset_m=pid_m*BLOCK_M+tl.arange(0,BLOCK_M)
    offset_n=pid_n*BLOCK_N+tl.arange(0,BLOCK_N)

    A_ptr=a_ptr+offset_m[:,None]*stride_am+offset_n[None,:]*stride_an
    A_mask=(offset_m[:,None]<M) & (offset_n[None,:]<N)
    A=tl.load(A_ptr,A_mask)
    out_ptrs = out_ptr + offset_n[:, None] * stride_om + offset_m[None, :] * stride_on
    out_mask=(offset_m[None,:]<M) & (offset_n[:,None]<N)
    tl.store(out_ptrs,tl.trans(A),out_mask)


def solve(A: torch.Tensor, out: torch.Tensor) -> None:
    """Launch transpose_kernel: out[j, i] = A[i, j]."""
    M, N = A.shape
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_kernel[grid](
        A, out,
        M, N,
        A.stride(0), A.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )