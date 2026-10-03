import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    a_ptr, x_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m=tl.program_id(0)
    offset_m=pid_m*BLOCK_M + tl.arange(0,BLOCK_M)
    offset_n=tl.arange(0,BLOCK_N)
    a_ptrs=a_ptr+ offset_m[:,None]*stride_am + offset_n[None,:]*stride_an
    x_ptrs=x_ptr+offset_n
    acc=tl.zeros((BLOCK_M,),tl.float32)
    for i in range(0,tl.cdiv(N,BLOCK_N)):
        offsets_x=i*BLOCK_N+offset_n
        mask_A=(offset_m[:,None]<M) & (offsets_x[None,:]<N)
        mask_X=offsets_x<N
        A=tl.load(a_ptrs,mask_A,other=0.0)
        X=tl.load(x_ptrs,mask_X,other=0.0)
        acc += tl.sum(A * X[None, :], axis=1)
        a_ptrs=a_ptrs+BLOCK_N*stride_an
        x_ptrs=x_ptrs+BLOCK_N

    out_mask=offset_m<M
    tl.store(out_ptr+offset_m,acc,out_mask)


def solve(A: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch gemv_kernel: out = A @ x."""
    M, N = A.shape
    BLOCK_M = 32
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M),)
    gemv_kernel[grid](
        A, x, out,
        M, N,
        A.stride(0), A.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )