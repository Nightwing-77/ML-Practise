import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, gamma_ptr, out_ptr,
    stride_x_row, stride_out_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid=tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset=pid*stride_x_row+ cols
    mask=cols<N
    x=tl.load(x_ptr+offset,mask,other=0.0)
    x_sum=tl.sum(x*x,axis=0)
    rms=tl.rsqrt(x_sum/N+eps)
    out=x*rms
    gamma = tl.load(gamma_ptr + cols, mask, other=0.0)
    out=out*gamma
    tl.store(out_ptr + pid * stride_out_row + cols,out,mask)



def solve(x: torch.Tensor, gamma: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    """Launch the RMSNorm forward kernel: one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    rmsnorm_fwd_kernel[grid](
        x, gamma, out,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )