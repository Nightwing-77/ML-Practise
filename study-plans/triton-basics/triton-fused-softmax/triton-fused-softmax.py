import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):

    pid=tl.program_id(0)
    cols=tl.arange(0,BLOCK_SIZE)
    offset=pid*x_row_stride+cols
    x_ptrs=x_ptr+offset
    mask=cols<n_cols
    x=tl.load(x_ptrs,mask,other=0.0)
    max= tl.max(tl.where(mask, x, float('-inf')), axis=0)
    exp_x=tl.exp(x-max)
    exp_x = tl.where(mask, exp_x, 0.0)
    sum=tl.sum(exp_x)
    y = exp_x / sum
    tl.store(out_ptr+pid*out_row_stride+cols,y,mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )