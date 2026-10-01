import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid=tl.program_id(0)
    block_start=BLOCK_SIZE*pid
    offset=block_start+tl.arange(0,BLOCK_SIZE)
    x_ptrs=x_ptr+offset
    mask=offset<n
    x=tl.load(x_ptrs,mask)
    y=x/(1+tl.exp(-x))
    y_ptrs=out_ptr+offset
    tl.store(y_ptrs,y,mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch silu_kernel: out = x / (1 + exp(-x))."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    silu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)