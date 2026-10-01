import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid=tl.program_id(0)
    block_start=pid*BLOCK_SIZE
    offset=block_start+tl.arange(0,BLOCK_SIZE)
    x_ptrs=x_ptr+offset
    mask=offset<n
    x=tl.load(x_ptrs,mask)
    y=tl.where(x>0,x,0)
    y_ptrs=out_ptr+offset
    tl.store(y_ptrs,y,mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch relu_kernel: out = max(x, 0)."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    relu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)