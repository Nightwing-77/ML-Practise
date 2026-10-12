import torch
import triton
import triton.language as tl


@triton.jit
def vectorized_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid_m=tl.program_id(0)
    offset_m=pid_m*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE)
    x_ptrs=x_ptr+offset_m
    y_ptrs=y_ptr+offset_m
    out_ptrs=out_ptr+offset_m
    mask=offset_m<n
    x=tl.load(x_ptrs,mask)
    y=tl.load(y_ptrs,mask)
    tl.store(out_ptrs,x+y,mask)


def solve(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Launch vectorized_add_kernel with a large BLOCK_SIZE for long arrays."""
    n = x.numel()
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    vectorized_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)