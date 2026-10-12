import torch
import triton
import triton.language as tl


@triton.jit
def fma_kernel(x_ptr, y_ptr, out_ptr, n, a, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid_m=tl.program_id(0)
    offset_m=pid_m*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE)
    x_ptrs=x_ptr+offset_m
    y_ptrs=y_ptr+offset_m
    out_ptrs=out_ptr+offset_m
    mask=offset_m<n
    X=tl.load(x_ptrs,mask)
    Y=tl.load(y_ptrs,mask)
    out=X*a + Y
    tl.store(out_ptrs,out,mask)

def solve(a: float, x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Launch fma_kernel: out = a * x + y."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fma_kernel[grid](x, y, out, n, a, BLOCK_SIZE=BLOCK_SIZE)