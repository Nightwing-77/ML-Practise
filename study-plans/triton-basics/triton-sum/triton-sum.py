import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid_m=tl.program_id(0)
    offset_m=pid_m*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE)
    x_ptrs=x_ptr+offset_m
    x_mask=offset_m<n
    X=tl.load(x_ptrs,x_mask)
    out=tl.sum(X)
    tl.atomic_add(out_ptr,out)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch sum_kernel on the provided tensors."""
    n = x.numel()
    out.zero_()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    sum_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)