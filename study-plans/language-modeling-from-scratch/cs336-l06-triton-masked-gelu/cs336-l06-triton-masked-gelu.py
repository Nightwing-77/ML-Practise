import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def masked_gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Returns nothing; writes GELU values through out_ptr.
    """
    pid=tl.program_id(0)
    offset=pid*BLOCK_SIZE+tl.arange(0,BLOCK_SIZE)
    x_ptrs=x_ptr+offset
    mask=offset<n_elements
    x=tl.load(x_ptrs,mask)
    y = 0.5*x*(1+tl.extra.cuda.libdevice.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
    tl.store(out_ptr+offset,y,mask)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    if x.shape != out.shape or x.device != out.device or x.dtype != out.dtype:
        raise ValueError("x and out must have matching shape, device, and dtype")
    if not x.is_cuda or x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("x must be a supported CUDA tensor")
    if not x.is_contiguous() or not out.is_contiguous():
        raise ValueError("x and out must be contiguous")
    if x.numel() == 0:
        return
    kernel_x = x
    kernel_out = out
    if x.dtype == torch.bfloat16:
        kernel_x = x.to(torch.float32)
        kernel_out = torch.empty_like(x, dtype=torch.float32)
    block_size = 256
    grid = (triton.cdiv(x.numel(), block_size),)
    masked_gelu_kernel[grid](kernel_x, kernel_out, x.numel(), BLOCK_SIZE=block_size)
    if kernel_out is not out:
        out.copy_(kernel_out)
