import torch
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols,
                     BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    offset = pid * x_row_stride + cols
    mask = cols < n_cols

    x = tl.load(x_ptr + offset, mask, other=0.0)

    x_max = tl.max(tl.where(mask, x, float('-inf')), axis=0)

    exp_x = tl.exp(x - x_max)
    exp_x = tl.where(mask, exp_x, 0.0)

    exp_sum = tl.sum(exp_x, axis=0)

    result = tl.log(exp_sum) + x_max

    tl.store(out_ptr + pid, result)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch logsumexp_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    logsumexp_kernel[grid](
        x, out, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )