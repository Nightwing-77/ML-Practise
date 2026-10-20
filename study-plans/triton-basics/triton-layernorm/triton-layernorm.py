import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_fwd_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    stride_x_row, stride_out_row,
    C, eps,
    BLOCK_SIZE: tl.constexpr,
):
    #doing this thing to get better at writing production kernels even tho ik our blocksize>N cause of how we launched it
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_x = x_ptr + pid * stride_x_row

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, C, BLOCK_SIZE):
        idx = offset + cols
        mask = idx < C
        x = tl.load(row_x + idx, mask=mask, other=0.0).to(tl.float32)
        acc += x
    mean = tl.sum(acc, axis=0) / C

    acc_ = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, C, BLOCK_SIZE):
        idx = offset + cols
        mask = idx < C
        x = tl.load(row_x + idx, mask=mask, other=0.0).to(tl.float32)
        d = tl.where(mask, x - mean, 0.0)
        acc_ += d * d
    variance = tl.sum(acc_, axis=0) / C
    rstd = tl.rsqrt(variance + eps)

    for offset in range(0, C, BLOCK_SIZE):
        idx = offset + cols
        mask = idx < C
        x = tl.load(row_x + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(gamma_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(beta_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + pid * stride_out_row + idx, y, mask=mask)



def solve(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    """Launch the LayerNorm forward kernel: one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    layernorm_fwd_kernel[grid](
        x, gamma, beta, out,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )