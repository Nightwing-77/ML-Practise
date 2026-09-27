import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    N, D,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Identify which row (token position N) this program instance handles
    row_idx = tl.program_id(0)

    # 2. Create offsets for the j-th channel pairs [0, 1, 2, ..., BLOCK_SIZE - 1]
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < (D // 2)

    # 3. Compute memory pointer offsets for even (2*j) and odd (2*j + 1) elements
    row_x_offset = row_idx * D
    x_even_ptrs = x_ptr + row_x_offset + (2 * j_offsets)
    x_odd_ptrs  = x_ptr + row_x_offset + (2 * j_offsets + 1)

    # 4. Compute memory pointer offsets for the precomputed cos and sin tables
    row_trig_offset = row_idx * (D // 2)
    cos_ptrs = cos_ptr + row_trig_offset + j_offsets
    sin_ptrs = sin_ptr + row_trig_offset + j_offsets

    # 5. Load the values from global GPU RAM into SRAM registers
    x_even = tl.load(x_even_ptrs, mask=mask)
    x_odd  = tl.load(x_odd_ptrs, mask=mask)
    cos_val = tl.load(cos_ptrs, mask=mask)
    sin_val = tl.load(sin_ptrs, mask=mask)

    # 6. Perform the 2D rotation math
    out_even = x_even * cos_val - x_odd * sin_val
    out_odd  = x_even * sin_val + x_odd * cos_val

    # 7. Write the transformed values into the output tensor pointers
    out_even_ptrs = out_ptr + row_x_offset + (2 * j_offsets)
    out_odd_ptrs  = out_ptr + row_x_offset + (2 * j_offsets + 1)

    tl.store(out_even_ptrs, out_even, mask=mask)
    tl.store(out_odd_ptrs, out_odd, mask=mask)


def solve(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the RoPE kernel: rotate pairs of channels with per-row (cos, sin) tables."""
    N, D = x.shape
    BLOCK_SIZE = 1
    while BLOCK_SIZE < D // 2:
        BLOCK_SIZE *= 2
    if BLOCK_SIZE < 1:
        BLOCK_SIZE = 1
    grid = (N,)
    rope_kernel[grid](x, cos, sin, out, N, D, BLOCK_SIZE=BLOCK_SIZE)