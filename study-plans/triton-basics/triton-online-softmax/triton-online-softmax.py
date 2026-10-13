import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(
    x_ptr, out_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    # similar to pid * block size + arange , just that n is row size and we're doing it row wise
    m = float("-inf")
    l = 0.0
    for i in range(0,N,BLOCK_SIZE):
        cols=i+tl.arange(0,BLOCK_SIZE)
        mask=cols<N
        offset=row*N+cols
        x_ptrs=x_ptr+offset   
        x=tl.load(x_ptrs,mask, other=-float("inf"))
        chunk_max=tl.max(x,axis=0)
        m_new = tl.maximum(m, chunk_max)
        chunk_sum=tl.sum(tl.exp(x-m_new),axis=0)
        l = l * tl.exp(m - m_new) + chunk_sum
        m=m_new
    for j in range(0,N,BLOCK_SIZE):   
        cols=j+tl.arange(0,BLOCK_SIZE)
        mask=cols<N
        offsets=row*N+cols
        x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))

        probs = tl.exp(x - m) / l
        tl.store(out_ptr + offsets, probs, mask=mask)
        

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the online softmax kernel row-wise."""
    M, N = x.shape
    BLOCK_SIZE = 1024
    grid = (M,)
    online_softmax_kernel[grid](x, out, M, N, BLOCK_SIZE=BLOCK_SIZE)