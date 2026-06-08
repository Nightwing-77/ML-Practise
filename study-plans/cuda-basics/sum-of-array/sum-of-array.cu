#include <cuda_runtime.h>

__global__ void sum_kernel(const float* input, float* result, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float Track[256];

    Track[threadIdx.x] =
        (idx < N) ? input[idx] : 0.0f;

    __syncthreads();

    for(int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1)
    {
        if(threadIdx.x < stride)
        {
            Track[threadIdx.x] +=
                Track[threadIdx.x + stride];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        atomicAdd(result, Track[0]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(result, 0, sizeof(float));
    sum_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}
