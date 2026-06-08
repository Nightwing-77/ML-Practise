#include <cuda_runtime.h>

__global__ void matrix_add_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (y < M && x < N)
    {
        int index = y * N + x;
        C[index] = A[index] + B[index];
    }
}

extern "C" void solve(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N)
{
    dim3 threads(16, 16);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    matrix_add_kernel<<<blocks, threads>>>(
        A, B, C, M, N
    );

    cudaDeviceSynchronize();
}