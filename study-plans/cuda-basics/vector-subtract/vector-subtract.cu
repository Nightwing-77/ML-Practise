#include <cuda_runtime.h>

__global__ void vector_sub(const float* A, const float* B, float* C, int N) {
    int index=blockIdx.x*blockDim.x + threadIdx.x;
    if (index<N){
        C[index]=A[index]-B[index];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_sub<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}