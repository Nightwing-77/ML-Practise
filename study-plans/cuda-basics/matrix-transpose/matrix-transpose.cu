#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* A, float* B, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    // FIX: x must be less than N (columns), y must be less than M (rows)
    if (x < N && y < M) {
        int input_num = y * N + x; 
        int output_num = x * M + y;
        B[output_num] = A[input_num];
    }
}

extern "C" void solve(const float* A, float* B, int M, int N) {
    dim3 threads(16, 16);
    // Your block dimensions are perfectly fine here! 
    // (N + 15) / 16 sets blocks.x according to columns (N)
    // (M + 15) / 16 sets blocks.y according to rows (M)
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    
    matrix_transpose_kernel<<<blocks, threads>>>(A, B, M, N);
    cudaDeviceSynchronize();
}