#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {

    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<N){
        if(input[index]>0){
            output[index]=input[index];
        } else{
            output[index]=0;
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}