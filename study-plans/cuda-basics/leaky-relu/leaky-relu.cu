#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float alpha, int N) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index<N){
        if(input[index]>0){
            output[index]=input[index];
        } else{
            output[index]=input[index]*alpha;
        }
    }
    
}

extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}