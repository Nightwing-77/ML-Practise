#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int N) {

    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<N){
        float x=input[index];
        output[index]=(expf(x)-expf(-x))/(expf(x)+expf(-x));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}