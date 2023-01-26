// ref: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

#include <stdint.h>
#include <cstddef>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_helper.hpp"
#include "reduction_gpu.hpp"
#include "reduction.hpp"


using namespace std;



__global__ void reduce_gpu_v4_kernel(const uint32_t *g_data, uint32_t*g_out){
    extern __shared__ uint32_t sdata[];
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x*blockDim.x*2 + threadIdx.x;
    sdata[tid] = g_data[i] + g_data[i+blockDim.x];

    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    } 
}

/*
uint32_t reduce_gpu_v4(const uint32_t *d_data, uint32_t *h_tmp_out, uint32_t *d_tmp_out, size_t n, int n_blocks, int n_threads, cudaDeviceProp &prop) {
    size_t n_blocks_v4 = n_blocks/2;
    reduce_gpu_v4_kernel<<<n_blocks_v4, n_threads, sizeof(uint32_t)*n_threads>>>(d_data, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(uint32_t)*n_blocks_v4, cudaMemcpyDefault));
    uint32_t ret = reduce(h_tmp_out, n_blocks_v4);
    return ret;
}

*/