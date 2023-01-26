// ref: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

#include <stdint.h>
#include <cstddef>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_helper.hpp"
#include "reduction_gpu.hpp"
#include "reduction.hpp"


using namespace std;

/*
__device__ void warpReduce(volatile uint32_t* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}


__global__ void reduce_gpu_v5_kernel(const uint32_t *g_data, const int warp_size, uint32_t*g_out){
    extern __shared__ uint32_t sdata[];
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x*blockDim.x*2 + threadIdx.x;
    sdata[tid] = g_data[i] + g_data[i+blockDim.x];

    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>warp_size; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < warp_size) {
        warpReduce(sdata, warp_size, tid);
    }
    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    } 
}


uint32_t reduce_gpu_v5(const uint32_t *d_data, uint32_t *h_tmp_out, uint32_t *d_tmp_out, size_t n, int n_blocks, int n_threads, cudaDeviceProp &prop) {
    size_t n_blocks_v5 = n_blocks/2;
    reduce_gpu_v5_kernel<<<n_blocks_v5, n_threads, sizeof(uint32_t)*n_threads>>>(d_data, prop.warpSize, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(uint32_t)*n_blocks_v5, cudaMemcpyDefault));
    uint32_t ret = reduce(h_tmp_out, n_blocks_v5);
    return ret;
}
*/
