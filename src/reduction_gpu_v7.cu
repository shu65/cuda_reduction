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

template <uint32_t blockSize>
__device__ void warpReduce(volatile uint32_t* sdata, int warp_size, int tid) {
    assert(warp_size == 32);
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1]; 
}

template <uint32_t blockSize>
__global__ void reduce_gpu_v7_kernel(const uint32_t *g_data, const size_t n, const int warp_size, uint32_t*g_out){
    extern __shared__ uint32_t sdata[];
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x*blockDim.x*2 + threadIdx.x;
    uint32_t grid_size = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { 
        sdata[tid] += g_data[i] + g_data[i+blockSize];
         i += grid_size; 
    }
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>512; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) {
        warpReduce<blockSize>(sdata, warp_size, tid);
    }
    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    } 
}

/*
uint32_t reduce_gpu_v7(const uint32_t *d_data, uint32_t *h_tmp_out, uint32_t *d_tmp_out, size_t n, int n_blocks, int n_threads, cudaDeviceProp &prop) {
    size_t n_blocks_v7 = 64;
    assert(512==n_threads);
    reduce_gpu_v7_kernel<512><<<n_blocks_v7, n_threads, sizeof(uint32_t)*n_threads>>>(d_data, n, prop.warpSize, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(uint32_t)*n_blocks_v7, cudaMemcpyDefault));
    uint32_t ret = reduce(h_tmp_out, n_blocks_v7);
    return ret;
}
*/
