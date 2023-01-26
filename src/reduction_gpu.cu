// ref: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

#include <stdint.h>
#include <cstddef>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "cuda_helper.hpp"
#include "reduction_gpu.hpp"
#include "reduction.hpp"


using namespace std;
namespace cg = cooperative_groups;

/*
template <uint32_t blockSize>
___device__ void warpReduce(volatile int* sdata, int tid) {
    if (blockSize >= 64) mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      mySum += tile32.shfl_down(mySum, offset);
    }
}
*/
__global__ void reduce_gpu_v1_kernel(const int *g_in, size_t n, int*g_out){
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        sdata[tid] = g_in[i];
    } else {
        sdata[tid] = 0;
    }

    cg::sync(cta);

    for(int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        cg::sync(cta);
    }
    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    } 
}

int reduce_gpu_v1(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads) {
    reduce_gpu_v1_kernel<<<n_blocks, n_threads, sizeof(int)*n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int)*n_blocks, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks);
    return ret;
}

__global__ void reduce_gpu_v2_kernel(const int *g_in, size_t n, int*g_out){
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        sdata[tid] = g_in[i];
    } else {
        sdata[tid] = 0;
    }

    cg::sync(cta);

    for(uint32_t s=1; s < blockDim.x; s *= 2) {
        uint32_t index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        cg::sync(cta);
    }
    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    } 
}

int reduce_gpu_v2(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads) {
    reduce_gpu_v2_kernel<<<n_blocks, n_threads, sizeof(int)*n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int)*n_blocks, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks);
    return ret;
}


__global__ void reduce_gpu_v3_kernel(const int *g_in, size_t n, int*g_out){
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        sdata[tid] = g_in[i];
    } else {
        sdata[tid] = 0;
    }

    cg::sync(cta);

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        cg::sync(cta);
    }
    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    } 
}

int reduce_gpu_v3(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads) {
    reduce_gpu_v3_kernel<<<n_blocks, n_threads, sizeof(int)*n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int)*n_blocks, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks);
    return ret;
}

__global__ void reduce_gpu_v4_kernel(const int *g_in, size_t n, int*g_out){
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    int sum_value = 0;
    if (i < n) {
        sum_value = g_in[i];
    }
    if ((i + blockDim.x) < n) {
        sum_value += g_in[i + blockDim.x];
    }
    sdata[tid] = sum_value;
    cg::sync(cta);

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        cg::sync(cta);
    }
    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    } 
}

int reduce_gpu_v4(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads) {
    reduce_gpu_v4_kernel<<<n_blocks, n_threads, sizeof(int)*n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int)*n_blocks, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks);
    return ret;
}

__global__ void reduce_gpu_v5_kernel(const int *g_in, size_t n, int*g_out){
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    int sum_value = 0;
    if (i < n) {
        sum_value = g_in[i];
    }
    if ((i + blockDim.x) < n) {
        sum_value += g_in[i + blockDim.x];
    }
    sdata[tid] = sum_value;
    cg::sync(cta);

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        cg::sync(cta);
    }
    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    } 
}

int reduce_gpu_v5(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads) {
    reduce_gpu_v5_kernel<<<n_blocks, n_threads, sizeof(int)*n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int)*n_blocks, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks);
    return ret;
}