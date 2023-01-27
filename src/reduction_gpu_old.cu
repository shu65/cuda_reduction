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

template <uint32_t kBlockSize>
__device__ void warpReduce(volatile int *sdata, int tid)
{
    assert(warpSize == 32);
    int v = sdata[tid];
    if (kBlockSize >= 64)
    {
        v += sdata[tid + 32];
        __syncwarp();
        sdata[tid] = v;
        __syncwarp();
    }
    if (kBlockSize >= 32)
    {
        v += sdata[tid + 16];
        __syncwarp();
        sdata[tid] = v;
        __syncwarp();
    }
    if (kBlockSize >= 16)
    {
        v += sdata[tid + 8];
        __syncwarp();
        sdata[tid] = v;
        __syncwarp();
    }
    if (kBlockSize >= 8)
    {
        v += sdata[tid + 4];
        __syncwarp();
        sdata[tid] = v;
        __syncwarp();
    }
    if (kBlockSize >= 4)
    {
        v += sdata[tid + 2];
        __syncwarp();
        sdata[tid] = v;
        __syncwarp();
    }
    if (kBlockSize >= 2)
    {
        v += sdata[tid + 1];
        __syncwarp();
        sdata[tid] = v;
        __syncwarp();
    }
}

__global__ void reduce_gpu_old_v1_kernel(const int *g_in, size_t n, int *g_out)
{
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        sdata[tid] = g_in[i];
    }
    else
    {
        sdata[tid] = 0;
    }

    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_out[blockIdx.x] = sdata[0];
    }
}

int reduce_gpu_old_v1(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads)
{
    reduce_gpu_old_v1_kernel<<<n_blocks, n_threads, sizeof(int) * n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int) * n_blocks, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks);
    return ret;
}

__global__ void reduce_gpu_old_v2_kernel(const int *g_in, size_t n, int *g_out)
{
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        sdata[tid] = g_in[i];
    }
    else
    {
        sdata[tid] = 0;
    }

    __syncthreads();

    for (uint32_t s = 1; s < blockDim.x; s *= 2)
    {
        uint32_t index = 2 * s * tid;
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_out[blockIdx.x] = sdata[0];
    }
}

int reduce_gpu_old_v2(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads)
{
    reduce_gpu_old_v2_kernel<<<n_blocks, n_threads, sizeof(int) * n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int) * n_blocks, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks);
    return ret;
}

__global__ void reduce_gpu_old_v3_kernel(const int *g_in, size_t n, int *g_out)
{
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        sdata[tid] = g_in[i];
    }
    else
    {
        sdata[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_out[blockIdx.x] = sdata[0];
    }
}

int reduce_gpu_old_v3(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads)
{
    reduce_gpu_old_v3_kernel<<<n_blocks, n_threads, sizeof(int) * n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int) * n_blocks, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks);
    return ret;
}

__global__ void reduce_gpu_old_v4_kernel(const int *g_in, size_t n, int *g_out)
{
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int sum_value = 0;
    if (i < n)
    {
        sum_value = g_in[i];
    }
    if ((i + blockDim.x) < n)
    {
        sum_value += g_in[i + blockDim.x];
    }
    sdata[tid] = sum_value;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        g_out[blockIdx.x] = sdata[0];
    }
}

int reduce_gpu_old_v4(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads)
{
    reduce_gpu_old_v4_kernel<<<n_blocks, n_threads, sizeof(int) * n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int) * n_blocks, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks);
    return ret;
}

template <uint32_t kBlockSize>
__global__ void reduce_gpu_old_v5_kernel(const int *g_in, size_t n, int *g_out)
{
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int sum_value = 0;
    if (i < n)
    {
        sum_value = g_in[i];
    }
    if ((i + blockDim.x) < n)
    {
        sum_value += g_in[i + blockDim.x];
    }
    sdata[tid] = sum_value;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > warpSize; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < warpSize)
    {
        warpReduce<kBlockSize>(sdata, tid);
    }
    if (tid == 0)
    {
        g_out[blockIdx.x] = sdata[0];
    }
}

int reduce_gpu_old_v5(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads)
{
    assert(512 == n_threads);
    reduce_gpu_old_v5_kernel<512><<<n_blocks, n_threads, sizeof(int) * n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int) * n_blocks, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks);
    return ret;
}

template <uint32_t kBlockSize>
__global__ void reduce_gpu_old_v6_kernel(const int *g_in, size_t n, int *g_out)
{
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int sum_value = 0;
    if (i < n)
    {
        sum_value = g_in[i];
    }
    if ((i + blockDim.x) < n)
    {
        sum_value += g_in[i + blockDim.x];
    }
    sdata[tid] = sum_value;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 512; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (kBlockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (kBlockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (kBlockSize >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    assert(warpSize == 32);
    if (tid < 32)
    {
        warpReduce<kBlockSize>(sdata, tid);
    }
    if (tid == 0)
    {
        g_out[blockIdx.x] = sdata[0];
    }
}

int reduce_gpu_old_v6(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads)
{
    assert(512 == n_threads);
    reduce_gpu_old_v6_kernel<512><<<n_blocks, n_threads, sizeof(int) * n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int) * n_blocks, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks);
    return ret;
}

template <uint32_t kBlockSize>
__global__ void reduce_gpu_old_v7_kernel(const int *g_in, size_t n, int *g_out)
{
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int grid_size = 2 * blockDim.x * gridDim.x;
    int sum_value = 0;

    while (i < n)
    {
        sum_value += g_in[i];
        if ((i + kBlockSize) < n)
        {
            sum_value += g_in[i + kBlockSize];
        }
        i += grid_size;
    }

    sdata[tid] = sum_value;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 512; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (kBlockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (kBlockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (kBlockSize >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    assert(warpSize == 32);
    if (tid < 32)
    {
        warpReduce<kBlockSize>(sdata, tid);
    }
    if (tid == 0)
    {
        g_out[blockIdx.x] = sdata[0];
    }
}

int reduce_gpu_old_v7(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads)
{
    size_t n_blocks_v7 = 128;
    assert(512 == n_threads);
    reduce_gpu_old_v7_kernel<512><<<n_blocks_v7, n_threads, sizeof(int) * n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int) * n_blocks_v7, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks_v7);
    return ret;
}