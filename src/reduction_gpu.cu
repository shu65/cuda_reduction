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


template <uint32_t kBlockSize>
__device__ __forceinline__ int warp_reduce_sum_with_warp_shuffle(unsigned int mask, int sum_value) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum_value += __shfl_down_sync(mask, sum_value, offset);
  }
  return sum_value;
}


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

template <uint32_t kBlockSize>
__global__ void reduce_gpu_v6_kernel(const int *g_in, size_t n, int*g_out){
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

    if (kBlockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } cg::sync(cta); }
    if (kBlockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } cg::sync(cta); }
    if (kBlockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } cg::sync(cta); }
    if (kBlockSize >= 64) {
        if (tid < 32) { sdata[tid] += sdata[tid + 32]; } cg::sync(cta); }
    if (kBlockSize >= 32) {
        if (tid < 16) { sdata[tid] += sdata[tid + 16]; } cg::sync(cta); }
    if (kBlockSize >= 16) {
        if (tid < 8) { sdata[tid] += sdata[tid + 8]; } cg::sync(cta); }
    if (kBlockSize >= 8) {
        if (tid < 4) { sdata[tid] += sdata[tid + 4]; } cg::sync(cta); }
    if (kBlockSize >= 4) {
        if (tid < 2) { sdata[tid] += sdata[tid + 2]; } cg::sync(cta); }
    if (kBlockSize >= 2) {
        if (tid < 1) { sdata[tid] += sdata[tid + 1]; } cg::sync(cta); }
    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    } 
}

int reduce_gpu_v6(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads) {
    assert(512==n_threads);
    reduce_gpu_v6_kernel<512><<<n_blocks, n_threads, sizeof(int)*n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int)*n_blocks, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks);
    return ret;
}


template <uint32_t kBlockSize>
__global__ void reduce_gpu_v7_kernel(const int *g_in, size_t n, int*g_out){
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    int grid_size = 2*blockDim.x * gridDim.x;
    int sum_value = 0;

    while (i < n) {
      sum_value += g_in[i];
      if ((i + kBlockSize) < n) {
        sum_value += g_in[i + kBlockSize];
      }
      i += grid_size;
    }
    sdata[tid] = sum_value;
    cg::sync(cta);

    if (kBlockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } cg::sync(cta); }
    if (kBlockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } cg::sync(cta); }
    if (kBlockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } cg::sync(cta); }
    if (kBlockSize >= 64) {
        if (tid < 32) { sdata[tid] += sdata[tid + 32]; } cg::sync(cta); }
    if (kBlockSize >= 32) {
        if (tid < 16) { sdata[tid] += sdata[tid + 16]; } cg::sync(cta); }
    if (kBlockSize >= 16) {
        if (tid < 8) { sdata[tid] += sdata[tid + 8]; } cg::sync(cta); }
    if (kBlockSize >= 8) {
        if (tid < 4) { sdata[tid] += sdata[tid + 4]; } cg::sync(cta); }
    if (kBlockSize >= 4) {
        if (tid < 2) { sdata[tid] += sdata[tid + 2]; } cg::sync(cta); }
    if (kBlockSize >= 2) {
        if (tid < 1) { sdata[tid] += sdata[tid + 1]; } cg::sync(cta); }
    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    } 
}

int reduce_gpu_v7(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads) {
    size_t n_blocks_v7 = 128;
    assert(512==n_threads);
    reduce_gpu_v7_kernel<512><<<n_blocks_v7, n_threads, sizeof(int)*n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int)*n_blocks_v7, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks_v7);
    return ret;
}

template <uint32_t kBlockSize>
__global__ void reduce_gpu_v8_kernel(const int *g_in, size_t n, int*g_out){
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    int grid_size = 2*blockDim.x * gridDim.x;
    int sum_value = 0;

    while (i < n) {
      sum_value += g_in[i];
      if ((i + kBlockSize) < n) {
        sum_value += g_in[i + kBlockSize];
      }
      i += grid_size;
    }
    sdata[tid] = sum_value;
    cg::sync(cta);

    if (kBlockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } cg::sync(cta); }
    if (kBlockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } cg::sync(cta); }
    if (kBlockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } cg::sync(cta); }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    if (cta.thread_rank() < 32) {
    if (kBlockSize >= 64) {
        if (tid < 32) { sdata[tid] += sdata[tid + 32]; } tile32.sync(); }
    if (kBlockSize >= 32) {
        if (tid < 16) { sdata[tid] += sdata[tid + 16]; } tile32.sync(); }
    if (kBlockSize >= 16) {
        if (tid < 8) { sdata[tid] += sdata[tid + 8]; } tile32.sync(); }
    if (kBlockSize >= 8) {
        if (tid < 4) { sdata[tid] += sdata[tid + 4]; } tile32.sync(); }
    if (kBlockSize >= 4) {
        if (tid < 2) { sdata[tid] += sdata[tid + 2]; } tile32.sync(); }
    if (kBlockSize >= 2) {
        if (tid < 1) { sdata[tid] += sdata[tid + 1]; } tile32.sync(); }
    }
    if (tid == 0) {
        g_out[blockIdx.x] = sdata[0];
    } 
}

int reduce_gpu_v8(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads) {
    size_t n_blocks_v7 = 128;
    assert(512==n_threads);
    reduce_gpu_v8_kernel<512><<<n_blocks_v7, n_threads, sizeof(int)*n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int)*n_blocks_v7, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks_v7);
    return ret;
}

template <uint32_t kBlockSize>
__global__ void reduce_gpu_v9_kernel(const int *g_in, size_t n, int*g_out){
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    int grid_size = 2*blockDim.x * gridDim.x;
    int sum_value = 0;

    while (i < n) {
      sum_value += g_in[i];
      if ((i + kBlockSize) < n) {
        sum_value += g_in[i + kBlockSize];
      }
      i += grid_size;
    }
    sdata[tid] = sum_value;
    cg::sync(cta);

    if (kBlockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } cg::sync(cta); }
    if (kBlockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } cg::sync(cta); }
    if (kBlockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } cg::sync(cta); }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    if (cta.thread_rank() < 32) {
        sum_value = sdata[tid];
        if (kBlockSize >= 64) {
            sum_value += sdata[tid + 32];
        }
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
            sum_value += tile32.shfl_down(sum_value, offset);
        }
    }
    if (tid == 0) {
        g_out[blockIdx.x] = sum_value;
    } 
}

int reduce_gpu_v9(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads) {
    size_t n_blocks_v7 = 128;
    assert(512==n_threads);
    reduce_gpu_v9_kernel<512><<<n_blocks_v7, n_threads, sizeof(int)*n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int)*n_blocks_v7, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks_v7);
    return ret;
}

template <uint32_t kBlockSize>
__global__ void reduce_gpu_v10_kernel(const int *g_in, size_t n, int*g_out){
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    int grid_size = 2*blockDim.x * gridDim.x;
    unsigned int maskLength = (kBlockSize & 31);  // 31 = warpSize-1
    maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
    const unsigned int mask = (0xffffffff) >> maskLength;
    int sum_value = 0;

    while (i < n) {
      sum_value += g_in[i];
      if ((i + kBlockSize) < n) {
        sum_value += g_in[i + kBlockSize];
      }
      i += grid_size;
    }
    sdata[tid] = sum_value;
    cg::sync(cta);

    if (kBlockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } cg::sync(cta); }
    if (kBlockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } cg::sync(cta); }
    if (kBlockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } cg::sync(cta); }

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    if (cta.thread_rank() < 32) {
        sum_value = sdata[tid];
        if (kBlockSize >= 64) {
            sum_value += sdata[tid + 32];
        }
        sum_value = warp_reduce_sum_with_warp_shuffle<kBlockSize>(mask, sum_value);
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_value += __shfl_down_sync(mask, sum_value, offset);
        }
    }
    if (tid == 0) {
        g_out[blockIdx.x] = sum_value;
    } 
}

int reduce_gpu_v10(const int *d_in, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads) {
    size_t n_blocks_v7 = 128;
    assert(512==n_threads);
    reduce_gpu_v10_kernel<512><<<n_blocks_v7, n_threads, sizeof(int)*n_threads>>>(d_in, n, d_tmp_out);
    checkCudaErrors(cudaMemcpy(h_tmp_out, d_tmp_out, sizeof(int)*n_blocks_v7, cudaMemcpyDefault));
    int ret = reduce(h_tmp_out, n_blocks_v7);
    return ret;
}