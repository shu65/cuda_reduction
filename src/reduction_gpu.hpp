#pragma once
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>

int reduce_gpu_v1(const int *d_data, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads);
int reduce_gpu_v2(const int *d_data, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads);
int reduce_gpu_v3(const int *d_data, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads);
int reduce_gpu_v4(const int *d_data, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads);
//int reduce_gpu_v5(const int *d_data, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads);
int reduce_gpu_v6(const int *d_data, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads);
int reduce_gpu_v7(const int *d_data, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads);
int reduce_gpu_v8(const int *d_data, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads);
int reduce_gpu_v9(const int *d_data, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads);
int reduce_gpu_v10(const int *d_data, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads);
