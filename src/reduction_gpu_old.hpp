#pragma once
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>

int reduce_gpu_old_v1(const int *d_data, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads);
int reduce_gpu_old_v2(const int *d_data, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads);
int reduce_gpu_old_v3(const int *d_data, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads);
int reduce_gpu_old_v4(const int *d_data, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads);
int reduce_gpu_old_v5(const int *d_data, int *h_tmp_out, int *d_tmp_out, size_t n, int n_blocks, int n_threads);


/*
uint32_t reduce_gpu_v2(const uint32_t *d_data, uint32_t *h_tmp_out, uint32_t *d_tmp_out, size_t n, int n_blocks, int n_threads, cudaDeviceProp &prop);

uint32_t reduce_gpu_v3(const uint32_t *d_data, uint32_t *h_tmp_out, uint32_t *d_tmp_out, size_t n, int n_blocks, int n_threads, cudaDeviceProp &prop);

uint32_t reduce_gpu_v4(const uint32_t *d_data, uint32_t *h_tmp_out, uint32_t *d_tmp_out, size_t n, int n_blocks, int n_threads, cudaDeviceProp &prop);

uint32_t reduce_gpu_v5(const uint32_t *d_data, uint32_t *h_tmp_out, uint32_t *d_tmp_out, size_t n, int n_blocks, int n_threads, cudaDeviceProp &prop);

uint32_t reduce_gpu_v6(const uint32_t *d_data, uint32_t *h_tmp_out, uint32_t *d_tmp_out, size_t n, int n_blocks, int n_threads, cudaDeviceProp &prop);

uint32_t reduce_gpu_v7(const uint32_t *d_data, uint32_t *h_tmp_out, uint32_t *d_tmp_out, size_t n, int n_blocks, int n_threads, cudaDeviceProp &prop);
*/