#pragma once
#include <stdint.h>

uint32_t reduce_gpu_v1(const uint32_t *d_data, uint32_t *h_tmp_out, uint32_t *d_tmp_out, size_t n, int n_blocks, int n_threads);
