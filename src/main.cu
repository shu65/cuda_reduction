#include <stdint.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

#include "reduction.hpp"
#include "reduction_gpu.hpp"
#include "cuda_helper.hpp"

using namespace std;


int main(int argc, char *argv[])
{
  const int n_blocks = 128;
  const int n_threads = 512;
  const int n = n_blocks * n_threads;
  const size_t array_size = sizeof(uint32_t) * n;
  const size_t tmp_size = sizeof(uint32_t) * n_blocks;

  vector<uint32_t> h_data(n);
  vector<uint32_t> h_tmp_out(n_blocks);

  for (int i = 0; i < n; ++i)
  {
    h_data[i] = i;
  }
  uint32_t *d_data = nullptr;
  uint32_t *d_tmp_out = nullptr;
  checkCudaErrors(cudaMalloc(&d_data, array_size));
  checkCudaErrors(cudaMalloc(&d_tmp_out, tmp_size));
  checkCudaErrors(cudaMemcpy(d_data, h_data.data(), array_size, cudaMemcpyDefault));

  uint32_t expected_value = 0;
  uint32_t actual_value = 0;

  std::chrono::system_clock::time_point start;
  std::chrono::system_clock::time_point end;
  double elapsed_time_msec;
  cudaDeviceSynchronize(); 

  start = std::chrono::system_clock::now(); 
  expected_value = reduce(h_data.data(), h_data.size());
  end = std::chrono::system_clock::now();  
  elapsed_time_msec = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0);
  cout << "CPU reduction: " << elapsed_time_msec << " msec." << endl;

  cudaDeviceSynchronize(); 
  start = std::chrono::system_clock::now(); 
  actual_value = reduce_gpu_v1(
    d_data, 
    h_tmp_out.data(),
    d_tmp_out,
    h_data.size(), 
    n_blocks, 
    n_threads
  );
  cudaDeviceSynchronize(); 
  end = std::chrono::system_clock::now();  
  elapsed_time_msec = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0);
  assert(actual_value == expected_value);
  cout << "GPU reduction (v1): " << elapsed_time_msec << " msec." << endl;

  checkCudaErrors(cudaFree(d_data));
  d_data = nullptr;

/*
  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;

  size_t array_size = sizeof(float) * n;

  checkCudaErrors(cudaMalloc(&d_a, array_size));
  checkCudaErrors(cudaMalloc(&d_b, array_size));
  checkCudaErrors(cudaMalloc(&d_c, array_size));

  checkCudaErrors(cudaMemcpy(d_a, h_a.data(), array_size, cudaMemcpyDefault));
  checkCudaErrors(cudaMemcpy(d_b, h_b.data(), array_size, cudaMemcpyDefault));

  vecAdd<<<n_blocks, n_threads>>>(d_a, d_b, d_c);

  checkCudaErrors(cudaMemcpy(h_c.data(), d_c, array_size, cudaMemcpyDefault));

  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(d_c));
  d_a = nullptr;
  d_b = nullptr;
  d_c = nullptr;

  for (int i = 0; i < n; ++i)
  {
    assert(h_c[i] == (h_a[i] + h_b[i]));
  }
  cout << "OK!" << endl;
*/
  return 0;
}