#include <stdint.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "reduction.hpp"
#include "reduction_gpu.hpp"
#include "reduction_gpu_old.hpp"
#include "cuda_helper.hpp"

using namespace std;

string getKernelName(int kernel_id, bool old)
{
  string old_str = "";
  if (old)
  {
    old_str = "Old";
  }
  else
  {
    old_str = "New";
  }
  if (kernel_id == 0)
  {
    return "CPU";
  }
  else
  {
    return "GPU-" + to_string(kernel_id) + "-" + old_str;
  }
}

int runReduceFunc(
    int kernel_id,
    bool old,
    const int *h_in,
    const int *d_in,
    int *h_tmp_out,
    int *d_tmp_out,
    size_t n,
    int n_blocks,
    int n_threads)
{
  int ret = 0;
  switch (kernel_id)
  {
  case 0:
    ret = reduce(h_in, n);
    break;
  case 1:
    if (old)
    {
      ret = reduce_gpu_old_v1(
          d_in,
          h_tmp_out,
          d_tmp_out,
          n,
          n_blocks,
          n_threads);
    }
    else
    {
      ret = reduce_gpu_v1(
          d_in,
          h_tmp_out,
          d_tmp_out,
          n,
          n_blocks,
          n_threads);
    }
    break;
  case 2:
    if (old)
    {
      ret = reduce_gpu_old_v2(
          d_in,
          h_tmp_out,
          d_tmp_out,
          n,
          n_blocks,
          n_threads);
    }
    else
    {
      ret = reduce_gpu_v2(
          d_in,
          h_tmp_out,
          d_tmp_out,
          n,
          n_blocks,
          n_threads);
    }
    break;
  case 3:
    if (old)
    {
      ret = reduce_gpu_old_v3(
          d_in,
          h_tmp_out,
          d_tmp_out,
          n,
          n_blocks,
          n_threads);
    }
    else
    {
      ret = reduce_gpu_v3(
          d_in,
          h_tmp_out,
          d_tmp_out,
          n,
          n_blocks,
          n_threads);
    }
    break;
  case 4:
    if (old)
    {
      ret = reduce_gpu_old_v4(
          d_in,
          h_tmp_out,
          d_tmp_out,
          n,
          n_blocks,
          n_threads);
    }
    else
    {
      ret = reduce_gpu_v4(
          d_in,
          h_tmp_out,
          d_tmp_out,
          n,
          n_blocks,
          n_threads);
    }
    break;
  case 5:
    if (old)
    {
      ret = reduce_gpu_old_v5(
          d_in,
          h_tmp_out,
          d_tmp_out,
          n,
          n_blocks,
          n_threads);
    }
    else
    {
      assert(false);
    }
    break;
  case 6:
    if (old)
    {
      ret = reduce_gpu_old_v6(
          d_in,
          h_tmp_out,
          d_tmp_out,
          n,
          n_blocks,
          n_threads);
    }
    else
    {
      ret = reduce_gpu_v6(
          d_in,
          h_tmp_out,
          d_tmp_out,
          n,
          n_blocks,
          n_threads);
    }
    break;
  case 7:
    if (old)
    {
      ret = reduce_gpu_old_v7(
          d_in,
          h_tmp_out,
          d_tmp_out,
          n,
          n_blocks,
          n_threads);
    }
    else
    {
      ret = reduce_gpu_v7(
          d_in,
          h_tmp_out,
          d_tmp_out,
          n,
          n_blocks,
          n_threads);
    }
    break;
  case 8:
    if (old)
    {
      assert(false);
    }
    else
    {
      ret = reduce_gpu_v8(
          d_in,
          h_tmp_out,
          d_tmp_out,
          n,
          n_blocks,
          n_threads);
    }
    break;
  case 9:
    if (old)
    {
      assert(false);
    }
    else
    {
      ret = reduce_gpu_v9(
          d_in,
          h_tmp_out,
          d_tmp_out,
          n,
          n_blocks,
          n_threads);
    }
    break;
  default:
    assert(false);
    break;
  }
  return ret;
}

double reduceBenchmark(
    int kernel_id,
    bool old,
    const int *h_in,
    const int *d_in,
    int *h_tmp_out,
    int *d_tmp_out,
    size_t n,
    int n_blocks,
    int n_threads,
    int n_trials,
    int expected_value,
    bool print_log)
{
  int actual_value = 0;
  std::chrono::system_clock::time_point start;
  std::chrono::system_clock::time_point end;
  double elapsed_time_msec;
  cudaDeviceSynchronize();
  const string kernel_name = getKernelName(kernel_id, old);
  start = std::chrono::system_clock::now();
  for (int i = 0; i < n_trials; ++i)
  {
    actual_value = runReduceFunc(
        kernel_id,
        old,
        h_in,
        d_in,
        h_tmp_out,
        d_tmp_out,
        n,
        n_blocks,
        n_threads);
  }
  // cudaDeviceSynchronize();
  end = std::chrono::system_clock::now();
  elapsed_time_msec = static_cast<double>(chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0);
  if (print_log)
  {
    cout << kernel_name << ": " << elapsed_time_msec / n_trials << " msec." << endl;
  }
  assert(actual_value == expected_value);

  return elapsed_time_msec;
}

int main(int argc, char *argv[])
{
  const int n_trials = 100;
  const int n_blocks = 512;
  const int n_threads = 512;
  const int n = n_blocks * n_threads;
  const size_t array_size = sizeof(int) * n;
  const size_t tmp_size = sizeof(int) * n_blocks;
  cout << "n:" << n << endl;
  vector<int> h_in(n);
  vector<int> h_tmp_out(n_blocks);

  for (int i = 0; i < n; ++i)
  {
    h_in[i] = 1;
  }

  int device;
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));

  int *d_in = nullptr;
  int *d_tmp_out = nullptr;
  checkCudaErrors(cudaMalloc(&d_in, array_size));
  checkCudaErrors(cudaMalloc(&d_tmp_out, tmp_size));
  checkCudaErrors(cudaMemcpy(d_in, h_in.data(), array_size, cudaMemcpyDefault));

  int expected_value = reduce(h_in.data(), h_in.size());
  int actual_value = 0;
  cout << "dummy gpu call" << endl;
  for (int i = 0; i < 10; ++i)
  {
    actual_value = reduceBenchmark(
        1,
        true,
        h_in.data(),
        d_in,
        h_tmp_out.data(),
        d_tmp_out,
        h_in.size(),
        n_blocks,
        n_threads,
        n_trials,
        expected_value,
        false);
  }
  bool old = false;
  double elapsed_time_msec;
  // "dummy gpu call"
  elapsed_time_msec = reduceBenchmark(
      0,
      old,
      h_in.data(),
      d_in,
      h_tmp_out.data(),
      d_tmp_out,
      h_in.size(),
      n_blocks,
      n_threads,
      n_trials,
      expected_value,
      false);
  // cpu
  elapsed_time_msec = reduceBenchmark(
      0,
      old,
      h_in.data(),
      d_in,
      h_tmp_out.data(),
      d_tmp_out,
      h_in.size(),
      n_blocks,
      n_threads,
      n_trials,
      expected_value,
      true);
  vector<bool> old_flags = {true, false};
  old = true;
  for (int kernel_id = 1; kernel_id < 8; ++kernel_id)
  {
    // dummy
    elapsed_time_msec = reduceBenchmark(
        kernel_id,
        old,
        h_in.data(),
        d_in,
        h_tmp_out.data(),
        d_tmp_out,
        h_in.size(),
        n_blocks,
        n_threads,
        n_trials,
        expected_value,
        false);

    elapsed_time_msec = reduceBenchmark(
        kernel_id,
        old,
        h_in.data(),
        d_in,
        h_tmp_out.data(),
        d_tmp_out,
        h_in.size(),
        n_blocks,
        n_threads,
        n_trials,
        expected_value,
        true);
  }

  old = false;
  for (int kernel_id = 1; kernel_id < 10; ++kernel_id)
  {
    if (kernel_id == 5)
    {
      continue;
    }
    // dummy
    elapsed_time_msec = reduceBenchmark(
        kernel_id,
        old,
        h_in.data(),
        d_in,
        h_tmp_out.data(),
        d_tmp_out,
        h_in.size(),
        n_blocks,
        n_threads,
        n_trials,
        expected_value,
        false);

    elapsed_time_msec = reduceBenchmark(
        kernel_id,
        old,
        h_in.data(),
        d_in,
        h_tmp_out.data(),
        d_tmp_out,
        h_in.size(),
        n_blocks,
        n_threads,
        n_trials,
        expected_value,
        true);
  }

  // free device memory
  checkCudaErrors(cudaFree(d_in));
  d_in = nullptr;
  return 0;
}