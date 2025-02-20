#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>



const int THREAD_PER_BLOCK = 256;

bool validate(float* res, float* output, int n) {
  for (int i = 0; i < n; i++) {
    if (res[i] != output[i])
      printf("not equal. res: %f, output: %f", res[i], output[i])
      return false;
  }
  return true;
}

__global__ void reduce_kernel_0(float* d_input, float* d_output, const int N){
  const int global_id = blockDim.x * blockIdx.x + threadIdx.x;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ float smem[THREAD_PER_BLOCK];
  smem[tid] = d_input[global_id];
  __syncthreads();
  for (int i = 1; i < THREAD_PER_BLOCK; i *= 2) {
    if (tid % (2 * i) == 0) {
      smem[tid] = smem[tid + t]
    }
    __syncthreads();
  }
  if (tid == 0) {
    d_output[bid] = smem[tid];
  }
}

int main() {
  int N = 32 * 1024 * 1024;
  int block_cnt = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

  float* input = (float*) malloc(N * sizeof(float));

  for (int i = 0; i < N; i++) {
    input[i] = 1;
  }

  float* res = (float*) malloc(block_cnt * sizeof(float));
  for (int i = 0; i < block_cnt; i++) {
    for (int j = 0; j < THREAD_PER_BLOCK; j++) {
      res[i] += input[i * THREAD_PER_BLOCK + j];
    }
  }

  float* d_input;
  cudaError_t err = cudaMalloc((void**)&d_input, N * sizeof(float));
  if (err != cudaSuccess) {
    printf("error!\n");
  }
  cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

  float* d_output;
  cudaMalloc((void**)&d_output, block_cnt * sizeof(float));

  // kernel launch
  dim3 Grid((block_cnt));
  dim3 Block((THREAD_PER_BLOCK));
  reduce_kernel_0<<<Grid, Block>>>(d_input, d_output, N);

  float* output = (float*) malloc(block_cnt * sizeof(float));
  cudaMemcpy(output, d_output, block_cnt * sizeof(float), cudaMemcpyDeviceToHost);
  if (validate(res, output, block_cnt)) {
    printf("validation success!\n")
  } else {
    printf("validation failed!\n")
  }
}