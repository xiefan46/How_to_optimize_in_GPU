#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>


const int THREAD_NUM_PER_BLOCK = 256;

__global__ void reduce_kernel_1(float* input, float* output, const int N) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int bd = blockDim.x;
  const int global_idx = blockDim.x * bid + tid;
  __shared__ float smem[THREAD_NUM_PER_BLOCK];
  if (global_idx < N) {
    smem[tid] = input[global_idx] + input[global_idx + bd];
  }
  __syncthreads();
  #pragma unroll
  for (int i = THREAD_NUM_PER_BLOCK / 2; i > 0; i /= 2) {
    if (tid < i) {
      smem[tid] += smem[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[bid] = smem[tid];
  }
}

bool validate(float* output, float* correct_res, int size) {
  for (int i = 0; i < size; i++) {
    if (output[i] != correct_res[i]) {
      printf("Error! output: %f, correct result: %f", output[i], correct_res[i]);
      return false;
    }
  }
  return true;
}

int main() {
  const int N = 32 * 1024 * 1024; // 32M data poiints
  float* input = (float*) malloc(N * sizeof(float));


  // Prepare host input data
  for (int i = 0; i < N; i++) {
    input[i] = 1;
  }
  const int num_per_block = 2 * THREAD_NUM_PER_BLOCK;
  const int block_cnt = (N + num_per_block - 1) / num_per_block;
  float* correct_res = (float*) malloc(block_cnt * sizeof(float));
  for (int i = 0; i < block_cnt; i++) {
    for (int j = 0; j < THREAD_NUM_PER_BLOCK; j++) {
      correct_res[i] += input[i * THREAD_NUM_PER_BLOCK + j];
    }
  }

  // Move host data to cpu
  float* d_input;
  if (cudaMalloc((void**)&d_input, N * sizeof(float)) != cudaSuccess) {
    printf("error allocaion memory!");
    exit(1);
  }
  float* d_output;
  if (cudaMalloc((void**)&d_output, block_cnt * sizeof(float)) != cudaSuccess) {
    printf("error allocaion memory!");
    exit(1);
  }

  if (cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
    printf("error copying memory!");
    exit(1);
  }

  // kernel launch
  dim3 Grid((block_cnt));
  dim3 Block((THREAD_NUM_PER_BLOCK));
  reduce_kernel_1<<<Grid, Block>>>(d_input, d_output, N);

  // copy data back and validate
  float* output = (float*) malloc(block_cnt * sizeof(float));
  cudaMemcpy(output, d_output, block_cnt * sizeof(float), cudaMemcpyDeviceToHost);

  if (validate(output, correct_res, block_cnt)) {
    printf("check success! \n");
  } else {
    printf("check failed! \n");
  }
}