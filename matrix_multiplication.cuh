#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
#pragma comment(lib, "cublas.lib")

#include "utils.cuh"


void matrixMultiplication(
  double* A, double* B, double* C,
  size_t M, size_t K, size_t N
) {
  double* A_d; // A_device
  double* B_d;
  double* C_d;

  size_t a_mem_size = sizeof(double) * K * M;
  size_t b_mem_size = sizeof(double) * K * N;
  size_t c_mem_size = sizeof(double) * M * N;

  cudaMalloc((void**)&A_d, a_mem_size);
  cudaMalloc((void**)&B_d, b_mem_size);
  cudaMalloc((void**)&C_d, c_mem_size);

  cudaMemcpy(A_d, A, a_mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, b_mem_size, cudaMemcpyHostToDevice);

  cuBLAS_mm(A_d, B_d, C_d, N, M, K);

  cudaMemcpy(C, C_d, c_mem_size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}


std::tuple<double*, size_t, size_t> matrixMultiplicationInParts(
    double* A, double* B, double* C,
    size_t M,size_t K, size_t N,
    size_t a_block_rows, size_t b_block_cols) {
  double* A_d; // A_device
  double* B_d;
  double* C_d;

  size_t a_mem_size = sizeof(double) * K * a_block_rows;
  size_t b_mem_size = sizeof(double) * K * b_block_cols;
  size_t c_mem_size = sizeof(double) * a_block_rows * b_block_cols;

  cudaMalloc((void**)&A_d, a_mem_size);
  cudaMalloc((void**)&B_d, b_mem_size);
  cudaMalloc((void**)&C_d, c_mem_size);

  // to copy submatrix from A or B
  double* A_temp;
  double* B_temp;
  double* C_temp;
  // double* temp;

  A_temp = (double*)malloc(a_mem_size);
  B_temp = (double*)malloc(b_mem_size);
  C_temp = (double*)malloc(c_mem_size);
  // temp = (double*)malloc(sizeof(double) * a_block_rows * b_block_cols);

  size_t submatrix_A_count = size_t(M / a_block_rows) + (M % a_block_rows != 0 ? 1 : 0);
  size_t submatrix_B_count = size_t(N / b_block_cols) + (N % b_block_cols != 0 ? 1 : 0);

  for (size_t i = 0; i < submatrix_A_count; ++i) {
    // better use A_temp = A + k * rows_size
    size_t curr_a_rows = min(a_block_rows, M - a_block_rows * i);

    // fill submatrix from A
    for (size_t q = 0; q < curr_a_rows; ++q) {
      for (size_t p = 0; p < K; ++p) {
        A_temp[q * K + p] = A[i * a_block_rows * K + q * K + p];
      }
    }

    // fill zeros if submatrix bigger than needed
    for (size_t q = curr_a_rows; q < a_block_rows; ++q) {
      for (size_t p = 0; p < K; ++p) {
        A_temp[q * K + p] = 0;
      }
    }

    for (size_t j = 0; j < submatrix_B_count; ++j) {
      size_t curr_b_cols = min(b_block_cols, N - b_block_cols * j);

      // fill submatrix from B
      for (size_t q = 0; q < curr_b_cols; ++q) {
        for (size_t p = 0; p < K; ++p) {
          B_temp[p * b_block_cols + q] = B[p * N + j * b_block_cols + q];
        }
      }

      // fill zeros if submatrix bigger than needed
      for (size_t q = curr_b_cols; q < b_block_cols; ++q) {
        for (size_t p = 0; p < K; ++p) {
          B_temp[p * b_block_cols + q] = 0;
        }
      }


      cudaMemcpy(A_d, A_temp, a_mem_size, cudaMemcpyHostToDevice);
      cudaMemcpy(B_d, B_temp, b_mem_size, cudaMemcpyHostToDevice);

      // mm(A_d, B_d, C_d, a_block_rows, a_cols, b_block_cols);

      cuBLAS_mm(A_d, B_d, C_d, a_block_rows, b_block_cols, K);

      cudaMemcpy(C_temp, C_d, c_mem_size, cudaMemcpyDeviceToHost);


//       toRowMajor(temp, C_temp, a_block_rows, b_block_cols);

      copyElements(C_temp, b_block_cols,
        C, N,
        i * a_block_rows, j * b_block_cols,
        curr_a_rows, curr_b_cols
      );
    }
  }

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  free(A_temp);
  free(B_temp);
  free(C_temp);
  // free(temp);

  return std::tie(C, M, N);
}