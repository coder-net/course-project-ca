#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
#pragma comment(lib, "cublas.lib")

#include "utils.cuh"


void copyElements(
  float* src, size_t src_cols,
  float* dst, size_t dst_cols,
  size_t row_offset, size_t col_offset,
  size_t rows_to_copy, size_t cols_to_copy
) {
  for (size_t i = 0; i < rows_to_copy; ++i) {
    for (size_t j = 0; j < cols_to_copy; ++j) {
      dst[(row_offset + i) * dst_cols + col_offset + j] = src[i * src_cols + j];
    }
  }
}


std::tuple<float*, size_t, size_t> partialMatrixMultiplication(float* A, size_t a_rows, size_t a_cols, float* B, size_t b_rows, size_t b_cols) {
  size_t a_partial = 2;
  size_t b_partial = 3;

  size_t n = a_rows, k = a_cols, m = b_cols;

  float* C = (float*)malloc(sizeof(float) * n * m); // answer

  float* A_d; // A_device
  float* B_d;
  float* C_d;

  cudaMalloc((void**)&A_d, sizeof(float) * k * a_partial);
  cudaMalloc((void**)&B_d, sizeof(float) * k * b_partial);
  cudaMalloc((void**)&C_d, sizeof(float) * a_partial * b_partial);

  // to copy submatrix from A or B
  float* A_temp;
  float* B_temp;
  float* C_temp;
  float* temp;

  A_temp = (float*)malloc(sizeof(float) * k * a_partial);
  B_temp = (float*)malloc(sizeof(float) * k * b_partial);
  C_temp = (float*)malloc(sizeof(float) * a_partial * b_partial);
  temp = (float*)malloc(sizeof(float) * a_partial * b_partial);

  size_t submatrix_A = size_t(a_rows / a_partial) + (a_rows % a_partial != 0 ? 1 : 0);
  size_t submatrix_B = size_t(b_cols / b_partial) + (b_cols % b_partial != 0 ? 1 : 0);

  for (size_t i = 0; i < submatrix_A; ++i) {
    // better use A_temp = A + k * rows_size
    size_t curr_a_rows = min(a_partial, a_rows - a_partial * i);

    // =============================================================
    // fill submatrix from A
    for (size_t q = 0; q < curr_a_rows; ++q) {
      for (size_t p = 0; p < k; ++p) {
        A_temp[q * a_cols + p] = A[i * a_partial * k + q * a_cols + p];
      }
    }

    // fill zeros if submatrix bigger than needed
    for (size_t q = curr_a_rows; q < a_partial; ++q) {
      for (size_t p = 0; p < k; ++p) {
        A_temp[q * a_cols + p] = 0;
      }
    }
    // ==============================================================

    for (size_t j = 0; j < submatrix_B; ++j) {
      size_t curr_b_cols = min(b_partial, b_cols - b_partial * j);

      // =============================================================
      // fill submatrix from B
      for (size_t q = 0; q < curr_b_cols; ++q) {
        for (size_t p = 0; p < k; ++p) {
          B_temp[p * b_partial + q] = B[p * b_cols + j * b_partial + q];
        }
      }

      // fill zeros if submatrix bigger than needed
      for (size_t q = curr_b_cols; q < b_partial; ++q) {
        for (size_t p = 0; p < k; ++p) {
          B_temp[p * b_partial + q] = 0;
        }
      }
      // =============================================================

    
      cudaMemcpy(A_d, A_temp, sizeof(float) * k * a_partial, cudaMemcpyHostToDevice);
      cudaMemcpy(B_d, B_temp, sizeof(float) * k * b_partial, cudaMemcpyHostToDevice);

      cudaMatrixMultiplication(A_d, a_partial, a_cols, B_d, b_rows, b_partial, C_d, 1);

      cudaMemcpy(temp, C_d, sizeof(float) * a_partial * b_partial, cudaMemcpyDeviceToHost);

      toRowMajor(temp, C_temp, a_partial, b_partial);

      copyElements(C_temp, b_partial,
        C, m,
        i * a_partial, j * b_partial,
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
  free(temp);

  return std::tie(C, n, m);
}