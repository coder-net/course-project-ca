#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda.h>

#include <cmath>

#include "utils.cuh"


size_t nearestTwoPower(size_t num) {
  size_t ones_count = 0;
  size_t count = 0;
  while (num) {
    ones_count += num & 1;
    ++count;
    num >>= 1;
  }
  if (ones_count == 1)
    return count - 1;
  return count;
}


// Winograd algorithm
void recursiveStrassen(double *A, double *B, double *C,
    int lda, int ldb, int ldc,
    int N,
    int depth) {
  if (N <= 8) {
    GPU_mul(A, B, C, lda, ldb, ldc, N, N, N, N ,N, N, 1, 0);
    return;
  }

  int XA2 = N / 2;
  int XB2 = N / 2;
  int XC2 = N / 2;

  int YA2 = N / 2;
  int YB2 = N / 2;
  int YC2 = N / 2;

  double *W_1, *W_2;
  int lw1 = XA2;
  int lw2 = XB2;
  cudaMalloc((void **)&W_1, lw1 * YA2 * sizeof(double));
  cudaMalloc((void **)&W_2, lw2 * YB2 * sizeof(double));

  int dXA = XA2;
  int dYA = YA2 * lda;
  int dXB = XB2;

  int dYB = YB2 * ldb;
  int dXC = XC2;
  int dYC = YC2 * ldc;

  double *A11, *A12, *A21, *A22;
  double *B11, *B12, *B21, *B22;
  double *C11, *C12, *C21, *C22;

  A11 = A;
  A12 = A + dXA;
  A21 = A + dYA;
  A22 = A + dXA + dYA;

  B11 = B;
  B12 = B + dXB;
  B21 = B + dYB;
  B22 = B + dXB + dYB;

  C11 = C;
  C12 = C + dXC;
  C21 = C + dYC;
  C22 = C + dXC + dYC;



  if (depth <= 1) {
    GPU_add(A11, A21, W_1, lda, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A11 - A21
    GPU_add(B22, B12, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - B12
    GPU_mul(W_1, W_2, C21, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C21 = W_1 * W_2
    GPU_add(A21, A22, W_1, lda, lda, lw1, XA2, YA2, 1.0, 1.0); // W_1 = A21 + A22
    GPU_add(B12, B11, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B12 - B11
    GPU_mul(W_1, W_2, C22, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C22 = W_1 * W_2
    GPU_add(W_1, A11, W_1, lw1, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = W_1- A11
    GPU_add(B22, W_2, W_2, ldb, lw2, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - W_2
    GPU_mul(W_1, W_2, C11, lw1, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C11 = W_1 * W_2
    GPU_add(A12, W_1, W_1, lda, lw1, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A12 - W_1
    GPU_mul(W_1, B22, C12, lw1, ldb, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C12 = W_1 * B22
    GPU_add(C22, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0, 1.0); // C12 = C22 + C12
    GPU_mul(A11, B11, W_1, lda, ldb, lw1, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // W_1= A11 * B11
    GPU_add(W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0, 1.0); // C11 = W_1 + C11
    GPU_add(C11, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0, 1.0); // C12 = C11 + C12
    GPU_add(C11, C21, C11, ldc, ldc, ldc, XC2, YC2, 1.0, 1.0); // C11 = C11 + C21
    GPU_add(W_2, B21, W_2, lw2, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = W_2- B21
    GPU_mul(A22, W_2, C21, lda, lw2, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C21 = A22 * W_2
    GPU_add(C11, C21, C21, ldc, ldc, ldc, XC2, YC2, 1.0, -1.0); // C11 = C11 - C21
    GPU_add(C11, C22, C22, ldc, ldc, ldc, XC2, YC2, 1.0, 1.0); // C22 = C11 + C22
    GPU_mul(A12, B21, C11, lda, ldb, ldc, XA2, XB2, XC2, YA2, YB2, YC2, 1.0, 0.0); // C11 = A12 * B21
    GPU_add(W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0, 1.0); // C11 = W_1+ C11
  }
  else {
    GPU_add(A11, A21, W_1, lda, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A11 - A21
    GPU_add(B22, B12, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - B12
    recursiveStrassen(W_1, W_2, C21, lw1, lw2, ldc, N / 2, depth - 1);
    GPU_add(A21, A22, W_1, lda, lda, lw1, XA2, YA2, 1.0, 1.0); // W_1 = A21 + A22
    GPU_add(B12, B11, W_2, ldb, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B12 - B11
    recursiveStrassen(W_1, W_2, C22, lw1, lw2, ldc,N / 2, depth - 1);
    GPU_add(W_1, A11, W_1, lw1, lda, lw1, XA2, YA2, 1.0, -1.0); // W_1 = W_1- A11
    GPU_add(B22, W_2, W_2, ldb, lw2, lw2, XB2, YB2, 1.0, -1.0); // W_2 = B22 - W_2
    recursiveStrassen(W_1, W_2, C11, lw1, lw2, ldc,N / 2, depth - 1);
    GPU_add(A12, W_1, W_1, lda, lw1, lw1, XA2, YA2, 1.0, -1.0); // W_1 = A12 - W_1
    recursiveStrassen(W_1, B22, C12, lw1, ldb, ldc, N / 2, depth - 1);
    GPU_add(C22, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0, 1.0); // C12 = C22 + C12
    recursiveStrassen(A11, B11, W_1, lda, ldb, lw1, N / 2, depth - 1);
    GPU_add(W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0, 1.0); // C11 = W_1 + C11
    GPU_add(C11, C12, C12, ldc, ldc, ldc, XC2, YC2, 1.0, 1.0); // C12 = C11 + C12
    GPU_add(C11, C21, C11, ldc, ldc, ldc, XC2, YC2, 1.0, 1.0); // C11 = C11 + C21
    GPU_add(W_2, B21, W_2, lw2, ldb, lw2, XB2, YB2, 1.0, -1.0); // W_2 = W_2- B21
    recursiveStrassen(A22, W_2, C21, lda, lw2, ldc, N / 2, depth - 1);
    GPU_add(C11, C21, C21, ldc, ldc, ldc, XC2, YC2, 1.0, -1.0); // C11 = C11 - C21
    GPU_add(C11, C22, C22, ldc, ldc, ldc, XC2, YC2, 1.0, 1.0); // C22 = C11 + C22
    recursiveStrassen(A12, B21, C11, lda, ldb, ldc, N / 2, depth - 1);
    GPU_add(W_1, C11, C11, lw1, ldc, ldc, XC2, YC2, 1.0, 1.0); // C11 = W_1+ C11
  }
  cudaFree(W_1);
  cudaFree(W_2);
}

void strassen_mm(
  double *A, double *B, double *C,
  size_t m, size_t k, size_t n,
  size_t level_num
) {
  double* A_square;
  double* B_square;
  double* C_square;
  size_t levels = nearestTwoPower(max(max(m, k), n));
  size_t N = 1 << levels;

  size_t mem_size = sizeof(double) * N * N;

  A_square = (double*)malloc(mem_size);
  B_square = (double*)malloc(mem_size);
  C_square = (double*)malloc(mem_size);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (i < m && j < k) {
        A_square[i * N + j] = A[i * k + j];
      }
      else {
        A_square[i * N + j] = 0;
      }

      if (i < k && j < n) {
        B_square[i * N + j] = B[i * n + j];
      }
      else {
        B_square[i * N + j] = 0;
      }
    }
  }

  double *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, mem_size);
  cudaMalloc((void**)&d_B, mem_size);
  cudaMalloc((void**)&d_C, mem_size);
  cudaMemcpy(d_A, A_square, mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B_square, mem_size, cudaMemcpyHostToDevice);

  recursiveStrassen(d_A, d_B, d_C, N, N, N, N, level_num);
  cudaDeviceSynchronize();

  cudaMemcpy(C_square, d_C, mem_size, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      C[i * n + j] = C_square[i * N + j];
    }
  }

  free(A_square);
  free(B_square);
  free(C_square);
}