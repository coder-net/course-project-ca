#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cublas_v2.h"
#pragma comment(lib, "cublas.lib")

#include "cuda_timer.cuh"
#include "utils.cuh"
#include "matrix_multiplication.cuh"

#include "strassen.cuh"



int main()
{
  cudaError_t cudaStatus;
  cudaStatus = cudaSetDevice(0);

  if (cudaStatus != cudaSuccess) {
  	fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    return -1;
  }

  size_t iter_number =  10;

  std::cout << "Iteration number: " << iter_number << std::endl << std::endl;

  double* h_A; // host A
  double* h_B;
  double* h_C;
  size_t xa, ya, xb, yb;
  
  std::tie(h_A, xa, ya) = readMatrixFromFile("matrix1.txt");
  std::tie(h_B, xb, yb) = readMatrixFromFile("matrix2.txt");

  if (ya != xb) {
    fprintf(stderr, "Cannot multiply these two matrices");
    return -1;

  }

  size_t M = xa, K = ya, N = yb;

  h_C = (double*)malloc(sizeof(double) * M * N);
  
  CudaTimer timer;

  timer.start();
  for (size_t i = 0; i < iter_number; ++i)
    matrixMultiplication(h_A, h_B, h_C, M, K, N);
  timer.stop();

  double simple_mm = timer.value() / iter_number;
  std::cout << "Simple MM time: " << simple_mm << "s" << std::endl;

  writeMatrixToFile("output_mm.txt", h_C, M, N);

  std::cout << std::endl;
  
  for (int k = 1; k < 5; ++k) {
    timer.start();
    for (size_t i = 0; i < iter_number; ++i)
      strassen_mm(h_A, h_B, h_C, M, K, N, k);
    timer.stop();

    double strassen_time = timer.value() / iter_number;
    std::cout << "Strassen time: " << strassen_time << "s for " << k << " recursion depth" << std::endl;
  }
  writeMatrixToFile("output_strassen.txt", h_C, M, N);

  std::cout << std::endl;
  
  for (int k = 1; k <= M; k *= 2) {
    for (int j = 1; j <= N; j *= 2) {
      timer.start();
      for (size_t i = 0; i < iter_number; ++i)
        matrixMultiplicationInParts(h_A, h_B, h_C, M, K, N, M / k, N / j);
      timer.stop();

      double parts_mm = timer.value() / iter_number;
      std::cout << "MM in parts time: " << parts_mm << "s for A block size: " << M / k << ", for B block size: " << N / j << std::endl;
    }
  }

  writeMatrixToFile("output_parts.txt", h_C, M, N);

  free(h_A);
  free(h_B);
  free(h_C);

	return 0;
}