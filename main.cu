#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
#pragma comment(lib, "cublas.lib")

#include "utils.cuh"
#include "algorithm1.cuh"
#include "algorithm2.cuh"



std::tuple<float*, size_t, size_t> matrixMultiplicationWithCuda(float* a, size_t a_row, size_t a_col, float* b, size_t b_row, size_t b_col) {
	float* cuda_a = nullptr;
	float* cuda_b = nullptr;
	float* cuda_c = nullptr;
	size_t a_size = a_row * a_col;
	size_t b_size = b_row * b_col;
	size_t c_size = a_row * b_col;

	cudaMalloc((void**)&cuda_a, sizeof(float) * a_size);
	cudaMalloc((void**)&cuda_b, sizeof(float) * b_size);
	cudaMalloc((void**)&cuda_c, sizeof(float) * c_size);

	cudaMemcpy(cuda_a, a, sizeof(float) * a_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_b, b, sizeof(float) * b_size, cudaMemcpyHostToDevice);

	cudaMatrixMultiplication(cuda_a, a_row, a_col, cuda_b, b_row, b_col, cuda_c, 1);

	float* temp_c = (float*)malloc(sizeof(float) * c_size);
	float* c = (float*)malloc(sizeof(float*) * c_size);

	cudaMemcpy(temp_c, cuda_c, sizeof(float) * c_size, cudaMemcpyDeviceToHost);

	toRowMajor(temp_c, c, a_row, b_col);

	free(temp_c);

	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_c);

  return std::tie(c, a_row, b_col);
}



int main()
{
  float* A;
  float* B;
  float* C;
  size_t a_rows, a_cols, b_rows, b_cols, c_rows, c_cols;

  std::tie(A, a_rows, a_cols) = readMatrixFromFile("matrix1.txt");
  std::tie(B, b_rows, b_cols) = readMatrixFromFile("matrix2.txt");

  if (a_cols != b_rows) {
    std::cout << "Impossible to multiply these two matrix";
    return -1;
  }

  /*std::cout << "A: " << std::endl;
  printMatrix(A, a_rows, a_cols);
  std::cout << std::endl << "B: " << std::endl;
  printMatrix(B, b_rows, b_cols);
  std::cout << std::endl;*/

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
	  fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    std::tie(C, c_rows, c_cols) = matrixMultiplication(A, a_rows, a_cols, B, b_rows, b_cols);
  }
  else {

    std::tie(C, c_rows, c_cols) = partialMatrixMultiplication1(A, a_rows, a_cols, B, b_rows, b_cols);
    cudaStatus = cudaDeviceSynchronize();

    writeMatrixToFile("algorithm1_out.txt", C, c_rows, c_cols);

    free(C);

    std::tie(C, c_rows, c_cols) = partialMatrixMultiplication2(A, a_rows, a_cols, B, b_rows, b_cols);
    cudaStatus = cudaDeviceSynchronize();

    writeMatrixToFile("algorithm2_out.txt", C, c_rows, c_cols);
  }

  // printMatrix(C, c_rows, c_cols);

  free(A);
  free(B);
  free(C);

	return 0;
}