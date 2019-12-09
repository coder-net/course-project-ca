#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
#pragma comment(lib, "cublas.lib")

#include "utils.cuh"
#include "cpp_utils.h"




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

	// print(temp_a, a_size);

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
	cudaError_t cudaStatus;
	
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}

  float* A;
  float* B;
  float* C;
  size_t a_rows, a_cols, b_rows, b_cols, c_rows, c_cols;

  std::tie(A, a_rows, a_cols) = readMatrixFromFile("matrix1.txt.txt");
  std::tie(B, b_rows, b_cols) = readMatrixFromFile("matrix2.txt.txt");

	std::tie(C, c_rows, c_cols) = matrixMultiplicationWithCuda(A, a_rows, a_cols, B, b_cols, b_rows);

	cudaStatus = cudaDeviceSynchronize();
	
  printMatrix(C, c_rows, c_cols);

	return 0;
}