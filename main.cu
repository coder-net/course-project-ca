#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
#pragma comment(lib, "cublas.lib")


#define ull unsigned long long

float* toColumnMajor(float* A, float* B, ull rows, ull cols) {
	for (ull i = 0; i < rows; i++) {
		for (ull j = 0; j < cols; ++j) {
			B[j * rows + i] = A[i * cols + j];
		}
	}
	return B;
}

float* toRowMajor(float* A, float* B, size_t rows, size_t cols) {
	for (ull i = 0; i < rows; i++) {
		for (ull j = 0; j < cols; ++j) {
			B[i * cols + j] = A[j * rows + i];
		}
	}
	return B;
}


void print(float* A, ull size) {
	for (int i = 0; i < size; ++i) {
		std::cout << A[i] << " ";
	}
	std::cout << std::endl;
}


float* cudaMatrixMultiplication(float* A, ull a_row, ull a_col, float* B, ull b_row, ull b_col, float* C, float alpha) {
	// formula: alpha * A * B  + beta * C
	float beta = 0;
	cublasHandle_t handle;

	cublasCreate(&handle);

	cublasSgemm(
		handle,
		CUBLAS_OP_C,
		CUBLAS_OP_C,
		a_row,
		b_col,
		a_col, // equal to a_col
		&alpha,
		A,
		a_col,
		B,
		b_col,
		&beta,
		C,
		a_row
	);

	cublasDestroy(handle);
	return C;
}


float* matrixMultiplication(float* a, ull a_row, ull a_col, float* b, ull b_row, ull b_col) {
	float* cuda_a = nullptr;
	float* cuda_b = nullptr;
	float* cuda_c = nullptr;
	ull a_size = a_row * a_col;
	ull b_size = b_row * b_col;
	ull c_size = a_row * b_col;

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

	return c;
}



int main()
{
	cudaError_t cudaStatus;
	
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 1;
	}
	float* a = new float[6];
	float* b = new float[4];
	a[0] = 1;
	a[1] = 2;
	a[2] = 3;
	a[3] = 4;
	a[4] = 5;
	a[5] = 6;
	b[0] = b[1] = b[2] = b[3] = 2;

	float *ans;

	ans = matrixMultiplication(a, 3, 2, b, 2, 2);

	cudaStatus = cudaDeviceSynchronize();
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 2; j++) {
			std::cout << ans[i * 2 + j] << " ";
		}
		std::cout << std::endl;
	}

	return 0;
}