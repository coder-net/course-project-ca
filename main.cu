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
#include "algorithm1.cuh"
#include "algorithm2.cuh"
#include "matrix_multiplication.cuh"

#include "strassen.cuh"



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
  double* h_A;
  double* h_B;
  double* h_C;
  size_t a_rows, a_cols, b_rows, b_cols, c_rows, c_cols;
  
  std::tie(h_A, a_rows, a_cols) = readMatrixFromFile("matrix3.txt");
  std::tie(h_B, b_rows, b_cols) = readMatrixFromFile("matrix4.txt");

  size_t M = a_rows, K = a_cols, N = b_cols;

  h_C = (double*)malloc(sizeof(double) * M * N);

  strassen_mm(h_A, h_B, h_C, M, K,N);

  printMatrix(h_C, M, N);
  //int iter = 1;
  //int check = 0;
  //int depth = 2;

  //int sizeA = M * K;
  //int sizeB = K * N;
  //int sizeC = M * N;
  //int memSizeA = sizeA * sizeof(double);
  //int memSizeB = sizeB * sizeof(double);
  //int memSizeC = sizeC * sizeof(double);

  //double *h_C = (double *)malloc(memSizeC);

  //printMatrix(h_A, M, K);
  //printMatrix(h_B, K, N);

  //double *d_A, *d_B, *d_C;
  //cudaMalloc((void**)&d_A, memSizeA);
  //cudaMalloc((void**)&d_B, memSizeB);
  //cudaMalloc((void**)&d_C, memSizeC);
  //cudaMemcpy(d_A, h_A, memSizeA, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_B, h_B, memSizeB, cudaMemcpyHostToDevice);



  //CudaTimer ct;
  //ct.start();
  //  strassen(d_A, d_B, d_C, K, N, N, K, N, N, M, K, M, 2);
  //ct.stop();

  //double strassenTime = ct.value() / iter;
  //cudaMemcpy(h_C, d_C, memSizeC, cudaMemcpyDeviceToHost);

  //printMatrix(h_C, M, N);
//  if (a_cols != b_rows) {
//    std::cout << "Impossible to multiply these two matrix";
//    return -1;
//  }
//
//  std::cout << "A: " << std::endl;
//  printMatrix(A, a_rows, a_cols);
//  std::cout << std::endl << "B: " << std::endl;
//  printMatrix(B, b_rows, b_cols);
//  std::cout << std::endl;
//
//	cudaError_t cudaStatus;
//	cudaStatus = cudaSetDevice(0);
//
//  float start = clock();
//  //std::tie(C, c_rows, c_cols) = matrixMultiplication(A, a_rows, a_cols, B, b_rows, b_cols);
//  float cpu_time = (clock() - start) / CLOCKS_PER_SEC;
//
//  std::cout << "CPU time: " << cpu_time << "sec" << std::endl;
//
//	if (cudaStatus != cudaSuccess) {
//	  fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//  }
//  else {
////    done_matrix_square(A, B, C, a_rows, a_cols, b_cols, 2);
//    CudaTimer timer;
//
//
//
//
//    double* A_d;
//    double* B_d;
//    double* C_d;
//
//cudaMalloc((void**)&A_d, sizeof(double) * a_rows * a_cols);
//cudaMalloc((void**)&B_d, sizeof(double) * b_rows * b_cols);
//cudaMalloc((void**)&C_d, sizeof(double) * N * M);
//cudaMemcpy(A_d, A, sizeof(double) * M * K, cudaMemcpyHostToDevice);
//cudaMemcpy(B_d, B, sizeof(double) * N * K, cudaMemcpyHostToDevice);
//// cudaMemcpy(C_d, C, sizeof(float) * M * N, cudaMemcpyHostToDevice);
//
//    timer.start();
//    strassen(A_d,  B_d, C_d, K, N, N, K, N, N, M,K, M, 5);
//cudaDeviceSynchronize();
//    timer.stop();
//
//    float time = timer.value();
//    
//    /*timer.start();
//    std::tie(C, c_rows, c_cols) = matrixMultiplicationInParts(A, a_rows, a_cols, B, b_rows, b_cols, 5, 5);
//    timer.stop();
//
//    float algo1_time = timer.value() ;*/
//
//    C = (double*)malloc(sizeof(double) * N * N);
//    cudaMemcpy(C, C_d, sizeof(double) * M * N, cudaMemcpyDeviceToHost);
//
//  }
//
//  printMatrix(C, M, N);
//
//  free(A);
//  free(B);
//  // free(C);

	return 0;
}