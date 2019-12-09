#include <iostream>
#include <fstream>
#include <tuple>
#include "cublas_v2.h"
#pragma comment(lib, "cublas.lib")


float* toColumnMajor(float* A, float* B, size_t rows, size_t cols) {
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; ++j) {
			B[j * rows + i] = A[i * cols + j];
		}
	}
	return B;
}

float* toRowMajor(float* A, float* B, size_t rows, size_t cols) {
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; ++j) {
			B[i * cols + j] = A[j * rows + i];
		}
	}
	return B;
}


float* cudaMatrixMultiplication(float* A, size_t a_row, size_t a_col, float* B, size_t b_row, size_t b_col, float* C, float alpha) {
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

std::tuple<float*, size_t, size_t> matrixMultiplication(
        const float* A, size_t a_rows, size_t a_cols,
        const float* B, size_t b_rows, size_t b_cols
) {
  size_t c_rows = a_rows;
  size_t c_cols = b_cols;
  float* C = (float*)malloc(sizeof(float) * c_rows * c_cols);
  for (size_t i = 0; i < a_rows; ++i) {
    for (size_t j = 0; j < b_cols; ++j) {
      float value = 0;
      for (size_t k = 0; k < a_cols; ++k) {
        value += A[i * a_cols + k] * B[k * b_cols + j];
      }
      C[i * c_cols + j] = value;
    }
  }
  return std::tie(C, c_rows, c_cols);
}


void printMatrix(float* A, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      std::cout << A[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
}

std::tuple<float*, size_t, size_t> readMatrixFromFile(std::string filename) {
  std::ifstream f(filename);
  if (!f.is_open()) {
    throw std::runtime_error("No such file");
  }
  size_t rows, cols;
  f >> rows >> cols;
  float* matrix = (float*)malloc(sizeof(float) * cols * rows);
  for (size_t i = 0; i < cols * rows; ++i) {
    f >> matrix[i];
  }
  f.close();
  return std::tie(matrix, rows, cols);
}