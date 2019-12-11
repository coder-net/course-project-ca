#include <iostream>
#include <fstream>
#include <tuple>
#include "cublas_v2.h"
#pragma comment(lib, "cublas.lib")




void GPU_mul(double *A, double *B, double *C,
  int lda, int ldb, int ldc,
  int XA, int XB, int XC,
  int YA, int YB, int YC,
  double alpha, double beta) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, XB, YA, XA, &alpha, B, ldb, A, lda, &beta, C, ldc);
  cublasDestroy(handle);
}

void GPU_add(double *A, double *B, double *C,
  int lda, int ldb, int ldc,
  int XA, int YA,
  double alpha, double beta) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, XA, YA, &alpha, A, lda, &beta, B, ldb, C, ldc);
  cublasDestroy(handle);
}

void cuBLAS_mm(double *d_A, double *d_B, double *d_C, size_t M, size_t N, size_t K) {
  double one = 1.0;
  double zero = 0.0;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, d_B, N, d_A, K, &zero, d_C, N);
  cublasDestroy(handle);
}

double* toColumnMajor(double* A, double* B, size_t rows, size_t cols) {
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; ++j) {
			B[j * rows + i] = A[i * cols + j];
		}
	}
	return B;
}

double* toRowMajor(double* A, double* B, size_t rows, size_t cols) {
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; ++j) {
			B[i * cols + j] = A[j * rows + i];
		}
	}
	return B;
}


double* mm(double* A, double* B, double* C,  size_t M, size_t K, size_t N) {
	// formula: alpha * A * B  + beta * C
  double alpha = 1.0;
	double beta = 0;
	cublasHandle_t handle;

	cublasCreate(&handle);

	cublasDgemm(
		handle,
		CUBLAS_OP_C,
		CUBLAS_OP_C,
		M,
		N,
		K, // equal to a_col
		&alpha,
		A,
		K,
		B,
		N,
		&beta,
		C,
		M
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


void printMatrix(double* A, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      std::cout << A[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
}

std::tuple<double*, size_t, size_t> readMatrixFromFile(std::string filename) {
  std::ifstream f(filename);
  if (!f.is_open()) {
    throw std::runtime_error("No such file");
  }
  size_t rows, cols;
  f >> rows >> cols;
  double* matrix = (double*)malloc(sizeof(double) * cols * rows);
  for (size_t i = 0; i < cols * rows; ++i) {
    f >> matrix[i];
  }
  f.close();
  return std::tie(matrix, rows, cols);
}


void writeMatrixToFile(std::string filename, double* matrix, size_t rows, size_t cols) {
  std::ofstream f(filename);
  if (!f.is_open()) {
    throw std::runtime_error("Error with writing matrix to file: " + filename);
  }
  f << rows << " " << cols << std::endl;
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      f << matrix[i * cols + j] << " ";
    }
    f << std::endl;
  }
}

template <typename T>
T min(const T& lhs, const T& rhs) {
  return lhs < rhs ? lhs : rhs;
}

template <typename T>
T max(const T& lhs, const T& rhs) {
  return lhs > rhs ? lhs : rhs;
}


void copyElements(
  double* src, size_t src_cols,
  double* dst, size_t dst_cols,
  size_t row_offset, size_t col_offset,
  size_t rows_to_copy, size_t cols_to_copy
) {
  for (size_t i = 0; i < rows_to_copy; ++i) {
    for (size_t j = 0; j < cols_to_copy; ++j) {
      dst[(row_offset + i) * dst_cols + col_offset + j] = src[i * src_cols + j];
    }
  }
}

