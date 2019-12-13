#include <iostream>
#include <fstream>
#include <tuple>
#include <assert.h>

void printMatrix(double* A, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      std::cout << A[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
}


std::tuple<double*, size_t, size_t> readMatrixFromFile(const std::string& filename) {
  std::fstream f(filename);
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


void writeMatrixToFile(const std::string& filename, double* matrix, size_t rows, size_t cols) {
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
  f.close();
}


void matrixAdd(const double *A, const double *B, double *C,
               size_t lda, size_t ldb, size_t ldc,
               size_t XA, size_t XB,
               double alpha, double beta
) {
  for (size_t i = 0; i < XA; ++i) {
    for (size_t j = 0; j < XB; ++j) {
      C[i * ldc + j] = alpha * A[i * lda + j] + beta * B[i * ldb + j];
    }
  }
}


void matrixMul(const double *A, const double *B, double *C,
               size_t lda, size_t ldb, size_t ldc,
               size_t XA, size_t XB, size_t XC,
               size_t YA, size_t YB, size_t YC,
               double alpha, double beta
) {
  assert(XA == XC && YA == XC && YC == YB && "Matrices sizes is incorrect. matrixAdd");
  for (size_t i = 0; i < XA; ++i) {
    for (size_t j = 0; j < YB; ++j) {
      double cij = C[i * ldc + j];
      C[i * ldc + j] = 0;
      for (size_t k = 0; k < XB; ++k) {
        C[i * ldc + j] += alpha * A[i * lda + k] * B[k * ldb + j];
      }
      C[i * ldc + j] += beta * cij;
    }
  }
}


void matrixMultiplication(const double *A, const double *B, double *C,
                          size_t M, size_t K, size_t N) {
  matrixMul(A, B, C, K, N, N, M, K, M, K, N, N, 1.0, 0.0);
}
