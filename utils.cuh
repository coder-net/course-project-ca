#include <tuple>
#include <string>

void GPU_mul(double *A, double *B, double *C,
  int lda, int ldb, int ldc,
  int XA, int XB, int XC,
  int YA, int YB, int YC,
  double alpha, double beta);

void GPU_add(double *A, double *B, double *C,
  int lda, int ldb, int ldc,
  int XA, int YA,
  double alpha, double beta);

float* toColumnMajor(float* A, float* B, size_t rows, size_t cols);

float* toRowMajor(float* A, float* B, size_t rows, size_t cols);

float* cudaMatrixMultiplication(float* A, size_t a_row, size_t a_col, float* B, size_t b_row, size_t b_col, float* C, float alpha);

std::tuple<float*, size_t, size_t> matrixMultiplication(
        const float* A, size_t a_rows, size_t a_cols,
        const float* B, size_t b_rows, size_t b_cols
);

void printMatrix(double* A, size_t rows, size_t cols);

std::tuple<double*, size_t, size_t> readMatrixFromFile(std::string filename);

void writeMatrixToFile(std::string filename, double* matrix, size_t rows, size_t cols);

template <typename T>
T min(const T& lhs, const T& rhs);

template <typename T>
T max(const T& lhs, const T& rhs);

void copyElements(
  float* src, size_t src_cols,
  float* dst, size_t dst_cols,
  size_t row_offset, size_t col_offset,
  size_t rows_to_copy, size_t cols_to_copy
);