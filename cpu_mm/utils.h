std::tuple<float*, size_t, size_t> matrixMultiplication(
        const float* A, size_t a_rows, size_t a_cols,
        const float* B, size_t b_rows, size_t b_cols
);

void printMatrix(double* A, size_t rows, size_t cols);

std::tuple<double*, size_t, size_t> readMatrixFromFile(const std::string& filename);

void writeMatrixToFile(const std::string& filename, double* matrix, size_t rows, size_t cols);

void copyElements(
        float* src, size_t src_cols,
        float* dst, size_t dst_cols ,
        size_t row_offset, size_t col_offset,
        size_t rows_to_copy, size_t cols_to_copy
);

void matrixAdd(const double *A, const double *B, double *C,
        size_t lda, size_t ldb, size_t ldc,
        size_t XA, size_t XB,
        double alpha, double beta
);

void matrixMul(const double *A, const double *B, double *C,
               size_t lda, size_t ldb, size_t ldc,
               size_t XA, size_t XB, size_t XC,
               size_t YA, size_t YB, size_t YC,
               double alpha, double beta
);

void matrixMultiplication(const double *A, const double *B, double *C,
                          size_t M, size_t K, size_t N);
