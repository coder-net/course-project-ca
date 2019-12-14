#include <iostream>
#include <tuple>
#include <time.h>
#include <ctime>
#include <chrono>

#include "utils.h"
#include "strassen.h"



int main() {
  double *A, *B, *C;
  size_t xa, ya, xb, yb;
  std::tie(A, xa, ya) = readMatrixFromFile("../test_matrices/matrix1024_1.txt");
  std::tie(B, xb, yb) = readMatrixFromFile("../test_matrices/matrix1024_2.txt");

  size_t iteration = 10;

  // std::cout << "A" << std::endl;
  // printMatrix(A, xa, ya);

  // std::cout << "B" << std::endl;
  // printMatrix(B, xb, yb);

  C = (double*)malloc(sizeof(double) * xa * yb);

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < iteration; ++i)
    strassen_mm(A, B, C, xa, ya, yb, 4);
  auto end = std::chrono::high_resolution_clock::now();
  double strassen_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "Strassen: " << strassen_time / iteration << "microseconds" << std::endl;

  writeMatrixToFile("test_results/strassen_res.txt", C, xa, yb);

  start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < iteration; ++i)
    matrixMultiplication(A, B, C, xa, ya, yb);
  end = std::chrono::high_resolution_clock::now();
  double mm_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "Simple matrix multiplication: " << mm_time / iteration << "microseconds" << std::endl;

  writeMatrixToFile("test_results/mm_res.txt", C, xa, yb);

  free(A);
  free(B);
  free(C);

  return 0;
}