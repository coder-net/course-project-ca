#include <iostream>
#include <tuple>
#include <time.h>
#include <ctime>
#include <chrono>
#include <fstream>
#include "utils.h"
#include "strassen.h"


int main() {
  double *A, *B, *C;
  size_t xa, ya, xb, yb;
  std::ofstream f("console_log.txt");

  size_t iteration = 100;
  for (int size = 2; size <= 1024; size *= 2) {
    std::string number = std::to_string(size);
    std::tie(A, xa, ya) = readMatrixFromFile("../test_matrices/matrix" + number + "_1.txt");
    std::tie(B, xb, yb) = readMatrixFromFile("../test_matrices/matrix" + number + "_2.txt");

    std::cout << size << std::endl;
    f << size << std::endl;

    C = (double *) malloc(sizeof(double) * xa * yb);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iteration; ++i)
      strassen_mm(A, B, C, xa, ya, yb, 4);
    auto end = std::chrono::high_resolution_clock::now();
    double strassen_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    f << "Strassen: " << strassen_time / iteration << " microseconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iteration; ++i)
      matrixMultiplication(A, B, C, xa, ya, yb);
    end = std::chrono::high_resolution_clock::now();
    double mm_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    f << "Simple_matrix_multiplication: " << mm_time / iteration << " microseconds" << std::endl;
  }

  free(A);
  free(B);
  free(C);

  return 0;
}
