#ifndef CUDA_MATMUL
#define CUDA_MATMUL

#include <vector>
#include <string>


// Function prototypes for CSR
extern "C" void multiplyCSR_CUDA(const CSRMatrix &A, CSRMatrix &B, CSRMatrix &C);

#endif // SPARSE_MATRIX_H
