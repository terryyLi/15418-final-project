#ifndef CUDA_MATMUL
#define CUDA_MATMUL

#include <vector>
#include <string>


// Function prototypes 
extern "C" void multiplyCSR_CUDA(const CSRMatrix &A, CSRMatrix &B, CSRMatrix &C);
extern "C" void multiplyCOO_CUDA(const COOMatrix &A, COOMatrix &B, COOMatrix &C);

#endif // SPARSE_MATRIX_H
