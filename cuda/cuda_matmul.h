#ifndef CUDA_MATMUL
#define CUDA_MATMUL

#include <vector>
#include <string>


// Function prototypes 
void multiplyCSR_CUDA(const CSRMatrix &A, CSRMatrix &B, CSRMatrix &C);
void multiplyCOO_CUDA(const COOMatrix &A, COOMatrix &B, COOMatrix &C);

#endif // SPARSE_MATRIX_H
