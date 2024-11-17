#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <vector>
#include <unordered_map>

// CSR Matrix representation
struct CSRMatrix {
    std::vector<int> row_ptr;    // Row pointers
    std::vector<int> col_idx;    // Column indices
    std::vector<double> values; // Non-zero values
    int rows;                    // Number of rows
    int cols;                    // Number of columns
};

// Multiply two CSR matrices
CSRMatrix multiplyCSR(const CSRMatrix &A, const CSRMatrix &B);

// Print a CSR matrix
void printCSR(const CSRMatrix &mat);

#endif // SPARSE_MATRIX_H