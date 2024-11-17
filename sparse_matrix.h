#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <vector>
#include <string>

// Define the CSRMatrix structure
struct CSRMatrix {
    int rows, cols;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
};

// Define the COOMatrix structure
struct COOMatrix {
    int rows, cols;
    std::vector<int> row_idx;    // Row indices
    std::vector<int> col_idx;    // Column indices
    std::vector<double> values;  // Non-zero values
};

// Function prototypes for CSR
CSRMatrix convertFullMatrixToCSR(const std::vector<std::vector<double>> &fullMatrix);
CSRMatrix multiplyCSR(const CSRMatrix &A, const CSRMatrix &B);
void printCSR(const CSRMatrix &mat);

// Function prototypes for COO
COOMatrix convertFullMatrixToCOO(const std::vector<std::vector<double>> &fullMatrix);
COOMatrix multiplyCOO(const COOMatrix &A, const COOMatrix &B);
void printCOO(const COOMatrix &mat);

#endif // SPARSE_MATRIX_H
