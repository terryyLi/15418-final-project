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

// Function prototypes
CSRMatrix convertFullMatrixToCSR(const std::vector<std::vector<double>> &fullMatrix);
CSRMatrix multiplyCSR(const CSRMatrix &A, const CSRMatrix &B);
void printCSR(const CSRMatrix &mat);

#endif // SPARSE_MATRIX_H
