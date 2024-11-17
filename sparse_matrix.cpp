#include "sparse_matrix.h"
#include <iostream>
#include <stdexcept>
#include <unordered_map>

CSRMatrix multiplyCSR(const CSRMatrix &A, const CSRMatrix &B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    // Transpose B for column-wise access
    std::vector<std::vector<int>> BT_col_idx(B.cols);
    std::vector<std::vector<double>> BT_values(B.cols);
    for (int i = 0; i < B.rows; ++i) {
        for (int j = B.row_ptr[i]; j < B.row_ptr[i + 1]; ++j) {
            int col = B.col_idx[j];
            double val = B.values[j];
            BT_col_idx[col].push_back(i);
            BT_values[col].push_back(val);
        }
    }

    // Initialize result matrix C
    CSRMatrix C;
    C.rows = A.rows;
    C.cols = B.cols;
    C.row_ptr.push_back(0);

    for (int i = 0; i < A.rows; ++i) {
        std::unordered_map<int, double> row_result; // Temporary storage for the current row
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int colA = A.col_idx[j];
            double valA = A.values[j];
            for (size_t k = 0; k < BT_col_idx[colA].size(); ++k) {
                int colB = BT_col_idx[colA][k];
                double valB = BT_values[colA][k];
                row_result[colB] += valA * valB;
            }
        }

        // Store non-zero results into C
        for (const auto &[col, value] : row_result) {
            if (value != 0.0) {
                C.col_idx.push_back(col);
                C.values.push_back(value);
            }
        }
        C.row_ptr.push_back(C.col_idx.size());
    }

    return C;
}

void printCSR(const CSRMatrix &mat) {
    std::cout << "Row Ptr: ";
    for (int x : mat.row_ptr) std::cout << x << " ";
    std::cout << "\nCol Idx: ";
    for (int x : mat.col_idx) std::cout << x << " ";
    std::cout << "\nValues: ";
    for (double x : mat.values) std::cout << x << " ";
    std::cout << std::endl;
}
