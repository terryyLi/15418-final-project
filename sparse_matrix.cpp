#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cassert>
#include "sparse_matrix.h"

// Convert a full matrix to CSR format
CSRMatrix convertFullMatrixToCSR(const std::vector<std::vector<double>> &fullMatrix) {
    int rows = fullMatrix.size();
    int cols = rows > 0 ? fullMatrix[0].size() : 0;

    CSRMatrix csr;
    csr.rows = rows;
    csr.cols = cols;
    csr.row_ptr.push_back(0); // Starting with the first row

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (fullMatrix[i][j] != 0.0) {
                csr.values.push_back(fullMatrix[i][j]);
                csr.col_idx.push_back(j);
            }
        }
        csr.row_ptr.push_back(csr.values.size());
    }
    
    return csr;
}

// Helper function to perform sparse row dot product
double dotProduct(const std::unordered_map<int, double> &rowA, 
                  const std::unordered_map<int, double> &rowB) {
    double result = 0.0;
    for (const auto &entry : rowA) {
        int col = entry.first;
        if (rowB.count(col)) {
            result += entry.second * rowB.at(col);
        }
    }
    return result;
}

// Multiply two CSR matrices
CSRMatrix multiplyCSR(const CSRMatrix &A, const CSRMatrix &B) {
    assert(A.cols == B.rows && "Matrix dimensions must align for multiplication.");

    // Transpose B to make accessing rows easier
    CSRMatrix B_T;
    B_T.rows = B.cols;
    B_T.cols = B.rows;
    B_T.row_ptr.resize(B.cols + 1, 0);

    std::vector<std::unordered_map<int, double>> B_rows(B.cols);
    
    for (int i = 0; i < B.rows; ++i) {
        for (int j = B.row_ptr[i]; j < B.row_ptr[i + 1]; ++j) {
            int col = B.col_idx[j];
            double value = B.values[j];
            B_rows[col][i] = value;
        }
    }

    for (int i = 0; i < B.cols; ++i) {
        B_T.row_ptr[i + 1] = B_T.row_ptr[i] + B_rows[i].size();
        for (const auto &entry : B_rows[i]) {
            B_T.col_idx.push_back(entry.first);
            B_T.values.push_back(entry.second);
        }
    }

    // Multiply A and B_T
    CSRMatrix result;
    result.rows = A.rows;
    result.cols = B_T.rows;
    result.row_ptr.resize(result.rows + 1, 0);

    for (int i = 0; i < A.rows; ++i) {
        std::unordered_map<int, double> rowA;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            rowA[A.col_idx[j]] = A.values[j];
        }

        for (int j = 0; j < B_T.rows; ++j) {
            double dot = dotProduct(rowA, B_rows[j]);
            if (dot != 0.0) {
                result.col_idx.push_back(j);
                result.values.push_back(dot);
            }
        }

        result.row_ptr[i + 1] = result.col_idx.size();
    }

    return result;
}

// Convert a full matrix to COO format
COOMatrix convertFullMatrixToCOO(const std::vector<std::vector<double>> &fullMatrix) {
    int rows = fullMatrix.size();
    int cols = rows > 0 ? fullMatrix[0].size() : 0;

    COOMatrix coo;
    coo.rows = rows;
    coo.cols = cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (fullMatrix[i][j] != 0.0) {
                coo.row_idx.push_back(i);
                coo.col_idx.push_back(j);
                coo.values.push_back(fullMatrix[i][j]);
            }
        }
    }

    return coo;
}

// Multiply two COO matrices directly (sequential version)
COOMatrix multiplyCOO(const COOMatrix &A, const COOMatrix &B) {
    assert(A.cols == B.rows && "Matrix dimensions must align for multiplication.");

    COOMatrix result;
    result.rows = A.rows;
    result.cols = B.cols;

    // Create a map for matrix B to easily find elements by row
    std::vector<std::vector<std::pair<int, double>>> B_by_row(B.rows);
    for (size_t i = 0; i < B.values.size(); ++i) {
        B_by_row[B.row_idx[i]].push_back({B.col_idx[i], B.values[i]});
    }

    // For each non-zero element in A
    for (size_t i = 0; i < A.values.size(); ++i) {
        int a_row = A.row_idx[i];
        int a_col = A.col_idx[i];
        double a_val = A.values[i];

        // Multiply with corresponding elements in B
        for (const auto &b_entry : B_by_row[a_col]) {
            int b_col = b_entry.first;
            double b_val = b_entry.second;
            double prod = a_val * b_val;

            if (prod != 0.0) {
                result.row_idx.push_back(a_row);
                result.col_idx.push_back(b_col);
                result.values.push_back(prod);
            }
        }
    }

    // Combine duplicate entries
    std::unordered_map<uint64_t, double> combined;
    for (size_t i = 0; i < result.values.size(); ++i) {
        uint64_t key = (static_cast<uint64_t>(result.row_idx[i]) << 32) | result.col_idx[i];
        combined[key] += result.values[i];
    }

    // Clear and rebuild result
    result.row_idx.clear();
    result.col_idx.clear();
    result.values.clear();

    for (const auto &entry : combined) {
        if (entry.second != 0.0) {
            result.row_idx.push_back(entry.first >> 32);
            result.col_idx.push_back(entry.first & 0xFFFFFFFF);
            result.values.push_back(entry.second);
        }
    }

    return result;
}

// Print a CSR matrix
void printCSR(const CSRMatrix &mat) {
    std::cout << "CSR Matrix:" << std::endl;
    std::cout << "Rows: " << mat.rows << ", Cols: " << mat.cols << std::endl;
    std::cout << "Row pointers: ";
    for (int val : mat.row_ptr) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "Column indices: ";
    for (int val : mat.col_idx) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "Values: ";
    for (double val : mat.values) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

// Print a COO matrix
void printCOO(const COOMatrix &mat) {
    std::cout << "COO Matrix:" << std::endl;
    std::cout << "Dimensions: " << mat.rows << " x " << mat.cols << std::endl;
    std::cout << "Non-zero elements:" << std::endl;
    for (size_t i = 0; i < mat.values.size(); ++i) {
        std::cout << "(" << mat.row_idx[i] << ", " << mat.col_idx[i] << "): " 
                 << mat.values[i] << std::endl;
    }
}
