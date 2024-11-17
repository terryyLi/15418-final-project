#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cassert>
#include "sparse_matrix.h"

// Load a full matrix from a file and convert it to CSR format
CSRMatrix loadFullMatrixToCSR(const std::string &filePath) {
    std::ifstream inFile(filePath);
    if (!inFile) {
        throw std::runtime_error("Unable to open file: " + filePath);
    }

    int rows, cols;
    inFile >> rows >> cols;

    std::vector<std::vector<double>> fullMatrix(rows, std::vector<double>(cols, 0.0));

    // Read the full matrix
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            inFile >> fullMatrix[i][j];
        }
    }

    // Convert to CSR format
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
