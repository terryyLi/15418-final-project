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

// Multiply two CSR matrices with OpenMP parallelization
CSRMatrix multiplyCSR(const CSRMatrix &A, const CSRMatrix &B) {
    assert(A.cols == B.rows && "Matrix dimensions must align for multiplication.");

    // Transpose B to make accessing rows easier
    CSRMatrix B_T;
    B_T.rows = B.cols;
    B_T.cols = B.rows;
    B_T.row_ptr.resize(B.cols + 1, 0);

    std::vector<std::unordered_map<int, double>> B_rows(B.cols);
    
    // Parallel preprocessing of matrix B
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < B.rows; ++i) {
        for (int j = B.row_ptr[i]; j < B.row_ptr[i + 1]; ++j) {
            int col = B.col_idx[j];
            double value = B.values[j];
            #pragma omp critical
            {
                B_rows[col][i] = value;
            }
        }
    }

    // Sequential part: compute row pointers for B_T
    for (int i = 0; i < B.cols; ++i) {
        B_T.row_ptr[i + 1] = B_T.row_ptr[i] + B_rows[i].size();
        for (const auto &entry : B_rows[i]) {
            B_T.col_idx.push_back(entry.first);
            B_T.values.push_back(entry.second);
        }
    }

    // Prepare result matrix
    CSRMatrix result;
    result.rows = A.rows;
    result.cols = B_T.rows;
    result.row_ptr.resize(result.rows + 1, 0);

    // Vector of vectors to store temporary results for each row
    std::vector<std::vector<std::pair<int, double>>> temp_results(A.rows);

    // Parallel multiplication
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < A.rows; ++i) {
        std::unordered_map<int, double> rowA;
        // Build row A
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            rowA[A.col_idx[j]] = A.values[j];
        }

        // Compute products for this row
        std::vector<std::pair<int, double>> row_results;
        for (int j = 0; j < B_T.rows; ++j) {
            double dot = dotProduct(rowA, B_rows[j]);
            if (dot != 0.0) {
                row_results.push_back({j, dot});
            }
        }
        temp_results[i] = std::move(row_results);
    }

    // Sequential part: combine results
    int current_pos = 0;
    for (int i = 0; i < A.rows; ++i) {
        result.row_ptr[i] = current_pos;
        for (const auto &entry : temp_results[i]) {
            result.col_idx.push_back(entry.first);
            result.values.push_back(entry.second);
            current_pos++;
        }
    }
    result.row_ptr[A.rows] = current_pos;

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

// Multiply two COO matrices with OpenMP parallelization
COOMatrix multiplyCOO(const COOMatrix &A, const COOMatrix &B) {
    assert(A.cols == B.rows && "Matrix dimensions must align for multiplication.");

    COOMatrix result;
    result.rows = A.rows;
    result.cols = B.cols;

    // Create a map for matrix B to easily find elements by row
    std::vector<std::vector<std::pair<int, double>>> B_by_row(B.rows);
    
    // Parallel preprocessing of matrix B
    #pragma omp parallel
    {
        // Create thread-local vectors to avoid synchronization during collection
        std::vector<std::vector<std::pair<int, double>>> local_B_by_row(B.rows);
        
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < B.values.size(); ++i) {
            local_B_by_row[B.row_idx[i]].push_back({B.col_idx[i], B.values[i]});
        }
        
        // Merge thread-local results into global B_by_row
        #pragma omp critical
        {
            for (int i = 0; i < B.rows; ++i) {
                B_by_row[i].insert(B_by_row[i].end(), 
                                 local_B_by_row[i].begin(), 
                                 local_B_by_row[i].end());
            }
        }
    }

    // Create thread-local storage for intermediate results
    std::vector<std::vector<std::tuple<int, int, double>>> thread_results;
    #pragma omp parallel
    {
        std::vector<std::tuple<int, int, double>> local_results;
        
        #pragma omp for schedule(dynamic)
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
                    local_results.emplace_back(a_row, b_col, prod);
                }
            }
        }

        // Combine thread-local results
        #pragma omp critical
        {
            thread_results.push_back(std::move(local_results));
        }
    }

    // Combine results from all threads using a hash map
    std::unordered_map<uint64_t, double> combined;
    for (const auto &thread_result : thread_results) {
        for (const auto &[row, col, val] : thread_result) {
            uint64_t key = (static_cast<uint64_t>(row) << 32) | col;
            #pragma omp atomic
            combined[key] += val;
        }
    }

    // Build final result
    for (const auto &entry : combined) {
        if (entry.second != 0.0) {
            result.row_idx.push_back(entry.first >> 32);
            result.col_idx.push_back(entry.first & 0xFFFFFFFF);
            result.values.push_back(entry.second);
        }
    }

    return result;
}

// Multiply a CSR matrix with a vector
std::vector<double> multiplyCSRVector(const CSRMatrix &A, const std::vector<double> &vec) {
    assert(A.cols == vec.size() && "Matrix and vector dimensions must align for multiplication.");

    std::vector<double> result(A.rows, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < A.rows; ++i) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            sum += A.values[j] * vec[A.col_idx[j]];
        }
        result[i] = sum;
    }

    return result;
}

// Multiply a COO matrix with a vector
std::vector<double> multiplyCOOVector(const COOMatrix &A, const std::vector<double> &vec) {
    assert(A.cols == vec.size() && "Matrix and vector dimensions must align for multiplication.");

    std::vector<double> result(A.rows, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < A.values.size(); ++i) {
        double value = A.values[i] * vec[A.col_idx[i]];
        #pragma omp atomic
        result[A.row_idx[i]] += value;
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
