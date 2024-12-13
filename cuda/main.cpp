#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include <unordered_map>
#include <unistd.h>
#include "../sparse_matrix.h" // Contains OpenMP functions
#include "cuda_matmul.h"   // Contains CUDA functions

void cooMatMulCPU(int nnzA, const std::vector<int>& rowA, const std::vector<int>& colA, const std::vector<double>& valA,
                  int nnzB, const std::vector<int>& rowB, const std::vector<int>& colB, const std::vector<double>& valB,
                  std::vector<int>& rowC, std::vector<int>& colC, std::vector<double>& valC, int& nnzC) {
    
    // Initialize nnzC to zero at the start
    nnzC = 0;

    // Iterate over all non-zero elements in matrix A
    for (int i = 0; i < nnzA; ++i) {
        int row = rowA[i];
        int col = colA[i];
        double val = valA[i];

        // Iterate over all non-zero elements in matrix B
        for (int j = 0; j < nnzB; ++j) {
            if (rowB[j] == col) { // Matching row in B with column in A
                int resultRow = row;
                int resultCol = colB[j];
                double resultVal = val * valB[j];

                // Check if we already have this entry in the result
                bool found = false;
                for (int k = 0; k < nnzC; ++k) {
                    if (rowC[k] == resultRow && colC[k] == resultCol) {
                        valC[k] += resultVal; // Add the value if the entry already exists
                        found = true;
                        break;
                    }
                }

                // If not found, add a new entry to the result matrix
                if (!found) {
                    rowC.push_back(resultRow);
                    colC.push_back(resultCol);
                    valC.push_back(resultVal);
                    ++nnzC;
                }
            }
        }
    }
}


void csrMatMulCPU(int rowsA, int colsB, 
                  const std::vector<int> &A_row_ptr, const std::vector<int> &A_col_idx, const std::vector<double> &A_values, 
                  const std::vector<int> &B_row_ptr, const std::vector<int> &B_col_idx, const std::vector<double> &B_values, 
                  std::vector<int> &C_row_ptr, std::vector<int> &C_col_idx, std::vector<double> &C_values) {

    // Initialize C_row_ptr
    C_row_ptr.resize(rowsA + 1, 0);

    // Temporary structure to store non-zero values in the current row
    std::unordered_map<int, double> rowAccumulator;

    // Loop over rows of A
    for (int row = 0; row < rowsA; ++row) {
        rowAccumulator.clear();

        // Iterate over non-zero elements in the current row of A
        for (int i = A_row_ptr[row]; i < A_row_ptr[row + 1]; ++i) {
            int A_col = A_col_idx[i];
            double A_val = A_values[i];

            // Iterate over non-zero elements in the corresponding row of B
            for (int j = B_row_ptr[A_col]; j < B_row_ptr[A_col + 1]; ++j) {
                int B_col = B_col_idx[j];
                double B_val = B_values[j];

                // Accumulate the product into the accumulator
                rowAccumulator[B_col] += A_val * B_val;
            }
        }

        // Store accumulated values into C
        for (const auto &entry : rowAccumulator) {
            C_col_idx.push_back(entry.first);
            C_values.push_back(entry.second);
        }

        // Update the row pointer for C
        C_row_ptr[row + 1] = C_col_idx.size();
    }
}


// Load a full matrix from a file
std::vector<std::vector<double>> loadFullMatrix(const std::string &filePath) {
    std::ifstream inFile(filePath);
    if (!inFile) {
        throw std::runtime_error("Unable to open file: " + filePath);
    }

    int rows, cols;
    inFile >> rows >> cols;

    std::vector<std::vector<double>> fullMatrix(rows, std::vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            inFile >> fullMatrix[i][j];
        }
    }
    return fullMatrix;
}

int main(int argc, char *argv[]) {
    const auto init_start = std::chrono::steady_clock::now();

    // Default parameters
    std::string fileA, fileB, outputFile;
    std::string format = "csr"; // Default format is CSR
    int num_threads = 1;

    int opt;
    while ((opt = getopt(argc, argv, "a:b:f:n:o:")) != -1) {
        switch (opt) {
            case 'a':
                fileA = optarg;
                break;
            case 'b':
                fileB = optarg;
                break;
            case 'f':
                format = optarg;
                if (format != "csr" && format != "coo") {
                    std::cerr << "Error: Format must be either 'csr' or 'coo'\n";
                    exit(EXIT_FAILURE);
                }
                break;
            case 'n':
                num_threads = atoi(optarg);
                break;
            case 'o':
                outputFile = optarg;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -a matrixA_file -b matrixB_file [-f format] [-n num_threads] [-o output_file]\n";
                std::cerr << "  format: csr (default) or coo\n";
                exit(EXIT_FAILURE);
        }
    }

    // Validate required inputs
    if (fileA.empty() || fileB.empty() || num_threads <= 0) {
        std::cerr << "Error: Both matrix A (-a) and matrix B (-b) files must be specified.\n";
        exit(EXIT_FAILURE);
    }

    std::cout << "Using format: " << format << std::endl;

    try {
        // Load full matrices
        auto fullMatrixA = loadFullMatrix(fileA);
        auto fullMatrixB = loadFullMatrix(fileB);

        const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - init_start).count();
        std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(6) << init_time << std::endl;

        if (format == "csr") {
            CSRMatrix A = convertFullMatrixToCSR(fullMatrixA);
            CSRMatrix B = convertFullMatrixToCSR(fullMatrixB);
            if (A.cols != B.rows) {
                throw std::runtime_error("Matrix dimensions mismatch for multiplication: A.cols must equal B.rows.");
            }
            CSRMatrix C_cpu, C_gpu;

            C_cpu.rows = A.rows;
            C_cpu.cols = B.rows;
            C_cpu.row_ptr.resize(C_cpu.rows + 1, 0);
            C_cpu.col_idx.resize(A.rows * B.cols);
            C_cpu.values.resize(A.rows * B.cols);

            C_gpu.rows = A.rows;
            C_gpu.cols = B.rows;
            C_gpu.row_ptr.resize(C_gpu.rows + 1, 0);
            C_gpu.col_idx.resize(A.rows * B.cols);
            C_gpu.values.resize(A.rows * B.cols);

            auto cpu_start = std::chrono::steady_clock::now();
            csrMatMulCPU(A.rows, B.cols, A.row_ptr, A.col_idx, A.values, 
                            B.row_ptr, B.col_idx, B.values, 
                            C_cpu.row_ptr, C_cpu.col_idx, C_cpu.values);
            const double cpu_compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::steady_clock::now() - cpu_start).count();

            std::cout << "CPU computation time: " << cpu_compute_time << " seconds" << std::endl;

            // GPU computation

            // Use CUDA to multiply matrices
            multiplyCSR_CUDA(A, B, C_gpu);


            if (!outputFile.empty()) {
                std::ofstream outFile(outputFile);
                if (!outFile) {
                    throw std::runtime_error("Unable to open output file: " + outputFile);
                }
                outFile << "Row pointers:\n";
                for (int val : C_cpu.row_ptr) outFile << val << " ";
                outFile << "\nColumn indices:\n";
                for (int val : C_cpu.col_idx) outFile << val << " ";
                outFile << "\nValues:\n";
                for (double val : C_cpu.values) outFile << val << " ";
                outFile << "\n";
                std::cout << "Result saved to " << outputFile << '\n';
            } else {
                std::cout << "Resultant Matrix C in CSR Format:\n";
                // comment out to not print too many output
                // printCSR(C);
            }
        }
        else {
            // Convert to COO and multiply
            COOMatrix A = convertFullMatrixToCOO(fullMatrixA);
            COOMatrix B = convertFullMatrixToCOO(fullMatrixB);

            if (A.cols != B.rows) {
                throw std::runtime_error("Matrix dimensions mismatch for multiplication: A.cols must equal B.rows.");
            }
            COOMatrix C_cpu, C_gpu;
            C_cpu.rows = A.rows;
            C_cpu.cols = B.rows;
            C_cpu.row_idx.resize(C_cpu.rows + 1, 0);
            C_cpu.col_idx.resize(A.rows * B.cols);
            C_cpu.values.resize(A.rows * B.cols);

            C_gpu.rows = A.rows;
            C_gpu.cols = B.rows;
            C_gpu.row_idx.resize(C_gpu.rows + 1, 0);
            C_gpu.col_idx.resize(A.rows * B.cols);
            C_gpu.values.resize(A.rows * B.cols);


            auto cpu_start = std::chrono::steady_clock::now();
        //     cooMatMulCPU( A.row_idx.size(), d_rowA, d_colA, d_valA,
        // B.row_idx.size(), d_rowB, d_colB, d_valB,
        // d_rowC, d_colC, d_valC, d_nnzC);
            const double cpu_compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::steady_clock::now() - cpu_start).count();

            std::cout << "CPU computation time: " << cpu_compute_time << " seconds" << std::endl;

            auto compute_start = std::chrono::steady_clock::now();
            // Use CUDA to multiply matrices
            multiplyCOO_CUDA(A, B, C_gpu);

            const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - compute_start).count();
            std::cout << "Computation time (sec): " << std::fixed << std::setprecision(6) << compute_time << std::endl;

            // Output result
            if (!outputFile.empty()) {
                std::ofstream outFile(outputFile);
                if (!outFile) {
                    throw std::runtime_error("Unable to open output file: " + outputFile);
                }
                outFile << "Row indices:\n";
                for (int val : C_cpu.row_idx) outFile << val << " ";
                outFile << "\nColumn indices:\n";
                for (int val : C_cpu.col_idx) outFile << val << " ";
                outFile << "\nValues:\n";
                for (double val : C_cpu.values) outFile << val << " ";
                outFile << "\n";
                std::cout << "Result saved to " << outputFile << '\n';
            } else {
                // std::cout << "Resultant Matrix C in COO Format:\n";
                // printCOO(C_cpu);
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
