#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include <unistd.h>
#include <omp.h>
#include "sparse_matrix.h"

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
    std::cout << "Number of threads: " << num_threads << '\n';
    // Set the number of threads for OpenMP
    omp_set_num_threads(num_threads);

    try {
        // Load full matrices
        auto fullMatrixA = loadFullMatrix(fileA);
        auto fullMatrixB = loadFullMatrix(fileB);

        const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - init_start).count();
        std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(6) << init_time << std::endl;

        const auto compute_start = std::chrono::steady_clock::now();

        if (format == "csr") {
            // Convert to CSR and multiply
            CSRMatrix A = convertFullMatrixToCSR(fullMatrixA);
            CSRMatrix B = convertFullMatrixToCSR(fullMatrixB);

            // Validate dimensions
            if (A.cols != B.rows) {
                throw std::runtime_error("Matrix dimensions mismatch for multiplication: A.cols must equal B.rows.");
            }

            // Perform matrix multiplication
            CSRMatrix C = multiplyCSR(A, B);

            const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - compute_start).count();
            std::cout << "Computation time (sec): " << std::fixed << std::setprecision(6) << compute_time << std::endl;

            // Output result
            if (!outputFile.empty()) {
                std::ofstream outFile(outputFile);
                if (!outFile) {
                    throw std::runtime_error("Unable to open output file: " + outputFile);
                }
                outFile << "Row pointers:\n";
                for (int val : C.row_ptr) outFile << val << " ";
                outFile << "\nColumn indices:\n";
                for (int val : C.col_idx) outFile << val << " ";
                outFile << "\nValues:\n";
                for (double val : C.values) outFile << val << " ";
                outFile << "\n";
                std::cout << "Result saved to " << outputFile << '\n';
            } else {
                std::cout << "Resultant Matrix C in CSR Format:\n";
                // comment out to not print too many output
                // printCSR(C);
            }
        } else { // format == "coo"
            // Convert to COO and multiply
            COOMatrix A = convertFullMatrixToCOO(fullMatrixA);
            COOMatrix B = convertFullMatrixToCOO(fullMatrixB);

            // Validate dimensions
            if (A.cols != B.rows) {
                throw std::runtime_error("Matrix dimensions mismatch for multiplication: A.cols must equal B.rows.");
            }

            // Perform matrix multiplication
            COOMatrix C = multiplyCOO(A, B);

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
                for (int val : C.row_idx) outFile << val << " ";
                outFile << "\nColumn indices:\n";
                for (int val : C.col_idx) outFile << val << " ";
                outFile << "\nValues:\n";
                for (double val : C.values) outFile << val << " ";
                outFile << "\n";
                std::cout << "Result saved to " << outputFile << '\n';
            } else {
                std::cout << "Resultant Matrix C in COO Format:\n";
                printCOO(C);
            }
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}