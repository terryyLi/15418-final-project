#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include <unistd.h>
#include "sparse_matrix.h"


int main(int argc, char *argv[]) {
    const auto init_start = std::chrono::steady_clock::now();

    // Default parameters
    std::string fileA, fileB, outputFile;
    int opt;
    while ((opt = getopt(argc, argv, "a:b:o:")) != -1) {
        switch (opt) {
            case 'a':
                fileA = optarg;
                break;
            case 'b':
                fileB = optarg;
                break;
            case 'o':
                outputFile = optarg;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -a matrixA_file -b matrixB_file [-o output_file]\n";
                exit(EXIT_FAILURE);
        }
    }

    // Validate required inputs
    if (fileA.empty() || fileB.empty()) {
        std::cerr << "Error: Both matrix A (-a) and matrix B (-b) files must be specified.\n";
        exit(EXIT_FAILURE);
    }

    try {
        // Load full matrices and convert to CSR
        CSRMatrix A = loadFullMatrixToCSR(fileA);
        CSRMatrix B = loadFullMatrixToCSR(fileB);

        // Validate dimensions
        if (A.cols != B.rows) {
            throw std::runtime_error("Matrix dimensions mismatch for multiplication: A.cols must equal B.rows.");
        }

        const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
        std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(6) << init_time << '\n';

        const auto compute_start = std::chrono::steady_clock::now();

        // Perform matrix multiplication
        CSRMatrix C = multiplyCSR(A, B);

        const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
        std::cout << "Computation time (sec): " << std::fixed << std::setprecision(6) << compute_time << '\n';

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
            printCSR(C);
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        exit(EXIT_FAILURE);
    }

    return 0;
}
