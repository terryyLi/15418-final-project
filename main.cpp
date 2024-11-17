#include <iostream>
#include "sparse_matrix.h"

int main() {
    // Example matrices in CSR format
    CSRMatrix A = {
        {0, 2, 4},        // row_ptr
        {0, 1, 1, 2},     // col_idx
        {1.0, 2.0, 3.0, 4.0}, // values
        2,                // rows
        3                 // cols
    };

    CSRMatrix B = {
        {0, 2, 3, 4},     // row_ptr
        {0, 2, 1, 2},     // col_idx
        {5.0, 6.0, 7.0, 8.0}, // values
        3,                // rows
        3                 // cols
    };

    try {
        // Perform multiplication
        CSRMatrix C = multiplyCSR(A, B);

        // Print the result
        std::cout << "Resultant Matrix C in CSR Format:\n";
        printCSR(C);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
