#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include "../sparse_matrix.h"
#include "cuda_matmul.h"


#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
if (code != cudaSuccess)
{
fprintf(stderr, "CUDA Error: %s at %s:%d\n",
cudaGetErrorString(code), file, line);
if (abort) exit(code);
}
}
#else
#define cudaCheckError(ans) ans
#endif

__global__ void csrMatMulKernel_naive(int rowsA, int colsB, 
                                      int *A_row_ptr, int *A_col_idx, double *A_values, 
                                      int *B_row_ptr, int *B_col_idx, double *B_values, 
                                      int *C_row_ptr, int *C_col_idx, double *C_values) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA) {
        double value;

        // i indicates the index of the non-zero value in A we currently are
        for (int i = A_row_ptr[row]; i < A_row_ptr[row + 1]; i++) {
            // A_col indicates the corresponding row in B
            int A_col = A_col_idx[i];  
            double A_val = A_values[i]; 

            // j indicates the index of the non_zero value in B we currently are
            for (int j = B_row_ptr[A_col]; j < B_row_ptr[A_col + 1]; j++) {
                // B_col gives the column index of the current non-zero value in B
                int B_col = B_col_idx[j];   
                double B_val = B_values[j]; 

                // Multiply and accumulate into C
                value = A_val * B_val;
                if (value != 0){
                    C_col_idx[row * colsB + B_col] = B_col;
                    C_row_ptr[row + 1] += 1;
                    C_values[row * colsB + B_col] = value;
                }
            }
        }
    }
}


extern "C" void multiplyCSR_CUDA(const CSRMatrix &A, CSRMatrix &B, CSRMatrix &C) {

    int threadsPerBlock = 512;
    int blocksPerGrid = (C.rows + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory
    int *d_B_row_ptr, *d_B_col_idx;
    double *d_B_values;

    int *d_A_row_ptr, *d_A_col_idx, *d_C_row_ptr;
    double *d_A_values, *d_C_values;
    int *d_C_col_idx;

    // Allocate and copy CSR data for A
    cudaMalloc(&d_A_row_ptr, A.row_ptr.size() * sizeof(int));
    cudaMalloc(&d_A_col_idx, A.col_idx.size() * sizeof(int));
    cudaMalloc(&d_A_values, A.values.size() * sizeof(double));
    cudaMemcpy(d_A_row_ptr, A.row_ptr.data(), A.row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_col_idx, A.col_idx.data(), A.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, A.values.data(), A.values.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate and copy CSR data for B
    cudaMalloc(&d_B_row_ptr, B.row_ptr.size() * sizeof(int));
    cudaMalloc(&d_B_col_idx, B.col_idx.size() * sizeof(int));
    cudaMalloc(&d_B_values, B.values.size() * sizeof(double));
    cudaMemcpy(d_B_row_ptr, B.row_ptr.data(), B.row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_col_idx, B.col_idx.data(), B.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_values, B.values.data(), B.values.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate memory for C
    cudaMalloc(&d_C_row_ptr, (C.rows + 1) * sizeof(int));
    cudaMalloc(&d_C_col_idx, (A.rows * B.cols) * sizeof(int));  // Estimate non-zero count for C
    cudaMalloc(&d_C_values, (A.rows * B.cols) * sizeof(double));

    // Kernel launch
    csrMatMulKernel_naive<<<blocksPerGrid, threadsPerBlock>>>(
        A.rows, B.cols, d_A_row_ptr, d_A_col_idx, d_A_values, 
        d_B_row_ptr, d_B_col_idx, d_B_values, 
        d_C_row_ptr, d_C_col_idx, d_C_values);

    // Copy results back to host
    cudaMemcpy(C.row_ptr.data(), d_C_row_ptr, (C.rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C.col_idx.data(), d_C_col_idx, (A.rows * B.cols) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C.values.data(), d_C_values, (A.rows * B.cols) * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A_row_ptr);
    cudaFree(d_A_col_idx);
    cudaFree(d_A_values);
    cudaFree(d_B_row_ptr);
    cudaFree(d_B_col_idx);
    cudaFree(d_B_values);
    cudaFree(d_C_row_ptr);
    cudaFree(d_C_col_idx);
    cudaFree(d_C_values);
}

