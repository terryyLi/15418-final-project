#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <unordered_map>

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

__global__ void cooMatMulKernel_naive(int nnzA, int *rowA, int *colA, double *valA,
                                int nnzB, int *rowB, int *colB, double *valB,
                                int *rowC, int *colC, double *valC, int *nnzC) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnzA) return;

    int row = rowA[idx];
    int col = colA[idx];
    double val = valA[idx];

    // Iterate over all non-zero elements in matrix B
    for (int j = 0; j < nnzB; j++) {
        if (rowB[j] == col) { // Matching row in B with column in A
            int resultRow = row;
            int resultCol = colB[j];
            double resultVal = val * valB[j];

            // Atomic addition to ensure correctness for parallel writes
            int resultIdx = atomicAdd(nnzC, 1);
            rowC[resultIdx] = resultRow;
            colC[resultIdx] = resultCol;
            valC[resultIdx] = resultVal;
        }
    }
}

__global__ void cooMatMulKernel_shared(int nnzA, int *rowA, int *colA, double *valA,
                                  int nnzB, int *rowB, int *colB, double *valB,
                                  int *rowC, int *colC, double *valC, int *nnzC) {
    // Warp thread identifier
    int lane = threadIdx.x & (31);

    // Global warp index, one warp per non-zero entry of A
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    // Shared memory for reduction within a warp
    __shared__ volatile double shared_vals[1024];
    __shared__ volatile int shared_rows[1024];
    __shared__ volatile int shared_cols[1024];

    if (warp_id < nnzA) {
        // Each warp processes one non-zero element of A
        int idx = warp_id;
        int row = rowA[idx];
        int col = colA[idx];
        double val = valA[idx];

        // Iterate over non-zero entries of B matching A's column
        for (int j = lane; j < nnzB; j += 32) {
            if (rowB[j] == col) {
                int resultCol = colB[j];
                double resultVal = val * valB[j];

                // Store partial results in shared memory
                shared_vals[threadIdx.x] = resultVal;
                shared_rows[threadIdx.x] = row;
                shared_cols[threadIdx.x] = resultCol;

                __syncthreads();

                // Warp-level reduction
                if (lane < 16) shared_vals[threadIdx.x] += shared_vals[threadIdx.x + 16];
                if (lane < 8) shared_vals[threadIdx.x] += shared_vals[threadIdx.x + 8];
                if (lane < 4) shared_vals[threadIdx.x] += shared_vals[threadIdx.x + 4];
                if (lane < 2) shared_vals[threadIdx.x] += shared_vals[threadIdx.x + 2];
                if (lane < 1) shared_vals[threadIdx.x] += shared_vals[threadIdx.x + 1];

                // First thread in the warp writes the result
                if (lane == 0) {
                    int resultIdx = atomicAdd(nnzC, 1);
                    rowC[resultIdx] = shared_rows[threadIdx.x];
                    colC[resultIdx] = shared_cols[threadIdx.x];
                    valC[resultIdx] = shared_vals[threadIdx.x];
                }
            }
        }
    }
}

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

__global__ void csrMatMulKernel_shared(int rowsA, int colsB, 
                                       int *A_row_ptr, int *A_col_idx, double *A_values, 
                                       int *B_row_ptr, int *B_col_idx, double *B_values, 
                                       int *C_row_ptr, int *C_col_idx, double *C_values) {
    // Warp thread identifier
    int lane = threadIdx.x & 31;

    // Global warp index, one row per warp
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    // Shared memory for reduction
    __shared__ volatile double vals[1024];

    if (row < rowsA) {
        // Initialize temporary storage for results
        double local_values[32] = {0.0};

        // Iterate over the non-zero entries of A in the current row
        int row_start_A = A_row_ptr[row];
        int row_end_A = A_row_ptr[row + 1];

        for (int i = row_start_A; i < row_end_A; i++) {
            int A_col = A_col_idx[i];
            double A_val = A_values[i];

            // Iterate over the non-zero entries in the corresponding row of B
            int row_start_B = B_row_ptr[A_col];
            int row_end_B = B_row_ptr[A_col + 1];

            for (int j = row_start_B + lane; j < row_end_B; j += 32) {
                int B_col = B_col_idx[j];
                double B_val = B_values[j];
                local_values[B_col] += A_val * B_val;
            }
        }

        // Reduction across warp threads
        for (int col = 0; col < colsB; col++) {
            vals[threadIdx.x] = (lane < 32) ? local_values[col] : 0.0;

            if (lane < 16) vals[threadIdx.x] += vals[threadIdx.x + 16];
            if (lane <  8) vals[threadIdx.x] += vals[threadIdx.x + 8];
            if (lane <  4) vals[threadIdx.x] += vals[threadIdx.x + 4];
            if (lane <  2) vals[threadIdx.x] += vals[threadIdx.x + 2];
            if (lane <  1) vals[threadIdx.x] += vals[threadIdx.x + 1];

            // First thread in warp writes result
            if (lane == 0 && vals[threadIdx.x] != 0.0) {
                C_col_idx[row * colsB + col] = col;
                C_values[row * colsB + col] = vals[threadIdx.x];
            }
            // Update row pointers for C
            if (lane == 0 && vals[threadIdx.x] != 0) {
                C_row_ptr[row + 1] += 1;
            }
        }
    }
}


extern "C" void multiplyCSR_CUDA(const CSRMatrix &A, const CSRMatrix &B, CSRMatrix &C) {
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
    int threadsPerBlock = 128; 
    int blocksPerGrid = (A.rows + threadsPerBlock - 1) / threadsPerBlock;

    size_t sharedMemorySize = 1024 * (sizeof(int) + sizeof(double)); // Shared memory for B

    auto compute_start = std::chrono::steady_clock::now();
    csrMatMulKernel_shared<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(
        A.rows, B.cols, d_A_row_ptr, d_A_col_idx, d_A_values, 
        d_B_row_ptr, d_B_col_idx, d_B_values, 
        d_C_row_ptr, d_C_col_idx, d_C_values);
    cudaDeviceSynchronize();
    const double gpu_compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - compute_start).count();

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

    std::cout << "GPU computation time: " << gpu_compute_time << " seconds" << std::endl;
}


extern "C" void multiplyCOO_CUDA(const COOMatrix &A, const COOMatrix &B, COOMatrix &C) {
    int *d_rowA, *d_colA, *d_rowB, *d_colB, *d_rowC, *d_colC;
    double *d_valA, *d_valB, *d_valC;
    int *d_nnzC;

    int nnzC_host = 0;  // Initial non-zero count for matrix C
    int threadsPerBlock = 256;
    int blocksPerGrid = (A.row_idx.size() + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory for A
    cudaMalloc(&d_rowA, A.row_idx.size() * sizeof(int));
    cudaMalloc(&d_colA, A.col_idx.size() * sizeof(int));
    cudaMalloc(&d_valA, A.values.size() * sizeof(double));
    cudaMemcpy(d_rowA, A.row_idx.data(), A.row_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colA, A.col_idx.data(), A.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valA, A.values.data(), A.values.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate device memory for B
    cudaMalloc(&d_rowB, B.row_idx.size() * sizeof(int));
    cudaMalloc(&d_colB, B.col_idx.size() * sizeof(int));
    cudaMalloc(&d_valB, B.values.size() * sizeof(double));
    cudaMemcpy(d_rowB, B.row_idx.data(), B.row_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colB, B.col_idx.data(), B.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valB, B.values.data(), B.values.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate device memory for C (result)
    cudaMalloc(&d_rowC, A.row_idx.size() * B.col_idx.size() * sizeof(int));  // Overestimate
    cudaMalloc(&d_colC, A.row_idx.size() * B.col_idx.size() * sizeof(int));
    cudaMalloc(&d_valC, A.row_idx.size() * B.col_idx.size() * sizeof(double));
    cudaMalloc(&d_nnzC, sizeof(int));
    cudaMemcpy(d_nnzC, &nnzC_host, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    auto compute_start = std::chrono::steady_clock::now();
    cooMatMulKernel_naive <<<blocksPerGrid, threadsPerBlock>>>(
        A.row_idx.size(), d_rowA, d_colA, d_valA,
        B.row_idx.size(), d_rowB, d_colB, d_valB,
        d_rowC, d_colC, d_valC, d_nnzC);
    cudaDeviceSynchronize();
    const double gpu_compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - compute_start).count();

    // Copy back results
    cudaMemcpy(&nnzC_host, d_nnzC, sizeof(int), cudaMemcpyDeviceToHost);
    C.row_idx.resize(nnzC_host);
    C.col_idx.resize(nnzC_host);
    C.values.resize(nnzC_host);

    cudaMemcpy(C.row_idx.data(), d_rowC, nnzC_host * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C.col_idx.data(), d_colC, nnzC_host * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C.values.data(), d_valC, nnzC_host * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_rowA);
    cudaFree(d_colA);
    cudaFree(d_valA);
    cudaFree(d_rowB);
    cudaFree(d_colB);
    cudaFree(d_valB);
    cudaFree(d_rowC);
    cudaFree(d_colC);
    cudaFree(d_valC);
    cudaFree(d_nnzC);

    std::cout << "GPU computation time: " << gpu_compute_time << " seconds" << std::endl;
}
