#include "gemm_launch.h"

__global__ void gemm_naive_kernel(int n, const double *A, const double *B, double *C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) {
        return;
    }

    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
}

void gemm_launch(int n, const double *d_A, const double *d_B, double *d_C, dim3 block_dim,
                 cudaStream_t stream) {
    const dim3 grid_dim((unsigned)((n + block_dim.x - 1) / block_dim.x),
                        (unsigned)((n + block_dim.y - 1) / block_dim.y));
    gemm_naive_kernel<<<grid_dim, block_dim, 0, stream>>>(n, d_A, d_B, d_C);
}
