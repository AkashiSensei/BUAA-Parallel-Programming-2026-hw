#include "gemm_launch.h"

__global__ void gemm_shared_kernel(int n, const double *A, const double *B, double *C) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tile = blockDim.x;
    const int row = blockIdx.y * tile + ty;
    const int col = blockIdx.x * tile + tx;

    extern __shared__ double shared_mem[];
    double *As = shared_mem;
    double *Bs = shared_mem + tile * tile;

    double sum = 0.0;
    const int num_tiles = (n + tile - 1) / tile;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * tile + tx;
        const int a_row = blockIdx.y * tile + ty;
        As[ty * tile + tx] = (a_row < n && a_col < n) ? A[a_row * n + a_col] : 0.0;

        const int b_row = t * tile + ty;
        const int b_col = blockIdx.x * tile + tx;
        Bs[ty * tile + tx] = (b_row < n && b_col < n) ? B[b_row * n + b_col] : 0.0;

        __syncthreads();

        for (int k = 0; k < tile; ++k) {
            sum += As[ty * tile + k] * Bs[k * tile + tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

void gemm_launch(int n, const double *d_A, const double *d_B, double *d_C, dim3 block_dim,
                 cudaStream_t stream) {
    const dim3 grid_dim((unsigned)((n + block_dim.x - 1) / block_dim.x),
                        (unsigned)((n + block_dim.y - 1) / block_dim.y));
    const size_t shmem_bytes = 2 * (size_t)block_dim.x * (size_t)block_dim.y * sizeof(double);
    gemm_shared_kernel<<<grid_dim, block_dim, shmem_bytes, stream>>>(n, d_A, d_B, d_C);
}
