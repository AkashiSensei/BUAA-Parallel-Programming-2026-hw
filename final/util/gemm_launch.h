#ifndef GEMM_LAUNCH_H
#define GEMM_LAUNCH_H

#include <cuda_runtime.h>

void gemm_launch(int n, const double *d_A, const double *d_B, double *d_C, dim3 block_dim,
                 cudaStream_t stream);

#endif
