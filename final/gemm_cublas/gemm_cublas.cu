#include "gemm_launch.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUBLAS_ABORT(call)                                                            \
    do {                                                                              \
        cublasStatus_t status = (call);                                               \
        if (status != CUBLAS_STATUS_SUCCESS) {                                        \
            std::fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__,       \
                         __LINE__, static_cast<int>(status));                         \
            std::abort();                                                             \
        }                                                                             \
    } while (0)

void gemm_launch(int n, const double *d_A, const double *d_B, double *d_C, dim3,
                 cudaStream_t stream) {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        CUBLAS_ABORT(cublasCreate(&handle));
    }
    CUBLAS_ABORT(cublasSetStream(handle, stream));

    const double alpha = 1.0;
    const double beta = 0.0;

    // cuBLAS uses column-major matrices. Row-major C = A * B has the same memory
    // layout as column-major C^T = B^T * A^T, so pass B before A without transpose.
    CUBLAS_ABORT(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_B, n,
                             d_A, n, &beta, d_C, n));
}
