#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t err = (call);                                                    \
        if (err != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA 错误 %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err));                                        \
            return 1;                                                                \
        }                                                                            \
    } while (0)

#endif
