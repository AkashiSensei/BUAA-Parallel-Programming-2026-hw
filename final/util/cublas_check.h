#ifndef CUBLAS_CHECK_H
#define CUBLAS_CHECK_H

#include <cublas_v2.h>
#include <stdio.h>

#define CUBLAS_CHECK(call)                                                           \
    do {                                                                             \
        cublasStatus_t st = (call);                                                  \
        if (st != CUBLAS_STATUS_SUCCESS) {                                           \
            fprintf(stderr, "cuBLAS 错误 %s:%d: 状态码 %d\n", __FILE__, __LINE__,  \
                    (int)st);                                                        \
            return 1;                                                                \
        }                                                                            \
    } while (0)

#endif
