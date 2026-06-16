#ifndef MAT_DATA_H
#define MAT_DATA_H

#include <stddef.h>
#include <stdint.h>

#define MAT_DATA_MAGIC 0x47454D4Du /* "GEMM" */
#define MAT_DATA_VERSION 1u
#define MAT_DATA_DEFAULT_SEED 20260616u

#define MAT_CREF_MAGIC 0x47454D43u /* "GEMC" */
#define MAT_CREF_VERSION 1u

typedef struct MatDataHeader {
    uint32_t magic;
    uint32_t version;
    int32_t n;
    uint32_t seed;
} MatDataHeader;

typedef struct MatCRefHeader {
    uint32_t magic;
    uint32_t version;
    int32_t n;
} MatCRefHeader;

static inline size_t mat_data_bytes(int32_t n) {
    return sizeof(MatDataHeader) + 2u * (size_t)n * (size_t)n * sizeof(double);
}

static inline size_t mat_cref_bytes(int32_t n) {
    return sizeof(MatCRefHeader) + (size_t)n * (size_t)n * sizeof(double);
}

#endif
