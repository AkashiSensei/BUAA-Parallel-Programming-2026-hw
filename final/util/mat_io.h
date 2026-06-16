#ifndef MAT_IO_H
#define MAT_IO_H

#include "mat_data.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int mat_input_path(const char *dir, int32_t n, char *buf, size_t buflen);
int mat_cref_path(const char *dir, int32_t n, char *buf, size_t buflen);

int read_mat_pair(const char *path, MatDataHeader *header, double **A, double **B);
int read_mat_cref(const char *path, MatCRefHeader *header, double **C);
int write_mat_cref(const char *path, int32_t n, const double *C);

#ifdef __cplusplus
}
#endif

#endif
