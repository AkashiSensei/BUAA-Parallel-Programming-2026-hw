#ifndef MAT_COMPARE_H
#define MAT_COMPARE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

double mat_max_rel_err(int32_t n, const double *got, const double *ref, double eps);

#ifdef __cplusplus
}
#endif

#endif
