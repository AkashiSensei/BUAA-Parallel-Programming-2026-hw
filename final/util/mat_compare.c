#include "mat_compare.h"

#include <math.h>
#include <stddef.h>

double mat_max_rel_err(int32_t n, const double *got, const double *ref, double eps) {
    if (!got || !ref || n <= 0) {
        return INFINITY;
    }

    const size_t count = (size_t)n * (size_t)n;
    double max_rel = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double denom = fabs(ref[i]) + eps;
        const double rel = fabs(got[i] - ref[i]) / denom;
        if (rel > max_rel) {
            max_rel = rel;
        }
    }
    return max_rel;
}
