// bench/compile/builders/simd_loop_harness/loop_003.h
// loop_003: FP64 inner product
#pragma once
#include <stdint.h>

struct loop_003_data {
    double *a;
    double *b;
    int n;
    double res;
};

#ifdef __cplusplus
extern "C" {
#endif
int armbench_entry_loop_003(void *a, void *b, int64_t n, void *res_out);
#ifdef __cplusplus
}
#endif
