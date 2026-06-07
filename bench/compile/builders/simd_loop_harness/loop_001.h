// bench/compile/builders/simd_loop_harness/loop_001.h
//
// Calling-convention contract for loop_001 (FP32 inner product) baselines.
// Matches bench/datasets/simd_loop.py SIGNATURES["loop_001"].

#pragma once
#include <stdint.h>

struct loop_001_data {
    float *a;
    float *b;
    int n;
    float res;
};

#ifdef __cplusplus
extern "C" {
#endif
// a, b: float arrays of length n; res_out: pointer to one float result.
int armbench_entry_loop_001(void *a, void *b, int64_t n, void *res_out);
#ifdef __cplusplus
}
#endif
