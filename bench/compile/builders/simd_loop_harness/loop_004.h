// bench/compile/builders/simd_loop_harness/loop_004.h
// loop_004: UINT64 inner product
#pragma once
#include <stdint.h>

struct loop_004_data {
    uint64_t *a;
    uint64_t *b;
    int n;
    uint64_t res;
};

#ifdef __cplusplus
extern "C" {
#endif
int armbench_entry_loop_004(void *a, void *b, int64_t n, void *res_out);
#ifdef __cplusplus
}
#endif
