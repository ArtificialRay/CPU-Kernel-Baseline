// bench/compile/builders/simd_loop_harness/loop_002.h
// loop_002: UINT32 inner product
#pragma once
#include <stdint.h>

struct loop_002_data {
    uint32_t *a;
    uint32_t *b;
    int n;
    uint32_t res;
};

#ifdef __cplusplus
extern "C" {
#endif
int armbench_entry_loop_002(void *a, void *b, int64_t n, void *res_out);
#ifdef __cplusplus
}
#endif
