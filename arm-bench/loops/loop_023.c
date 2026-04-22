/*----------------------------------------------------------------------------
#
#   Loop 023: Conjugate Gradient
#
#   Purpose:
#     Use of gathers load instruction.
#
#   This file is part of the SIMD Loops project. For more information, visit:
#     https://gitlab.arm.com/architecture/simd-loops
#
#   Copyright (c) 2025, Arm Limited. All rights reserved.
#
#   SPDX-License-Identifier: BSD-3-Clause
#
----------------------------------------------------------------------------*/

#include "helpers.h"
#include "loops.h"


struct loop_023_data {
  double *restrict a;
  double *restrict b;
  uint32_t *restrict indexes;
  int n;
  double res;
};


#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_023(struct loop_023_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_023(struct loop_023_data *restrict data) {
  double *a = data->a;
  double *b = data->b;
  uint32_t *indexes = data->indexes;
  int n = data->n;

  double res = 0;
  for (int i = 0; i < n; i++) {
    res = res + a[indexes[i]] * b[i];
  }
  data->res = res;
}
#elif defined(HAVE_SVE_INTRINSICS)

static void inner_loop_023(struct loop_023_data *restrict data) {
  double *a = data->a;
  double *b = data->b;
  uint32_t *indexes = data->indexes;
  int n = data->n;

  svfloat64_t acc = svdup_f64(0.0);

  svbool_t p;
  FOR_LOOP_64(int, i, 0, n, p) {
    svuint64_t idx = svld1uw_u64(p, indexes + i);
    svfloat64_t a_vec = svld1_gather_index(p, a, idx);
    svfloat64_t b_vec = svld1(p, b + i);
    acc = svmla_m(p, acc, a_vec, b_vec);
  }
  data->res = svaddv(svptrue_b64(), acc);
}
#elif defined(__ARM_FEATURE_SVE2p1)

static void inner_loop_023(struct loop_023_data *restrict data) {
  double *a = data->a;
  double *b = data->b;
  uint32_t *indexes = data->indexes;
  uint64_t n = data->n;

  uint64_t i = 0, c1 = 0;
  double res;
  asm volatile(
      "   ptrue   p2.d                                        \n"
      "   rdvl    %[c1], #1                                   \n"
      "   add     %[c1], %[c0], %[c1], lsr #1                 \n"
      "   mov     z0.d, #0                                    \n"
      "   mov     z1.d, #0                                    \n"
      "   whilelt pn8.d, %[i], %[n], vlx2                     \n"
      "   b.none  2f                                          \n"
      "1:                                                     \n"
      "   ld1d    {z2.d-z3.d}, pn8/z, [%[b], %[i], lsl #3]    \n"
      "   pext    {p0.d,p1.d}, pn8[0]                         \n"
      "   ld1w    {z4.d}, p0/z, [%[c0], %[i], lsl #2]         \n"
      "   ld1w    {z5.d}, p1/z, [%[c1], %[i], lsl #2]         \n"
      "   ld1d    {z6.d}, p0/z, [%[a], z4.d, lsl #3]          \n"
      "   ld1d    {z7.d}, p1/z, [%[a], z5.d, lsl #3]          \n"
      "   incw    %[i]                                        \n"
      "   fmla    z0.d, p0/m, z2.d, z6.d                      \n"
      "   fmla    z1.d, p1/m, z3.d, z7.d                      \n"
      "   whilelt pn8.d, %[i], %[n], vlx2                     \n"
      "   b.first 1b                                          \n"
      "2:                                                     \n"
      "   fadd    z0.d, p2/m, z0.d, z1.d                      \n"
      "   faddv   %d[res], p2, z0.d                           \n"
      : [i] "+&r"(i), [res] "=&w"(res), [c1] "=&r"(c1)
      : [n] "r"(n), [a] "r"(a), [b] "r"(b), [c0] "r"(indexes)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "p0", "p1", "p2", "p8",
        "cc", "memory");

  data->res = res;
}
#elif defined(__ARM_FEATURE_SVE)

static void inner_loop_023(struct loop_023_data *restrict data) {
  double *a = data->a;
  double *b = data->b;
  uint32_t *indexes = data->indexes;
  int n = data->n;

  double res;
  int64_t i;
  int64_t indexes2;
  int64_t b2;

  asm volatile(
      "     mov     z0.d, #0                                      \n"
      "     mov     z10.d, #0                                     \n"
      "     mov     %[i], #0                                      \n"
      "     addvl   %[b2], %[b], #1                               \n"
      "     rdvl    %[indexes2], #1                               \n"
      "     add     %[indexes2], %[indexes], %[indexes2], lsr #1  \n"
      "     ptrue   p1.b                                          \n"
      "     whilelo p0.d, %[i], %x[n]                             \n"
      "1:   ld1sw   {z3.d},  p1/z, [%[indexes], %[i], lsl #2]     \n"
      "     ld1d    {z2.d},  p1/z, [%[b], %[i], lsl #3]           \n"
      "     ld1sw   {z13.d}, p0/z, [%[indexes2], %[i], lsl #2]    \n"
      "     ld1d    {z12.d}, p0/z, [%[b2], %[i], lsl #3]          \n"
      "     ld1d    {z1.d},  p1/z, [%[a], z3.d, lsl #3]           \n"
      "     ld1d    {z11.d}, p0/z, [%[a], z13.d, lsl #3]          \n"
      "     incw    %[i]                                          \n"
      "     fmla    z0.d,  p1/m, z1.d,  z2.d                      \n"
      "     fmla    z10.d, p0/m, z11.d, z12.d                     \n"
      "     whilelo p0.d, %[i], %x[n]                             \n"
      "     b.any   1b                                            \n"
      "     fadd    z0.d, p1/m, z0.d, z10.d                       \n"
      "     faddv   %d[res], p1, z0.d                             \n"
      // output operands, source operands, and clobber list
      :
      [i] "=&r"(i), [res] "=&w"(res), [b2] "=&r"(b2), [indexes2] "=&r"(indexes2)
      : [n] "r"(n), [a] "r"(a), [b] "r"(b), [indexes] "r"(indexes)
      : "v0", "v1", "v2", "v3", "v10", "v11", "v12", "v13", "p0", "p1",
        "cc", "memory");

  data->res = res;
}
#elif defined(__ARM_NEON)

// Neon version
static void inner_loop_023(struct loop_023_data *restrict data) {
  double *a = data->a;
  double *b = data->b;
  uint32_t *indexes = data->indexes;
  int n = data->n;

  double res;
  int64_t idx1;
  int64_t idx2;
  int64_t idx3;
  int64_t idx4;
  int64_t ptr;
  int64_t ptr2;

  asm volatile(
      "     movi    v0.16b, #0                            \n"
      "     movi    v10.16b, #0                           \n"
      "1:   ldpsw   %[idx1], %[idx2], [%[indexes]]        \n"
      "     ldpsw   %[idx3], %[idx4], [%[indexes], #8]    \n"
      "     ldr     q2,  [%[b]]                           \n"
      "     ldr     q12, [%[b], #16]                      \n"
      "     ldr     d1,  [%[a], %[idx1], lsl #3]          \n"
      "     ldr     d11, [%[a], %[idx3], lsl #3]          \n"
      "     add     %[ptr],  %[a], %[idx2], lsl #3        \n"
      "     add     %[ptr2], %[a], %[idx4], lsl #3        \n"
      "     ld1     {v1.d}[1], [%[ptr]]                   \n"
      "     ld1     {v11.d}[1], [%[ptr2]]                 \n"
      "     fmla    v0.2d, v1.2d, v2.2d                   \n"
      "     fmla    v10.2d, v11.2d, v12.2d                \n"
      "     add     %[indexes], %[indexes], #16           \n"
      "     add     %[b], %[b], #32                       \n"
      "     cmp     %[indexes], %[lmt]                    \n"
      "     b.ne    1b                                    \n"
      "     fadd    v0.2d, v0.2d, v10.2d                  \n"
      "     faddp   %[res].2d, v0.2d, v0.2d               \n"
      // output operands, source operands, and clobber list
      : [idx1] "=&r"(idx1), [idx2] "=&r"(idx2), [idx3] "=&r"(idx3),
        [idx4] "=&r"(idx4), [indexes] "+&r"(indexes), [res] "=&w"(res),
        [b] "+&r"(b), [ptr] "=&r"(ptr), [ptr2] "=&r"(ptr2)
      : [lmt] "r"(indexes + n), [a] "r"(a)
      : "v0", "v1", "v2", "v10", "v11", "v12", "cc", "memory");

  data->res = res;
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)

// Scalar version
static void inner_loop_023(struct loop_023_data *restrict data) {
  double *a = data->a;
  double *b = data->b;
  uint32_t *indexes = data->indexes;
  int n = data->n;

  double res;
  int64_t idx1;
  int64_t idx2;
  int64_t idx3;
  int64_t idx4;

  asm volatile(
      "     fmov    d0, xzr                               \n"
      "     fmov    d1, xzr                               \n"
      "     fmov    d10, xzr                              \n"
      "     fmov    d11, xzr                              \n"
      "1:   ldpsw   %[idx1], %[idx2], [%[indexes]]        \n"
      "     ldpsw   %[idx3], %[idx4], [%[indexes], #8]    \n"
      "     ldp     d4, d5, [%[b]]                        \n"
      "     ldp     d14, d15, [%[b], #16]                 \n"
      "     ldr     d2, [%[a], %[idx1], lsl #3]           \n"
      "     ldr     d3, [%[a], %[idx2], lsl #3]           \n"
      "     ldr     d12, [%[a], %[idx3], lsl #3]          \n"
      "     ldr     d13, [%[a], %[idx4], lsl #3]          \n"
      "     fmadd   d0, d2, d4, d0                        \n"
      "     fmadd   d1, d3, d5, d1                        \n"
      "     fmadd   d10, d12, d14, d10                    \n"
      "     fmadd   d11, d13, d15, d11                    \n"
      "     add     %[indexes], %[indexes], #16           \n"
      "     add     %[b], %[b], #32                       \n"
      "     cmp     %[indexes], %[lmt]                    \n"
      "     b.ne    1b                                    \n"
      "     fadd    d0, d0, d1                            \n"
      "     fadd    d10, d10, d11                         \n"
      "     fadd    %d[res], d0, d10                      \n"
      // output operands, source operands, and clobber list
      : [idx1] "=&r"(idx1), [idx2] "=&r"(idx2), [idx3] "=&r"(idx3),
        [idx4] "=&r"(idx4), [indexes] "+&r"(indexes), [res] "=&w"(res),
        [b] "+&r"(b)
      : [lmt] "r"(indexes + n), [a] "r"(a)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v10", "v11", "v12", "v13", "v14",
        "v15", "cc", "memory");

  data->res = res;
}
#else
static void inner_loop_023(struct loop_023_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 4096
#endif
_Static_assert(SIZE % 16 == 0, "SIZE must be a multiple of 16");

LOOP_DECL(023, NS_SVE_LOOP_ATTR)
{
  struct loop_023_data data = { .n = SIZE, .res = DBL_MAX, };

  ALLOC_64B(data.a, SIZE, "A vector");
  ALLOC_64B(data.b, SIZE, "B vector");
  ALLOC_64B(data.indexes, SIZE, "index buffer");

  fill_double(data.a, SIZE);
  fill_double(data.b, SIZE);
  fill_uint32(data.indexes, SIZE);
  for (int i = 0; i < SIZE; i++) {
    data.indexes[i] %= SIZE;
  }

  inner_loops_023(iters, &data);

  double res = data.res;
  bool passed = check_double(res, 1040.0, 0.1);
#ifndef STANDALONE
  FINALISE_LOOP_F(23, passed, "%9.6f", 1040.0, 0.1, res)
#endif
  return passed ? 0 : 1;
}
