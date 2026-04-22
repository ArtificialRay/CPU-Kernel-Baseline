/*----------------------------------------------------------------------------
#
#   Loop 038: Fp16 convolution
#
#   Purpose:
#     Use of fp16 multiply-add instructions.
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


struct loop_038_data {
  float16_t *restrict a;
  float16_t *restrict b;
  float16_t *restrict c;
  int dim;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_038(struct loop_038_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_038(struct loop_038_data *restrict data) {
  float16_t *a = data->a;
  float16_t *b = data->b;
  float16_t *c = data->c;
  int dim = data->dim;

  for (int row = 0; row < dim - 1; row++) {
    for (int col = 0; col < dim - 1; col++) {
      FLOAT16_t s0 = fp16_to_native(a[row * dim + col]);
      FLOAT16_t s1 = fp16_to_native(a[row * dim + col + 1]);
      FLOAT16_t s2 = fp16_to_native(a[(row + 1) * dim + col]);
      FLOAT16_t s3 = fp16_to_native(a[(row + 1) * dim + col + 1]);
      FLOAT16_t ac = fp16_to_native(b[row * dim + col]);
      FLOAT16_t k = 0.25f;
      FLOAT16_t r = ac + s0 * k + s1 * k + s2 * k + s3 * k;
      c[row * dim + col] = native_to_fp16(r);
    }
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))
static void inner_loop_038(struct loop_038_data *restrict data)
LOOP_ATTR
{
  float16_t *a = data->a;
  float16_t *b = data->b;
  float16_t *c = data->c;
  int dim = data->dim;
  svfloat16_t k_vec = svdup_f16(0.25f);

  for (int row = 0; row < dim - 1; row++) {
    float16_t *s0 = a + (row * dim);
    float16_t *s1 = a + (row * dim) + 1;
    float16_t *s2 = a + ((row + 1) * dim);
    float16_t *s3 = a + ((row + 1) * dim) + 1;
    float16_t *ac = b + (row * dim);
    float16_t *ad = c + (row * dim);
    svbool_t p;
    FOR_LOOP_16(int32_t, col, 0, dim - 1, p) {
      svfloat16_t s0_vec = svld1(p, s0 + col);
      svfloat16_t s1_vec = svld1(p, s1 + col);
      svfloat16_t s2_vec = svld1(p, s2 + col);
      svfloat16_t s3_vec = svld1(p, s3 + col);
      svfloat16_t ac_vec = svld1(p, ac + col);
      ac_vec = svadd_x(p, ac_vec, svmul_x(p, s0_vec, k_vec));
      ac_vec = svadd_x(p, ac_vec, svmul_x(p, s1_vec, k_vec));
      ac_vec = svadd_x(p, ac_vec, svmul_x(p, s2_vec, k_vec));
      ac_vec = svadd_x(p, ac_vec, svmul_x(p, s3_vec, k_vec));
      svst1(p, ad + col, ac_vec);
    }
  }
}
#elif (defined(__ARM_FEATURE_SVE2p1) || defined(__ARM_FEATURE_SME))
static void inner_loop_038(struct loop_038_data *restrict data)
LOOP_ATTR
{
  float16_t *a = data->a;
  float16_t *b = data->b;
  float16_t *c = data->c;
  uint64_t dim = data->dim;
  svfloat16_t k = svdup_f16(0.25f);

  for (int row = 0; row < dim - 1; row++) {
    float16_t *s0 = a + (row * dim);
    float16_t *s1 = a + (row * dim) + 1;
    float16_t *s2 = a + ((row + 1) * dim);
    float16_t *s3 = a + ((row + 1) * dim) + 1;
    float16_t *ac = b + (row * dim);
    float16_t *ad = c + (row * dim);
    uint64_t i = 0;
    asm volatile(
        "       ptrue   p0.h                                        \n"
        "       whilelt pn8.h, %[i], %[dim], vlx2                   \n"
        "       b.none  2f                                          \n"
        "1:                                                         \n"
        "       ld1h    {z8.h-z9.h}, pn8/z, [%[ac], %[i], lsl #1]   \n"
        "       ld1h    {z0.h-z1.h}, pn8/z, [%[s0], %[i], lsl #1]   \n"
        "       ld1h    {z2.h-z3.h}, pn8/z, [%[s1], %[i], lsl #1]   \n"
        "       ld1h    {z4.h-z5.h}, pn8/z, [%[s2], %[i], lsl #1]   \n"
        "       ld1h    {z6.h-z7.h}, pn8/z, [%[s3], %[i], lsl #1]   \n"
        "       fmla    z8.h, p0/m, z0.h, %[k].h                    \n"
        "       fmla    z9.h, p0/m, z1.h, %[k].h                    \n"
        "       fmla    z8.h, p0/m, z2.h, %[k].h                    \n"
        "       fmla    z9.h, p0/m, z3.h, %[k].h                    \n"
        "       fmla    z8.h, p0/m, z4.h, %[k].h                    \n"
        "       fmla    z9.h, p0/m, z5.h, %[k].h                    \n"
        "       fmla    z8.h, p0/m, z6.h, %[k].h                    \n"
        "       fmla    z9.h, p0/m, z7.h, %[k].h                    \n"
        "       st1h    {z8.h-z9.h}, pn8, [%[ad], %[i], lsl #1]     \n"
        "       incb    %[i]                                        \n"
        "       whilelt pn8.h, %[i], %[dim], vlx2                   \n"
        "       b.first 1b                                          \n"
        "2:                                                         \n"
        // output operands, source operands, and clobber list
        : [i] "+&r"(i)
        : [s0] "r"(s0), [s1] "r"(s1), [s2] "r"(s2), [s3] "r"(s3), [ac] "r"(ac),
          [ad] "r"(ad), [dim] "r"(dim - 1), [k] "w"(k)
        : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",
          "p0", "p8", "cc", "memory");
  }
}
#elif defined(__ARM_FEATURE_SVE)
static void inner_loop_038(struct loop_038_data *restrict data)
LOOP_ATTR
{
  float16_t *a = data->a;
  float16_t *b = data->b;
  float16_t *c = data->c;
  int64_t dim = data->dim;
  svfloat16_t k = svdup_f16(0.25f);

  for (int row = 0; row < dim - 1; row++) {
    int64_t i0 = 0;
    int64_t i1 = svcnth();
    float16_t *s0 = a + (row * dim);
    float16_t *s1 = a + (row * dim) + 1;
    float16_t *s2 = a + ((row + 1) * dim);
    float16_t *s3 = a + ((row + 1) * dim) + 1;
    float16_t *ac = b + (row * dim);
    float16_t *ad = c + (row * dim);

    asm volatile(
        "       ptrue   p0.h                                    \n"
        "       b       2f                                      \n"

        "1:     ld1h    {z10.h}, p0/z, [%[ac], %[i0], lsl #1]   \n"
        "       ld1h    {z11.h}, p1/z, [%[ac], %[i1], lsl #1]   \n"
        "       ld1h    {z0.h},  p0/z, [%[s0], %[i0], lsl #1]   \n"
        "       ld1h    {z1.h},  p1/z, [%[s0], %[i1], lsl #1]   \n"
        "       ld1h    {z2.h},  p0/z, [%[s1], %[i0], lsl #1]   \n"
        "       ld1h    {z3.h},  p1/z, [%[s1], %[i1], lsl #1]   \n"
        "       ld1h    {z4.h},  p0/z, [%[s2], %[i0], lsl #1]   \n"
        "       ld1h    {z5.h},  p1/z, [%[s2], %[i1], lsl #1]   \n"
        "       ld1h    {z6.h},  p0/z, [%[s3], %[i0], lsl #1]   \n"
        "       ld1h    {z7.h},  p1/z, [%[s3], %[i1], lsl #1]   \n"

        "       fmla    z10.h, p0/m, z0.h, %[k].h               \n"
        "       fmla    z11.h, p0/m, z1.h, %[k].h               \n"
        "       fmla    z10.h, p0/m, z2.h, %[k].h               \n"
        "       fmla    z11.h, p0/m, z3.h, %[k].h               \n"
        "       fmla    z10.h, p0/m, z4.h, %[k].h               \n"
        "       fmla    z11.h, p0/m, z5.h, %[k].h               \n"
        "       fmla    z10.h, p0/m, z6.h, %[k].h               \n"
        "       fmla    z11.h, p0/m, z7.h, %[k].h               \n"

        "       st1h    {z10.h}, p0, [%[ad], %[i0], lsl #1]     \n"
        "       st1h    {z11.h}, p1, [%[ad], %[i1], lsl #1]     \n"

        "       incb    %[i0], all                              \n"
        "       incb    %[i1], all                              \n"

        // loop back if at least one full vector
        "2:     whilelo p1.h, %[i1], %[dim]                     \n"
        "       b.first    1b                                   \n"

        // output operands, source operands, and clobber list
        : [i0] "+&r"(i0), [i1] "+&r"(i1)
        : [s0] "r"(s0), [s1] "r"(s1), [s2] "r"(s2), [s3] "r"(s3), [ac] "r"(ac),
          [ad] "r"(ad), [dim] "r"(dim - 1), [k] "w"(k)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v10", "v11", "cc",
          "p0", "p1", "cc", "memory");
  }
}
#elif defined(__ARM_NEON)
static void inner_loop_038(struct loop_038_data *restrict data) {
  _Float16 *a = (_Float16 *)data->a;
  _Float16 *b = (_Float16 *)data->b;
  _Float16 *c = (_Float16 *)data->c;
  int64_t dim = data->dim;
  int64_t iters = (dim - 1) / 16;
  int64_t lmt = iters * 32;
  float16x8_t k = vdupq_n_f16(0.25f);

  for (int row = 0; row < dim - 1; row++) {
    int64_t i0 = 0;
    int64_t i1 = 16;  // Bytes
    _Float16 *s0 = a + (row * dim);
    _Float16 *s1 = a + (row * dim) + 1;
    _Float16 *s2 = a + ((row + 1) * dim);
    _Float16 *s3 = a + ((row + 1) * dim) + 1;
    _Float16 *ac = b + (row * dim);
    _Float16 *ad = c + (row * dim);

    asm volatile(
        "       b       2f                        \n"

        "1:     ldr     q10, [%[ac], %[i0]]       \n"
        "       ldr     q11, [%[ac], %[i1]]       \n"
        "       ldr     q0,  [%[s0], %[i0]]       \n"
        "       ldr     q1,  [%[s0], %[i1]]       \n"
        "       ldr     q2,  [%[s1], %[i0]]       \n"
        "       ldr     q3,  [%[s1], %[i1]]       \n"
        "       ldr     q4,  [%[s2], %[i0]]       \n"
        "       ldr     q5,  [%[s2], %[i1]]       \n"
        "       ldr     q6,  [%[s3], %[i0]]       \n"
        "       ldr     q7,  [%[s3], %[i1]]       \n"

        "       fmla    v10.8h, v0.8h, %[k].8h    \n"
        "       fmla    v11.8h, v1.8h, %[k].8h    \n"
        "       fmla    v10.8h, v2.8h, %[k].8h    \n"
        "       fmla    v11.8h, v3.8h, %[k].8h    \n"
        "       fmla    v10.8h, v4.8h, %[k].8h    \n"
        "       fmla    v11.8h, v5.8h, %[k].8h    \n"
        "       fmla    v10.8h, v6.8h, %[k].8h    \n"
        "       fmla    v11.8h, v7.8h, %[k].8h    \n"

        "       str     q10, [%[ad], %[i0]]       \n"
        "       str     q11, [%[ad], %[i1]]       \n"
        "       add    %[i0], %[i0], #32          \n"
        "       add    %[i1], %[i1], #32          \n"

        // loop back
        "2:     cmp     %[i0], %[lmt]             \n"
        "       b.lt    1b                        \n"
        // output operands, source operands, and clobber list
        : [i0] "+&r"(i0), [i1] "+&r"(i1)
        : [s0] "r"(s0), [s1] "r"(s1), [s2] "r"(s2), [s3] "r"(s3), [ac] "r"(ac),
          [ad] "r"(ad), [lmt] "r"(lmt), [k] "w"(k)
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v10", "v11", "cc",
          "memory");

    // Tail loop
    for (int col = iters * 16; col < dim - 1; col++) {
      _Float16 s0 = a[row * dim + col];
      _Float16 s1 = a[row * dim + col + 1];
      _Float16 s2 = a[(row + 1) * dim + col];
      _Float16 s3 = a[(row + 1) * dim + col + 1];
      _Float16 ac = b[row * dim + col];
      _Float16 k = 0.25f;
      _Float16 r = ac + s0 * k + s1 * k + s2 * k + s3 * k;
      c[row * dim + col] = r;
    }
  }
}
#else
static void inner_loop_038(struct loop_038_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#define DIM 128

LOOP_DECL(038, SC_SVE_LOOP_ATTR)
{
  struct loop_038_data data = { .dim = DIM };

  ALLOC_64B(data.a, DIM * DIM, "A matrix");
  ALLOC_64B(data.b, DIM * DIM, "B matrix");
  ALLOC_64B(data.c, DIM * DIM, "C matrix");

  fill_fp16(data.a, DIM * DIM);
  fill_fp16(data.b, DIM * DIM);
  fill_fp16(data.c, DIM * DIM);

  inner_loops_038(iters, &data);

  float res = fp16_to_native(data.c[DIM * DIM / 2]);
  bool passed = check_float(res, -0.183350f, 0.01f);
#ifndef STANDALONE
  FINALISE_LOOP_F(38, passed, "%9.6f", -0.183350f, 0.01f, res)
#endif
  return passed ? 0 : 1;
}
