/*----------------------------------------------------------------------------
#
#   Loop 036: Sparse matrix Gauss Step
#
#   Purpose:
#     Use of gather-based matrix processing.
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


struct loop_036_data {
  double *restrict matrix;
  int32_t *restrict indexes;
  int32_t *restrict non_zeros;
  double *restrict diag;
  double *restrict xv;
  double *restrict rv;
  double *restrict res;
  int64_t dim;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_036(struct loop_036_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_036(struct loop_036_data *restrict input) {
  double *matrix = input->matrix;
  int32_t *indexes = input->indexes;
  int32_t *non_zeros = input->non_zeros;
  double *diag = input->diag;
  double *xv = input->xv;
  double *rv = input->rv;
  double *res = input->res;
  int64_t dim = input->dim;

  for (int i = 0; i < dim; i++) {
    const double *const row = matrix + (i * dim);
    const int32_t *const row_indexes = indexes + (i * dim);
    const int32_t row_non_zeros = non_zeros[i];
    const double row_diag = diag[i];
    double sum = rv[i];

    for (int32_t j = 0; j < row_non_zeros; j++) {
      int32_t col = row_indexes[j];
      sum -= row[col] * xv[j];
    }
    sum += xv[i] * row_diag;
    res[i] = sum / row_diag;
  }
}
#elif defined(HAVE_SVE_INTRINSICS)
static void inner_loop_036(struct loop_036_data *restrict input) {
  double *matrix = input->matrix;
  int32_t *indexes = input->indexes;
  int32_t *non_zeros = input->non_zeros;
  double *diag = input->diag;
  double *xv = input->xv;
  double *rv = input->rv;
  double *res = input->res;
  int64_t dim = input->dim;

  svbool_t p;
  for (int64_t i = 0; i < dim; i++) {
    const double *row_val = matrix + (i * dim);
    const int32_t *row_idx = indexes + (i * dim);
    svfloat64_t acc = svdup_f64(0.0);

    int32_t n = non_zeros[i];
    FOR_LOOP_64(int32_t, j, 0, n, p) {
      svint64_t vec_i = svld1sw_s64(p, row_idx + j);
      svfloat64_t vec_x = svld1(p, xv + j);
      svfloat64_t vec_m = svld1_gather_index(p, row_val, vec_i);
      acc = svmls_m(p, acc, vec_m, vec_x);
    }

    double sum = rv[i] + svaddv(svptrue_b64(), acc) + xv[i] * diag[i];
    res[i] = sum / diag[i];
  }
}
#elif defined(__ARM_FEATURE_SVE)
static void inner_loop_036(struct loop_036_data *restrict input) {
  double *matrix = input->matrix;
  int32_t *indexes = input->indexes;
  int32_t *non_zeros = input->non_zeros;
  double *diag = input->diag;
  double *xv = input->xv;
  double *rv = input->rv;
  double *res = input->res;
  int64_t dim = input->dim;

  int64_t i = 0;
  int64_t j, rnz, row, rix;

  asm volatile(
      "     mov     %[rix],   %[ix]                         \n"
      "     mov     %[row],   %[mt]                         \n"
      "     ptrue   p1.b                                    \n"
      "1:   ldr     %w[rnz],  [%[nz], %[i], lsl #2]         \n"
      "     ldr     d4,       [%[dg], %[i], lsl #3]         \n"
      "     ldr     d3,       [%[rv], %[i], lsl #3]         \n"
      "     mov     %[j],     #0x0                          \n"
      "     whilelo p0.d,     wzr,  %w[rnz]                 \n"
      "     mov     z1.d,     #0                            \n"
      "2:   ld1sw   {z0.d},   p0/z, [%[rix], %[j], lsl #2]  \n"
      "     ld1d    {z2.d},   p0/z, [%[xv], %[j], lsl #3]   \n"
      "     ld1d    {z0.d},   p0/z, [%[row], z0.d, lsl #3]  \n"
      "     incd    %[j]                                    \n"
      "     fmls    z1.d,     p0/m, z0.d, z2.d              \n"
      "     whilelo p0.d,     %w[j], %w[rnz]                \n"
      "     b.any   2b                                      \n"
      "     faddv   d1,       p1,   z1.d                    \n"
      "     fadd    d3,       d3,   d1                      \n"
      "     ldr     d0,       [%[xv], %[i], lsl #3]         \n"
      "     add     %[rix],   %[rix], %[dim], lsl #2        \n"
      "     add     %[row],   %[row], %[dim], lsl #3        \n"
      "     fmadd   d3,       d4,   d0,   d3                \n"
      "     fdiv    d3,       d3,   d4                      \n"
      "     str     d3,       [%[res], %[i], lsl #3]        \n"
      "     add     %[i],     %[i], #0x1                    \n"
      "     cmp     %[dim],   %[i]                          \n"
      "     b.ne 1b                                         \n"
      // output operands, source operands, and clobber list
      : [i] "+&r"(i), [j] "=&r"(j), [rnz] "=&r"(rnz), [rix] "=&r"(rix),
        [row] "=&r"(row)
      : [mt] "r"(matrix), [ix] "r"(indexes), [nz] "r"(non_zeros),
        [dg] "r"(diag), [xv] "r"(xv), [rv] "r"(rv), [res] "r"(res),
        [dim] "r"(dim)
      : "v0", "v1", "v2", "v3", "p0", "p1", "memory", "cc");
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)
static void inner_loop_036(struct loop_036_data *restrict input) {
  double *matrix = input->matrix;
  int32_t *indexes = input->indexes;
  int32_t *non_zeros = input->non_zeros;
  double *diag = input->diag;
  double *xv = input->xv;
  double *rv = input->rv;
  double *res = input->res;
  int64_t dim = input->dim;

  int64_t i = 0;
  int64_t j, rnz, col, row, rix;

  asm volatile(
      "     mov   %[row],   #0x0                    \n"
      "     mov   %[rix],   %[ix]                   \n"
      "1:   ldr   %w[rnz],  [%[nz], %[i], lsl #2]   \n"
      "     ldr   d3,       [%[dg], %[i], lsl #3]   \n"
      "     ldr   d0,       [%[rv], %[i], lsl #3]   \n"
      "     cmp   %w[rnz],  #0x0                    \n"
      "     b.le  3f                                \n"
      "     mov   %[j],     #0x0                    \n"
      "2:   ldrsw %[col],   [%[rix], %[j], lsl #2]  \n"
      "     ldr   d2,       [%[xv], %[j], lsl #3]   \n"
      "     add   %[j],     %[j],   #0x1            \n"
      "     add   %[col],   %[col], %[row]          \n"
      "     ldr   d1,       [%[mt], %[col], lsl #3] \n"
      "     fmsub d0,       d2,     d1,   d0        \n"
      "     cmp   %w[rnz],  %w[j]                   \n"
      "     b.gt  2b                                \n"
      "3:   ldr   d1,       [%[xv], %[i], lsl #3]   \n"
      "     add   %[rix],   %[rix], %[dim], lsl #2  \n"
      "     add   %[row],   %[row], %[dim]          \n"
      "     fmadd d0,       d1,     d3,   d0        \n"
      "     fdiv  d0,       d0,     d3              \n"
      "     str   d0,       [%[res], %[i], lsl #3]  \n"
      "     add   %[i],     %[i],   #0x1            \n"
      "     cmp   %[dim],   %[i]                    \n"
      "     b.ne  1b                                \n"
      // output operands, source operands, and clobber list
      : [i] "+&r"(i), [j] "=&r"(j), [rnz] "=&r"(rnz), [col] "=&r"(col),
        [row] "=&r"(row), [rix] "=&r"(rix)
      : [mt] "r"(matrix), [ix] "r"(indexes), [nz] "r"(non_zeros),
        [dg] "r"(diag), [xv] "r"(xv), [rv] "r"(rv), [res] "r"(res),
        [dim] "r"(dim)
      : "v0", "v1", "v2", "v3", "memory", "cc");
}
#else
static void inner_loop_036(struct loop_036_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#define DIM 200
_Static_assert(DIM >= 100, "DIM must be at least 20");

LOOP_DECL(036, NS_SVE_LOOP_ATTR)
{
  struct loop_036_data data = { .dim = DIM };

  ALLOC_64B(data.matrix   , DIM * DIM, "input matrix");
  ALLOC_64B(data.indexes  , DIM * DIM, "index buffer");
  ALLOC_64B(data.non_zeros, DIM, "non-zeros buffer");
  ALLOC_64B(data.diag     , DIM, "diagonal buffer");
  ALLOC_64B(data.xv       , DIM, "X vector");
  ALLOC_64B(data.rv       , DIM, "R vector");
  ALLOC_64B(data.res      , DIM, "result vector");

  fill_double(data.matrix, DIM * DIM);
  fill_double(data.diag, DIM);
  fill_double(data.xv, DIM);
  fill_double(data.rv, DIM);
  fill_double(data.res, DIM);

  // Fill non-zero and indexes
  for (int i = 0; i < DIM; i++) {
    data.non_zeros[i] = i % 2 ? 17 : 33;
    for (int j = 0; j < data.non_zeros[i]; j++) {
      data.indexes[j] = (j * DIM) / data.non_zeros[i];
    }
  }

  inner_loops_036(iters, &data);

  double res = 0.0f;
  for (int i = 0; i < DIM; i++) {
    res += i * data.res[i];
  }

  bool passed = check_double(res, -526902.6, 1.0);
#ifndef STANDALONE
  FINALISE_LOOP_F(36, passed, "%9.6f", -526902.6, 1.0, res)
#endif
  return passed ? 0 : 1;
}
