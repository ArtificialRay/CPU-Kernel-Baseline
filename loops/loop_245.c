/*----------------------------------------------------------------------------
#
#   Loop 245: INT8-INT32 matrix-matrix multiply with rearrangement
#
#   Purpose:
#     Use of MOPA, DOT & MMLA instructions with matrix rearrangements.
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

/*
  Data format -
    A: column-major
    B: row-major
    C: row-major
  Constraints -
    M: multiple of 16
    N: multiple of 2*SVLs
    K: multiple of SVLb
*/

struct loop_245_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  int8_t *restrict a;
  int8_t *restrict a_mod;
  int8_t *restrict b;
  int8_t *restrict b_mod;
  int32_t *restrict c;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_245(struct loop_245_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#define LOOP_ATTR SC_SVE_ATTR
#define OUTER_LOOP_ATTR SC_SVE_LOOP_ATTR
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#elif defined(__ARM_FEATURE_SME2)
#define LOOP_ATTR SME_ZA_ATTR
#define OUTER_LOOP_ATTR S_LOOP_ATTR
#elif defined(__ARM_FEATURE_SVE2)
#define LOOP_ATTR SC_SVE_ATTR
#define OUTER_LOOP_ATTR SC_SVE_LOOP_ATTR
#else
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#endif



static inline int32_t int8_to_int32(uint64_t i, uint64_t j, uint64_t k,
                                    struct loop_245_data *data) {
  return (int32_t)(data->a[k * data->m + i] * data->b[k * data->n + j]);
}

#if !defined(HAVE_CANDIDATE)
#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_245(struct loop_245_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  int32_t *restrict c = data->c;

  for (uint64_t x = 0; x < m; x++)
    for (uint64_t y = 0; y < n; y++) c[x * n + y] = 0;

  // Loops ordered for contiguous memory access in inner loop
  for (uint64_t z = 0; z < k; z++)
    for (uint64_t x = 0; x < m; x++)
      for (uint64_t y = 0; y < n; y++)
        c[x * n + y] += int8_to_int32(x, y, z, data);
}
#elif defined(__ARM_FEATURE_SME2)

static void sme_interleave_vla(uint64_t rows, uint64_t cols, uint64_t matrix,
                               uint64_t matrix_new) LOOP_ATTR {
  register uint64_t row = rows;
  register uint64_t col = cols;
  register uint64_t mat = matrix;
  register uint64_t mat_mod = matrix_new;

  register uint64_t mat_off2 = col * 2;
  register uint64_t mat_off3 = col * 3;
  register uint64_t mat1ptr;
  register uint64_t mat_mod1ptr;
  register uint64_t cnd;
  register uint64_t row_idx;

  asm volatile(

      "   ptrue   p0.b                                              \n"
      "   ptrue   pn8.b                                             \n"
      "   mov     %[row_idx], #0                                    \n"

      // Row loop head
      "1:                                                           \n"
      "   mov     %[mat_mod1ptr], %[mat_mod0ptr]                    \n"
      "   mov     %[mat1ptr], %[mat0ptr]                            \n"
      "   add     %[cnd], %[mat1ptr], %[col]                        \n"

      // Column loop head
      "2:                                                           \n"
      "   ld1b    {z0.b}, p0/z, [%[mat1ptr]]                        \n"
      "   ld1b    {z1.b}, p0/z, [%[mat1ptr], %[mat_off1]]           \n"
      "   ld1b    {z2.b}, p0/z, [%[mat1ptr], %[mat_off2]]           \n"
      "   ld1b    {z3.b}, p0/z, [%[mat1ptr], %[mat_off3]]           \n"

      "   zip     {z0.b-z3.b},  {z0.b-z3.b}                         \n"
      "   st1b    {z0.b-z3.b},  pn8, [%[mat_mod1ptr]]               \n"

      "   addvl   %[mat_mod1ptr], %[mat_mod1ptr], #4                \n"
      "   addvl   %[mat1ptr], %[mat1ptr], #1                        \n"
      "   cmp     %[mat1ptr], %[cnd]                                \n"
      "   b.mi    2b                                                \n"

      "   add     %[mat_mod0ptr], %[mat_mod0ptr], %[col], lsl #2    \n"
      "   add     %[mat0ptr], %[mat0ptr], %[col], lsl #2            \n"
      "   add     %[row_idx], %[row_idx], #4                        \n"
      "   cmp     %[row_idx], %[row]                                \n"
      "   b.mi    1b                                                \n"

      : [mat0ptr] "+&r"(mat), [mat1ptr] "=&r"(mat1ptr),
        [mat_mod0ptr] "+&r"(mat_mod), [mat_mod1ptr] "=&r"(mat_mod1ptr),
        [cnd] "=&r"(cnd), [row_idx] "=&r"(row_idx)
      : [row] "r"(row), [col] "r"(col), [mat_off1] "r"(cols),
        [mat_off2] "r"(mat_off2), [mat_off3] "r"(mat_off3)
      : "z0", "z1", "z2", "z3", "p0", "p8", "cc", "memory");
}

static void sme_gemm(uint64_t M, uint64_t K, uint64_t N, uint64_t A_new,
                     uint64_t B_new, uint64_t C) LOOP_ATTR {
  register uint64_t m = M;
  register uint64_t k = K;
  register uint64_t n = N;
  register uint64_t a = A_new;
  register uint64_t b = B_new;
  register uint64_t c = C;
  register uint64_t nx4 = n * 4;
  register uint64_t mx4 = m * 4;

  register uint64_t svl_s;
  asm volatile("cntw %[v]" : [v] "=r"(svl_s)::);
  register uint64_t l_cnd = svl_s * 4 - 8;
  register uint64_t a_cnd = a + (m * k);
  register uint64_t c_blk = svl_s * n;
  register uint64_t c_off = c_blk + n;
  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  // x12: slice index register for tile-to-vec mova

  asm volatile(
      // M loop head
      "   mov     %[m_idx], #0                                              \n"
      "   ptrue   pn8.b                                                     \n"
      "   ptrue   p0.b                                                      \n"

#if defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                      \n"
#endif
      "1:                                                                   \n"

      // N loop head
      "   mov     %[n_idx], #0                                              \n"
      "2:                                                                   \n"
      "   add     %[a_ptr], %[a_src], %[m_idx]                              \n"
      "   add     %[b_ptr], %[b_src], %[n_idx]                              \n"
      "   add     %[c_ptr], %[c_dst], %[n_idx]                              \n"

      // K loop
      "   ld1b    {z2.b-z3.b}, pn8/z, [%[a_ptr]]                            \n"
#if !defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                      \n"
#endif
      "   ld1b    {z0.b-z1.b}, pn8/z, [%[b_ptr]]                            \n"
      "   smopa   za0.s, p0/m, p0/m, z2.b, z0.b                             \n"
      "   smopa   za1.s, p0/m, p0/m, z2.b, z1.b                             \n"
      "   ld1b    {z6.b-z7.b}, pn8/z, [%[a_ptr], %[mx4]]                    \n"
      "   ld1b    {z4.b-z5.b}, pn8/z, [%[b_ptr], %[nx4]]                    \n"
      "   smopa   za2.s, p0/m, p0/m, z3.b, z0.b                             \n"
      "   smopa   za3.s, p0/m, p0/m, z3.b, z1.b                             \n"
      "   add     %[a_ptr], %[a_ptr], %[mx4], lsl #1                        \n"
      "   add     %[b_ptr], %[b_ptr], %[nx4], lsl #1                        \n"
      "3:                                                                   \n"
      "   smopa   za0.s, p0/m, p0/m, z6.b, z4.b                             \n"
      "   smopa   za1.s, p0/m, p0/m, z6.b, z5.b                             \n"
      "   ld1b    {z2.b-z3.b}, pn8/z, [%[a_ptr]]                            \n"
      "   ld1b    {z0.b-z1.b}, pn8/z, [%[b_ptr]]                            \n"
      "   smopa   za2.s, p0/m, p0/m, z7.b, z4.b                             \n"
      "   smopa   za3.s, p0/m, p0/m, z7.b, z5.b                             \n"
      "   ld1b    {z6.b-z7.b}, pn8/z, [%[a_ptr], %[mx4]]                    \n"
      "   ld1b    {z4.b-z5.b}, pn8/z, [%[b_ptr], %[nx4]]                    \n"
      "   smopa   za0.s, p0/m, p0/m, z2.b, z0.b                             \n"
      "   smopa   za1.s, p0/m, p0/m, z2.b, z1.b                             \n"
      "   add     %[a_ptr], %[a_ptr], %[mx4], lsl #1                        \n"
      "   add     %[b_ptr], %[b_ptr], %[nx4], lsl #1                        \n"
      "   smopa   za2.s, p0/m, p0/m, z3.b, z0.b                             \n"
      "   smopa   za3.s, p0/m, p0/m, z3.b, z1.b                             \n"
      "   cmp     %[a_ptr], %[a_cnd]                                        \n"
      "   b.lt    3b                                                        \n"
      "   smopa   za0.s, p0/m, p0/m, z6.b, z4.b                             \n"
      "   smopa   za1.s, p0/m, p0/m, z6.b, z5.b                             \n"
      "   smopa   za2.s, p0/m, p0/m, z7.b, z4.b                             \n"
      "   smopa   za3.s, p0/m, p0/m, z7.b, z5.b                             \n"

      // Store loop
      "   mov     x12, #0                                                   \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z0.b-z3.b}, za0h.b[w12, 0:3]                             \n"
#else
      "   mova    {z0.b-z3.b}, za0h.b[w12, 0:3]                             \n"
#endif
      "   st1w    {z0.s-z1.s}, pn8, [%[c_ptr]]                              \n"
      "   st1w    {z2.s-z3.s}, pn8, [%[c_ptr], %[c_blk], lsl #2]            \n"

      "4:                                                                   \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z0.b-z3.b}, za0h.b[w12, 4:7]                             \n"
#else
      "   mova    {z0.b-z3.b}, za0h.b[w12, 4:7]                             \n"
#endif
      "   st1w    {z0.s-z1.s}, pn8, [%[c_ptr], %[n], lsl #2]                \n"
      "   st1w    {z2.s-z3.s}, pn8, [%[c_ptr], %[c_off], lsl #2]            \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #3                          \n"
      "   add     x12, x12, #8                                              \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z0.b-z3.b}, za0h.b[w12, 0:3]                             \n"
#else
      "   mova    {z0.b-z3.b}, za0h.b[w12, 0:3]                             \n"
#endif
      "   st1w    {z0.s-z1.s}, pn8, [%[c_ptr]]                              \n"
      "   st1w    {z2.s-z3.s}, pn8, [%[c_ptr], %[c_blk], lsl #2]            \n"
      "   cmp     x12, %[l_cnd]                                             \n"
      "   b.mi    4b                                                        \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z0.b-z3.b}, za0h.b[w12, 4:7]                             \n"
#else
      "   mova    {z0.b-z3.b}, za0h.b[w12, 4:7]                             \n"
#endif
      "   st1w    {z0.s-z1.s}, pn8, [%[c_ptr], %[n], lsl #2]                \n"
      "   st1w    {z2.s-z3.s}, pn8, [%[c_ptr], %[c_off], lsl #2]            \n"

      // N loop tail
      "   addvl   %[n_idx], %[n_idx], #2                                    \n"
      "   cmp     %[n_idx], %[nx4]                                          \n"
      "   b.mi 2b                                                           \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[c_blk], lsl #3                      \n"
      "   addvl   %[m_idx], %[m_idx], #2                                    \n"
      "   cmp     %[m_idx], %[mx4]                                          \n"
      "   b.mi 1b                                                           \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [c_ptr] "=&r"(c_ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [mx4] "r"(mx4), [nx4] "r"(nx4),
        [svl_s] "r"(svl_s), [c_blk] "r"(c_blk), [c_off] "r"(c_off),
        [l_cnd] "r"(l_cnd), [a_cnd] "r"(a_cnd), [a_src] "r"(a), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "x12", "p0", "p8",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

static void inner_loop_245(struct loop_245_data *data) LOOP_ATTR {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;
  register uint64_t a_mod = (uint64_t)data->a_mod;
  register uint64_t b_mod = (uint64_t)data->b_mod;

  sme_interleave_vla(k, m, a, a_mod);
  sme_interleave_vla(k, n, b, b_mod);
  sme_gemm(m, k, n, a_mod, b_mod, c);
}

#elif defined(__ARM_FEATURE_SVE2)
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)

static void sve_mmla_interleave_vla(uint64_t rows, uint64_t cols,
                                    uint64_t matrix, uint64_t matrix_new) {
  register uint64_t row = rows;
  register uint64_t col = cols;
  register uint64_t mat = matrix;
  register uint64_t mat_mod = matrix_new;

  register uint64_t mat_off2 = col * 2;
  register uint64_t mat_off3 = col * 3;
  register uint64_t mat_off4 = col * 4;
  register uint64_t mat_off5 = col * 5;
  register uint64_t mat_off6 = col * 6;
  register uint64_t mat_off7 = col * 7;
  register uint64_t mat1ptr;
  register uint64_t mat_mod1ptr;
  register uint64_t cnd;
  register uint64_t row_idx;

  asm volatile(

#if defined(__ARM_FEATURE_SVE2p1)
      "   ptrue   pn8.b                                             \n"
#endif
      "   ptrue   p0.b                                              \n"
      "   mov     %[row_idx], #0                                    \n"

      // Row loop head
      "1:                                                           \n"
      "   mov     %[mat_mod1ptr], %[mat_mod0ptr]                    \n"
      "   mov     %[mat1ptr], %[mat0ptr]                            \n"
      "   add     %[cnd], %[mat1ptr], %[col]                        \n"

      // Column loop head
      "2:                                                           \n"
      "   ld1b    {z0.b}, p0/z, [%[mat1ptr]]                        \n"
      "   ld1b    {z1.b}, p0/z, [%[mat1ptr], %[mat_off1]]           \n"
      "   ld1b    {z2.b}, p0/z, [%[mat1ptr], %[mat_off2]]           \n"
      "   ld1b    {z3.b}, p0/z, [%[mat1ptr], %[mat_off3]]           \n"
      "   ld1b    {z4.b}, p0/z, [%[mat1ptr], %[mat_off4]]           \n"
      "   ld1b    {z5.b}, p0/z, [%[mat1ptr], %[mat_off5]]           \n"
      "   ld1b    {z6.b}, p0/z, [%[mat1ptr], %[mat_off6]]           \n"
      "   ld1b    {z7.b}, p0/z, [%[mat1ptr], %[mat_off7]]           \n"

      "   zip1    z16.b,  z0.b,  z4.b                               \n"
      "   zip1    z17.b,  z2.b,  z6.b                               \n"
      "   zip1    z18.b,  z1.b,  z5.b                               \n"
      "   zip1    z19.b,  z3.b,  z7.b                               \n"
      "   zip2    z20.b,  z0.b,  z4.b                               \n"
      "   zip2    z21.b,  z2.b,  z6.b                               \n"
      "   zip2    z22.b,  z1.b,  z5.b                               \n"
      "   zip2    z23.b,  z3.b,  z7.b                               \n"

      "   zip1    z24.b,  z16.b,  z17.b                             \n"
      "   zip1    z25.b,  z18.b,  z19.b                             \n"
      "   zip2    z26.b,  z16.b,  z17.b                             \n"
      "   zip2    z27.b,  z18.b,  z19.b                             \n"
      "   zip1    z0.b,   z24.b,  z25.b                             \n"
      "   zip2    z1.b,   z24.b,  z25.b                             \n"
      "   zip1    z2.b,   z26.b,  z27.b                             \n"
      "   zip2    z3.b,   z26.b,  z27.b                             \n"

      "   zip1    z24.b,  z20.b,  z21.b                             \n"
      "   zip1    z25.b,  z22.b,  z23.b                             \n"
      "   zip2    z26.b,  z20.b,  z21.b                             \n"
      "   zip2    z27.b,  z22.b,  z23.b                             \n"
      "   zip1    z4.b,   z24.b,  z25.b                             \n"
      "   zip2    z5.b,   z24.b,  z25.b                             \n"
      "   zip1    z6.b,   z26.b,  z27.b                             \n"
      "   zip2    z7.b,   z26.b,  z27.b                             \n"

#if defined(__ARM_FEATURE_SVE2p1)
      "   st1b    {z0.b-z3.b}, pn8, [%[mat_mod1ptr]]                \n"
      "   st1b    {z4.b-z7.b}, pn8, [%[mat_mod1ptr], #4, mul vl]    \n"
#else
      "   st1b    {z0.b}, p0, [%[mat_mod1ptr]]                      \n"
      "   st1b    {z1.b}, p0, [%[mat_mod1ptr], #1, mul vl]          \n"
      "   st1b    {z2.b}, p0, [%[mat_mod1ptr], #2, mul vl]          \n"
      "   st1b    {z3.b}, p0, [%[mat_mod1ptr], #3, mul vl]          \n"
      "   st1b    {z4.b}, p0, [%[mat_mod1ptr], #4, mul vl]          \n"
      "   st1b    {z5.b}, p0, [%[mat_mod1ptr], #5, mul vl]          \n"
      "   st1b    {z6.b}, p0, [%[mat_mod1ptr], #6, mul vl]          \n"
      "   st1b    {z7.b}, p0, [%[mat_mod1ptr], #7, mul vl]          \n"
#endif

      "   addvl   %[mat_mod1ptr], %[mat_mod1ptr], #8                \n"
      "   addvl   %[mat1ptr], %[mat1ptr], #1                        \n"
      "   cmp     %[mat1ptr], %[cnd]                                \n"
      "   b.mi    2b                                                \n"

      "   add     %[mat_mod0ptr], %[mat_mod0ptr], %[col], lsl #3    \n"
      "   add     %[mat0ptr], %[mat0ptr], %[col], lsl #3            \n"
      "   add     %[row_idx], %[row_idx], #8                        \n"
      "   cmp     %[row_idx], %[row]                                \n"
      "   b.mi    1b                                                \n"

      : [mat0ptr] "+&r"(mat), [mat1ptr] "=&r"(mat1ptr),
        [mat_mod0ptr] "+&r"(mat_mod), [mat_mod1ptr] "=&r"(mat_mod1ptr),
        [cnd] "=&r"(cnd), [row_idx] "=&r"(row_idx)
      : [row] "r"(row), [col] "r"(col), [mat_off1] "r"(cols),
        [mat_off2] "r"(mat_off2), [mat_off3] "r"(mat_off3),
        [mat_off4] "r"(mat_off4), [mat_off5] "r"(mat_off5),
        [mat_off6] "r"(mat_off6), [mat_off7] "r"(mat_off7)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z16", "z17", "z18",
        "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28",
        "z29", "z30", "z31", "p0", "p8", "cc", "memory");
}

static void sve_mmla_gemm(uint64_t M, uint64_t K, uint64_t N, uint64_t A_new,
                          uint64_t B_new, uint64_t C) {
  register uint64_t m = M;
  register uint64_t k = K;
  register uint64_t n = N;
  register uint64_t a = A_new;
  register uint64_t b = B_new;
  register uint64_t c = C;

  register uint64_t c2off = n * 2;
  register uint64_t c3off = n * 3;
  register uint64_t a_cnd = a + (m * k);

  register uint64_t n_idx;
  register uint64_t m_idx;
  register uint64_t a_ptr;
  register uint64_t b_ptr;
#if defined(__ARM_FEATURE_SVE2p1)
  register uint64_t c_ptr;
#else
  register uint64_t c0ptr;
  register uint64_t c1ptr;
#endif

  asm volatile(
#if defined(__ARM_FEATURE_SVE2p1)
      "   ptrue   pn8.b                                             \n"
#endif
      "   ptrue   p0.b                                              \n"

      // M loop head
      "   mov     %[m_idx], #0                                      \n"
      "1:                                                           \n"

      // N loop head
      "   mov     %[n_idx], #0                                      \n"
      "2:                                                           \n"

      // Accumulators
      "   mov     z8.s,  #0                                         \n"
      "   mov     z9.s,  #0                                         \n"
      "   mov     z10.s, #0                                         \n"
      "   mov     z11.s, #0                                         \n"
      "   mov     z12.s, #0                                         \n"
      "   mov     z13.s, #0                                         \n"
      "   mov     z14.s, #0                                         \n"
      "   mov     z15.s, #0                                         \n"
      "   mov     z16.s, #0                                         \n"
      "   mov     z17.s, #0                                         \n"
      "   mov     z18.s, #0                                         \n"
      "   mov     z19.s, #0                                         \n"
      "   mov     z20.s, #0                                         \n"
      "   mov     z21.s, #0                                         \n"
      "   mov     z22.s, #0                                         \n"
      "   mov     z23.s, #0                                         \n"
      "   mov     z24.s, #0                                         \n"
      "   mov     z25.s, #0                                         \n"
      "   mov     z26.s, #0                                         \n"
      "   mov     z27.s, #0                                         \n"
      "   mov     z28.s, #0                                         \n"
      "   mov     z29.s, #0                                         \n"
      "   mov     z30.s, #0                                         \n"
      "   mov     z31.s, #0                                         \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #3              \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #3              \n"
      "3:                                                           \n"
      "   ld1rqb  {z0.b}, p0/z, [%[a_ptr]]                          \n"
      "   ld1rqb  {z1.b}, p0/z, [%[a_ptr], #16]                     \n"
#if defined(__ARM_FEATURE_SVE2p1)
      "   ld1b    {z4.b-z7.b}, pn8/z, [%[b_ptr]]                    \n"
#else
      "   ld1b    {z4.b}, p0/z, [%[b_ptr], #0, mul vl]              \n"
      "   ld1b    {z5.b}, p0/z, [%[b_ptr], #1, mul vl]              \n"
      "   ld1b    {z6.b}, p0/z, [%[b_ptr], #2, mul vl]              \n"
      "   ld1b    {z7.b}, p0/z, [%[b_ptr], #3, mul vl]              \n"
#endif
      "   smmla   z16.s, z0.b, z4.b                                 \n"
      "   smmla   z20.s, z0.b, z5.b                                 \n"
      "   smmla   z24.s, z0.b, z6.b                                 \n"
      "   smmla   z28.s, z0.b, z7.b                                 \n"

      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                  \n"
      "   ld1rqb  {z2.b}, p0/z, [%[a_ptr], #32]                     \n"
      "   smmla   z17.s, z1.b, z4.b                                 \n"
      "   smmla   z21.s, z1.b, z5.b                                 \n"
      "   smmla   z25.s, z1.b, z6.b                                 \n"
      "   smmla   z29.s, z1.b, z7.b                                 \n"

      "   ld1rqb  {z3.b}, p0/z, [%[a_ptr], #48]                     \n"
      "   smmla   z18.s, z2.b, z4.b                                 \n"
      "   smmla   z22.s, z2.b, z5.b                                 \n"
      "   smmla   z26.s, z2.b, z6.b                                 \n"
      "   smmla   z30.s, z2.b, z7.b                                 \n"

      "   ld1rqb  {z0.b}, p0/z, [%[a_ptr], #64]                     \n"
      "   smmla   z19.s, z3.b, z4.b                                 \n"
      "   smmla   z23.s, z3.b, z5.b                                 \n"
      "   smmla   z27.s, z3.b, z6.b                                 \n"
      "   smmla   z31.s, z3.b, z7.b                                 \n"

      "   ld1rqb  {z1.b}, p0/z, [%[a_ptr], #80]                     \n"
      "   smmla   z8.s,  z0.b, z4.b                                 \n"
      "   smmla   z10.s, z0.b, z5.b                                 \n"
      "   smmla   z12.s, z0.b, z6.b                                 \n"
      "   smmla   z14.s, z0.b, z7.b                                 \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   smmla   z9.s,  z1.b, z4.b                                 \n"
      "   smmla   z11.s, z1.b, z5.b                                 \n"
      "   smmla   z13.s, z1.b, z6.b                                 \n"
      "   smmla   z15.s, z1.b, z7.b                                 \n"
      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

  // Store
#if defined(__ARM_FEATURE_SVE2p1)
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #2              \n"
#else
      "   add     %[c0ptr], %[c_dst], %[n_idx], lsl #2              \n"
      "   addvl   %[c1ptr], %[c0ptr], #1                            \n"
#endif
      "   uzp1    z0.d, z16.d, z20.d                                \n"
      "   uzp1    z1.d, z24.d, z28.d                                \n"
      "   uzp2    z2.d, z16.d, z20.d                                \n"
      "   uzp2    z3.d, z24.d, z28.d                                \n"
      "   uzp1    z4.d, z17.d, z21.d                                \n"
      "   uzp1    z5.d, z25.d, z29.d                                \n"
      "   uzp2    z6.d, z17.d, z21.d                                \n"
      "   uzp2    z7.d, z25.d, z29.d                                \n"

#if defined(__ARM_FEATURE_SVE2p1)
      "   st1w    {z0.s-z1.s}, pn8, [%[c_ptr]]                      \n"
      "   st1w    {z2.s-z3.s}, pn8, [%[c_ptr], %[c1off], lsl #2]    \n"
      "   st1w    {z4.s-z5.s}, pn8, [%[c_ptr], %[c2off], lsl #2]    \n"
      "   st1w    {z6.s-z7.s}, pn8, [%[c_ptr], %[c3off], lsl #2]    \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #4                  \n"
#else
      "   st1w    {z0.s}, p0, [%[c0ptr]]                            \n"
      "   st1w    {z1.s}, p0, [%[c1ptr]]                            \n"
      "   st1w    {z2.s}, p0, [%[c0ptr], %[c1off], lsl #2]          \n"
      "   st1w    {z3.s}, p0, [%[c1ptr], %[c1off], lsl #2]          \n"
      "   st1w    {z4.s}, p0, [%[c0ptr], %[c2off], lsl #2]          \n"
      "   st1w    {z5.s}, p0, [%[c1ptr], %[c2off], lsl #2]          \n"
      "   st1w    {z6.s}, p0, [%[c0ptr], %[c3off], lsl #2]          \n"
      "   st1w    {z7.s}, p0, [%[c1ptr], %[c3off], lsl #2]          \n"
      "   add     %[c0ptr], %[c0ptr], %[n], lsl #4                  \n"
      "   add     %[c1ptr], %[c1ptr], %[n], lsl #4                  \n"
#endif

      "   uzp1    z0.d, z18.d, z22.d                                \n"
      "   uzp1    z1.d, z26.d, z30.d                                \n"
      "   uzp2    z2.d, z18.d, z22.d                                \n"
      "   uzp2    z3.d, z26.d, z30.d                                \n"
      "   uzp1    z4.d, z19.d, z23.d                                \n"
      "   uzp1    z5.d, z27.d, z31.d                                \n"
      "   uzp2    z6.d, z19.d, z23.d                                \n"
      "   uzp2    z7.d, z27.d, z31.d                                \n"

#if defined(__ARM_FEATURE_SVE2p1)
      "   st1w    {z0.s-z1.s}, pn8, [%[c_ptr]]                      \n"
      "   st1w    {z2.s-z3.s}, pn8, [%[c_ptr], %[c1off], lsl #2]    \n"
      "   st1w    {z4.s-z5.s}, pn8, [%[c_ptr], %[c2off], lsl #2]    \n"
      "   st1w    {z6.s-z7.s}, pn8, [%[c_ptr], %[c3off], lsl #2]    \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #4                  \n"
#else
      "   st1w    {z0.s}, p0, [%[c0ptr]]                            \n"
      "   st1w    {z1.s}, p0, [%[c1ptr]]                            \n"
      "   st1w    {z2.s}, p0, [%[c0ptr], %[c1off], lsl #2]          \n"
      "   st1w    {z3.s}, p0, [%[c1ptr], %[c1off], lsl #2]          \n"
      "   st1w    {z4.s}, p0, [%[c0ptr], %[c2off], lsl #2]          \n"
      "   st1w    {z5.s}, p0, [%[c1ptr], %[c2off], lsl #2]          \n"
      "   st1w    {z6.s}, p0, [%[c0ptr], %[c3off], lsl #2]          \n"
      "   st1w    {z7.s}, p0, [%[c1ptr], %[c3off], lsl #2]          \n"
      "   add     %[c0ptr], %[c0ptr], %[n], lsl #4                  \n"
      "   add     %[c1ptr], %[c1ptr], %[n], lsl #4                  \n"
#endif

      "   uzp1    z16.d, z8.d,  z10.d                               \n"
      "   uzp1    z17.d, z12.d, z14.d                               \n"
      "   uzp2    z20.d, z8.d,  z10.d                               \n"
      "   uzp2    z21.d, z12.d, z14.d                               \n"

#if defined(__ARM_FEATURE_SVE2p1)
      "   st1w    {z16.s-z17.s}, pn8, [%[c_ptr]]                    \n"
      "   st1w    {z20.s-z21.s}, pn8, [%[c_ptr], %[c1off], lsl #2]  \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #3                  \n"
#else
      "   st1w    {z16.s}, p0, [%[c0ptr]]                           \n"
      "   st1w    {z17.s}, p0, [%[c1ptr]]                           \n"
      "   st1w    {z20.s}, p0, [%[c0ptr], %[c1off], lsl #2]         \n"
      "   st1w    {z21.s}, p0, [%[c1ptr], %[c1off], lsl #2]         \n"
      "   add     %[c0ptr], %[c0ptr], %[n], lsl #3                  \n"
      "   add     %[c1ptr], %[c1ptr], %[n], lsl #3                  \n"
#endif

      "   uzp1    z18.d, z9.d,  z11.d                               \n"
      "   uzp1    z19.d, z13.d, z15.d                               \n"
      "   uzp2    z22.d, z9.d,  z11.d                               \n"
      "   uzp2    z23.d, z13.d, z15.d                               \n"

#if defined(__ARM_FEATURE_SVE2p1)
      "   st1w    {z18.s-z19.s}, pn8, [%[c_ptr]]                    \n"
      "   st1w    {z22.s-z23.s}, pn8, [%[c_ptr], %[c1off], lsl #2]  \n"
#else
      "   st1w    {z18.s}, p0, [%[c0ptr]]                           \n"
      "   st1w    {z19.s}, p0, [%[c1ptr]]                           \n"
      "   st1w    {z22.s}, p0, [%[c0ptr], %[c1off], lsl #2]         \n"
      "   st1w    {z23.s}, p0, [%[c1ptr], %[c1off], lsl #2]         \n"
#endif

      // N loop tail
      "   incw    %[n_idx], all, mul #2                             \n"
      "   cmp     %[n_idx], %[n]                                    \n"
      "   b.mi    2b                                                \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #5                  \n"
      "   add     %[c_dst], %[c_dst], %[n], lsl #4                  \n"
      "   add     %[m_idx], %[m_idx], #12                           \n"
      "   cmp     %[m_idx], %[m]                                    \n"
      "   b.mi    1b                                                \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [m_idx] "=&r"(m_idx),
#if defined(__ARM_FEATURE_SVE2p1)
        [c_ptr] "=&r"(c_ptr), [n_idx] "=&r"(n_idx),
#else
        [c0ptr] "=&r"(c0ptr), [c1ptr] "=&r"(c1ptr), [n_idx] "=&r"(n_idx),
#endif
        [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [c2off] "r"(c2off), [c3off] "r"(c3off), [c1off] "r"(n), [a_src] "r"(a),
        [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10",
        "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20",
        "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30",
        "z31",
#if defined(__ARM_FEATURE_SVE2p1)
        "p8", "p0",
#else
        "p0",
#endif
        "cc", "memory");
}

static void inner_loop_245(struct loop_245_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;
  register uint64_t a_mod = (uint64_t)data->a_mod;
  register uint64_t b_mod = (uint64_t)data->b_mod;

  sve_mmla_interleave_vla(k, m, a, a_mod);
  sve_mmla_interleave_vla(k, n, b, b_mod);
  sve_mmla_gemm(m, k, n, a_mod, b_mod, c);
}

#else

static void sve_dot_interleave_vla(uint64_t rows, uint64_t cols,
                                   uint64_t matrix,
                                   uint64_t matrix_new) LOOP_ATTR {
  register uint64_t row = rows;
  register uint64_t col = cols;
  register uint64_t mat = matrix;
  register uint64_t mat_mod = matrix_new;

  register uint64_t mat_off2 = col * 2;
  register uint64_t mat_off3 = col * 3;
  register uint64_t mat1ptr;
  register uint64_t mat_mod1ptr;
  register uint64_t cnd;
  register uint64_t row_idx;

  asm volatile(

#if defined(__ARM_FEATURE_SVE2p1)
      "   ptrue   pn8.b                                             \n"
#endif
      "   ptrue   p0.b                                              \n"
      "   mov     %[row_idx], #0                                    \n"

      // Row loop head
      "1:                                                           \n"
      "   mov     %[mat_mod1ptr], %[mat_mod0ptr]                    \n"
      "   mov     %[mat1ptr], %[mat0ptr]                            \n"
      "   add     %[cnd], %[mat1ptr], %[col]                        \n"

      // Column loop head
      "2:                                                           \n"
      "   ld1b    {z0.b}, p0/z, [%[mat1ptr]]                        \n"
      "   ld1b    {z1.b}, p0/z, [%[mat1ptr], %[mat_off1]]           \n"
      "   ld1b    {z2.b}, p0/z, [%[mat1ptr], %[mat_off2]]           \n"
      "   ld1b    {z3.b}, p0/z, [%[mat1ptr], %[mat_off3]]           \n"

      "   zip1    z4.b,  z0.b,  z2.b                                \n"
      "   zip1    z5.b,  z1.b,  z3.b                                \n"
      "   zip2    z6.b,  z0.b,  z2.b                                \n"
      "   zip2    z7.b,  z1.b,  z3.b                                \n"

      "   zip1    z0.b,  z4.b,  z5.b                                \n"
      "   zip2    z1.b,  z4.b,  z5.b                                \n"
      "   zip1    z2.b,  z6.b,  z7.b                                \n"
      "   zip2    z3.b,  z6.b,  z7.b                                \n"

#if defined(__ARM_FEATURE_SVE2p1)
      "   st1b    {z0.b-z3.b}, pn8, [%[mat_mod1ptr]]                \n"
#else
      "   st1b    {z0.b}, p0, [%[mat_mod1ptr]]                      \n"
      "   st1b    {z1.b}, p0, [%[mat_mod1ptr], #1, mul vl]          \n"
      "   st1b    {z2.b}, p0, [%[mat_mod1ptr], #2, mul vl]          \n"
      "   st1b    {z3.b}, p0, [%[mat_mod1ptr], #3, mul vl]          \n"
#endif

      "   addvl   %[mat_mod1ptr], %[mat_mod1ptr], #4                \n"
      "   addvl   %[mat1ptr], %[mat1ptr], #1                        \n"
      "   cmp     %[mat1ptr], %[cnd]                                \n"
      "   b.mi    2b                                                \n"

      "   add     %[mat_mod0ptr], %[mat_mod0ptr], %[col], lsl #2    \n"
      "   add     %[mat0ptr], %[mat0ptr], %[col], lsl #2            \n"
      "   add     %[row_idx], %[row_idx], #4                        \n"
      "   cmp     %[row_idx], %[row]                                \n"
      "   b.mi    1b                                                \n"

      : [mat0ptr] "+&r"(mat), [mat1ptr] "=&r"(mat1ptr),
        [mat_mod0ptr] "+&r"(mat_mod), [mat_mod1ptr] "=&r"(mat_mod1ptr),
        [cnd] "=&r"(cnd), [row_idx] "=&r"(row_idx)
      : [row] "r"(row), [col] "r"(col), [mat_off1] "r"(cols),
        [mat_off2] "r"(mat_off2), [mat_off3] "r"(mat_off3)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "p0", "p8", "cc",
        "memory");
}

static void sve_dot_gemm(uint64_t M, uint64_t K, uint64_t N, uint64_t A_new,
                         uint64_t B_new, uint64_t C) LOOP_ATTR {
  register uint64_t m = M;
  register uint64_t k = K;
  register uint64_t n = N;
  register uint64_t a = A_new;
  register uint64_t b = B_new;
  register uint64_t c = C;

  register uint64_t c2off = n * 2;
  register uint64_t c3off = n * 3;
  register uint64_t a_cnd = a + (m * k);
  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t n_idx;
  register uint64_t m_idx;
#if defined(__ARM_FEATURE_SVE2p1)
  register uint64_t c_ptr;
#else
  register uint64_t c0ptr;
  register uint64_t c1ptr;
#endif

  asm volatile(
#if defined(__ARM_FEATURE_SVE2p1)
      "   ptrue   pn8.b                                             \n"
#endif
      "   ptrue   p0.b                                              \n"

      // M loop head
      "   mov     %[m_idx], #0                                      \n"
      "1:                                                           \n"

      // N loop head
      "   mov     %[n_idx], #0                                      \n"
      "2:                                                           \n"

      // Accumulators
      "   mov     z8.s,  #0                                         \n"
      "   mov     z9.s,  #0                                         \n"
      "   mov     z10.s, #0                                         \n"
      "   mov     z11.s, #0                                         \n"
      "   mov     z12.s, #0                                         \n"
      "   mov     z13.s, #0                                         \n"
      "   mov     z14.s, #0                                         \n"
      "   mov     z15.s, #0                                         \n"
      "   mov     z16.s, #0                                         \n"
      "   mov     z17.s, #0                                         \n"
      "   mov     z18.s, #0                                         \n"
      "   mov     z19.s, #0                                         \n"
      "   mov     z20.s, #0                                         \n"
      "   mov     z21.s, #0                                         \n"
      "   mov     z22.s, #0                                         \n"
      "   mov     z23.s, #0                                         \n"
      "   mov     z24.s, #0                                         \n"
      "   mov     z25.s, #0                                         \n"
      "   mov     z26.s, #0                                         \n"
      "   mov     z27.s, #0                                         \n"
      "   mov     z28.s, #0                                         \n"
      "   mov     z29.s, #0                                         \n"
      "   mov     z30.s, #0                                         \n"
      "   mov     z31.s, #0                                         \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2              \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #2              \n"
      "3:                                                           \n"
      "   ld1rqb  {z0.b}, p0/z, [%[a_ptr]]                          \n"
      "   ld1rqb  {z1.b}, p0/z, [%[a_ptr], #16]                     \n"
      "   ld1rqb  {z2.b}, p0/z, [%[a_ptr], #32]                     \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                  \n"
#if defined(__ARM_FEATURE_SVE2p1)
      "   ld1b    {z4.b-z5.b}, pn8/z, [%[b_ptr]]                    \n"
#else
      "   ld1b    {z4.b}, p0/z, [%[b_ptr]]                          \n"
      "   ld1b    {z5.b}, p0/z, [%[b_ptr], #1, mul vl]              \n"
#endif
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                  \n"

      "   sdot    z10.s, z4.b, z0.b[0]                              \n"
      "   sdot    z12.s, z4.b, z0.b[1]                              \n"
      "   sdot    z14.s, z4.b, z0.b[2]                              \n"
      "   sdot    z16.s, z4.b, z0.b[3]                              \n"

      "   sdot    z20.s, z4.b, z1.b[0]                              \n"
      "   sdot    z22.s, z4.b, z1.b[1]                              \n"
      "   sdot    z24.s, z4.b, z1.b[2]                              \n"
      "   sdot    z26.s, z4.b, z1.b[3]                              \n"

      "   sdot    z28.s, z4.b, z2.b[0]                              \n"
      "   sdot    z30.s, z4.b, z2.b[1]                              \n"
      "   sdot    z8.s,  z4.b, z2.b[2]                              \n"
      "   sdot    z18.s, z4.b, z2.b[3]                              \n"

      "   sdot    z11.s, z5.b, z0.b[0]                              \n"
      "   sdot    z13.s, z5.b, z0.b[1]                              \n"
      "   sdot    z15.s, z5.b, z0.b[2]                              \n"
      "   sdot    z17.s, z5.b, z0.b[3]                              \n"

      "   sdot    z21.s, z5.b, z1.b[0]                              \n"
      "   sdot    z23.s, z5.b, z1.b[1]                              \n"
      "   sdot    z25.s, z5.b, z1.b[2]                              \n"
      "   sdot    z27.s, z5.b, z1.b[3]                              \n"

      "   sdot    z29.s, z5.b, z2.b[0]                              \n"
      "   sdot    z31.s, z5.b, z2.b[1]                              \n"
      "   sdot    z9.s,  z5.b, z2.b[2]                              \n"
      "   sdot    z19.s, z5.b, z2.b[3]                              \n"

      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

#if defined(__ARM_FEATURE_SVE2p1)
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #2              \n"
#else
      "   add     %[c0ptr], %[c_dst], %[n_idx], lsl #2              \n"
      "   addvl   %[c1ptr], %[c0ptr], #1                            \n"
#endif

#if defined(__ARM_FEATURE_SVE2p1)
      "   st1w    {z10.s-z11.s}, pn8, [%[c_ptr]]                    \n"
      "   st1w    {z12.s-z13.s}, pn8, [%[c_ptr], %[c1off], lsl #2]  \n"
      "   st1w    {z14.s-z15.s}, pn8, [%[c_ptr], %[c2off], lsl #2]  \n"
      "   st1w    {z16.s-z17.s}, pn8, [%[c_ptr], %[c3off], lsl #2]  \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #4                  \n"
#else
      "   st1w    {z10.s}, p0, [%[c0ptr]]                           \n"
      "   st1w    {z11.s}, p0, [%[c1ptr]]                           \n"
      "   st1w    {z12.s}, p0, [%[c0ptr], %[c1off], lsl #2]         \n"
      "   st1w    {z13.s}, p0, [%[c1ptr], %[c1off], lsl #2]         \n"
      "   st1w    {z14.s}, p0, [%[c0ptr], %[c2off], lsl #2]         \n"
      "   st1w    {z15.s}, p0, [%[c1ptr], %[c2off], lsl #2]         \n"
      "   st1w    {z16.s}, p0, [%[c0ptr], %[c3off], lsl #2]         \n"
      "   st1w    {z17.s}, p0, [%[c1ptr], %[c3off], lsl #2]         \n"
      "   add     %[c0ptr], %[c0ptr], %[n], lsl #4                  \n"
      "   add     %[c1ptr], %[c1ptr], %[n], lsl #4                  \n"
#endif

#if defined(__ARM_FEATURE_SVE2p1)
      "   st1w    {z20.s-z21.s}, pn8, [%[c_ptr]]                    \n"
      "   st1w    {z22.s-z23.s}, pn8, [%[c_ptr], %[c1off], lsl #2]  \n"
      "   st1w    {z24.s-z25.s}, pn8, [%[c_ptr], %[c2off], lsl #2]  \n"
      "   st1w    {z26.s-z27.s}, pn8, [%[c_ptr], %[c3off], lsl #2]  \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #4                  \n"
#else
      "   st1w    {z20.s}, p0, [%[c0ptr]]                           \n"
      "   st1w    {z21.s}, p0, [%[c1ptr]]                           \n"
      "   st1w    {z22.s}, p0, [%[c0ptr], %[c1off], lsl #2]         \n"
      "   st1w    {z23.s}, p0, [%[c1ptr], %[c1off], lsl #2]         \n"
      "   st1w    {z24.s}, p0, [%[c0ptr], %[c2off], lsl #2]         \n"
      "   st1w    {z25.s}, p0, [%[c1ptr], %[c2off], lsl #2]         \n"
      "   st1w    {z26.s}, p0, [%[c0ptr], %[c3off], lsl #2]         \n"
      "   st1w    {z27.s}, p0, [%[c1ptr], %[c3off], lsl #2]         \n"
      "   add     %[c0ptr], %[c0ptr], %[n], lsl #4                  \n"
      "   add     %[c1ptr], %[c1ptr], %[n], lsl #4                  \n"
#endif

#if defined(__ARM_FEATURE_SVE2p1)
      "   st1w    {z28.s-z29.s}, pn8, [%[c_ptr]]                    \n"
      "   st1w    {z30.s-z31.s}, pn8, [%[c_ptr], %[c1off], lsl #2]  \n"
      "   st1w    {z8.s-z9.s},   pn8, [%[c_ptr], %[c2off], lsl #2]  \n"
      "   st1w    {z18.s-z19.s}, pn8, [%[c_ptr], %[c3off], lsl #2]  \n"
#else
      "   st1w    {z28.s}, p0, [%[c0ptr]]                           \n"
      "   st1w    {z29.s}, p0, [%[c1ptr]]                           \n"
      "   st1w    {z30.s}, p0, [%[c0ptr], %[c1off], lsl #2]         \n"
      "   st1w    {z31.s}, p0, [%[c1ptr], %[c1off], lsl #2]         \n"
      "   st1w    {z8.s},  p0, [%[c0ptr], %[c2off], lsl #2]         \n"
      "   st1w    {z9.s},  p0, [%[c1ptr], %[c2off], lsl #2]         \n"
      "   st1w    {z18.s}, p0, [%[c0ptr], %[c3off], lsl #2]         \n"
      "   st1w    {z19.s}, p0, [%[c1ptr], %[c3off], lsl #2]         \n"
#endif

      // N loop tail
      "   incw    %[n_idx], all, mul #2                             \n"
      "   cmp     %[n_idx], %[n]                                    \n"
      "   b.mi    2b                                                \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #5                  \n"
      "   add     %[c_dst], %[c_dst], %[n], lsl #4                  \n"
      "   add     %[m_idx], %[m_idx], #12                           \n"
      "   cmp     %[m_idx], %[m]                                    \n"
      "   b.mi    1b                                                \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [m_idx] "=&r"(m_idx),
#if defined(__ARM_FEATURE_SVE2p1)
        [c_ptr] "=&r"(c_ptr), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
#else
        [n_idx] "=&r"(n_idx), [c0ptr] "=&r"(c0ptr), [c1ptr] "=&r"(c1ptr),
        [c_dst] "+&r"(c)
#endif
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [c2off] "r"(c2off),
        [c3off] "r"(c3off), [c1off] "r"(n), [a_src] "r"(a), [b_src] "r"(b),
        [a_cnd] "r"(a_cnd)
      : "z0", "z1", "z2", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11",
        "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21",
        "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",
#if defined(__ARM_FEATURE_SVE2p1)
        "p8", "p0",
#else
        "p0",
#endif
        "cc", "memory");
}

static void inner_loop_245(struct loop_245_data *data) LOOP_ATTR {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;
  register uint64_t a_mod = (uint64_t)data->a_mod;
  register uint64_t b_mod = (uint64_t)data->b_mod;

  sve_dot_interleave_vla(k, m, a, a_mod);
  sve_dot_interleave_vla(k, n, b, b_mod);
  sve_dot_gemm(m, k, n, a_mod, b_mod, c);
}

#endif

#elif (defined(__ARM_NEON) && defined (__ARM_FEATURE_DOTPROD))
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)

static void neon_mmla_interleave(uint64_t rows, uint64_t cols, uint64_t matrix,
                                 uint64_t matrix_new) {
  register uint64_t row = rows;
  register uint64_t col = cols;
  register uint64_t mat = matrix;
  register uint64_t mat_mod = matrix_new;

  register uint64_t mat1ptr;
  register uint64_t mat_mod1ptr;
  register uint64_t cnd;
  register uint64_t row_idx;

  asm volatile(

      "   mov     %[row_idx], #0                                    \n"

      // Row loop head
      "1:                                                           \n"
      "   mov     %[mat_mod1ptr], %[mat_mod0ptr]                    \n"
      "   add     %[cnd], %[mat0ptr], %[col]                        \n"

      // Column loop head
      "2:                                                           \n"
      "   mov     %[mat1ptr], %[mat0ptr]                            \n"
      "   ld1     {v0.16b}, [%[mat1ptr]], %[mat_off1]               \n"
      "   ld1     {v1.16b}, [%[mat1ptr]], %[mat_off1]               \n"
      "   ld1     {v2.16b}, [%[mat1ptr]], %[mat_off1]               \n"
      "   ld1     {v3.16b}, [%[mat1ptr]], %[mat_off1]               \n"
      "   ld1     {v4.16b}, [%[mat1ptr]], %[mat_off1]               \n"
      "   ld1     {v5.16b}, [%[mat1ptr]], %[mat_off1]               \n"
      "   ld1     {v6.16b}, [%[mat1ptr]], %[mat_off1]               \n"
      "   ld1     {v7.16b}, [%[mat1ptr]]                            \n"

      "   zip1    v16.16b,  v0.16b,  v4.16b                         \n"
      "   zip1    v17.16b,  v2.16b,  v6.16b                         \n"
      "   zip1    v18.16b,  v1.16b,  v5.16b                         \n"
      "   zip1    v19.16b,  v3.16b,  v7.16b                         \n"
      "   zip2    v20.16b,  v0.16b,  v4.16b                         \n"
      "   zip2    v21.16b,  v2.16b,  v6.16b                         \n"
      "   zip2    v22.16b,  v1.16b,  v5.16b                         \n"
      "   zip2    v23.16b,  v3.16b,  v7.16b                         \n"

      "   zip1    v24.16b,  v16.16b,  v17.16b                       \n"
      "   zip1    v25.16b,  v18.16b,  v19.16b                       \n"
      "   zip2    v26.16b,  v16.16b,  v17.16b                       \n"
      "   zip2    v27.16b,  v18.16b,  v19.16b                       \n"
      "   zip1    v0.16b,   v24.16b,  v25.16b                       \n"
      "   zip2    v1.16b,   v24.16b,  v25.16b                       \n"
      "   zip1    v2.16b,   v26.16b,  v27.16b                       \n"
      "   zip2    v3.16b,   v26.16b,  v27.16b                       \n"

      "   zip1    v24.16b,  v20.16b,  v21.16b                       \n"
      "   zip1    v25.16b,  v22.16b,  v23.16b                       \n"
      "   zip2    v26.16b,  v20.16b,  v21.16b                       \n"
      "   zip2    v27.16b,  v22.16b,  v23.16b                       \n"
      "   zip1    v4.16b,   v24.16b,  v25.16b                       \n"
      "   zip2    v5.16b,   v24.16b,  v25.16b                       \n"
      "   zip1    v6.16b,   v26.16b,  v27.16b                       \n"
      "   zip2    v7.16b,   v26.16b,  v27.16b                       \n"

      "   st1     {v0.16b,v1.16b,v2.16b,v3.16b}, [%[mat_mod1ptr]], #64 \n"
      "   st1     {v4.16b,v5.16b,v6.16b,v7.16b}, [%[mat_mod1ptr]], #64 \n"

      "   add    %[mat0ptr], %[mat0ptr], #16                        \n"
      "   cmp    %[mat0ptr], %[cnd]                                 \n"
      "   b.mi    2b                                                \n"

      "   add     %[mat_mod0ptr], %[mat_mod0ptr], %[col], lsl #3    \n"
      "   add     %[mat0ptr], %[mat1ptr], #16                       \n"
      "   add     %[row_idx], %[row_idx], #8                        \n"
      "   cmp     %[row_idx], %[row]                                \n"
      "   b.mi    1b                                                \n"

      : [mat0ptr] "+&r"(mat), [mat1ptr] "=&r"(mat1ptr),
        [mat_mod0ptr] "+&r"(mat_mod), [mat_mod1ptr] "+&r"(mat_mod1ptr),
        [cnd] "=&r"(cnd), [row_idx] "=&r"(row_idx)
      : [row] "r"(row), [col] "r"(col), [mat_off1] "r"(cols)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18",
        "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
        "v29", "v30", "v31", "cc", "memory");
}

static void neon_mmla_gemm(uint64_t M, uint64_t K, uint64_t N, uint64_t A_new,
                           uint64_t B_new, uint64_t C) {
  register uint64_t m = M;
  register uint64_t k = K;
  register uint64_t n = N;
  register uint64_t a = A_new;
  register uint64_t b = B_new;
  register uint64_t c = C;

  register uint64_t c_inc = n * 4;
  register uint64_t a_cnd = a + (m * k);

  register uint64_t n_idx;
  register uint64_t m_idx;
  register uint64_t a_ptr;
  register uint64_t a1ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;

  asm volatile(

      // M loop head
      "   mov     %[m_idx], #0                                      \n"
      "1:                                                           \n"

      // N loop head
      "   mov     %[n_idx], #0                                      \n"
      "2:                                                           \n"

      // Accumulators
      "   movi     v8.4s,  #0                                       \n"
      "   movi     v9.4s,  #0                                       \n"
      "   movi     v10.4s, #0                                       \n"
      "   movi     v11.4s, #0                                       \n"
      "   movi     v12.4s, #0                                       \n"
      "   movi     v13.4s, #0                                       \n"
      "   movi     v14.4s, #0                                       \n"
      "   movi     v15.4s, #0                                       \n"
      "   movi     v16.4s, #0                                       \n"
      "   movi     v17.4s, #0                                       \n"
      "   movi     v18.4s, #0                                       \n"
      "   movi     v19.4s, #0                                       \n"
      "   movi     v20.4s, #0                                       \n"
      "   movi     v21.4s, #0                                       \n"
      "   movi     v22.4s, #0                                       \n"
      "   movi     v23.4s, #0                                       \n"
      "   movi     v24.4s, #0                                       \n"
      "   movi     v25.4s, #0                                       \n"
      "   movi     v26.4s, #0                                       \n"
      "   movi     v27.4s, #0                                       \n"
      "   movi     v28.4s, #0                                       \n"
      "   movi     v29.4s, #0                                       \n"
      "   movi     v30.4s, #0                                       \n"
      "   movi     v31.4s, #0                                       \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #3              \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #3              \n"
      "3:                                                           \n"
      "   ld1     {v0.16b,v1.16b,v2.16b,v3.16b}, [%[a_ptr]]         \n"
      "   ld1     {v4.16b,v5.16b,v6.16b,v7.16b}, [%[b_ptr]]         \n"
      "   smmla   v16.4s, v0.16b, v4.16b                            \n"
      "   smmla   v20.4s, v0.16b, v5.16b                            \n"
      "   smmla   v24.4s, v0.16b, v6.16b                            \n"
      "   smmla   v28.4s, v0.16b, v7.16b                            \n"

      "   add     %[a1ptr], %[a_ptr], #64                           \n"
      "   smmla   v17.4s, v1.16b, v4.16b                            \n"
      "   smmla   v21.4s, v1.16b, v5.16b                            \n"
      "   smmla   v25.4s, v1.16b, v6.16b                            \n"
      "   smmla   v29.4s, v1.16b, v7.16b                            \n"

      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                  \n"
      "   smmla   v18.4s, v2.16b, v4.16b                            \n"
      "   smmla   v22.4s, v2.16b, v5.16b                            \n"
      "   smmla   v26.4s, v2.16b, v6.16b                            \n"
      "   smmla   v30.4s, v2.16b, v7.16b                            \n"

      "   ld1     {v0.16b,v1.16b}, [%[a1ptr]]                       \n"
      "   smmla   v19.4s, v3.16b, v4.16b                            \n"
      "   smmla   v23.4s, v3.16b, v5.16b                            \n"
      "   smmla   v27.4s, v3.16b, v6.16b                            \n"
      "   smmla   v31.4s, v3.16b, v7.16b                            \n"

      "   smmla   v8.4s,  v0.16b, v4.16b                            \n"
      "   smmla   v10.4s, v0.16b, v5.16b                            \n"
      "   smmla   v12.4s, v0.16b, v6.16b                            \n"
      "   smmla   v14.4s, v0.16b, v7.16b                            \n"

      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   smmla   v9.4s,  v1.16b, v4.16b                            \n"
      "   smmla   v11.4s, v1.16b, v5.16b                            \n"
      "   smmla   v13.4s, v1.16b, v6.16b                            \n"
      "   smmla   v15.4s, v1.16b, v7.16b                            \n"
      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #2              \n"
      "   uzp1    v0.2d, v16.2d, v20.2d                             \n"
      "   uzp1    v1.2d, v24.2d, v28.2d                             \n"
      "   uzp2    v2.2d, v16.2d, v20.2d                             \n"
      "   uzp2    v3.2d, v24.2d, v28.2d                             \n"
      "   uzp1    v4.2d, v17.2d, v21.2d                             \n"
      "   uzp1    v5.2d, v25.2d, v29.2d                             \n"
      "   uzp2    v6.2d, v17.2d, v21.2d                             \n"
      "   uzp2    v7.2d, v25.2d, v29.2d                             \n"
      "   st1     {v0.4s,v1.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   st1     {v2.4s,v3.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   st1     {v4.4s,v5.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   st1     {v6.4s,v7.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   uzp1    v0.2d, v18.2d, v22.2d                             \n"
      "   uzp1    v1.2d, v26.2d, v30.2d                             \n"
      "   uzp2    v2.2d, v18.2d, v22.2d                             \n"
      "   uzp2    v3.2d, v26.2d, v30.2d                             \n"
      "   uzp1    v4.2d, v19.2d, v23.2d                             \n"
      "   uzp1    v5.2d, v27.2d, v31.2d                             \n"
      "   uzp2    v6.2d, v19.2d, v23.2d                             \n"
      "   uzp2    v7.2d, v27.2d, v31.2d                             \n"
      "   st1     {v0.4s,v1.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   st1     {v2.4s,v3.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   st1     {v4.4s,v5.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   st1     {v6.4s,v7.4s}, [%[c_ptr]], %[c_inc]               \n"
      "   uzp1    v16.2d, v8.2d,  v10.2d                            \n"
      "   uzp1    v17.2d, v12.2d, v14.2d                            \n"
      "   uzp2    v20.2d, v8.2d,  v10.2d                            \n"
      "   uzp2    v21.2d, v12.2d, v14.2d                            \n"
      "   st1     {v16.4s,v17.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   st1     {v20.4s,v21.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   uzp1    v18.2d, v9.2d,  v11.2d                            \n"
      "   uzp1    v19.2d, v13.2d, v15.2d                            \n"
      "   uzp2    v22.2d, v9.2d,  v11.2d                            \n"
      "   uzp2    v23.2d, v13.2d, v15.2d                            \n"
      "   st1     {v18.4s,v19.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   st1     {v22.4s,v23.4s}, [%[c_ptr]], %[c_inc]             \n"

      // N loop tail
      "   add     %[n_idx], %[n_idx], #8                            \n"
      "   cmp     %[n_idx], %[n]                                    \n"
      "   b.mi    2b                                                \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #5                  \n"
      "   add     %[c_dst], %[c_dst], %[n], lsl #4                  \n"
      "   add     %[m_idx], %[m_idx], #12                           \n"
      "   cmp     %[m_idx], %[m]                                    \n"
      "   b.mi    1b                                                \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [m_idx] "=&r"(m_idx),
        [c_ptr] "+&r"(c_ptr), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c),
        [a1ptr] "=&r"(a1ptr)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd), [a_src] "r"(a),
        [c_inc] "r"(c_inc), [b_src] "r"(b)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
        "v30", "v31", "cc", "memory");
}

static void inner_loop_245(struct loop_245_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;
  register uint64_t a_mod = (uint64_t)data->a_mod;
  register uint64_t b_mod = (uint64_t)data->b_mod;

  neon_mmla_interleave(k, m, a, a_mod);
  neon_mmla_interleave(k, n, b, b_mod);
  neon_mmla_gemm(m, k, n, a_mod, b_mod, c);
}

#else

static void neon_dot_interleave(uint64_t rows, uint64_t cols, uint64_t matrix,
                                uint64_t matrix_new) {
  register uint64_t row = rows;
  register uint64_t col = cols;
  register uint64_t mat = matrix;
  register uint64_t mat_mod = matrix_new;

  register uint64_t mat1ptr;
  register uint64_t mat_mod1ptr;
  register uint64_t cnd;
  register uint64_t row_idx;

  asm volatile(

      "   mov     %[row_idx], #0                                        \n"

      // Row loop head
      "1:                                                               \n"
      "   mov     %[mat_mod1ptr], %[mat_mod0ptr]                        \n"
      "   add     %[cnd], %[mat0ptr], %[col]                            \n"

      // Column loop head
      "2:                                                               \n"
      "   mov     %[mat1ptr], %[mat0ptr]                                \n"
      "   ld1     {v0.16b}, [%[mat1ptr]], %[mat_off1]                   \n"
      "   ld1     {v1.16b}, [%[mat1ptr]], %[mat_off1]                   \n"
      "   ld1     {v2.16b}, [%[mat1ptr]], %[mat_off1]                   \n"
      "   ld1     {v3.16b}, [%[mat1ptr]]                                \n"

      "   zip1    v4.16b,  v0.16b,  v2.16b                              \n"
      "   zip1    v5.16b,  v1.16b,  v3.16b                              \n"
      "   zip2    v6.16b,  v0.16b,  v2.16b                              \n"
      "   zip2    v7.16b,  v1.16b,  v3.16b                              \n"

      "   zip1    v0.16b,  v4.16b,  v5.16b                              \n"
      "   zip2    v1.16b,  v4.16b,  v5.16b                              \n"
      "   zip1    v2.16b,  v6.16b,  v7.16b                              \n"
      "   zip2    v3.16b,  v6.16b,  v7.16b                              \n"

      "   st1     {v0.16b,v1.16b,v2.16b,v3.16b}, [%[mat_mod1ptr]], #64  \n"

      "   add     %[mat0ptr], %[mat0ptr], #16                           \n"
      "   cmp     %[mat0ptr], %[cnd]                                    \n"
      "   b.mi    2b                                                    \n"

      "   add     %[mat_mod0ptr], %[mat_mod0ptr], %[col], lsl #2        \n"
      "   add     %[mat0ptr], %[mat1ptr], #16                           \n"
      "   add     %[row_idx], %[row_idx], #4                            \n"
      "   cmp     %[row_idx], %[row]                                    \n"
      "   b.mi    1b                                                    \n"

      : [mat0ptr] "+&r"(mat), [mat1ptr] "=&r"(mat1ptr),
        [mat_mod0ptr] "+&r"(mat_mod), [mat_mod1ptr] "+&r"(mat_mod1ptr),
        [cnd] "=&r"(cnd), [row_idx] "=&r"(row_idx)
      : [row] "r"(row), [col] "r"(col), [mat_off1] "r"(cols)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "cc", "memory");
}

static void neon_dot_gemm(uint64_t M, uint64_t K, uint64_t N, uint64_t A_new,
                          uint64_t B_new, uint64_t C) {
  register uint64_t m = M;
  register uint64_t k = K;
  register uint64_t n = N;
  register uint64_t a = A_new;
  register uint64_t b = B_new;
  register uint64_t c = C;

  register uint64_t c_inc = n * 4;
  register uint64_t a_cnd = a + (m * k);

  register uint64_t a_ptr;
  register uint64_t a1ptr;
  register uint64_t b_ptr;
  register uint64_t n_idx;
  register uint64_t m_idx;
  register uint64_t c_ptr;

  asm volatile(

      // M loop head
      "   mov     %[m_idx], #0                                      \n"
      "1:                                                           \n"

      // N loop head
      "   mov     %[n_idx], #0                                      \n"
      "2:                                                           \n"

      // Accumulators
      "   movi     v8.4s,  #0                                       \n"
      "   movi     v9.4s,  #0                                       \n"
      "   movi     v10.4s, #0                                       \n"
      "   movi     v11.4s, #0                                       \n"
      "   movi     v12.4s, #0                                       \n"
      "   movi     v13.4s, #0                                       \n"
      "   movi     v14.4s, #0                                       \n"
      "   movi     v15.4s, #0                                       \n"
      "   movi     v16.4s, #0                                       \n"
      "   movi     v17.4s, #0                                       \n"
      "   movi     v18.4s, #0                                       \n"
      "   movi     v19.4s, #0                                       \n"
      "   movi     v20.4s, #0                                       \n"
      "   movi     v21.4s, #0                                       \n"
      "   movi     v22.4s, #0                                       \n"
      "   movi     v23.4s, #0                                       \n"
      "   movi     v24.4s, #0                                       \n"
      "   movi     v25.4s, #0                                       \n"
      "   movi     v26.4s, #0                                       \n"
      "   movi     v27.4s, #0                                       \n"
      "   movi     v28.4s, #0                                       \n"
      "   movi     v29.4s, #0                                       \n"
      "   movi     v30.4s, #0                                       \n"
      "   movi     v31.4s, #0                                       \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2              \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #2              \n"

      "3:                                                           \n"
      "   ld1     {v0.16b,v1.16b}, [%[a_ptr]]                       \n"
      "   add     %[a1ptr], %[a_ptr], #32                           \n"
      "   ld1     {v2.16b}, [%[a1ptr]]                              \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                  \n"

      "   ld1     {v3.16b,v4.16b}, [%[b_ptr]]                       \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                  \n"

      "   sdot    v10.4s, v3.16b, v0.4b[0]                          \n"
      "   sdot    v12.4s, v3.16b, v0.4b[1]                          \n"
      "   sdot    v14.4s, v3.16b, v0.4b[2]                          \n"
      "   sdot    v16.4s, v3.16b, v0.4b[3]                          \n"

      "   sdot    v20.4s, v3.16b, v1.4b[0]                          \n"
      "   sdot    v22.4s, v3.16b, v1.4b[1]                          \n"
      "   sdot    v24.4s, v3.16b, v1.4b[2]                          \n"
      "   sdot    v26.4s, v3.16b, v1.4b[3]                          \n"

      "   sdot    v28.4s, v3.16b, v2.4b[0]                          \n"
      "   sdot    v30.4s, v3.16b, v2.4b[1]                          \n"
      "   sdot    v8.4s,  v3.16b, v2.4b[2]                          \n"
      "   sdot    v18.4s, v3.16b, v2.4b[3]                          \n"

      "   sdot    v11.4s, v4.16b, v0.4b[0]                          \n"
      "   sdot    v13.4s, v4.16b, v0.4b[1]                          \n"
      "   sdot    v15.4s, v4.16b, v0.4b[2]                          \n"
      "   sdot    v17.4s, v4.16b, v0.4b[3]                          \n"

      "   sdot    v21.4s, v4.16b, v1.4b[0]                          \n"
      "   sdot    v23.4s, v4.16b, v1.4b[1]                          \n"
      "   sdot    v25.4s, v4.16b, v1.4b[2]                          \n"
      "   sdot    v27.4s, v4.16b, v1.4b[3]                          \n"

      "   sdot    v29.4s, v4.16b, v2.4b[0]                          \n"
      "   sdot    v31.4s, v4.16b, v2.4b[1]                          \n"
      "   sdot    v9.4s,  v4.16b, v2.4b[2]                          \n"
      "   sdot    v19.4s, v4.16b, v2.4b[3]                          \n"

      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #2              \n"
      "   st1     {v10.4s,v11.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   st1     {v12.4s,v13.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   st1     {v14.4s,v15.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   st1     {v16.4s,v17.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   st1     {v20.4s,v21.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   st1     {v22.4s,v23.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   st1     {v24.4s,v25.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   st1     {v26.4s,v27.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   st1     {v28.4s,v29.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   st1     {v30.4s,v31.4s}, [%[c_ptr]], %[c_inc]             \n"
      "   st1     {v8.4s,v9.4s},   [%[c_ptr]], %[c_inc]             \n"
      "   st1     {v18.4s,v19.4s}, [%[c_ptr]], %[c_inc]             \n"

      // N loop tail
      "   add     %[n_idx], %[n_idx], #8                            \n"
      "   cmp     %[n_idx], %[n]                                    \n"
      "   b.mi    2b                                                \n"

      "   add     %[c_dst], %[c_dst], %[n], lsl #5                  \n"
      "   add     %[c_dst], %[c_dst], %[n], lsl #4                  \n"
      "   add     %[m_idx], %[m_idx], #12                           \n"
      "   cmp     %[m_idx], %[m]                                    \n"
      "   b.mi    1b                                                \n"

      : [a_ptr] "=&r"(a_ptr), [a1ptr] "=&r"(a1ptr), [b_ptr] "=&r"(b_ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_ptr] "+&r"(c_ptr),
        [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [c_inc] "r"(c_inc), [a_src] "r"(a),
        [b_src] "r"(b), [a_cnd] "r"(a_cnd)
      : "v0", "v1", "v2", "v3", "v4", "v6", "v7", "v8", "v9", "v10", "v11",
        "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
        "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
        "cc", "memory");
}

static void inner_loop_245(struct loop_245_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;
  register uint64_t a_mod = (uint64_t)data->a_mod;
  register uint64_t b_mod = (uint64_t)data->b_mod;

  neon_dot_interleave(k, m, a, a_mod);
  neon_dot_interleave(k, n, b, b_mod);
  neon_dot_gemm(m, k, n, a_mod, b_mod, c);
}
#endif

#else

static void inner_loop_245(struct loop_245_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif


// Ensure the max SVL that will be targetted is defined
#if (!defined(MAX_VL) || MAX_VL == 0)
#undef MAX_VL
#define MAX_VL 2048
#endif

// Re-define PROBLEM_SIZE_LIMIT_KIB if it has been set to 0
// Default of 192KiB equates to original problem size (M=256, K=256, N=512)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 197
#endif
#endif /* !HAVE_CANDIDATE */

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m, n, k) ((k) * ((m) + (n)) * sizeof(int8_t))

LOOP_DECL(245, OUTER_LOOP_ATTR) {
  // Work out values for M, K and N to fit within problem size limit
  uint64_t M = 0;  // multiple of 16
  uint64_t K = 0;  // multiple of SVLb
  uint64_t N = 0;  // multiple of 2*SVLs

  const uint64_t K_base = MAX_VL / 8;
  uint64_t m = 7 * 16;

  while (true) {
    uint64_t k = K + K_base;
    uint64_t n = k / 2;
    if (PROBLEM_SIZE_ACTUAL(m, n, k) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
      M = m;
      N = n;
      K = k;
    } else {
      break;
    }
  }

  struct loop_245_data data = {
      .m = M,
      .n = N,
      .k = K,
  };

  ALLOC_64B(data.a, M * K, "A matrix");
  ALLOC_64B(data.a_mod, M * K, "A Preprocessed matrix");
  ALLOC_64B(data.b, K * N, "B matrix");
  ALLOC_64B(data.b_mod, K * N, "B Preprocessed matrix");
  ALLOC_64B(data.c, M * N, "C matrix");

  fill_int8(data.a, M * K);
  fill_int8(data.b, K * N);

  inner_loops_245(iters, &data);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", K = %" PRIu64 ", N = %" PRIu64 "\n", M, K, N);
  printf("\t%" PRIu64 " x %" PRIu64 " * %" PRIu64 " x %" PRIu64 "\n", M, K, K, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M, N, K) / 1024.0f);
#endif

  int checksum = 0;
#define CHECK(x, y)                                \
  {                                                \
    int32_t d = 0;                                 \
    for (int k = 0; k < K; k++) {                  \
      d += int8_to_int32(x, y, k, &data);          \
    }                                              \
    checksum += (int)(d != data.c[(x) * N + (y)]); \
  }

#ifdef FULL_CHECK
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) CHECK(m, n);
#else
  CHECK(0, 0);
  CHECK(M - 1, 0);
  CHECK(0, N - 1);
  CHECK(M - 1, N - 1);
  CHECK(M / 2, N / 2);
#endif

  bool passed = (checksum == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(245, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
