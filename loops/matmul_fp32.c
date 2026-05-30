/*----------------------------------------------------------------------------
#
#   This file is part of the SIMD Loops project. For more information, visit:
#     https://gitlab.arm.com/architecture/simd-loops
#
#   Copyright (c) 2025, Arm Limited. All rights reserved.
#
#   SPDX-License-Identifier: BSD-3-Clause
#
----------------------------------------------------------------------------*/

#include <stdint.h>
#include "loops.h"

/*
  FP32 Matrix Multiplication

  Data format -
    A: column-major
    B: row-major
    C: row-major
  Constraints -
    M: multiple of SVLh
    N: multiple of SVLh
    K: even
*/

void matmul_fp32(uint64_t m, uint64_t n, uint64_t k, float *restrict a,
                 float *restrict b, float *restrict c)
#if defined(__ARM_FEATURE_SME2)
SME_ZA_ATTR
#elif defined(__ARM_FEATURE_SVE2)
SC_SVE_ATTR
#endif
{


#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

  for (uint64_t x = 0; x < m; x++) {
    for (uint64_t y = 0; y < n; y++) {
      c[x * n + y] = 0.0f;
    }
  }

  // Loops ordered for contiguous memory access in inner loop
  for (uint64_t z = 0; z < k; z++)
    for (uint64_t x = 0; x < m; x++) {
      for (uint64_t y = 0; y < n; y++) {
        c[x * n + y] += a[z * m + x] * b[z * n + y];
      }
    }

#elif defined(HAVE_SME_INTRINSICS)

  float32_t *ptr_a, *ptr_b, *ptr_c;
  float32_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx, l_idx;
  uint64_t svl_s = svcntw();
  uint64_t l_cnd = svl_s * 4;
  uint64_t c_blk = svl_s * n;
  uint64_t c_off = c_blk + n;

  svcount_t c_all = svptrue_c32();
  svbool_t p_all = svptrue_b32();

  svuint8x4_t vec_c0, vec_c1;
  svfloat32x2_t vec_a0, vec_a1, vec_b0, vec_b1;

#define MOPA_TILE(t, x, i, j) \
  svmopa_za32_m(t, p_all, p_all, svget2(vec_a##x, i), svget2(vec_b##x, j))

#define EXTR(x, i) svreinterpret_f32(svget4(vec_c##x, i))
#define STORE_PAIR(x, i, j, o) \
  svst1(c_all, &ptr_c[o], svcreate2(EXTR(x, i), EXTR(x, j)))

#if defined(__ARM_FEATURE_SME2p1)
  svzero_za();
#endif

  for (m_idx = 0; m_idx < m; m_idx += svl_s * 2) {
    for (n_idx = 0; n_idx < n; n_idx += svl_s * 2) {
#if !defined(__ARM_FEATURE_SME2p1)
      svzero_za();
#endif

      ptr_a = &a[m_idx];
      ptr_b = &b[n_idx];
      while (ptr_a < cnd_k) {
        vec_a0 = svld1_x2(c_all, &ptr_a[0]);
        vec_b0 = svld1_x2(c_all, &ptr_b[0]);
        vec_a1 = svld1_x2(c_all, &ptr_a[m]);
        vec_b1 = svld1_x2(c_all, &ptr_b[n]);

        MOPA_TILE(0, 0, 0, 0);
        MOPA_TILE(1, 0, 0, 1);
        MOPA_TILE(2, 0, 1, 0);
        MOPA_TILE(3, 0, 1, 1);
        MOPA_TILE(0, 1, 0, 0);
        MOPA_TILE(1, 1, 0, 1);
        MOPA_TILE(2, 1, 1, 0);
        MOPA_TILE(3, 1, 1, 1);

        ptr_a += m * 2;
        ptr_b += n * 2;
      }

      ptr_c = &c[n_idx];
      for (l_idx = 0; l_idx < l_cnd; l_idx += 8) {
#if defined(__ARM_FEATURE_SME2p1)
        vec_c0 = svreadz_hor_za8_u8_vg4(0, l_idx + 0);
        vec_c1 = svreadz_hor_za8_u8_vg4(0, l_idx + 4);
#else
        vec_c0 = svread_hor_za8_u8_vg4(0, l_idx + 0);
        vec_c1 = svread_hor_za8_u8_vg4(0, l_idx + 4);
#endif

        STORE_PAIR(0, 0, 1, 0);
        STORE_PAIR(1, 0, 1, n);
        STORE_PAIR(0, 2, 3, c_blk);
        STORE_PAIR(1, 2, 3, c_off);

        ptr_c += n * 2;
      }
    }
    c += c_blk * 2;
  }

#elif defined(HAVE_SVE_INTRINSICS)

  float32_t *ptr_a, *ptr_b, *ptr_c;
  float32_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx;
  svbool_t p_all = svptrue_b32();

  svfloat32x2_t acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7;
  svfloat32x2_t lda_0, lda_1;
  svfloat32x2_t ldb_0, ldb_1;

#define ZERO svdup_f32(0.0f)
#define ZERO_PAIR(y) acc_##y = svcreate2(ZERO, ZERO)

#define LOADA(y, z) svld1rq(p_all, &ptr_a[z * m + y * 4])
#define LOADA_PAIR(z) svcreate2(LOADA(0, z), LOADA(1, z))

#define GETA(y, z) svget2(lda_##z, y / 4)
#define GETB(x, z) svget2(ldb_##z, x)
#define GETC(x, y) svget2(acc_##y, x)

#define MLA(x, y, z) svmla_lane(GETC(x, y), GETB(x, z), GETA(y, z), y % 4)
#define MLA_PAIR(y, z) acc_##y = svcreate2(MLA(0, y, z), MLA(1, y, z));
#define MLA_GROUP(y) MLA_PAIR(y, 0) MLA_PAIR(y, 1)

#if defined(__ARM_FEATURE_SVE2p1)
  svcount_t c_all = svptrue_c32();
#define LOADB_PAIR(y) svld1_x2(c_all, &ptr_b[n * y]);
#define STORE_PAIR(y) svst1(c_all, &ptr_c[n * y], acc_##y)
#else
#define LOADB(x, y) svld1_vnum(p_all, &ptr_b[n * y], x)
#define LOADB_PAIR(y) svcreate2(LOADB(0, y), LOADB(1, y))
#define STORE(x, y) svst1_vnum(p_all, &ptr_c[n * y], x, GETC(x, y));
#define STORE_PAIR(y) STORE(0, y) STORE(1, y)
#endif

  for (m_idx = 0; m_idx < m; m_idx += 8) {
    for (n_idx = 0; n_idx < n; n_idx += svcntw() * 2) {
      ZERO_PAIR(0);
      ZERO_PAIR(1);
      ZERO_PAIR(2);
      ZERO_PAIR(3);
      ZERO_PAIR(4);
      ZERO_PAIR(5);
      ZERO_PAIR(6);
      ZERO_PAIR(7);

      ptr_a = &a[m_idx];
      ptr_b = &b[n_idx];
      while (ptr_a < cnd_k) {
        lda_0 = LOADA_PAIR(0);
        lda_1 = LOADA_PAIR(1);
        ldb_0 = LOADB_PAIR(0);
        ldb_1 = LOADB_PAIR(1);

        MLA_GROUP(0);
        MLA_GROUP(1);
        MLA_GROUP(2);
        MLA_GROUP(3);
        MLA_GROUP(4);
        MLA_GROUP(5);
        MLA_GROUP(6);
        MLA_GROUP(7);

        ptr_a += m * 2;
        ptr_b += n * 2;
      }

      ptr_c = &c[n_idx];
      STORE_PAIR(0);
      STORE_PAIR(1);
      STORE_PAIR(2);
      STORE_PAIR(3);
      STORE_PAIR(4);
      STORE_PAIR(5);
      STORE_PAIR(6);
      STORE_PAIR(7);
    }
    c += n * 8;
  }

#elif defined(__ARM_FEATURE_SME2)

  register uint64_t svl_s;
  asm volatile("cntw %[v]" : [v] "=&r"(svl_s)::);

  register uint64_t a_cnd = (uint64_t)&a[m * k];
  register uint64_t l_cnd = svl_s * 4 - 8;
  register uint64_t c_blk = svl_s * n;
  register uint64_t c_off = c_blk + n;

  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  // x12: slice index register for tile-to-vec mova

  asm volatile(
      "   ptrue   p0.s                                                      \n"
      "   ptrue   pn8.s                                                     \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                      \n"
#endif

      // M loop head
      "   mov     %[m_idx], #0                                              \n"
      "1:                                                                   \n"

      // N loop head
      "   mov     %[n_idx], #0                                              \n"
      "2:                                                                   \n"

      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2                      \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #2                      \n"
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #2                      \n"

      // K loop
      "   ld1w    { z2.s-z3.s }, pn8/z, [%[a_ptr]]                          \n"
#if !defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                      \n"
#endif
      "   ld1w    { z0.s-z1.s }, pn8/z, [%[b_ptr]]                          \n"
      "   fmopa   za0.s, p0/m, p0/m, z2.s, z0.s                             \n"
      "   fmopa   za1.s, p0/m, p0/m, z2.s, z1.s                             \n"
      "   ld1w    { z6.s-z7.s }, pn8/z, [%[a_ptr], %[m], lsl #2]            \n"
      "   ld1w    { z4.s-z5.s }, pn8/z, [%[b_ptr], %[n], lsl #2]            \n"
      "   fmopa   za2.s, p0/m, p0/m, z3.s, z0.s                             \n"
      "   fmopa   za3.s, p0/m, p0/m, z3.s, z1.s                             \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                          \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                          \n"
      "3:                                                                   \n"
      "   fmopa   za0.s, p0/m, p0/m, z6.s, z4.s                             \n"
      "   fmopa   za1.s, p0/m, p0/m, z6.s, z5.s                             \n"
      "   ld1w    { z2.s-z3.s }, pn8/z, [%[a_ptr]]                          \n"
      "   ld1w    { z0.s-z1.s }, pn8/z, [%[b_ptr]]                          \n"
      "   fmopa   za2.s, p0/m, p0/m, z7.s, z4.s                             \n"
      "   fmopa   za3.s, p0/m, p0/m, z7.s, z5.s                             \n"
      "   ld1w    { z6.s-z7.s }, pn8/z, [%[a_ptr], %[m], lsl #2]            \n"
      "   ld1w    { z4.s-z5.s }, pn8/z, [%[b_ptr], %[n], lsl #2]            \n"
      "   fmopa   za0.s, p0/m, p0/m, z2.s, z0.s                             \n"
      "   fmopa   za1.s, p0/m, p0/m, z2.s, z1.s                             \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                          \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                          \n"
      "   fmopa   za2.s, p0/m, p0/m, z3.s, z0.s                             \n"
      "   fmopa   za3.s, p0/m, p0/m, z3.s, z1.s                             \n"
      "   cmp     %[a_ptr], %[a_cnd]                                        \n"
      "   b.mi    3b                                                        \n"
      "   fmopa   za0.s, p0/m, p0/m, z6.s, z4.s                             \n"
      "   fmopa   za1.s, p0/m, p0/m, z6.s, z5.s                             \n"
      "   fmopa   za2.s, p0/m, p0/m, z7.s, z4.s                             \n"
      "   fmopa   za3.s, p0/m, p0/m, z7.s, z5.s                             \n"

      // Store loop
      "   mov     x12, #0                                                   \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   { z0.b-z3.b }, za0h.b[w12, 0:3]                           \n"
#else
      "   mova    { z0.b-z3.b }, za0h.b[w12, 0:3]                           \n"
#endif
      "   st1w    { z0.s-z1.s }, pn8, [%[c_ptr]]                            \n"
      "   st1w    { z2.s-z3.s }, pn8, [%[c_ptr], %[c_blk], lsl #2]          \n"
      "4:                                                                   \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   { z0.b-z3.b }, za0h.b[w12, 4:7]                           \n"
#else
      "   mova    { z0.b-z3.b }, za0h.b[w12, 4:7]                           \n"
#endif
      "   st1w    { z0.s-z1.s }, pn8, [%[c_ptr], %[n], lsl #2]              \n"
      "   st1w    { z2.s-z3.s }, pn8, [%[c_ptr], %[c_off], lsl #2]          \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #3                          \n"
      "   add     x12, x12, #8                                              \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   { z0.b-z3.b }, za0h.b[w12, 0:3]                           \n"
#else
      "   mova    { z0.b-z3.b }, za0h.b[w12, 0:3]                           \n"
#endif
      "   st1w    { z0.s-z1.s }, pn8, [%[c_ptr]]                            \n"
      "   st1w    { z2.s-z3.s }, pn8, [%[c_ptr], %[c_blk], lsl #2]          \n"
      "   cmp     x12, %[l_cnd]                                             \n"
      "   b.mi    4b                                                        \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   { z0.b-z3.b }, za0h.b[w12, 4:7]                           \n"
#else
      "   mova    { z0.b-z3.b }, za0h.b[w12, 4:7]                           \n"
#endif
      "   st1w    { z0.s-z1.s }, pn8, [%[c_ptr], %[n], lsl #2]              \n"
      "   st1w    { z2.s-z3.s }, pn8, [%[c_ptr], %[c_off], lsl #2]          \n"

      // N loop tail
      "   incw    %[n_idx], all, mul #2                                     \n"
      "   cmp     %[n_idx], %[n]                                            \n"
      "   b.mi    2b                                                        \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[c_blk], lsl #3                      \n"
      "   incw    %[m_idx], all, mul #2                                     \n"
      "   cmp     %[m_idx], %[m]                                            \n"
      "   b.mi    1b                                                        \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [c_ptr] "=&r"(c_ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [c_blk] "r"(c_blk),
        [c_off] "r"(c_off), [l_cnd] "r"(l_cnd), [a_cnd] "r"(a_cnd),
        [a_src] "r"(a), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "x12", "p0", "p8",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");

#elif defined(__ARM_FEATURE_SVE2p1)

  register uint64_t a_cnd = (uint64_t)&a[m * k];
  register uint64_t off_2 = n * 2;
  register uint64_t off_3 = n * 3;
  register uint64_t off_r = m + 4;

  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  register uint64_t n_idx;
  register uint64_t m_idx;

  asm volatile(
      "   ptrue   p0.s                                              \n"
      "   ptrue   pn8.s                                             \n"

      // M loop head
      "   mov     %[m_idx], #0                                      \n"
      "1:                                                           \n"

      // N loop head
      "   mov     %[n_idx], #0                                      \n"
      "2:                                                           \n"

      // Accumulators
      "   mov     z10.s, #0                                         \n"
      "   mov     z11.s, #0                                         \n"
      "   mov     z12.s, #0                                         \n"
      "   mov     z13.s, #0                                         \n"
      "   mov     z14.s, #0                                         \n"
      "   mov     z15.s, #0                                         \n"
      "   mov     z16.s, #0                                         \n"
      "   mov     z17.s, #0                                         \n"
      "   mov     z20.s, #0                                         \n"
      "   mov     z21.s, #0                                         \n"
      "   mov     z22.s, #0                                         \n"
      "   mov     z23.s, #0                                         \n"
      "   mov     z24.s, #0                                         \n"
      "   mov     z25.s, #0                                         \n"
      "   mov     z26.s, #0                                         \n"
      "   mov     z27.s, #0                                         \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2              \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #2              \n"
      "3:                                                           \n"
      "   ld1rqw  {z0.s}, p0/z, [%[a_ptr], #0 ]                     \n"
      "   ld1rqw  {z1.s}, p0/z, [%[a_ptr], #16]                     \n"
      "   ld1rqw  {z2.s}, p0/z, [%[a_ptr], %[off_l], lsl #2]        \n"
      "   ld1rqw  {z3.s}, p0/z, [%[a_ptr], %[off_r], lsl #2]        \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   ld1w    {z4.s-z5.s}, pn8/z, [%[b_ptr]]                    \n"
      "   ld1w    {z6.s-z7.s}, pn8/z, [%[b_ptr], %[n], lsl #2]      \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                  \n"
      "   fmla    z10.s, z4.s, z0.s[0]                              \n"
      "   fmla    z12.s, z4.s, z0.s[1]                              \n"
      "   fmla    z14.s, z4.s, z0.s[2]                              \n"
      "   fmla    z16.s, z4.s, z0.s[3]                              \n"
      "   fmla    z11.s, z5.s, z0.s[0]                              \n"
      "   fmla    z13.s, z5.s, z0.s[1]                              \n"
      "   fmla    z15.s, z5.s, z0.s[2]                              \n"
      "   fmla    z17.s, z5.s, z0.s[3]                              \n"
      "   fmla    z20.s, z4.s, z1.s[0]                              \n"
      "   fmla    z22.s, z4.s, z1.s[1]                              \n"
      "   fmla    z24.s, z4.s, z1.s[2]                              \n"
      "   fmla    z26.s, z4.s, z1.s[3]                              \n"
      "   fmla    z21.s, z5.s, z1.s[0]                              \n"
      "   fmla    z23.s, z5.s, z1.s[1]                              \n"
      "   fmla    z25.s, z5.s, z1.s[2]                              \n"
      "   fmla    z27.s, z5.s, z1.s[3]                              \n"
      "   fmla    z10.s, z6.s, z2.s[0]                              \n"
      "   fmla    z12.s, z6.s, z2.s[1]                              \n"
      "   fmla    z14.s, z6.s, z2.s[2]                              \n"
      "   fmla    z16.s, z6.s, z2.s[3]                              \n"
      "   fmla    z11.s, z7.s, z2.s[0]                              \n"
      "   fmla    z13.s, z7.s, z2.s[1]                              \n"
      "   fmla    z15.s, z7.s, z2.s[2]                              \n"
      "   fmla    z17.s, z7.s, z2.s[3]                              \n"
      "   fmla    z20.s, z6.s, z3.s[0]                              \n"
      "   fmla    z22.s, z6.s, z3.s[1]                              \n"
      "   fmla    z24.s, z6.s, z3.s[2]                              \n"
      "   fmla    z26.s, z6.s, z3.s[3]                              \n"
      "   fmla    z21.s, z7.s, z3.s[0]                              \n"
      "   fmla    z23.s, z7.s, z3.s[1]                              \n"
      "   fmla    z25.s, z7.s, z3.s[2]                              \n"
      "   fmla    z27.s, z7.s, z3.s[3]                              \n"
      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #2              \n"
      "   st1w    {z10.s-z11.s}, pn8, [%[c_ptr]]                    \n"
      "   st1w    {z12.s-z13.s}, pn8, [%[c_ptr], %[off_1], lsl #2]  \n"
      "   st1w    {z14.s-z15.s}, pn8, [%[c_ptr], %[off_2], lsl #2]  \n"
      "   st1w    {z16.s-z17.s}, pn8, [%[c_ptr], %[off_3], lsl #2]  \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #4                  \n"
      "   st1w    {z20.s-z21.s}, pn8, [%[c_ptr]]                    \n"
      "   st1w    {z22.s-z23.s}, pn8, [%[c_ptr], %[off_1], lsl #2]  \n"
      "   st1w    {z24.s-z25.s}, pn8, [%[c_ptr], %[off_2], lsl #2]  \n"
      "   st1w    {z26.s-z27.s}, pn8, [%[c_ptr], %[off_3], lsl #2]  \n"

      // N loop tail
      "   incw    %[n_idx], all, mul #2                             \n"
      "   cmp     %[n_idx], %[n]                                    \n"
      "   b.mi    2b                                                \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #5                  \n"
      "   add     %[m_idx], %[m_idx], #8                            \n"
      "   cmp     %[m_idx], %[m]                                    \n"
      "   b.mi    1b                                                \n"

      : [m_idx] "=&r"(m_idx), [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr),
        [n_idx] "=&r"(n_idx), [c_ptr] "=&r"(c_ptr), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [off_r] "r"(off_r), [off_l] "r"(m), [off_2] "r"(off_2),
        [off_3] "r"(off_3), [off_1] "r"(n), [a_src] "r"(a), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "z14", "z15", "z16", "z17", "z20", "z21", "z22", "z23", "z24",
        "z25", "z26", "z27", "p0", "p8", "cc", "memory");

#elif defined(__ARM_FEATURE_SVE2)

  register uint64_t a_cnd = (uint64_t)&a[m * k];
  register uint64_t off_2 = n * 2;
  register uint64_t off_3 = n * 3;
  register uint64_t off_r = m + 4;

  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c0ptr;
  register uint64_t c1ptr;
  register uint64_t n_idx;
  register uint64_t m_idx;

  asm volatile(
      "   ptrue   p0.s                                              \n"

      // M loop head
      "   mov     %[m_idx], #0                                      \n"
      "1:                                                           \n"

      // N loop head
      "   mov     %[n_idx], #0                                      \n"
      "2:                                                           \n"

      // Accumulators
      "   mov     z10.s, #0                                         \n"
      "   mov     z11.s, #0                                         \n"
      "   mov     z12.s, #0                                         \n"
      "   mov     z13.s, #0                                         \n"
      "   mov     z14.s, #0                                         \n"
      "   mov     z15.s, #0                                         \n"
      "   mov     z16.s, #0                                         \n"
      "   mov     z17.s, #0                                         \n"
      "   mov     z20.s, #0                                         \n"
      "   mov     z21.s, #0                                         \n"
      "   mov     z22.s, #0                                         \n"
      "   mov     z23.s, #0                                         \n"
      "   mov     z24.s, #0                                         \n"
      "   mov     z25.s, #0                                         \n"
      "   mov     z26.s, #0                                         \n"
      "   mov     z27.s, #0                                         \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2              \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #2              \n"
      "3:                                                           \n"
      "   ld1rqw  {z0.s}, p0/z, [%[a_ptr], #0 ]                     \n"
      "   ld1rqw  {z1.s}, p0/z, [%[a_ptr], #16]                     \n"
      "   ld1rqw  {z2.s}, p0/z, [%[a_ptr], %[off_l], lsl #2]        \n"
      "   ld1rqw  {z3.s}, p0/z, [%[a_ptr], %[off_r], lsl #2]        \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   ld1w    {z4.s}, p0/z, [%[b_ptr]]                          \n"
      "   ld1w    {z5.s}, p0/z, [%[b_ptr], #1, mul vl]              \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                  \n"
      "   ld1w    {z6.s}, p0/z, [%[b_ptr]]                          \n"
      "   ld1w    {z7.s}, p0/z, [%[b_ptr], #1, mul vl]              \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                  \n"
      "   fmla    z10.s, z4.s, z0.s[0]                              \n"
      "   fmla    z12.s, z4.s, z0.s[1]                              \n"
      "   fmla    z14.s, z4.s, z0.s[2]                              \n"
      "   fmla    z16.s, z4.s, z0.s[3]                              \n"
      "   fmla    z11.s, z5.s, z0.s[0]                              \n"
      "   fmla    z13.s, z5.s, z0.s[1]                              \n"
      "   fmla    z15.s, z5.s, z0.s[2]                              \n"
      "   fmla    z17.s, z5.s, z0.s[3]                              \n"
      "   fmla    z20.s, z4.s, z1.s[0]                              \n"
      "   fmla    z22.s, z4.s, z1.s[1]                              \n"
      "   fmla    z24.s, z4.s, z1.s[2]                              \n"
      "   fmla    z26.s, z4.s, z1.s[3]                              \n"
      "   fmla    z21.s, z5.s, z1.s[0]                              \n"
      "   fmla    z23.s, z5.s, z1.s[1]                              \n"
      "   fmla    z25.s, z5.s, z1.s[2]                              \n"
      "   fmla    z27.s, z5.s, z1.s[3]                              \n"
      "   fmla    z10.s, z6.s, z2.s[0]                              \n"
      "   fmla    z12.s, z6.s, z2.s[1]                              \n"
      "   fmla    z14.s, z6.s, z2.s[2]                              \n"
      "   fmla    z16.s, z6.s, z2.s[3]                              \n"
      "   fmla    z11.s, z7.s, z2.s[0]                              \n"
      "   fmla    z13.s, z7.s, z2.s[1]                              \n"
      "   fmla    z15.s, z7.s, z2.s[2]                              \n"
      "   fmla    z17.s, z7.s, z2.s[3]                              \n"
      "   fmla    z20.s, z6.s, z3.s[0]                              \n"
      "   fmla    z22.s, z6.s, z3.s[1]                              \n"
      "   fmla    z24.s, z6.s, z3.s[2]                              \n"
      "   fmla    z26.s, z6.s, z3.s[3]                              \n"
      "   fmla    z21.s, z7.s, z3.s[0]                              \n"
      "   fmla    z23.s, z7.s, z3.s[1]                              \n"
      "   fmla    z25.s, z7.s, z3.s[2]                              \n"
      "   fmla    z27.s, z7.s, z3.s[3]                              \n"
      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

      // Store
      "   add     %[c0ptr], %[c_dst], %[n_idx], lsl #2              \n"
      "   addvl   %[c1ptr], %[c0ptr], #1                            \n"
      "   st1w    {z10.s}, p0, [%[c0ptr]]                           \n"
      "   st1w    {z11.s}, p0, [%[c1ptr]]                           \n"
      "   st1w    {z12.s}, p0, [%[c0ptr], %[off_1], lsl #2]         \n"
      "   st1w    {z13.s}, p0, [%[c1ptr], %[off_1], lsl #2]         \n"
      "   st1w    {z14.s}, p0, [%[c0ptr], %[off_2], lsl #2]         \n"
      "   st1w    {z15.s}, p0, [%[c1ptr], %[off_2], lsl #2]         \n"
      "   st1w    {z16.s}, p0, [%[c0ptr], %[off_3], lsl #2]         \n"
      "   st1w    {z17.s}, p0, [%[c1ptr], %[off_3], lsl #2]         \n"
      "   add     %[c0ptr], %[c0ptr], %[n], lsl #4                  \n"
      "   add     %[c1ptr], %[c1ptr], %[n], lsl #4                  \n"
      "   st1w    {z20.s}, p0, [%[c0ptr]]                           \n"
      "   st1w    {z21.s}, p0, [%[c1ptr]]                           \n"
      "   st1w    {z22.s}, p0, [%[c0ptr], %[off_1], lsl #2]         \n"
      "   st1w    {z23.s}, p0, [%[c1ptr], %[off_1], lsl #2]         \n"
      "   st1w    {z24.s}, p0, [%[c0ptr], %[off_2], lsl #2]         \n"
      "   st1w    {z25.s}, p0, [%[c1ptr], %[off_2], lsl #2]         \n"
      "   st1w    {z26.s}, p0, [%[c0ptr], %[off_3], lsl #2]         \n"
      "   st1w    {z27.s}, p0, [%[c1ptr], %[off_3], lsl #2]         \n"

      // N loop tail
      "   incw    %[n_idx], all, mul #2                             \n"
      "   cmp     %[n_idx], %[n]                                    \n"
      "   b.mi    2b                                                \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #5                  \n"
      "   add     %[m_idx], %[m_idx], #8                            \n"
      "   cmp     %[m_idx], %[m]                                    \n"
      "   b.mi    1b                                                \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [m_idx] "=&r"(m_idx),
        [c0ptr] "=&r"(c0ptr), [c1ptr] "=&r"(c1ptr), [n_idx] "=&r"(n_idx),
        [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [off_r] "r"(off_r), [off_l] "r"(m), [off_2] "r"(off_2),
        [off_3] "r"(off_3), [off_1] "r"(n), [a_src] "r"(a), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "z14", "z15", "z16", "z17", "z20", "z21", "z22", "z23", "z24",
        "z25", "z26", "z27", "p0", "cc", "memory");

#elif defined(__ARM_NEON)

  register uint64_t a_cnd = (uint64_t)&a[m * k];
  register uint64_t m_inc = m * 4;
  register uint64_t n_inc = n * 4;

  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  register uint64_t m_idx;
  register uint64_t n_idx;

  asm volatile(
      // M loop head
      "   mov     %[m_idx], #0                                \n"
      "1:                                                     \n"

      // N loop head
      "   mov     %[n_idx], #0                                \n"
      "2:                                                     \n"

      // Accumulators
      "   movi    v10.4s, #0                                  \n"
      "   movi    v11.4s, #0                                  \n"
      "   movi    v12.4s, #0                                  \n"
      "   movi    v13.4s, #0                                  \n"
      "   movi    v14.4s, #0                                  \n"
      "   movi    v15.4s, #0                                  \n"
      "   movi    v16.4s, #0                                  \n"
      "   movi    v17.4s, #0                                  \n"
      "   movi    v20.4s, #0                                  \n"
      "   movi    v21.4s, #0                                  \n"
      "   movi    v22.4s, #0                                  \n"
      "   movi    v23.4s, #0                                  \n"
      "   movi    v24.4s, #0                                  \n"
      "   movi    v25.4s, #0                                  \n"
      "   movi    v26.4s, #0                                  \n"
      "   movi    v27.4s, #0                                  \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2        \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #2        \n"
      "3:                                                     \n"
      "   ld1     {v0.4s,v1.4s}, [%[a_ptr]], %[m_inc]         \n"
      "   ld1     {v2.4s,v3.4s}, [%[a_ptr]], %[m_inc]         \n"
      "   ld1     {v4.4s,v5.4s}, [%[b_ptr]], %[n_inc]         \n"
      "   ld1     {v6.4s,v7.4s}, [%[b_ptr]], %[n_inc]         \n"
      "   fmla    v10.4s, v4.4s, v0.s[0]                      \n"
      "   fmla    v12.4s, v4.4s, v0.s[1]                      \n"
      "   fmla    v14.4s, v4.4s, v0.s[2]                      \n"
      "   fmla    v16.4s, v4.4s, v0.s[3]                      \n"
      "   fmla    v20.4s, v4.4s, v1.s[0]                      \n"
      "   fmla    v22.4s, v4.4s, v1.s[1]                      \n"
      "   fmla    v24.4s, v4.4s, v1.s[2]                      \n"
      "   fmla    v26.4s, v4.4s, v1.s[3]                      \n"
      "   fmla    v11.4s, v5.4s, v0.s[0]                      \n"
      "   fmla    v13.4s, v5.4s, v0.s[1]                      \n"
      "   fmla    v15.4s, v5.4s, v0.s[2]                      \n"
      "   fmla    v17.4s, v5.4s, v0.s[3]                      \n"
      "   fmla    v21.4s, v5.4s, v1.s[0]                      \n"
      "   fmla    v23.4s, v5.4s, v1.s[1]                      \n"
      "   fmla    v25.4s, v5.4s, v1.s[2]                      \n"
      "   fmla    v27.4s, v5.4s, v1.s[3]                      \n"
      "   fmla    v10.4s, v6.4s, v2.s[0]                      \n"
      "   fmla    v12.4s, v6.4s, v2.s[1]                      \n"
      "   fmla    v14.4s, v6.4s, v2.s[2]                      \n"
      "   fmla    v16.4s, v6.4s, v2.s[3]                      \n"
      "   fmla    v20.4s, v6.4s, v3.s[0]                      \n"
      "   fmla    v22.4s, v6.4s, v3.s[1]                      \n"
      "   fmla    v24.4s, v6.4s, v3.s[2]                      \n"
      "   fmla    v26.4s, v6.4s, v3.s[3]                      \n"
      "   fmla    v11.4s, v7.4s, v2.s[0]                      \n"
      "   fmla    v13.4s, v7.4s, v2.s[1]                      \n"
      "   fmla    v15.4s, v7.4s, v2.s[2]                      \n"
      "   fmla    v17.4s, v7.4s, v2.s[3]                      \n"
      "   fmla    v21.4s, v7.4s, v3.s[0]                      \n"
      "   fmla    v23.4s, v7.4s, v3.s[1]                      \n"
      "   fmla    v25.4s, v7.4s, v3.s[2]                      \n"
      "   fmla    v27.4s, v7.4s, v3.s[3]                      \n"
      "   cmp     %[a_ptr], %[a_cnd]                          \n"
      "   b.mi    3b                                          \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #2        \n"
      "   st1     {v10.4s,v11.4s}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v12.4s,v13.4s}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v14.4s,v15.4s}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v16.4s,v17.4s}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v20.4s,v21.4s}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v22.4s,v23.4s}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v24.4s,v25.4s}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v26.4s,v27.4s}, [%[c_ptr]], %[n_inc]       \n"

      // N loop tail
      "   add     %[n_idx], %[n_idx], #8                      \n"
      "   cmp     %[n_idx], %[n]                              \n"
      "   b.mi    2b                                          \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #5            \n"
      "   add     %[m_idx], %[m_idx], #8                      \n"
      "   cmp     %[m_idx], %[m]                              \n"
      "   b.mi    1b                                          \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [c_ptr] "=&r"(c_ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [m_inc] "r"(m_inc), [n_inc] "r"(n_inc), [a_src] "r"(a), [b_src] "r"(b)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v10", "v11", "v12",
        "v13", "v14", "v15", "v16", "v17", "v20", "v21", "v22", "v23", "v24",
        "v25", "v26", "v27", "cc", "memory");
#endif

}  // matmul_fp32
