/*----------------------------------------------------------------------------
#
#   Loop 206: INT16-INT64 matrix-matrix multiply using MOPA / DOT
#
#   Purpose:
#     Use of i16 to i64 MOPA (or DOT) instructions.
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
    M: multiple of SVLh
    N: multiple of SVLs
    K: multiple of 8

  Note: A and B matrices are considered to be re-arranged,
        as required by the INT16 -> INT64 matrix multiplication.
*/

struct loop_206_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  uint16_t *restrict a;
  uint16_t *restrict b;
  uint64_t *restrict c;
};

#if (defined(__ARM_FEATURE_SME2) && defined(__ARM_FEATURE_SME_I16I64))
#define LOOP_206_SME
#endif

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_206(struct loop_206_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#define LOOP_ATTR SC_SVE_ATTR
#define OUTER_LOOP_ATTR SC_SVE_LOOP_ATTR
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#elif defined(LOOP_206_SME)
#define LOOP_ATTR SME_ZA_ATTR
#define OUTER_LOOP_ATTR S_LOOP_ATTR
#elif defined(__ARM_FEATURE_SVE2)
#define LOOP_ATTR SC_SVE_ATTR
#define OUTER_LOOP_ATTR SC_SVE_LOOP_ATTR
#else
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#endif

#if !defined(HAVE_CANDIDATE)

#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_206(struct loop_206_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  uint16_t *restrict a = data->a;
  uint16_t *restrict b = data->b;
  uint64_t *restrict c = data->c;
  for (uint64_t x = 0; x < m; x++) {
    for (uint64_t y = 0; y < n; y++) {
      uint64_t d = 0;
      for (uint64_t z = 0; z < k; z += 4) {
        uint64_t i = z * m + 4 * x, j = z * n + 4 * y;
        d += (uint64_t)a[i + 0] * (uint64_t)b[j + 0];
        d += (uint64_t)a[i + 1] * (uint64_t)b[j + 1];
        d += (uint64_t)a[i + 2] * (uint64_t)b[j + 2];
        d += (uint64_t)a[i + 3] * (uint64_t)b[j + 3];
      }
      c[x * n + y] = d;
    }
  }
}

#elif (defined(LOOP_206_SME) && defined(HAVE_SME_INTRINSICS))

static void inner_loop_206(struct loop_206_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  uint16_t *a = data->a;
  uint16_t *b = data->b;
  uint64_t *c = data->c;

  uint16_t *ptr_a, *ptr_b;
  uint64_t *ptr_c;
  uint16_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx, l_idx;
  uint64_t svl_d = svcntd();
  uint64_t l_cnd = svcntb();
  uint64_t c_bl1 = svl_d * n;
  uint64_t c_bl2 = 2 * svl_d * n;
  uint64_t c_bl3 = 3 * svl_d * n;

  svcount_t c_all = svptrue_c16();
  svbool_t p_all = svptrue_b16();

  svuint8x4_t vec_c0, vec_c1;
  svuint16x4_t vec_a0, vec_a1;
  svuint16x2_t vec_b0, vec_b1;

#define MOPA_TILE(t, x, i, j) \
  svmopa_za64_u16_m(t, p_all, p_all, svget4(vec_a##x, i), svget2(vec_b##x, j))

#define EXTR(x, i) svreinterpret_u64(svget4(vec_c##x, i))
#define STORE_PAIR(x, i, j, o) \
  svst1(c_all, &ptr_c[o], svcreate2(EXTR(x, i), EXTR(x, j)))

#if defined(__ARM_FEATURE_SME2p1)
  svzero_za();
#endif

  for (m_idx = 0; m_idx < m; m_idx += svcntd() * 4) {
    for (n_idx = 0; n_idx < n; n_idx += svcntd() * 2) {
#if !defined(__ARM_FEATURE_SME2p1)
      svzero_za();
#endif

      ptr_a = &a[m_idx << 2];
      ptr_b = &b[n_idx << 2];
      while (ptr_a < cnd_k) {
        vec_a0 = svld1_x4(c_all, &ptr_a[0]);
        vec_b0 = svld1_x2(c_all, &ptr_b[0]);
        vec_a1 = svld1_x4(c_all, &ptr_a[4 * m]);
        vec_b1 = svld1_x2(c_all, &ptr_b[4 * n]);

        MOPA_TILE(0, 0, 0, 0);
        MOPA_TILE(1, 0, 0, 1);
        MOPA_TILE(2, 0, 1, 0);
        MOPA_TILE(3, 0, 1, 1);
        MOPA_TILE(4, 0, 2, 0);
        MOPA_TILE(5, 0, 2, 1);
        MOPA_TILE(6, 0, 3, 0);
        MOPA_TILE(7, 0, 3, 1);
        MOPA_TILE(0, 1, 0, 0);
        MOPA_TILE(1, 1, 0, 1);
        MOPA_TILE(2, 1, 1, 0);
        MOPA_TILE(3, 1, 1, 1);
        MOPA_TILE(4, 1, 2, 0);
        MOPA_TILE(5, 1, 2, 1);
        MOPA_TILE(6, 1, 3, 0);
        MOPA_TILE(7, 1, 3, 1);

        ptr_a += m * 8;
        ptr_b += n * 8;
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
        STORE_PAIR(0, 2, 3, c_bl1);
        STORE_PAIR(1, 0, 1, c_bl2);
        STORE_PAIR(1, 2, 3, c_bl3);

        ptr_c += n;
      }
    }
    c += c_bl1 * 4;
  }
}

#elif defined(HAVE_SVE_INTRINSICS)

static void inner_loop_206(struct loop_206_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  uint16_t *a = data->a;
  uint16_t *b = data->b;
  uint64_t *c = data->c;

  uint16_t *ptr_a, *ptr_b;
  uint64_t *ptr_c;
  uint16_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx;
  svbool_t p_all = svptrue_b16();

  svuint64x2_t acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7;
  svuint16x4_t lda_0, lda_1;
  svuint16x2_t ldb_0, ldb_1;

#define ZERO svdup_u64(0)
#define ZERO_PAIR(y) acc_##y = svcreate2(ZERO, ZERO)

#define GETA(y, z) svget4(lda_##z, y / 2)
#define GETB(x, z) svget2(ldb_##z, x)
#define GETC(x, y) svget2(acc_##y, x)

#define LDA(y, z) svld1rq(p_all, &ptr_a[z * m * 4 + y * 8])
#define LOADA_QUAD(z) svcreate4(LDA(0, z), LDA(1, z), LDA(2, z), LDA(3, z))

#define DOT(x, y, z) svdot_lane(GETC(x, y), GETB(x, z), GETA(y, z), y % 2)
#define DOT_PAIR(y, z) acc_##y = svcreate2(DOT(0, y, z), DOT(1, y, z));
#define DOT_GROUP(y) DOT_PAIR(y, 0) DOT_PAIR(y, 1)

#if defined(__ARM_FEATURE_SVE2p1)
  svcount_t c_all = svptrue_c16();
#define LOADB_PAIR(y) svld1_x2(c_all, &ptr_b[n * y * 4]);
#define STORE_PAIR(y) svst1(c_all, &ptr_c[n * y], acc_##y)
#else
#define LOADB(y, p) svld1_vnum(p_all, &ptr_b[n * y * 4], p)
#define LOADB_PAIR(y) svcreate2(LOADB(y, 0), LOADB(y, 1))
#define STORE(y, p) svst1_vnum(p_all, &ptr_c[n * y], p, GETC(p, y));
#define STORE_PAIR(y) STORE(y, 0) STORE(y, 1)
#endif

  for (m_idx = 0; m_idx < m; m_idx += 8) {
    for (n_idx = 0; n_idx < n; n_idx += svcntd() * 2) {
      ZERO_PAIR(0);
      ZERO_PAIR(1);
      ZERO_PAIR(2);
      ZERO_PAIR(3);
      ZERO_PAIR(4);
      ZERO_PAIR(5);
      ZERO_PAIR(6);
      ZERO_PAIR(7);

      ptr_a = &a[m_idx * 4];
      ptr_b = &b[n_idx * 4];
      while (ptr_a < cnd_k) {
        lda_0 = LOADA_QUAD(0);
        lda_1 = LOADA_QUAD(1);
        ldb_0 = LOADB_PAIR(0);
        ldb_1 = LOADB_PAIR(1);

        DOT_GROUP(0);
        DOT_GROUP(1);
        DOT_GROUP(2);
        DOT_GROUP(3);
        DOT_GROUP(4);
        DOT_GROUP(5);
        DOT_GROUP(6);
        DOT_GROUP(7);

        ptr_a += m * 8;
        ptr_b += n * 8;
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
}

#elif defined(LOOP_206_SME)

static void inner_loop_206(struct loop_206_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t svl_d;
  asm volatile("cntd %[v]" : [v] "=&r"(svl_d)::);

  register uint64_t c_bl1 = svl_d * n;
  register uint64_t c_bl2 = c_bl1 * 2;
  register uint64_t c_bl3 = c_bl1 * 3;
  register uint64_t l_cnd = svl_d * 8 - 8;
  register uint64_t a_cnd = a + 2 * (m * k);

  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  // x12: slice index register for tile-to-vec mova

  asm volatile(
      "   ptrue   p0.b                                                      \n"
      "   ptrue   pn8.b                                                     \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                      \n"
#endif

      // M loop head
      "   mov     %[m_idx], #0                                              \n"
      "1:                                                                   \n"

      // N loop head
      "   mov     %[n_idx], #0                                              \n"
      "2:                                                                   \n"

      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #3                      \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #3                      \n"
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #3                      \n"

      // K loop
      "   ld1h    {z0.h-z3.h}, pn8/z, [%[a_ptr]]                            \n"
#if !defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                      \n"
#endif
      "   ld1h    {z4.h-z5.h}, pn8/z, [%[b_ptr]]                            \n"
      "   umopa   za0.d, p0/m, p0/m, z0.h, z4.h                             \n"
      "   umopa   za1.d, p0/m, p0/m, z0.h, z5.h                             \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                          \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                          \n"
      "   umopa   za2.d, p0/m, p0/m, z1.h, z4.h                             \n"
      "   umopa   za3.d, p0/m, p0/m, z1.h, z5.h                             \n"
      "3:                                                                   \n"
      "   umopa   za4.d, p0/m, p0/m, z2.h, z4.h                             \n"
      "   umopa   za5.d, p0/m, p0/m, z2.h, z5.h                             \n"
      "   umopa   za6.d, p0/m, p0/m, z3.h, z4.h                             \n"
      "   umopa   za7.d, p0/m, p0/m, z3.h, z5.h                             \n"
      "   ld1h    {z0.h-z3.h}, pn8/z, [%[a_ptr]]                            \n"
      "   ld1h    {z4.h-z5.h}, pn8/z, [%[b_ptr]]                            \n"
      "   umopa   za0.d, p0/m, p0/m, z0.h, z4.h                             \n"
      "   umopa   za1.d, p0/m, p0/m, z0.h, z5.h                             \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                          \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                          \n"
      "   umopa   za2.d, p0/m, p0/m, z1.h, z4.h                             \n"
      "   umopa   za3.d, p0/m, p0/m, z1.h, z5.h                             \n"
      "   cmp     %[a_ptr], %[a_cnd]                                        \n"
      "   b.mi    3b                                                        \n"
      "   umopa   za4.d, p0/m, p0/m, z2.h, z4.h                             \n"
      "   umopa   za5.d, p0/m, p0/m, z2.h, z5.h                             \n"
      "   umopa   za6.d, p0/m, p0/m, z3.h, z4.h                             \n"
      "   umopa   za7.d, p0/m, p0/m, z3.h, z5.h                             \n"

      // Store loop
      "   mov     x12, #0                                                   \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z0.b-z3.b}, za0h.b[w12, 0:3]                             \n"
#else
      "   mova    {z0.b-z3.b}, za0h.b[w12, 0:3]                             \n"
#endif
      "   st1d    {z0.d-z1.d}, pn8, [%[c_ptr]]                              \n"
      "   st1d    {z2.d-z3.d}, pn8, [%[c_ptr], %[c_bl1], lsl #3]            \n"
      "4:                                                                   \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z4.b-z7.b}, za0h.b[w12, 4:7]                             \n"
#else
      "   mova    {z4.b-z7.b}, za0h.b[w12, 4:7]                             \n"
#endif
      "   st1d    {z4.d-z5.d}, pn8, [%[c_ptr], %[c_bl2], lsl #3]            \n"
      "   st1d    {z6.d-z7.d}, pn8, [%[c_ptr], %[c_bl3], lsl #3]            \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #3                          \n"
      "   add     x12, x12, #8                                              \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z0.b-z3.b}, za0h.b[w12, 0:3]                             \n"
#else
      "   mova    {z0.b-z3.b}, za0h.b[w12, 0:3]                             \n"
#endif
      "   st1d    {z0.d-z1.d}, pn8, [%[c_ptr]]                              \n"
      "   st1d    {z2.d-z3.d}, pn8, [%[c_ptr], %[c_bl1], lsl #3]            \n"
      "   cmp     x12, %[l_cnd]                                             \n"
      "   b.mi    4b                                                        \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   {z4.b-z7.b}, za0h.b[w12, 4:7]                             \n"
#else
      "   mova    {z4.b-z7.b}, za0h.b[w12, 4:7]                             \n"
#endif
      "   st1d    {z4.d-z5.d}, pn8, [%[c_ptr], %[c_bl2], lsl #3]            \n"
      "   st1d    {z6.d-z7.d}, pn8, [%[c_ptr], %[c_bl3], lsl #3]            \n"

      // N loop tail
      "   incd    %[n_idx], all, mul #2                                     \n"
      "   cmp     %[n_idx], %[n]                                            \n"
      "   b.mi    2b                                                        \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[c_bl1], lsl #5                      \n"
      "   incd    %[m_idx], all, mul #4                                     \n"
      "   cmp     %[m_idx], %[m]                                            \n"
      "   b.mi    1b                                                        \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [c_ptr] "=&r"(c_ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [c_bl1] "r"(c_bl1),
        [c_bl2] "r"(c_bl2), [c_bl3] "r"(c_bl3), [l_cnd] "r"(l_cnd),
        [a_cnd] "r"(a_cnd), [a_src] "r"(a), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "p0", "p8", "x12",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2p1)

static void inner_loop_206(struct loop_206_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t svl_d;
  asm volatile("cntd %[v]" : [v] "=&r"(svl_d)::);

  register uint64_t n_idx;
  register uint64_t m_idx;
  register uint64_t b1off = n * 4;
  register uint64_t c2off = n * 2;
  register uint64_t c3off = n * 3;
  register uint64_t a_cnd = a + 2 * (m * k);
  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;

  asm volatile(
      "   ptrue   p0.b                                              \n"
      "   ptrue   pn8.b                                             \n"

      // M loop head
      "   mov     %[m_idx], #0                                      \n"
      "1:                                                           \n"

      // N loop head
      "   mov     %[n_idx], #0                                      \n"
      "2:                                                           \n"

      // Accumulators
      "   mov     z10.d, #0                                         \n"
      "   mov     z11.d, #0                                         \n"
      "   mov     z12.d, #0                                         \n"
      "   mov     z13.d, #0                                         \n"
      "   mov     z14.d, #0                                         \n"
      "   mov     z15.d, #0                                         \n"
      "   mov     z16.d, #0                                         \n"
      "   mov     z17.d, #0                                         \n"
      "   mov     z20.d, #0                                         \n"
      "   mov     z21.d, #0                                         \n"
      "   mov     z22.d, #0                                         \n"
      "   mov     z23.d, #0                                         \n"
      "   mov     z24.d, #0                                         \n"
      "   mov     z25.d, #0                                         \n"
      "   mov     z26.d, #0                                         \n"
      "   mov     z27.d, #0                                         \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #3              \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #3              \n"
      "3:                                                           \n"
      "   ld1rqh  {z0.h}, p0/z, [%[a_ptr]]                          \n"
      "   ld1rqh  {z1.h}, p0/z, [%[a_ptr], #16]                     \n"
      "   ld1rqh  {z2.h}, p0/z, [%[a_ptr], #32]                     \n"
      "   ld1rqh  {z3.h}, p0/z, [%[a_ptr], #48]                     \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   ld1rqh  {z4.h}, p0/z, [%[a_ptr]]                          \n"
      "   ld1rqh  {z5.h}, p0/z, [%[a_ptr], #16]                     \n"
      "   ld1rqh  {z6.h}, p0/z, [%[a_ptr], #32]                     \n"
      "   ld1rqh  {z7.h}, p0/z, [%[a_ptr], #48]                     \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   ld1h    {z18.h-z19.h}, pn8/z, [%[b_ptr]]                  \n"
      "   ld1h    {z28.h-z29.h}, pn8/z, [%[b_ptr], %[b1off], lsl #1]\n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #4                  \n"
      "   udot    z10.d, z18.h, z0.h[0]                             \n"
      "   udot    z12.d, z18.h, z0.h[1]                             \n"
      "   udot    z14.d, z18.h, z1.h[0]                             \n"
      "   udot    z16.d, z18.h, z1.h[1]                             \n"
      "   udot    z20.d, z18.h, z2.h[0]                             \n"
      "   udot    z22.d, z18.h, z2.h[1]                             \n"
      "   udot    z24.d, z18.h, z3.h[0]                             \n"
      "   udot    z26.d, z18.h, z3.h[1]                             \n"
      "   udot    z11.d, z19.h, z0.h[0]                             \n"
      "   udot    z13.d, z19.h, z0.h[1]                             \n"
      "   udot    z15.d, z19.h, z1.h[0]                             \n"
      "   udot    z17.d, z19.h, z1.h[1]                             \n"
      "   udot    z21.d, z19.h, z2.h[0]                             \n"
      "   udot    z23.d, z19.h, z2.h[1]                             \n"
      "   udot    z25.d, z19.h, z3.h[0]                             \n"
      "   udot    z27.d, z19.h, z3.h[1]                             \n"
      "   udot    z10.d, z28.h, z4.h[0]                             \n"
      "   udot    z12.d, z28.h, z4.h[1]                             \n"
      "   udot    z14.d, z28.h, z5.h[0]                             \n"
      "   udot    z16.d, z28.h, z5.h[1]                             \n"
      "   udot    z20.d, z28.h, z6.h[0]                             \n"
      "   udot    z22.d, z28.h, z6.h[1]                             \n"
      "   udot    z24.d, z28.h, z7.h[0]                             \n"
      "   udot    z26.d, z28.h, z7.h[1]                             \n"
      "   udot    z11.d, z29.h, z4.h[0]                             \n"
      "   udot    z13.d, z29.h, z4.h[1]                             \n"
      "   udot    z15.d, z29.h, z5.h[0]                             \n"
      "   udot    z17.d, z29.h, z5.h[1]                             \n"
      "   udot    z21.d, z29.h, z6.h[0]                             \n"
      "   udot    z23.d, z29.h, z6.h[1]                             \n"
      "   udot    z25.d, z29.h, z7.h[0]                             \n"
      "   udot    z27.d, z29.h, z7.h[1]                             \n"
      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #3              \n"
      "   st1d    {z10.d-z11.d}, pn8, [%[c_ptr]]                    \n"
      "   st1d    {z12.d-z13.d}, pn8, [%[c_ptr], %[c1off], lsl #3]  \n"
      "   st1d    {z14.d-z15.d}, pn8, [%[c_ptr], %[c2off], lsl #3]  \n"
      "   st1d    {z16.d-z17.d}, pn8, [%[c_ptr], %[c3off], lsl #3]  \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #5                  \n"
      "   st1d    {z20.d-z21.d}, pn8, [%[c_ptr]]                    \n"
      "   st1d    {z22.d-z23.d}, pn8, [%[c_ptr], %[c1off], lsl #3]  \n"
      "   st1d    {z24.d-z25.d}, pn8, [%[c_ptr], %[c2off], lsl #3]  \n"
      "   st1d    {z26.d-z27.d}, pn8, [%[c_ptr], %[c3off], lsl #3]  \n"

      // N loop tail
      "   incd    %[n_idx], all, mul #2                             \n"
      "   cmp     %[n_idx], %[n]                                    \n"
      "   b.mi    2b                                                \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #6                  \n"
      "   add     %[m_idx], %[m_idx], #8                            \n"
      "   cmp     %[m_idx], %[m]                                    \n"
      "   b.mi    1b                                                \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [c_ptr] "=&r"(c_ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [c2off] "r"(c2off), [c3off] "r"(c3off), [c1off] "r"(n),
        [b1off] "r"(b1off), [a_src] "r"(a), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22",
        "z23", "z24", "z25", "z26", "z27", "z28", "z29",
        "p0", "p8", "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_206(struct loop_206_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t svl_d;
  asm volatile("cntd %[v]" : [v] "=&r"(svl_d)::);

  register uint64_t n_idx;
  register uint64_t m_idx;
  register uint64_t b2off = n * 4;
  register uint64_t b3off = n * 4 + svl_d * 4;
  register uint64_t c2off = n * 2;
  register uint64_t c3off = n * 3;
  register uint64_t a_cnd = a + 2 * (m * k);
  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c0ptr;
  register uint64_t c1ptr;

  asm volatile(
      "   ptrue   p0.b                                              \n"

      // M loop head
      "   mov     %[m_idx], #0                                      \n"
      "1:                                                           \n"

      // N loop head
      "   mov     %[n_idx], #0                                      \n"
      "2:                                                           \n"

      // Accumulators
      "   mov     z10.d, #0                                         \n"
      "   mov     z11.d, #0                                         \n"
      "   mov     z12.d, #0                                         \n"
      "   mov     z13.d, #0                                         \n"
      "   mov     z14.d, #0                                         \n"
      "   mov     z15.d, #0                                         \n"
      "   mov     z16.d, #0                                         \n"
      "   mov     z17.d, #0                                         \n"
      "   mov     z20.d, #0                                         \n"
      "   mov     z21.d, #0                                         \n"
      "   mov     z22.d, #0                                         \n"
      "   mov     z23.d, #0                                         \n"
      "   mov     z24.d, #0                                         \n"
      "   mov     z25.d, #0                                         \n"
      "   mov     z26.d, #0                                         \n"
      "   mov     z27.d, #0                                         \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #3              \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #3              \n"
      "3:                                                           \n"
      "   ld1rqh  {z0.h}, p0/z, [%[a_ptr]]                          \n"
      "   ld1rqh  {z1.h}, p0/z, [%[a_ptr], #16]                     \n"
      "   ld1rqh  {z2.h}, p0/z, [%[a_ptr], #32]                     \n"
      "   ld1rqh  {z3.h}, p0/z, [%[a_ptr], #48]                     \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   ld1rqh  {z4.h}, p0/z, [%[a_ptr]]                          \n"
      "   ld1rqh  {z5.h}, p0/z, [%[a_ptr], #16]                     \n"
      "   ld1rqh  {z6.h}, p0/z, [%[a_ptr], #32]                     \n"
      "   ld1rqh  {z7.h}, p0/z, [%[a_ptr], #48]                     \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   ld1h    {z18.h}, p0/z, [%[b_ptr]]                         \n"
      "   ld1h    {z19.h}, p0/z, [%[b_ptr], #1, mul vl]             \n"
      "   ld1h    {z28.h}, p0/z, [%[b_ptr], %[b2off], lsl #1]       \n"
      "   ld1h    {z29.h}, p0/z, [%[b_ptr], %[b3off], lsl #1]       \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #4                  \n"
      "   udot    z10.d, z18.h, z0.h[0]                             \n"
      "   udot    z12.d, z18.h, z0.h[1]                             \n"
      "   udot    z14.d, z18.h, z1.h[0]                             \n"
      "   udot    z16.d, z18.h, z1.h[1]                             \n"
      "   udot    z20.d, z18.h, z2.h[0]                             \n"
      "   udot    z22.d, z18.h, z2.h[1]                             \n"
      "   udot    z24.d, z18.h, z3.h[0]                             \n"
      "   udot    z26.d, z18.h, z3.h[1]                             \n"
      "   udot    z11.d, z19.h, z0.h[0]                             \n"
      "   udot    z13.d, z19.h, z0.h[1]                             \n"
      "   udot    z15.d, z19.h, z1.h[0]                             \n"
      "   udot    z17.d, z19.h, z1.h[1]                             \n"
      "   udot    z21.d, z19.h, z2.h[0]                             \n"
      "   udot    z23.d, z19.h, z2.h[1]                             \n"
      "   udot    z25.d, z19.h, z3.h[0]                             \n"
      "   udot    z27.d, z19.h, z3.h[1]                             \n"
      "   udot    z10.d, z28.h, z4.h[0]                             \n"
      "   udot    z12.d, z28.h, z4.h[1]                             \n"
      "   udot    z14.d, z28.h, z5.h[0]                             \n"
      "   udot    z16.d, z28.h, z5.h[1]                             \n"
      "   udot    z20.d, z28.h, z6.h[0]                             \n"
      "   udot    z22.d, z28.h, z6.h[1]                             \n"
      "   udot    z24.d, z28.h, z7.h[0]                             \n"
      "   udot    z26.d, z28.h, z7.h[1]                             \n"
      "   udot    z11.d, z29.h, z4.h[0]                             \n"
      "   udot    z13.d, z29.h, z4.h[1]                             \n"
      "   udot    z15.d, z29.h, z5.h[0]                             \n"
      "   udot    z17.d, z29.h, z5.h[1]                             \n"
      "   udot    z21.d, z29.h, z6.h[0]                             \n"
      "   udot    z23.d, z29.h, z6.h[1]                             \n"
      "   udot    z25.d, z29.h, z7.h[0]                             \n"
      "   udot    z27.d, z29.h, z7.h[1]                             \n"
      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

      // Store
      "   add     %[c0ptr], %[c_dst], %[n_idx], lsl #3              \n"
      "   addvl   %[c1ptr], %[c0ptr], #1                            \n"
      "   st1d    {z10.d}, p0, [%[c0ptr]]                           \n"
      "   st1d    {z11.d}, p0, [%[c1ptr]]                           \n"
      "   st1d    {z12.d}, p0, [%[c0ptr], %[c1off], lsl #3]         \n"
      "   st1d    {z13.d}, p0, [%[c1ptr], %[c1off], lsl #3]         \n"
      "   st1d    {z14.d}, p0, [%[c0ptr], %[c2off], lsl #3]         \n"
      "   st1d    {z15.d}, p0, [%[c1ptr], %[c2off], lsl #3]         \n"
      "   st1d    {z16.d}, p0, [%[c0ptr], %[c3off], lsl #3]         \n"
      "   st1d    {z17.d}, p0, [%[c1ptr], %[c3off], lsl #3]         \n"
      "   add     %[c0ptr], %[c0ptr], %[n], lsl #5                  \n"
      "   add     %[c1ptr], %[c1ptr], %[n], lsl #5                  \n"
      "   st1d    {z20.d}, p0, [%[c0ptr]]                           \n"
      "   st1d    {z21.d}, p0, [%[c1ptr]]                           \n"
      "   st1d    {z22.d}, p0, [%[c0ptr], %[c1off], lsl #3]         \n"
      "   st1d    {z23.d}, p0, [%[c1ptr], %[c1off], lsl #3]         \n"
      "   st1d    {z24.d}, p0, [%[c0ptr], %[c2off], lsl #3]         \n"
      "   st1d    {z25.d}, p0, [%[c1ptr], %[c2off], lsl #3]         \n"
      "   st1d    {z26.d}, p0, [%[c0ptr], %[c3off], lsl #3]         \n"
      "   st1d    {z27.d}, p0, [%[c1ptr], %[c3off], lsl #3]         \n"

      // N loop tail
      "   incd    %[n_idx], all, mul #2                             \n"
      "   cmp     %[n_idx], %[n]                                    \n"
      "   b.mi    2b                                                \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #6                  \n"
      "   add     %[m_idx], %[m_idx], #8                            \n"
      "   cmp     %[m_idx], %[m]                                    \n"
      "   b.mi    1b                                                \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [m_idx] "=&r"(m_idx),
        [c0ptr] "=&r"(c0ptr), [c1ptr] "=&r"(c1ptr), [n_idx] "=&r"(n_idx),
        [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [c2off] "r"(c2off), [c3off] "r"(c3off), [c1off] "r"(n),
        [b2off] "r"(b2off), [b3off] "r"(b3off), [a_src] "r"(a), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22",
        "z23", "z24", "z25", "z26", "z27", "z28", "z29", "p0", "cc", "memory");
}

#elif defined(__ARM_NEON)

static void inner_loop_206(struct loop_206_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t m_inc = m * 8;
  register uint64_t n_inc = n * 8;
  register uint64_t a_cnd = a + 2 * (m * k);

  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a_ptr;
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
      "   movi    v16.16b, #0                                       \n"
      "   movi    v17.16b, #0                                       \n"
      "   movi    v18.16b, #0                                       \n"
      "   movi    v19.16b, #0                                       \n"
      "   movi    v20.16b, #0                                       \n"
      "   movi    v21.16b, #0                                       \n"
      "   movi    v22.16b, #0                                       \n"
      "   movi    v23.16b, #0                                       \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #3              \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #3              \n"
      "3:                                                           \n"
      "   ld1     {v0.8h, v1.8h}, [%[a_ptr]], %[m_inc]              \n"
      "   ld1     {v2.8h, v3.8h}, [%[b_ptr]], %[n_inc]              \n"

      "   uxtl    v4.4s, v0.4h                                      \n"
      "   uxtl2   v5.4s, v0.8h                                      \n"
      "   uxtl    v14.4s, v1.4h                                     \n"
      "   uxtl2   v15.4s, v1.8h                                     \n"

      "   zip1    v6.8h, v2.8h, v3.8h                               \n"
      "   zip2    v7.8h, v2.8h, v3.8h                               \n"
      "   zip1    v8.8h, v6.8h, v7.8h                               \n"
      "   zip2    v9.8h, v6.8h, v7.8h                               \n"

      "   uxtl    v10.4s, v8.4h                                     \n"
      "   uxtl2   v11.4s, v8.8h                                     \n"
      "   uxtl    v12.4s, v9.4h                                     \n"
      "   uxtl2   v13.4s, v9.8h                                     \n"

      "   umlal    v16.2d, v10.2s, v4.s[0]                          \n"
      "   umlal2   v17.2d, v10.4s, v4.s[0]                          \n"
      "   umlal    v18.2d, v10.2s, v5.s[0]                          \n"
      "   umlal2   v19.2d, v10.4s, v5.s[0]                          \n"
      "   umlal    v20.2d, v10.2s, v14.s[0]                         \n"
      "   umlal2   v21.2d, v10.4s, v14.s[0]                         \n"
      "   umlal    v22.2d, v10.2s, v15.s[0]                         \n"
      "   umlal2   v23.2d, v10.4s, v15.s[0]                         \n"

      "   umlal    v16.2d, v11.2s, v4.s[1]                          \n"
      "   umlal2   v17.2d, v11.4s, v4.s[1]                          \n"
      "   umlal    v18.2d, v11.2s, v5.s[1]                          \n"
      "   umlal2   v19.2d, v11.4s, v5.s[1]                          \n"
      "   umlal    v20.2d, v11.2s, v14.s[1]                         \n"
      "   umlal2   v21.2d, v11.4s, v14.s[1]                         \n"
      "   umlal    v22.2d, v11.2s, v15.s[1]                         \n"
      "   umlal2   v23.2d, v11.4s, v15.s[1]                         \n"

      "   umlal    v16.2d, v12.2s, v4.s[2]                          \n"
      "   umlal2   v17.2d, v12.4s, v4.s[2]                          \n"
      "   umlal    v18.2d, v12.2s, v5.s[2]                          \n"
      "   umlal2   v19.2d, v12.4s, v5.s[2]                          \n"
      "   umlal    v20.2d, v12.2s, v14.s[2]                         \n"
      "   umlal2   v21.2d, v12.4s, v14.s[2]                         \n"
      "   umlal    v22.2d, v12.2s, v15.s[2]                         \n"
      "   umlal2   v23.2d, v12.4s, v15.s[2]                         \n"

      "   umlal    v16.2d, v13.2s, v4.s[3]                          \n"
      "   umlal2   v17.2d, v13.4s, v4.s[3]                          \n"
      "   umlal    v18.2d, v13.2s, v5.s[3]                          \n"
      "   umlal2   v19.2d, v13.4s, v5.s[3]                          \n"
      "   umlal    v20.2d, v13.2s, v14.s[3]                         \n"
      "   umlal2   v21.2d, v13.4s, v14.s[3]                         \n"
      "   umlal    v22.2d, v13.2s, v15.s[3]                         \n"
      "   umlal2   v23.2d, v13.4s, v15.s[3]                         \n"

      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #3              \n"
      "   st1     {v16.2d, v17.2d}, [%[c_ptr]], %[n_inc]            \n"
      "   st1     {v18.2d, v19.2d}, [%[c_ptr]], %[n_inc]            \n"
      "   st1     {v20.2d, v21.2d}, [%[c_ptr]], %[n_inc]            \n"
      "   st1     {v22.2d, v23.2d}, [%[c_ptr]], %[n_inc]            \n"

      // N loop tail
      "   add     %[n_idx], %[n_idx], #4                            \n"
      "   cmp     %[n_idx], %[n]                                    \n"
      "   b.mi    2b                                                \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #5                  \n"
      "   add     %[m_idx], %[m_idx], #4                            \n"
      "   cmp     %[m_idx], %[m]                                    \n"
      "   b.mi    1b                                                \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [c_ptr] "=&r"(c_ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [m_inc] "r"(m_inc), [n_inc] "r"(n_inc), [a_src] "r"(a), [b_src] "r"(b)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v10", "v11", "v12",
        "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
        "v23", "v24", "v25", "v26", "v27", "v28", "v29", "cc", "memory");
}

#else

static void inner_loop_206(struct loop_206_data *data) {
    printf("ABORT: No implementations available for this target.\n");
  exit(2);
}

#endif


// Ensure the max SVL that will be targetted is defined
#if (!defined(MAX_VL) || MAX_VL == 0)
#undef  MAX_VL
#define MAX_VL 2048
#endif

// Re-define PROBLEM_SIZE_LIMIT_KIB if it has been set to 0
// Default of 96KiB equates to original problem size (M=128, K=256, N=64)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef  PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 96
#endif
#endif /* !HAVE_CANDIDATE */

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m,n,k) ((k)*((m)+(n))*sizeof(uint16_t))

LOOP_DECL(206, OUTER_LOOP_ATTR)
{
  // Work out values for M, K and N to fit within problem size limit
  uint64_t M = 0;  // multiple of SVLh
  uint64_t N = 0;  // multiple of SVLs
  uint64_t K = 0;  // multiple of 8

  // For this loop, N should be M/2, K should be 2*M
  const uint64_t M_base = MAX_VL / 16;
  while (true) {
    uint64_t m = M + M_base;
    uint64_t n = m / 2;   // Automatically a multiple of SVLs
    uint64_t k = m * 2;   // Automatically a multiple of 8
    if (PROBLEM_SIZE_ACTUAL(m,n,k) <= PROBLEM_SIZE_LIMIT_KIB*1024) {
      M = m;
      N = n;
      K = k;
    } else {
      break;
    }
  }

  struct loop_206_data data = { .m = M, .n = N, .k = K, };
  ALLOC_64B(data.a, M * K, "A matrix");
  ALLOC_64B(data.b, K * N, "B matrix");
  ALLOC_64B(data.c, M * N, "C matrix");

  fill_uint16(data.a, M * K);
  fill_uint16(data.b, K * N);

  inner_loops_206(iters, &data);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", K = %" PRIu64 ", N = %" PRIu64 "\n", M, K, N);
  printf("\t%" PRIu64 " x %" PRIu64 " * %" PRIu64 " x %" PRIu64 "\n", M, K, K, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M,N,K)/1024.0f);
#endif

  int checksum = 0;
#define CHECK(x, y)                                             \
  {                                                             \
    uint64_t d = 0;                                             \
    for (int k = 0; k < K; k += 4) {                            \
      int i = k * M + 4 * (x), j = k * N + 4 * (y);             \
      d += (uint64_t)data.a[i + 0] * (uint64_t)data.b[j + 0];   \
      d += (uint64_t)data.a[i + 1] * (uint64_t)data.b[j + 1];   \
      d += (uint64_t)data.a[i + 2] * (uint64_t)data.b[j + 2];   \
      d += (uint64_t)data.a[i + 3] * (uint64_t)data.b[j + 3];   \
    }                                                           \
    checksum += (int)(d != data.c[(x) * N + (y)]);              \
  }
#ifdef FULL_CHECK
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) CHECK(m, n);
#else
  CHECK(0, 0);
  CHECK(0, N - 1);
  CHECK(M - 1, 0);
  CHECK(M - 1, N - 1);
  CHECK(M / 2, N / 2);
#endif

  bool passed = (checksum == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(206, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
