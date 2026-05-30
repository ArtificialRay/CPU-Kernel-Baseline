/*----------------------------------------------------------------------------
#
#   Loop 201: FP64 matrix-matrix multiply using MOPA / DOT
#
#   Purpose:
#     Use of fp64 MOPA (or MLA) instructions.
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
    K: even
*/

struct loop_201_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  double *restrict a;
  double *restrict b;
  double *restrict c;
};

#if (defined(__ARM_FEATURE_SME2) && defined(__ARM_FEATURE_SME_F64F64))
#define LOOP_201_SME
#endif

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_201(struct loop_201_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#define LOOP_ATTR SC_SVE_ATTR
#define OUTER_LOOP_ATTR SC_SVE_LOOP_ATTR
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#elif defined(LOOP_201_SME)
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

static void inner_loop_201(struct loop_201_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  double *restrict a = data->a;
  double *restrict b = data->b;
  double *restrict c = data->c;
  for (uint64_t x = 0; x < m; x++) {
    for (uint64_t y = 0; y < n; y++) {
      c[x * n + y] = 0.0;
    }
  }

  // Loops ordered for contiguous memory access in inner loop
  for (uint64_t z = 0; z < k; z++)
    for (uint64_t x = 0; x < m; x++) {
      for (uint64_t y = 0; y < n; y++) {
        c[x * n + y] += a[z * m + x] * b[z * n + y];
      }
    }
}

#elif (defined(LOOP_201_SME) && defined(HAVE_SME_INTRINSICS))

static void inner_loop_201(struct loop_201_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  float64_t *a = data->a;
  float64_t *b = data->b;
  float64_t *c = data->c;

  float64_t *ptr_a, *ptr_b, *ptr_c;
  float64_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx, l_idx;
  uint64_t svl_d = svcntd();
  uint64_t l_cnd = svl_d * 8;
  uint64_t blk_1 = svl_d * n;
  uint64_t blk_2 = blk_1 * 2;
  uint64_t blk_3 = blk_1 * 3;

  svcount_t c_all = svptrue_c64();
  svbool_t p_all = svptrue_b64();

  svfloat64x4_t vec_a0, vec_a1;
  svfloat64x2_t vec_b0, vec_b1;
  svuint8x4_t vec_c0, vec_c1;

#define MOPA_TILE(t, x, i, j) \
  svmopa_za64_m(t, p_all, p_all, svget4(vec_a##x, i), svget2(vec_b##x, j))

#define EXTR(x, i) svreinterpret_f64(svget4(vec_c##x, i))
#define STORE_PAIR(x, i, j, o) \
  svst1(c_all, &ptr_c[o], svcreate2(EXTR(x, i), EXTR(x, j)))

#if defined(__ARM_FEATURE_SME2p1)
  svzero_za();
#endif

  for (m_idx = 0; m_idx < m; m_idx += svl_d * 4) {
    for (n_idx = 0; n_idx < n; n_idx += svl_d * 2) {
#if !defined(__ARM_FEATURE_SME2p1)
      svzero_za();
#endif
      ptr_a = &a[m_idx];
      ptr_b = &b[n_idx];
      while (ptr_a < cnd_k) {
        vec_a0 = svld1_x4(c_all, &ptr_a[0]);
        vec_b0 = svld1_x2(c_all, &ptr_b[0]);
        vec_a1 = svld1_x4(c_all, &ptr_a[m]);
        vec_b1 = svld1_x2(c_all, &ptr_b[n]);

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
        STORE_PAIR(0, 2, 3, blk_1);
        STORE_PAIR(1, 0, 1, blk_2);
        STORE_PAIR(1, 2, 3, blk_3);

        ptr_c += n;
      }
    }
    c += blk_1 * 4;
  }
}

#elif defined(HAVE_SVE_INTRINSICS)

static void inner_loop_201(struct loop_201_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  float64_t *a = data->a;
  float64_t *b = data->b;
  float64_t *c = data->c;

  float64_t *ptr_a, *ptr_b, *ptr_c;
  float64_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx;
  svbool_t p_all = svptrue_b64();

  svfloat64x2_t acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7;
  svfloat64x4_t lda_0, lda_1;
  svfloat64x2_t ldb_0, ldb_1;

#define ZERO svdup_f64(0.0)
#define ZERO_PAIR(y) acc_##y = svcreate2(ZERO, ZERO)

#define LDA(y, z) svld1rq(p_all, &ptr_a[z * m + y * 2])
#define LOADA_QUAD(z) svcreate4(LDA(0, z), LDA(1, z), LDA(2, z), LDA(3, z))

#define GETA(y, z) svget4(lda_##z, y / 2)
#define GETB(x, z) svget2(ldb_##z, x)
#define GETC(x, y) svget2(acc_##y, x)

#define MLA(x, y, z) svmla_lane(GETC(x, y), GETB(x, z), GETA(y, z), y % 2)
#define MLA_PAIR(y, z) acc_##y = svcreate2(MLA(0, y, z), MLA(1, y, z));
#define MLA_GROUP(y) MLA_PAIR(y, 0) MLA_PAIR(y, 1)

#if defined(__ARM_FEATURE_SVE2p1)
  svcount_t c_all = svptrue_c64();
#define LOADB_PAIR(y) svld1_x2(c_all, &ptr_b[n * y]);
#define STORE_PAIR(y) svst1(c_all, &ptr_c[n * y], acc_##y)
#else
#define LOADB(x, y) svld1_vnum(p_all, &ptr_b[n * y], x)
#define LOADB_PAIR(y) svcreate2(LOADB(0, y), LOADB(1, y))
#define STORE(x, y) svst1_vnum(p_all, &ptr_c[n * y], x, GETC(x, y));
#define STORE_PAIR(y) STORE(0, y) STORE(1, y)
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

      ptr_a = &a[m_idx];
      ptr_b = &b[n_idx];
      while (ptr_a < cnd_k) {
        lda_0 = LOADA_QUAD(0);
        lda_1 = LOADA_QUAD(1);
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
}

#elif defined(LOOP_201_SME)

static void inner_loop_201(struct loop_201_data *data)
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
  register uint64_t a_cnd = a + 8 * (m * k);

  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  // x12: slice index register for tile-to-vec mova

  asm volatile(
      "   ptrue   p0.d                                                      \n"
      "   ptrue   pn8.d                                                     \n"
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
      "   ld1d    {z0.d-z3.d}, pn8/z, [%[a_ptr]]                            \n"
#if !defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                      \n"
#endif
      "   ld1d    {z4.d-z5.d}, pn8/z, [%[b_ptr]]                            \n"
      "   fmopa   za0.d, p0/m, p0/m, z0.d, z4.d                             \n"
      "   fmopa   za1.d, p0/m, p0/m, z0.d, z5.d                             \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                          \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                          \n"
      "   fmopa   za2.d, p0/m, p0/m, z1.d, z4.d                             \n"
      "   fmopa   za3.d, p0/m, p0/m, z1.d, z5.d                             \n"
      "3:                                                                   \n"
      "   fmopa   za4.d, p0/m, p0/m, z2.d, z4.d                             \n"
      "   fmopa   za5.d, p0/m, p0/m, z2.d, z5.d                             \n"
      "   fmopa   za6.d, p0/m, p0/m, z3.d, z4.d                             \n"
      "   fmopa   za7.d, p0/m, p0/m, z3.d, z5.d                             \n"
      "   ld1d    {z0.d-z3.d}, pn8/z, [%[a_ptr]]                            \n"
      "   ld1d    {z4.d-z5.d}, pn8/z, [%[b_ptr]]                            \n"
      "   fmopa   za0.d, p0/m, p0/m, z0.d, z4.d                             \n"
      "   fmopa   za1.d, p0/m, p0/m, z0.d, z5.d                             \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                          \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                          \n"
      "   fmopa   za2.d, p0/m, p0/m, z1.d, z4.d                             \n"
      "   fmopa   za3.d, p0/m, p0/m, z1.d, z5.d                             \n"
      "   cmp     %[a_ptr], %[a_cnd]                                        \n"
      "   b.mi    3b                                                        \n"
      "   fmopa   za4.d, p0/m, p0/m, z2.d, z4.d                             \n"
      "   fmopa   za5.d, p0/m, p0/m, z2.d, z5.d                             \n"
      "   fmopa   za6.d, p0/m, p0/m, z3.d, z4.d                             \n"
      "   fmopa   za7.d, p0/m, p0/m, z3.d, z5.d                             \n"

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
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "x12", "p0", "p8",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2p1)

static void inner_loop_201(struct loop_201_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t off_2 = n * 2;
  register uint64_t off_3 = n * 3;
  register uint64_t a_cnd = a + 8 * (m * k);

  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  register uint64_t n_idx;
  register uint64_t m_idx;

  asm volatile(
      "   ptrue   p0.d                                              \n"
      "   ptrue   pn8.d                                             \n"

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
      "   ld1rqd  {z0.d}, p0/z, [%[a_ptr]]                          \n"
      "   ld1rqd  {z1.d}, p0/z, [%[a_ptr], #16]                     \n"
      "   ld1rqd  {z2.d}, p0/z, [%[a_ptr], #32]                     \n"
      "   ld1rqd  {z3.d}, p0/z, [%[a_ptr], #48]                     \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   ld1rqd  {z4.d}, p0/z, [%[a_ptr]]                          \n"
      "   ld1rqd  {z5.d}, p0/z, [%[a_ptr], #16]                     \n"
      "   ld1rqd  {z6.d}, p0/z, [%[a_ptr], #32]                     \n"
      "   ld1rqd  {z7.d}, p0/z, [%[a_ptr], #48]                     \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   ld1d    {z18.d-z19.d}, pn8/z, [%[b_ptr]]                  \n"
      "   ld1d    {z28.d-z29.d}, pn8/z, [%[b_ptr], %[n], lsl #3]    \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #4                  \n"
      "   fmla    z10.d, z18.d, z0.d[0]                             \n"
      "   fmla    z12.d, z18.d, z0.d[1]                             \n"
      "   fmla    z14.d, z18.d, z1.d[0]                             \n"
      "   fmla    z16.d, z18.d, z1.d[1]                             \n"
      "   fmla    z20.d, z18.d, z2.d[0]                             \n"
      "   fmla    z22.d, z18.d, z2.d[1]                             \n"
      "   fmla    z24.d, z18.d, z3.d[0]                             \n"
      "   fmla    z26.d, z18.d, z3.d[1]                             \n"
      "   fmla    z11.d, z19.d, z0.d[0]                             \n"
      "   fmla    z13.d, z19.d, z0.d[1]                             \n"
      "   fmla    z15.d, z19.d, z1.d[0]                             \n"
      "   fmla    z17.d, z19.d, z1.d[1]                             \n"
      "   fmla    z21.d, z19.d, z2.d[0]                             \n"
      "   fmla    z23.d, z19.d, z2.d[1]                             \n"
      "   fmla    z25.d, z19.d, z3.d[0]                             \n"
      "   fmla    z27.d, z19.d, z3.d[1]                             \n"
      "   fmla    z10.d, z28.d, z4.d[0]                             \n"
      "   fmla    z12.d, z28.d, z4.d[1]                             \n"
      "   fmla    z14.d, z28.d, z5.d[0]                             \n"
      "   fmla    z16.d, z28.d, z5.d[1]                             \n"
      "   fmla    z20.d, z28.d, z6.d[0]                             \n"
      "   fmla    z22.d, z28.d, z6.d[1]                             \n"
      "   fmla    z24.d, z28.d, z7.d[0]                             \n"
      "   fmla    z26.d, z28.d, z7.d[1]                             \n"
      "   fmla    z11.d, z29.d, z4.d[0]                             \n"
      "   fmla    z13.d, z29.d, z4.d[1]                             \n"
      "   fmla    z15.d, z29.d, z5.d[0]                             \n"
      "   fmla    z17.d, z29.d, z5.d[1]                             \n"
      "   fmla    z21.d, z29.d, z6.d[0]                             \n"
      "   fmla    z23.d, z29.d, z6.d[1]                             \n"
      "   fmla    z25.d, z29.d, z7.d[0]                             \n"
      "   fmla    z27.d, z29.d, z7.d[1]                             \n"
      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #3              \n"
      "   st1d    {z10.d-z11.d}, pn8, [%[c_ptr]]                    \n"
      "   st1d    {z12.d-z13.d}, pn8, [%[c_ptr], %[off_1], lsl #3]  \n"
      "   st1d    {z14.d-z15.d}, pn8, [%[c_ptr], %[off_2], lsl #3]  \n"
      "   st1d    {z16.d-z17.d}, pn8, [%[c_ptr], %[off_3], lsl #3]  \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #5                  \n"
      "   st1d    {z20.d-z21.d}, pn8, [%[c_ptr]]                    \n"
      "   st1d    {z22.d-z23.d}, pn8, [%[c_ptr], %[off_1], lsl #3]  \n"
      "   st1d    {z24.d-z25.d}, pn8, [%[c_ptr], %[off_2], lsl #3]  \n"
      "   st1d    {z26.d-z27.d}, pn8, [%[c_ptr], %[off_3], lsl #3]  \n"

      // N loop tail
      "   incd    %[n_idx], all, mul #2                             \n"
      "   cmp     %[n_idx], %[n]                                    \n"
      "   b.mi    2b                                                \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #6                  \n"
      "   add     %[m_idx], %[m_idx], #8                            \n"
      "   cmp     %[m_idx], %[m]                                    \n"
      "   b.mi    1b                                                \n"

      : [m_idx] "=&r"(m_idx), [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr),
        [n_idx] "=&r"(n_idx), [c_ptr] "=&r"(c_ptr), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [off_2] "r"(off_2), [off_3] "r"(off_3), [off_1] "r"(n), [a_src] "r"(a),
        [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22",
        "z23", "z24", "z25", "z26", "z27", "z28", "z29",
        "p0", "p8", "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_201(struct loop_201_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t off_2 = n * 2;
  register uint64_t off_3 = n * 3;
  register uint64_t a_cnd = a + 8 * (m * k);

  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c0ptr;
  register uint64_t c1ptr;
  register uint64_t n_idx;
  register uint64_t m_idx;

  asm volatile(
      "   ptrue   p0.d                                              \n"

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
      "   ld1rqd  {z0.d}, p0/z, [%[a_ptr]]                          \n"
      "   ld1rqd  {z1.d}, p0/z, [%[a_ptr], #16]                     \n"
      "   ld1rqd  {z2.d}, p0/z, [%[a_ptr], #32]                     \n"
      "   ld1rqd  {z3.d}, p0/z, [%[a_ptr], #48]                     \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   ld1rqd  {z4.d}, p0/z, [%[a_ptr]]                          \n"
      "   ld1rqd  {z5.d}, p0/z, [%[a_ptr], #16]                     \n"
      "   ld1rqd  {z6.d}, p0/z, [%[a_ptr], #32]                     \n"
      "   ld1rqd  {z7.d}, p0/z, [%[a_ptr], #48]                     \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   ld1d    {z18.d}, p0/z, [%[b_ptr]]                         \n"
      "   ld1d    {z19.d}, p0/z, [%[b_ptr], #1, mul vl]             \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                  \n"
      "   ld1d    {z28.d}, p0/z, [%[b_ptr]]                         \n"
      "   ld1d    {z29.d}, p0/z, [%[b_ptr], #1, mul vl]             \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                  \n"
      "   fmla    z10.d, z18.d, z0.d[0]                             \n"
      "   fmla    z12.d, z18.d, z0.d[1]                             \n"
      "   fmla    z14.d, z18.d, z1.d[0]                             \n"
      "   fmla    z16.d, z18.d, z1.d[1]                             \n"
      "   fmla    z20.d, z18.d, z2.d[0]                             \n"
      "   fmla    z22.d, z18.d, z2.d[1]                             \n"
      "   fmla    z24.d, z18.d, z3.d[0]                             \n"
      "   fmla    z26.d, z18.d, z3.d[1]                             \n"
      "   fmla    z11.d, z19.d, z0.d[0]                             \n"
      "   fmla    z13.d, z19.d, z0.d[1]                             \n"
      "   fmla    z15.d, z19.d, z1.d[0]                             \n"
      "   fmla    z17.d, z19.d, z1.d[1]                             \n"
      "   fmla    z21.d, z19.d, z2.d[0]                             \n"
      "   fmla    z23.d, z19.d, z2.d[1]                             \n"
      "   fmla    z25.d, z19.d, z3.d[0]                             \n"
      "   fmla    z27.d, z19.d, z3.d[1]                             \n"
      "   fmla    z10.d, z28.d, z4.d[0]                             \n"
      "   fmla    z12.d, z28.d, z4.d[1]                             \n"
      "   fmla    z14.d, z28.d, z5.d[0]                             \n"
      "   fmla    z16.d, z28.d, z5.d[1]                             \n"
      "   fmla    z20.d, z28.d, z6.d[0]                             \n"
      "   fmla    z22.d, z28.d, z6.d[1]                             \n"
      "   fmla    z24.d, z28.d, z7.d[0]                             \n"
      "   fmla    z26.d, z28.d, z7.d[1]                             \n"
      "   fmla    z11.d, z29.d, z4.d[0]                             \n"
      "   fmla    z13.d, z29.d, z4.d[1]                             \n"
      "   fmla    z15.d, z29.d, z5.d[0]                             \n"
      "   fmla    z17.d, z29.d, z5.d[1]                             \n"
      "   fmla    z21.d, z29.d, z6.d[0]                             \n"
      "   fmla    z23.d, z29.d, z6.d[1]                             \n"
      "   fmla    z25.d, z29.d, z7.d[0]                             \n"
      "   fmla    z27.d, z29.d, z7.d[1]                             \n"
      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

      // Store
      "   add     %[c0ptr], %[c_dst], %[n_idx], lsl #3              \n"
      "   addvl   %[c1ptr], %[c0ptr], #1                            \n"
      "   st1d    {z10.d}, p0, [%[c0ptr]]                           \n"
      "   st1d    {z11.d}, p0, [%[c1ptr]]                           \n"
      "   st1d    {z12.d}, p0, [%[c0ptr], %[off_1], lsl #3]         \n"
      "   st1d    {z13.d}, p0, [%[c1ptr], %[off_1], lsl #3]         \n"
      "   st1d    {z14.d}, p0, [%[c0ptr], %[off_2], lsl #3]         \n"
      "   st1d    {z15.d}, p0, [%[c1ptr], %[off_2], lsl #3]         \n"
      "   st1d    {z16.d}, p0, [%[c0ptr], %[off_3], lsl #3]         \n"
      "   st1d    {z17.d}, p0, [%[c1ptr], %[off_3], lsl #3]         \n"
      "   add     %[c0ptr], %[c0ptr], %[n], lsl #5                  \n"
      "   add     %[c1ptr], %[c1ptr], %[n], lsl #5                  \n"
      "   st1d    {z20.d}, p0, [%[c0ptr]]                           \n"
      "   st1d    {z21.d}, p0, [%[c1ptr]]                           \n"
      "   st1d    {z22.d}, p0, [%[c0ptr], %[off_1], lsl #3]         \n"
      "   st1d    {z23.d}, p0, [%[c1ptr], %[off_1], lsl #3]         \n"
      "   st1d    {z24.d}, p0, [%[c0ptr], %[off_2], lsl #3]         \n"
      "   st1d    {z25.d}, p0, [%[c1ptr], %[off_2], lsl #3]         \n"
      "   st1d    {z26.d}, p0, [%[c0ptr], %[off_3], lsl #3]         \n"
      "   st1d    {z27.d}, p0, [%[c1ptr], %[off_3], lsl #3]         \n"

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
        [off_2] "r"(off_2), [off_3] "r"(off_3), [off_1] "r"(n), [a_src] "r"(a),
        [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22",
        "z23", "z24", "z25", "z26", "z27", "z28", "z29", "p0", "cc", "memory");
}

#elif defined(__ARM_NEON)

static void inner_loop_201(struct loop_201_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t m_inc = m * 8;
  register uint64_t n_inc = n * 8;
  register uint64_t a_cnd = a + (m_inc * k);

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
      "   movi    v10.16b, #0                                       \n"
      "   movi    v11.16b, #0                                       \n"
      "   movi    v12.16b, #0                                       \n"
      "   movi    v13.16b, #0                                       \n"
      "   movi    v14.16b, #0                                       \n"
      "   movi    v15.16b, #0                                       \n"
      "   movi    v16.16b, #0                                       \n"
      "   movi    v17.16b, #0                                       \n"
      "   movi    v20.16b, #0                                       \n"
      "   movi    v21.16b, #0                                       \n"
      "   movi    v22.16b, #0                                       \n"
      "   movi    v23.16b, #0                                       \n"
      "   movi    v24.16b, #0                                       \n"
      "   movi    v25.16b, #0                                       \n"
      "   movi    v26.16b, #0                                       \n"
      "   movi    v27.16b, #0                                       \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #3              \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #3              \n"
      "3:                                                           \n"
      "   ld1     {v0.2d,v1.2d,v2.2d,v3.2d}, [%[a_ptr]], %[m_inc]   \n"
      "   ld1     {v4.2d,v5.2d,v6.2d,v7.2d}, [%[a_ptr]], %[m_inc]   \n"
      "   ld1     {v18.2d,v19.2d}, [%[b_ptr]], %[n_inc]             \n"
      "   ld1     {v28.2d,v29.2d}, [%[b_ptr]], %[n_inc]             \n"
      "   fmla    v10.2d, v18.2d, v0.d[0]                           \n"
      "   fmla    v12.2d, v18.2d, v0.d[1]                           \n"
      "   fmla    v14.2d, v18.2d, v1.d[0]                           \n"
      "   fmla    v16.2d, v18.2d, v1.d[1]                           \n"
      "   fmla    v20.2d, v18.2d, v2.d[0]                           \n"
      "   fmla    v22.2d, v18.2d, v2.d[1]                           \n"
      "   fmla    v24.2d, v18.2d, v3.d[0]                           \n"
      "   fmla    v26.2d, v18.2d, v3.d[1]                           \n"
      "   fmla    v11.2d, v19.2d, v0.d[0]                           \n"
      "   fmla    v13.2d, v19.2d, v0.d[1]                           \n"
      "   fmla    v15.2d, v19.2d, v1.d[0]                           \n"
      "   fmla    v17.2d, v19.2d, v1.d[1]                           \n"
      "   fmla    v21.2d, v19.2d, v2.d[0]                           \n"
      "   fmla    v23.2d, v19.2d, v2.d[1]                           \n"
      "   fmla    v25.2d, v19.2d, v3.d[0]                           \n"
      "   fmla    v27.2d, v19.2d, v3.d[1]                           \n"
      "   fmla    v10.2d, v28.2d, v4.d[0]                           \n"
      "   fmla    v12.2d, v28.2d, v4.d[1]                           \n"
      "   fmla    v14.2d, v28.2d, v5.d[0]                           \n"
      "   fmla    v16.2d, v28.2d, v5.d[1]                           \n"
      "   fmla    v20.2d, v28.2d, v6.d[0]                           \n"
      "   fmla    v22.2d, v28.2d, v6.d[1]                           \n"
      "   fmla    v24.2d, v28.2d, v7.d[0]                           \n"
      "   fmla    v26.2d, v28.2d, v7.d[1]                           \n"
      "   fmla    v11.2d, v29.2d, v4.d[0]                           \n"
      "   fmla    v13.2d, v29.2d, v4.d[1]                           \n"
      "   fmla    v15.2d, v29.2d, v5.d[0]                           \n"
      "   fmla    v17.2d, v29.2d, v5.d[1]                           \n"
      "   fmla    v21.2d, v29.2d, v6.d[0]                           \n"
      "   fmla    v23.2d, v29.2d, v6.d[1]                           \n"
      "   fmla    v25.2d, v29.2d, v7.d[0]                           \n"
      "   fmla    v27.2d, v29.2d, v7.d[1]                           \n"
      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #3              \n"
      "   st1     {v10.2d,v11.2d}, [%[c_ptr]], %[n_inc]             \n"
      "   st1     {v12.2d,v13.2d}, [%[c_ptr]], %[n_inc]             \n"
      "   st1     {v14.2d,v15.2d}, [%[c_ptr]], %[n_inc]             \n"
      "   st1     {v16.2d,v17.2d}, [%[c_ptr]], %[n_inc]             \n"
      "   st1     {v20.2d,v21.2d}, [%[c_ptr]], %[n_inc]             \n"
      "   st1     {v22.2d,v23.2d}, [%[c_ptr]], %[n_inc]             \n"
      "   st1     {v24.2d,v25.2d}, [%[c_ptr]], %[n_inc]             \n"
      "   st1     {v26.2d,v27.2d}, [%[c_ptr]], %[n_inc]             \n"

      // N loop tail
      "   add     %[n_idx], %[n_idx], #4                            \n"
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
        [m_inc] "r"(m_inc), [n_inc] "r"(n_inc), [a_src] "r"(a), [b_src] "r"(b)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v10", "v11", "v12",
        "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
        "v23", "v24", "v25", "v26", "v27", "v28", "v29", "cc", "memory");
}

#else

static void inner_loop_201(struct loop_201_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}

#endif
#endif /* !HAVE_CANDIDATE */


// Ensure the max SVL that will be targetted is defined
#if (!defined(MAX_VL) || MAX_VL == 0)
#undef  MAX_VL
#define MAX_VL 2048
#endif

// Re-define PROBLEM_SIZE_LIMIT_KIB if it has been set to 0
// Default of 192KiB equates to original problem size (M=128, K=128, N=64)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef  PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 192
#endif

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m,n,k) ((k)*((m)+(n))*sizeof(double))

LOOP_DECL(201, OUTER_LOOP_ATTR)
{
  // Work out values for M, K and N to fit within problem size limit
  uint64_t M = 0;  // multiple of SVLh
  uint64_t N = 0;  // multiple of SVLs
  uint64_t K = 0;  // even

  // For this loop, K should equal to M, N must be equal to M/2
  const uint64_t M_base = MAX_VL / 16;
  while (true) {
    uint64_t m = M + M_base;
    uint64_t n = m / 2;
    uint64_t k = m;
    if (PROBLEM_SIZE_ACTUAL(m,n,k) <= PROBLEM_SIZE_LIMIT_KIB*1024) {
      M = m;
      N = n;
      K = k;
    } else {
      break;
    }
  }

  struct loop_201_data data = { .m = M, .n = N, .k = K, };
  ALLOC_64B(data.a, M * K, "A matrix");
  ALLOC_64B(data.b, K * N, "B matrix");
  ALLOC_64B(data.c, M * N, "C matrix");

  fill_double(data.a, M * K);
  fill_double(data.b, K * N);

  inner_loops_201(iters, &data);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", K = %" PRIu64 ", N = %" PRIu64 "\n", M, K, N);
  printf("\t%" PRIu64 " x %" PRIu64 " * %" PRIu64 " x %" PRIu64 "\n", M, K, K, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M,N,K)/1024.0f);
#endif

  int checksum = 0;
#define CHECK(x, y)                                                 \
  {                                                                 \
    double d = 0.0;                                                 \
    for (int k = 0; k < K; k++)                                     \
      d += data.a[k * M + (x)] * data.b[k * N + (y)];               \
    checksum += (int)!check_double(d, data.c[(x) * N + (y)], 1e-6); \
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
  FINALISE_LOOP_I(201, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
