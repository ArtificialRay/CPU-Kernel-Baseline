/*----------------------------------------------------------------------------
#
#   Loop 205: INT8-INT32 matrix-matrix multiply using MOPA / DOT
#
#   Purpose:
#     Use of i8 to i32 MOPA (or DOT) instructions.
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
    N: multiple of SVLh
    K: multiple of 8

  Note: A and B matrices are considered to be re-arranged,
        as required by the INT8 -> INT32 matrix multiplication.
*/

struct loop_205_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  uint8_t *restrict a;
  uint8_t *restrict b;
  uint32_t *restrict c;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_205(struct loop_205_data *restrict data) {
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

#if !defined(HAVE_CANDIDATE)


#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_205(struct loop_205_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  uint8_t *restrict a = data->a;
  uint8_t *restrict b = data->b;
  uint32_t *restrict c = data->c;
  for (uint64_t x = 0; x < m; x++) {
    for (uint64_t y = 0; y < n; y++) {
      uint32_t d = 0;
      for (uint64_t z = 0; z < k; z += 4) {
        uint64_t i = z * m + 4 * x, j = z * n + 4 * y;
        d += (uint32_t)a[i + 0] * (uint32_t)b[j + 0];
        d += (uint32_t)a[i + 1] * (uint32_t)b[j + 1];
        d += (uint32_t)a[i + 2] * (uint32_t)b[j + 2];
        d += (uint32_t)a[i + 3] * (uint32_t)b[j + 3];
      }
      c[x * n + y] = d;
    }
  }
}

#elif defined(HAVE_SME_INTRINSICS)

static void inner_loop_205(struct loop_205_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  uint8_t *a = data->a;
  uint8_t *b = data->b;
  uint32_t *c = data->c;

  uint8_t *ptr_a, *ptr_b;
  uint32_t *ptr_c;
  uint8_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx, l_idx;
  uint64_t svl_s = svcntw();
  uint64_t l_cnd = svl_s * 4;
  uint64_t c_blk = svl_s * n;
  uint64_t c_off = c_blk + n;

  svcount_t c_all = svptrue_c8();
  svbool_t p_all = svptrue_b8();

  svuint8x4_t vec_c0, vec_c1;
  svuint8x2_t vec_a0, vec_a1, vec_b0, vec_b1;

#define MOPA_TILE(t, x, i, j) \
  svmopa_za32_u8_m(t, p_all, p_all, svget2(vec_a##x, i), svget2(vec_b##x, j))

#define EXTR(x, i) svreinterpret_u32(svget4(vec_c##x, i))
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

      ptr_a = &a[m_idx << 2];
      ptr_b = &b[n_idx << 2];
      while (ptr_a < cnd_k) {
        vec_a0 = svld1_x2(c_all, &ptr_a[0]);
        vec_b0 = svld1_x2(c_all, &ptr_b[0]);
        vec_a1 = svld1_x2(c_all, &ptr_a[4 * m]);
        vec_b1 = svld1_x2(c_all, &ptr_b[4 * n]);

        MOPA_TILE(0, 0, 0, 0);
        MOPA_TILE(1, 0, 0, 1);
        MOPA_TILE(2, 0, 1, 0);
        MOPA_TILE(3, 0, 1, 1);
        MOPA_TILE(0, 1, 0, 0);
        MOPA_TILE(1, 1, 0, 1);
        MOPA_TILE(2, 1, 1, 0);
        MOPA_TILE(3, 1, 1, 1);

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
        STORE_PAIR(1, 0, 1, n);
        STORE_PAIR(0, 2, 3, c_blk);
        STORE_PAIR(1, 2, 3, c_off);

        ptr_c += n * 2;
      }
    }
    c += c_blk * 2;
  }
}

#elif defined(HAVE_SVE_INTRINSICS)

static void inner_loop_205(struct loop_205_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  uint8_t *a = data->a;
  uint8_t *b = data->b;
  uint32_t *c = data->c;

  uint32_t *ptr_c;
  uint8_t *ptr_a, *ptr_b;
  uint8_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx;
  svbool_t p_all = svptrue_b8();

  svuint32x2_t acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7;
  svuint8x2_t lda_0, lda_1;
  svuint8x2_t ldb_0, ldb_1;

#define ZERO svdup_u32(0)
#define ZERO_PAIR(y) acc_##y = svcreate2(ZERO, ZERO)

#define LOADA(y, z) svld1rq(p_all, &ptr_a[z * m * 4 + y * 16])
#define LOADA_PAIR(z) svcreate2(LOADA(0, z), LOADA(1, z))

#define GETA(y, z) svget2(lda_##z, y / 4)
#define GETB(x, z) svget2(ldb_##z, x)
#define GETC(x, y) svget2(acc_##y, x)

#define DOT(x, y, z) svdot_lane(GETC(x, y), GETB(x, z), GETA(y, z), y % 4)
#define DOT_PAIR(y, z) acc_##y = svcreate2(DOT(0, y, z), DOT(1, y, z));
#define DOT_GROUP(y) DOT_PAIR(y, 0) DOT_PAIR(y, 1)

#if defined(__ARM_FEATURE_SVE2p1)
  svcount_t c_all = svptrue_c8();
#define LOADB_PAIR(y) svld1_x2(c_all, &ptr_b[n * y * 4]);
#define STORE_PAIR(y) svst1(c_all, &ptr_c[n * y], acc_##y)
#else
#define LOADB(y, p) svld1_vnum(p_all, &ptr_b[n * y * 4], p)
#define LOADB_PAIR(y) svcreate2(LOADB(y, 0), LOADB(y, 1))
#define STORE(y, p) svst1_vnum(p_all, &ptr_c[n * y], p, GETC(p, y));
#define STORE_PAIR(y) STORE(y, 0) STORE(y, 1)
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

      ptr_a = &a[m_idx * 4];
      ptr_b = &b[n_idx * 4];
      while (ptr_a < cnd_k) {
        lda_0 = LOADA_PAIR(0);
        lda_1 = LOADA_PAIR(1);
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

#elif defined(__ARM_FEATURE_SME2)

static void inner_loop_205(struct loop_205_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;
  register uint64_t nx4 = (uint64_t)data->n * 4;
  register uint64_t mx4 = (uint64_t)data->m * 4;

  register uint64_t svl_s;
  asm volatile("cntw %[v]" : [v] "=&r"(svl_s)::);
  register uint64_t l_cnd = svl_s * 4 - 8;
  register uint64_t a_cnd = (uint64_t) & (data->a[m * k]);
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
      "   ld1b    { z2.b-z3.b }, pn8/z, [%[a_ptr]]                          \n"
#if !defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                      \n"
#endif
      "   ld1b    { z0.b-z1.b }, pn8/z, [%[b_ptr]]                          \n"
      "   umopa   za0.s, p0/m, p0/m, z2.b, z0.b                             \n"
      "   umopa   za1.s, p0/m, p0/m, z2.b, z1.b                             \n"
      "   ld1b    { z6.b-z7.b }, pn8/z, [%[a_ptr], %[mx4]]                  \n"
      "   ld1b    { z4.b-z5.b }, pn8/z, [%[b_ptr], %[nx4]]                  \n"
      "   umopa   za2.s, p0/m, p0/m, z3.b, z0.b                             \n"
      "   umopa   za3.s, p0/m, p0/m, z3.b, z1.b                             \n"
      "   add     %[a_ptr], %[a_ptr], %[mx4], lsl #1                        \n"
      "   add     %[b_ptr], %[b_ptr], %[nx4], lsl #1                        \n"
      "3:                                                                   \n"
      "   umopa   za0.s, p0/m, p0/m, z6.b, z4.b                             \n"
      "   umopa   za1.s, p0/m, p0/m, z6.b, z5.b                             \n"
      "   ld1b    { z2.b-z3.b }, pn8/z, [%[a_ptr]]                          \n"
      "   ld1b    { z0.b-z1.b }, pn8/z, [%[b_ptr]]                          \n"
      "   umopa   za2.s, p0/m, p0/m, z7.b, z4.b                             \n"
      "   umopa   za3.s, p0/m, p0/m, z7.b, z5.b                             \n"
      "   ld1b    { z6.b-z7.b }, pn8/z, [%[a_ptr], %[mx4]]                  \n"
      "   ld1b    { z4.b-z5.b }, pn8/z, [%[b_ptr], %[nx4]]                  \n"
      "   umopa   za0.s, p0/m, p0/m, z2.b, z0.b                             \n"
      "   umopa   za1.s, p0/m, p0/m, z2.b, z1.b                             \n"
      "   add     %[a_ptr], %[a_ptr], %[mx4], lsl #1                        \n"
      "   add     %[b_ptr], %[b_ptr], %[nx4], lsl #1                        \n"
      "   umopa   za2.s, p0/m, p0/m, z3.b, z0.b                             \n"
      "   umopa   za3.s, p0/m, p0/m, z3.b, z1.b                             \n"
      "   cmp     %[a_ptr], %[a_cnd]                                        \n"
      "   b.lt    3b                                                        \n"
      "   umopa   za0.s, p0/m, p0/m, z6.b, z4.b                             \n"
      "   umopa   za1.s, p0/m, p0/m, z6.b, z5.b                             \n"
      "   umopa   za2.s, p0/m, p0/m, z7.b, z4.b                             \n"
      "   umopa   za3.s, p0/m, p0/m, z7.b, z5.b                             \n"

      // Store loop
      "   mov     x12, #0                                                   \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   { z0.b-z3.b }, za0h.b[w12, 0:3]                           \n"
#else
      "   mova    { z0.b-z3.b }, za0h.b[w12, 0:3]                           \n"
#endif
      "   st1w    { z0.s-z1.s }, pn8, [%[c_ptr]]                            \n"
      "   st1w    { z2.s-z3.s }, pn8, [%[c_ptr], %[c_blk], lsl #2]          \n"

      "5:                                                                   \n"
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
      "   b.mi    5b                                                        \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   movaz   { z0.b-z3.b }, za0h.b[w12, 4:7]                           \n"
#else
      "   mova    { z0.b-z3.b }, za0h.b[w12, 4:7]                           \n"
#endif
      "   st1w    { z0.s-z1.s }, pn8, [%[c_ptr], %[n], lsl #2]              \n"
      "   st1w    { z2.s-z3.s }, pn8, [%[c_ptr], %[c_off], lsl #2]          \n"

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
      : "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
        "p11", "p12", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "x12",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2p1)

static void inner_loop_205(struct loop_205_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t svl_s;
  asm volatile("cntw %[v]" : [v] "=&r"(svl_s)::);

  register uint64_t n_idx;
  register uint64_t m_idx;
  register uint64_t a2off = 4 * m;
  register uint64_t a3off = 4 * m + 16;
  register uint64_t b1off = n * 4;
  register uint64_t c2off = n * 2;
  register uint64_t c3off = n * 3;
  register uint64_t a_cnd = a + m * k;
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
      "   ld1rqb  {z0.b}, p0/z, [%[a_ptr]]                          \n"
      "   ld1rqb  {z1.b}, p0/z, [%[a_ptr], %[a1off]]                \n"
      "   ld1rqb  {z2.b}, p0/z, [%[a_ptr], %[a2off]]                \n"
      "   ld1rqb  {z3.b}, p0/z, [%[a_ptr], %[a3off]]                \n"
      "   ld1b    {z4.b-z5.b}, pn8/z, [%[b_ptr]]                    \n"
      "   ld1b    {z6.b-z7.b}, pn8/z, [%[b_ptr], %[b1off]]          \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                  \n"
      "   udot    z10.s, z4.b, z0.b[0]                              \n"
      "   udot    z12.s, z4.b, z0.b[1]                              \n"
      "   udot    z14.s, z4.b, z0.b[2]                              \n"
      "   udot    z16.s, z4.b, z0.b[3]                              \n"
      "   udot    z11.s, z5.b, z0.b[0]                              \n"
      "   udot    z13.s, z5.b, z0.b[1]                              \n"
      "   udot    z15.s, z5.b, z0.b[2]                              \n"
      "   udot    z17.s, z5.b, z0.b[3]                              \n"
      "   udot    z20.s, z4.b, z1.b[0]                              \n"
      "   udot    z22.s, z4.b, z1.b[1]                              \n"
      "   udot    z24.s, z4.b, z1.b[2]                              \n"
      "   udot    z26.s, z4.b, z1.b[3]                              \n"
      "   udot    z21.s, z5.b, z1.b[0]                              \n"
      "   udot    z23.s, z5.b, z1.b[1]                              \n"
      "   udot    z25.s, z5.b, z1.b[2]                              \n"
      "   udot    z27.s, z5.b, z1.b[3]                              \n"
      "   udot    z10.s, z6.b, z2.b[0]                              \n"
      "   udot    z12.s, z6.b, z2.b[1]                              \n"
      "   udot    z14.s, z6.b, z2.b[2]                              \n"
      "   udot    z16.s, z6.b, z2.b[3]                              \n"
      "   udot    z11.s, z7.b, z2.b[0]                              \n"
      "   udot    z13.s, z7.b, z2.b[1]                              \n"
      "   udot    z15.s, z7.b, z2.b[2]                              \n"
      "   udot    z17.s, z7.b, z2.b[3]                              \n"
      "   udot    z20.s, z6.b, z3.b[0]                              \n"
      "   udot    z22.s, z6.b, z3.b[1]                              \n"
      "   udot    z24.s, z6.b, z3.b[2]                              \n"
      "   udot    z26.s, z6.b, z3.b[3]                              \n"
      "   udot    z21.s, z7.b, z3.b[0]                              \n"
      "   udot    z23.s, z7.b, z3.b[1]                              \n"
      "   udot    z25.s, z7.b, z3.b[2]                              \n"
      "   udot    z27.s, z7.b, z3.b[3]                              \n"
      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #2              \n"
      "   st1w    {z10.s-z11.s}, pn8, [%[c_ptr]]                    \n"
      "   st1w    {z12.s-z13.s}, pn8, [%[c_ptr], %[c1off], lsl #2]  \n"
      "   st1w    {z14.s-z15.s}, pn8, [%[c_ptr], %[c2off], lsl #2]  \n"
      "   st1w    {z16.s-z17.s}, pn8, [%[c_ptr], %[c3off], lsl #2]  \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #4                  \n"
      "   st1w    {z20.s-z21.s}, pn8, [%[c_ptr]]                    \n"
      "   st1w    {z22.s-z23.s}, pn8, [%[c_ptr], %[c1off], lsl #2]  \n"
      "   st1w    {z24.s-z25.s}, pn8, [%[c_ptr], %[c2off], lsl #2]  \n"
      "   st1w    {z26.s-z27.s}, pn8, [%[c_ptr], %[c3off], lsl #2]  \n"

      // N loop tail
      "   incw    %[n_idx], all, mul #2                             \n"
      "   cmp     %[n_idx], %[n]                                    \n"
      "   b.mi    2b                                                \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #5                  \n"
      "   add     %[m_idx], %[m_idx], #8                            \n"
      "   cmp     %[m_idx], %[m]                                    \n"
      "   b.mi    1b                                                \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [c_ptr] "=&r"(c_ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [a2off] "r"(a2off), [a3off] "r"(a3off), [a1off] "i"(16),
        [c2off] "r"(c2off), [c3off] "r"(c3off), [c1off] "r"(n),
        [b1off] "r"(b1off), [a_src] "r"(a), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "z14", "z15", "z16", "z17", "z20", "z21", "z22", "z23", "z24",
        "z25", "z26", "z27", "p0", "p8", "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_205(struct loop_205_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t svl_s;
  asm volatile("cntw %[v]" : [v] "=&r"(svl_s)::);

  register uint64_t n_idx;
  register uint64_t m_idx;
  register uint64_t a2off = 4 * m;
  register uint64_t a3off = 4 * m + 16;
  register uint64_t b2off = n * 4;
  register uint64_t b3off = n * 4 + svl_s * 4;
  register uint64_t c2off = n * 2;
  register uint64_t c3off = n * 3;
  register uint64_t a_cnd = a + m * k;
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
      "   ld1rqb  {z0.b}, p0/z, [%[a_ptr]]                          \n"
      "   ld1rqb  {z1.b}, p0/z, [%[a_ptr], %[a1off]]                \n"
      "   ld1rqb  {z2.b}, p0/z, [%[a_ptr], %[a2off]]                \n"
      "   ld1rqb  {z3.b}, p0/z, [%[a_ptr], %[a3off]]                \n"
      "   ld1b    {z4.b}, p0/z, [%[b_ptr]]                          \n"
      "   ld1b    {z5.b}, p0/z, [%[b_ptr], #1, mul vl]              \n"
      "   ld1b    {z6.b}, p0/z, [%[b_ptr], %[b2off]]                \n"
      "   ld1b    {z7.b}, p0/z, [%[b_ptr], %[b3off]]                \n"
      "   udot    z10.s, z4.b, z0.b[0]                              \n"
      "   udot    z12.s, z4.b, z0.b[1]                              \n"
      "   udot    z14.s, z4.b, z0.b[2]                              \n"
      "   udot    z16.s, z4.b, z0.b[3]                              \n"
      "   udot    z11.s, z5.b, z0.b[0]                              \n"
      "   udot    z13.s, z5.b, z0.b[1]                              \n"
      "   udot    z15.s, z5.b, z0.b[2]                              \n"
      "   udot    z17.s, z5.b, z0.b[3]                              \n"
      "   udot    z20.s, z4.b, z1.b[0]                              \n"
      "   udot    z22.s, z4.b, z1.b[1]                              \n"
      "   udot    z24.s, z4.b, z1.b[2]                              \n"
      "   udot    z26.s, z4.b, z1.b[3]                              \n"
      "   udot    z21.s, z5.b, z1.b[0]                              \n"
      "   udot    z23.s, z5.b, z1.b[1]                              \n"
      "   udot    z25.s, z5.b, z1.b[2]                              \n"
      "   udot    z27.s, z5.b, z1.b[3]                              \n"
      "   udot    z10.s, z6.b, z2.b[0]                              \n"
      "   udot    z12.s, z6.b, z2.b[1]                              \n"
      "   udot    z14.s, z6.b, z2.b[2]                              \n"
      "   udot    z16.s, z6.b, z2.b[3]                              \n"
      "   udot    z11.s, z7.b, z2.b[0]                              \n"
      "   udot    z13.s, z7.b, z2.b[1]                              \n"
      "   udot    z15.s, z7.b, z2.b[2]                              \n"
      "   udot    z17.s, z7.b, z2.b[3]                              \n"
      "   udot    z20.s, z6.b, z3.b[0]                              \n"
      "   udot    z22.s, z6.b, z3.b[1]                              \n"
      "   udot    z24.s, z6.b, z3.b[2]                              \n"
      "   udot    z26.s, z6.b, z3.b[3]                              \n"
      "   udot    z21.s, z7.b, z3.b[0]                              \n"
      "   udot    z23.s, z7.b, z3.b[1]                              \n"
      "   udot    z25.s, z7.b, z3.b[2]                              \n"
      "   udot    z27.s, z7.b, z3.b[3]                              \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #3                  \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #3                  \n"
      "   cmp     %[a_ptr], %[a_cnd]                                \n"
      "   b.mi    3b                                                \n"

      // Store
      "   add     %[c0ptr], %[c_dst], %[n_idx], lsl #2              \n"
      "   addvl   %[c1ptr], %[c0ptr], #1                            \n"
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
      "   st1w    {z20.s}, p0, [%[c0ptr]]                           \n"
      "   st1w    {z21.s}, p0, [%[c1ptr]]                           \n"
      "   st1w    {z22.s}, p0, [%[c0ptr], %[c1off], lsl #2]         \n"
      "   st1w    {z23.s}, p0, [%[c1ptr], %[c1off], lsl #2]         \n"
      "   st1w    {z24.s}, p0, [%[c0ptr], %[c2off], lsl #2]         \n"
      "   st1w    {z25.s}, p0, [%[c1ptr], %[c2off], lsl #2]         \n"
      "   st1w    {z26.s}, p0, [%[c0ptr], %[c3off], lsl #2]         \n"
      "   st1w    {z27.s}, p0, [%[c1ptr], %[c3off], lsl #2]         \n"

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
        [a2off] "r"(a2off), [a3off] "r"(a3off), [a1off] "i"(16),
        [c2off] "r"(c2off), [c3off] "r"(c3off), [c1off] "r"(n),
        [b2off] "r"(b2off), [b3off] "r"(b3off), [a_src] "r"(a), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z10", "z11", "z12",
        "z13", "z14", "z15", "z16", "z17", "z20", "z21", "z22", "z23", "z24",
        "z25", "z26", "z27", "p0", "cc", "memory");
}

#elif (defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_FML))

static void inner_loop_205(struct loop_205_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t a_cnd = (uint64_t)&data->a[m * k - 1];
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
      "   movi    v18.4s, #0                                  \n"
      "   movi    v19.4s, #0                                  \n"
      "   movi    v20.4s, #0                                  \n"
      "   movi    v21.4s, #0                                  \n"
      "   movi    v22.4s, #0                                  \n"
      "   movi    v23.4s, #0                                  \n"
      "   movi    v24.4s, #0                                  \n"
      "   movi    v25.4s, #0                                  \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #2        \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #2        \n"
      "3:                                                     \n"
      "   ld1     {v0.16b,v1.16b}, [%[a_ptr]], %[m_inc]         \n"
      "   ld1     {v2.16b,v3.16b}, [%[a_ptr]], %[m_inc]         \n"
      "   ld1     {v4.16b,v5.16b}, [%[b_ptr]], %[n_inc]         \n"
      "   ld1     {v6.16b,v7.16b}, [%[b_ptr]], %[n_inc]         \n"

      "   udot    v10.4s, v4.16b, v0.4b[0]                    \n"
      "   udot    v12.4s, v4.16b, v0.4b[1]                    \n"
      "   udot    v14.4s, v4.16b, v0.4b[2]                    \n"
      "   udot    v16.4s, v4.16b, v0.4b[3]                    \n"
      "   udot    v18.4s, v4.16b, v1.4b[0]                    \n"
      "   udot    v20.4s, v4.16b, v1.4b[1]                    \n"
      "   udot    v22.4s, v4.16b, v1.4b[2]                    \n"
      "   udot    v24.4s, v4.16b, v1.4b[3]                    \n"
      "   udot    v10.4s, v6.16b, v2.4b[0]                    \n"
      "   udot    v12.4s, v6.16b, v2.4b[1]                    \n"
      "   udot    v14.4s, v6.16b, v2.4b[2]                    \n"
      "   udot    v16.4s, v6.16b, v2.4b[3]                    \n"
      "   udot    v18.4s, v6.16b, v3.4b[0]                    \n"
      "   udot    v20.4s, v6.16b, v3.4b[1]                    \n"
      "   udot    v22.4s, v6.16b, v3.4b[2]                    \n"
      "   udot    v24.4s, v6.16b, v3.4b[3]                    \n"

      "   udot    v11.4s, v5.16b, v0.4b[0]                    \n"
      "   udot    v13.4s, v5.16b, v0.4b[1]                    \n"
      "   udot    v15.4s, v5.16b, v0.4b[2]                    \n"
      "   udot    v17.4s, v5.16b, v0.4b[3]                    \n"
      "   udot    v19.4s, v5.16b, v1.4b[0]                    \n"
      "   udot    v21.4s, v5.16b, v1.4b[1]                    \n"
      "   udot    v23.4s, v5.16b, v1.4b[2]                    \n"
      "   udot    v25.4s, v5.16b, v1.4b[3]                    \n"
      "   udot    v11.4s, v7.16b, v2.4b[0]                    \n"
      "   udot    v13.4s, v7.16b, v2.4b[1]                    \n"
      "   udot    v15.4s, v7.16b, v2.4b[2]                    \n"
      "   udot    v17.4s, v7.16b, v2.4b[3]                    \n"
      "   udot    v19.4s, v7.16b, v3.4b[0]                    \n"
      "   udot    v21.4s, v7.16b, v3.4b[1]                    \n"
      "   udot    v23.4s, v7.16b, v3.4b[2]                    \n"
      "   udot    v25.4s, v7.16b, v3.4b[3]                    \n"

      "   cmp     %[a_ptr], %[a_cnd]                          \n"
      "   b.mi    3b                                          \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #2        \n"
      "   st1     {v10.4s,v11.4s}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v12.4s,v13.4s}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v14.4s,v15.4s}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v16.4s,v17.4s}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v18.4s,v19.4s}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v20.4s,v21.4s}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v22.4s,v23.4s}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v24.4s,v25.4s}, [%[c_ptr]], %[n_inc]       \n"

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
}

#else

static void inner_loop_205(struct loop_205_data *data) {
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
// Default of 64KiB equates to original problem size (M=128, K=256, N=128)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef  PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 64
#endif
#endif /* !HAVE_CANDIDATE */

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m,n,k) ((k)*((m)+(n))*sizeof(uint8_t))

LOOP_DECL(205, OUTER_LOOP_ATTR)
{
  // Work out values for M, K and N to fit within problem size limit
  uint64_t M = 0;  // multiple of SVLh
  uint64_t N = 0;  // multiple of SVLh
  uint64_t K = 0;  // multiple of 8 (implicit)

  // Work out values for M and N to fit within problem size limit
  // Implicitly converts between KiB and byte-elements
  const uint64_t M_base = MAX_VL / 16;
  while (true) {
    uint64_t m = M + M_base;
    uint64_t n = m;
    uint64_t k = m * 2;
    if (PROBLEM_SIZE_ACTUAL(m,n,k) <= PROBLEM_SIZE_LIMIT_KIB*1024) {
      M = m;
      N = n;
      K = k;
    } else {
      break;
    }
  }

  struct loop_205_data data = { .m = M, .n = N, .k = K, };
  ALLOC_64B(data.a, M * K, "A matrix");
  ALLOC_64B(data.b, K * N, "B matrix");
  ALLOC_64B(data.c, M * N, "C matrix");

  fill_uint8(data.a, M * K);
  fill_uint8(data.b, K * N);

  inner_loops_205(iters, &data);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", K = %" PRIu64 ", N = %" PRIu64 "\n", M, K, N);
  printf("\t%" PRIu64 " x %" PRIu64 " * %" PRIu64 " x %" PRIu64 "\n", M, K, K, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M,N,K)/1024.0f);
#endif

  int checksum = 0;
#define CHECK(x, y)                                             \
  {                                                             \
    uint32_t d = 0;                                             \
    for (int k = 0; k < K; k += 4) {                            \
      int i = k * M + 4 * (x), j = k * N + 4 * (y);             \
      d += (uint32_t)data.a[i + 0] * (uint32_t)data.b[j + 0];   \
      d += (uint32_t)data.a[i + 1] * (uint32_t)data.b[j + 1];   \
      d += (uint32_t)data.a[i + 2] * (uint32_t)data.b[j + 2];   \
      d += (uint32_t)data.a[i + 3] * (uint32_t)data.b[j + 3];   \
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
  FINALISE_LOOP_I(205, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
