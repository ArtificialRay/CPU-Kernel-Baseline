/*----------------------------------------------------------------------------
#
#   Loop 204: FP16 matrix-matrix multiply using MOPA / DOT
#
#   Purpose:
#     Use of fp16 to fp16 MOPA (or MLA) instructions.
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
    N: multiple of SVLb
    K: even
*/

struct loop_204_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  float16_t *restrict a;
  float16_t *restrict b;
  float16_t *restrict c;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_204(struct loop_204_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#define LOOP_ATTR SC_SVE_ATTR
#define OUTER_LOOP_ATTR SC_SVE_LOOP_ATTR
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#elif defined(__ARM_FEATURE_SME2p1)
#define LOOP_ATTR SME_ZA_ATTR
#define OUTER_LOOP_ATTR S_LOOP_ATTR
#elif defined(__ARM_FEATURE_SVE2)
#define LOOP_ATTR SC_SVE_ATTR
#define OUTER_LOOP_ATTR SC_SVE_LOOP_ATTR
#elif defined(__ARM_NEON)
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#else
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
#endif

#define FP16_MUL(u, v) (fp16_to_native(u) * fp16_to_native(v))

#if !defined(HAVE_CANDIDATE)

#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void inner_loop_204(struct loop_204_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  float16_t *a = data->a;
  float16_t *b = data->b;
  float16_t *c = data->c;

  for (uint64_t x = 0; x < m; x++)
    for (uint64_t y = 0; y < n; y++)
      c[x * n + y] = native_to_fp16(0.0f);

  for (uint64_t z = 0; z < k; z++)
    for (uint64_t x = 0; x < m; x++)
      for (uint64_t y = 0; y < n; y++) {
        FLOAT16_t d = fp16_to_native(c[x * n + y]);
        FLOAT16_t e = FP16_MUL(a[z * m + x], b[z * n + y]);
        c[x * n + y] = native_to_fp16(d + e);
      }
}

#elif (defined(HAVE_SME_INTRINSICS) && defined(__ARM_FEATURE_SME2p1))


static void inner_loop_204(struct loop_204_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  float16_t *a = data->a;
  float16_t *b = data->b;
  float16_t *c = data->c;

  float16_t *ptr_a, *ptr_b, *ptr_c;
  float16_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx, l_idx;
  uint64_t svl_h = svcnth();
  uint64_t l_cnd = svl_h * 2;
  uint64_t c_blk = svl_h * n;

  svcount_t c_all = svptrue_c16();
  svbool_t  p_all = svptrue_b16();

  svfloat16_t   vec_a0, vec_a1;
  svfloat16x2_t vec_b0, vec_b1;
  svuint8x4_t   vec_c0, vec_c1;

#define MOPA_TILE(t, x, i) \
  svmopa_za16_m(t, p_all, p_all, vec_a##x, svget2(vec_b##x, i))

#define EXTR(x, i) svreinterpret_f16(svget4(vec_c##x, i))
#define STORE_PAIR(x, i, j, y) \
  svst1(c_all, &ptr_c[n * (y)], svcreate2(EXTR(x, i), EXTR(x, j)))
#if defined(__ARM_FEATURE_SME2p1)
  svzero_za();
#endif

  for (m_idx = 0; m_idx < m; m_idx += svl_h) {
    for (n_idx = 0; n_idx < n; n_idx += svl_h * 2) {
#if !defined(__ARM_FEATURE_SME2p1)
      svzero_za();
#endif

      ptr_a = &a[m_idx];
      ptr_b = &b[n_idx];
      while (ptr_a < cnd_k) {
        vec_a0 = svld1(p_all, &ptr_a[0]);
        vec_a1 = svld1(p_all, &ptr_a[m]);
        vec_b0 = svld1_x2(c_all, &ptr_b[0]);
        vec_b1 = svld1_x2(c_all, &ptr_b[n]);

        MOPA_TILE(0, 0, 0);
        MOPA_TILE(1, 0, 1);
        MOPA_TILE(0, 1, 0);
        MOPA_TILE(1, 1, 1);

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
        STORE_PAIR(0, 2, 3, 1);
        STORE_PAIR(1, 0, 1, 2);
        STORE_PAIR(1, 2, 3, 3);

        ptr_c += n * 4;
      }
    }
    c += c_blk;
  }
}

#elif (defined(HAVE_SVE_INTRINSICS) && defined(__ARM_FEATURE_SVE2p1))

static void inner_loop_204(struct loop_204_data *data)
LOOP_ATTR
{
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  float16_t *a = data->a;
  float16_t *b = data->b;
  float16_t *c = data->c;

  float16_t *ptr_a, *ptr_b, *ptr_c;
  float16_t *cnd_k = &a[m * k];

  uint64_t m_idx, n_idx;
  svbool_t  p_all = svptrue_b16();
  svcount_t c_all = svptrue_c16();

  svfloat16x2_t acc_0, acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, acc_7;
  svfloat16_t   lda_0, lda_1;
  svfloat16x2_t ldb_0, ldb_1;

#define ZERO svdup_f16(0.0f)
#define ZERO_PAIR(y) acc_##y = svcreate2(ZERO, ZERO)

#define GETB(x, z) svget2(ldb_##z, x)
#define GETC(x, y) svget2(acc_##y, x)

#define MLA(x, y, z) svmla_lane(GETC(x, y), GETB(x, z), lda_##z, y)
#define MLA_PAIR(y, z) acc_##y = svcreate2(MLA(0, y, z), MLA(1, y, z));
#define MLA_GROUP(y) MLA_PAIR(y, 0) MLA_PAIR(y, 1)

#define STORE_PAIR(y) svst1(c_all, &ptr_c[n * y], acc_##y)

  for (m_idx = 0; m_idx < m; m_idx += 8) {
    for (n_idx = 0; n_idx < n; n_idx += svcnth() * 2) {
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
        lda_0 = svld1rq(p_all, &ptr_a[0]);
        lda_1 = svld1rq(p_all, &ptr_a[m]);
        ldb_0 = svld1_x2(c_all, &ptr_b[0]);
        ldb_1 = svld1_x2(c_all, &ptr_b[n]);

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

#elif defined(__ARM_FEATURE_SME2p1)


static void inner_loop_204(struct loop_204_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t svl_h;
  asm volatile("cnth %[v]" : [v] "=&r"(svl_h)::);

  register uint64_t c_blk = svl_h * n;
  register uint64_t l_cnd = svl_h * 2 - 8;
  register uint64_t a_cnd = a + 2 * (m * k);
  register uint64_t off_2 = n * 2;
  register uint64_t off_3 = n * 3;

  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  // x12: slice index register for tile-to-vec mova/movaz

  asm volatile(
      "   ptrue   p0.h                                                \n"
      "   ptrue   pn8.h                                               \n"
      "   zero    {za}                                                \n"

      // M loop head
      "   mov     %[m_idx], #0                                        \n"
      "1:                                                             \n"

      // N loop head
      "   mov     %[n_idx], #0                                        \n"
      "2:                                                             \n"

      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #1                \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #1                \n"
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #1                \n"

      // K loop
      "   ld1h    {z4.h}, p0/z, [%[a_ptr]]                            \n"
      "   ld1h    {z5.h}, p0/z, [%[a_ptr], %[m], lsl #1]              \n"
      "   ld1h    {z0.h-z1.h}, pn8/z, [%[b_ptr]]                      \n"
      "   ld1h    {z2.h-z3.h}, pn8/z, [%[b_ptr], %[n], lsl #1]        \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                    \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                    \n"
      "   fmopa   za0.h, p0/m, p0/m, z4.h, z0.h                       \n"
      "   fmopa   za1.h, p0/m, p0/m, z4.h, z1.h                       \n"
      "3:                                                             \n"
      "   ld1h    {z4.h}, p0/z, [%[a_ptr]]                            \n"
      "   ld1h    {z0.h-z1.h}, pn8/z, [%[b_ptr]]                      \n"
      "   fmopa   za0.h, p0/m, p0/m, z5.h, z2.h                       \n"
      "   fmopa   za1.h, p0/m, p0/m, z5.h, z3.h                       \n"
      "   ld1h    {z5.h}, p0/z, [%[a_ptr], %[m], lsl #1]              \n"
      "   ld1h    {z2.h-z3.h}, pn8/z, [%[b_ptr], %[n], lsl #1]        \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                    \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                    \n"
      "   fmopa   za0.h, p0/m, p0/m, z4.h, z0.h                       \n"
      "   fmopa   za1.h, p0/m, p0/m, z4.h, z1.h                       \n"
      "   cmp     %[a_ptr], %[a_cnd]                                  \n"
      "   b.mi    3b                                                  \n"
      "   fmopa   za0.h, p0/m, p0/m, z5.h, z2.h                       \n"
      "   fmopa   za1.h, p0/m, p0/m, z5.h, z3.h                       \n"

      // Store loop
      "   mov     x12, #0                                             \n"
      "   movaz   {z0.b-z3.b}, za0h.b[w12, 0:3]                       \n"
      "   movaz   {z4.b-z7.b}, za0h.b[w12, 4:7]                       \n"
      "4:                                                             \n"
      "   st1h    {z0.h-z1.h}, pn8, [%[c_ptr]]                        \n"
      "   st1h    {z2.h-z3.h}, pn8, [%[c_ptr], %[off_1], lsl #1]      \n"
      "   st1h    {z4.h-z5.h}, pn8, [%[c_ptr], %[off_2], lsl #1]      \n"
      "   st1h    {z6.h-z7.h}, pn8, [%[c_ptr], %[off_3], lsl #1]      \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #3                    \n"
      "   add     x12, x12, #8                                        \n"
      "   movaz   {z0.b-z3.b}, za0h.b[w12, 0:3]                       \n"
      "   movaz   {z4.b-z7.b}, za0h.b[w12, 4:7]                       \n"
      "   cmp     x12, %[l_cnd]                                       \n"
      "   b.mi    4b                                                  \n"
      "   st1h    {z0.h-z1.h}, pn8, [%[c_ptr]]                        \n"
      "   st1h    {z2.h-z3.h}, pn8, [%[c_ptr], %[off_1], lsl #1]      \n"
      "   st1h    {z4.h-z5.h}, pn8, [%[c_ptr], %[off_2], lsl #1]      \n"
      "   st1h    {z6.h-z7.h}, pn8, [%[c_ptr], %[off_3], lsl #1]      \n"

      // N loop tail
      "   inch    %[n_idx], all, mul #2                               \n"
      "   cmp     %[n_idx], %[n]                                      \n"
      "   b.mi    2b                                                  \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[c_blk], lsl #1                \n"
      "   inch    %[m_idx]                                            \n"
      "   cmp     %[m_idx], %[m]                                      \n"
      "   b.mi    1b                                                  \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [c_ptr] "=&r"(c_ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [l_cnd] "r"(l_cnd),
        [a_cnd] "r"(a_cnd), [c_blk] "r"(c_blk), [off_2] "r"(off_2),
        [off_3] "r"(off_3), [off_1] "r"(n), [a_src] "r"(a), [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "p0", "p8", "x12",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2p1)

static void inner_loop_204(struct loop_204_data *data)
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
  register uint64_t a_cnd = a + 2 * (m * k);

  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  register uint64_t n_idx;
  register uint64_t m_idx;

  asm volatile(
      "   ptrue   p0.h                                                \n"
      "   ptrue   pn8.h                                               \n"

      // M loop head
      "   mov     %[m_idx], #0                                        \n"
      "1:                                                             \n"

      // N loop head
      "   mov     %[n_idx], #0                                        \n"
      "2:                                                             \n"

      // Accumulators
      "   mov     z10.h, #0                                           \n"
      "   mov     z11.h, #0                                           \n"
      "   mov     z12.h, #0                                           \n"
      "   mov     z13.h, #0                                           \n"
      "   mov     z14.h, #0                                           \n"
      "   mov     z15.h, #0                                           \n"
      "   mov     z16.h, #0                                           \n"
      "   mov     z17.h, #0                                           \n"
      "   mov     z20.h, #0                                           \n"
      "   mov     z21.h, #0                                           \n"
      "   mov     z22.h, #0                                           \n"
      "   mov     z23.h, #0                                           \n"
      "   mov     z24.h, #0                                           \n"
      "   mov     z25.h, #0                                           \n"
      "   mov     z26.h, #0                                           \n"
      "   mov     z27.h, #0                                           \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #1                \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #1                \n"
      "3:                                                             \n"
      "   ld1rqh  {z4.h}, p0/z, [%[a_ptr]]                            \n"
      "   ld1rqh  {z5.h}, p0/z, [%[a_ptr], %[m], lsl #1]              \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                    \n"
      "   ld1h    {z0.h-z1.h}, pn8/z, [%[b_ptr]]                      \n"
      "   ld1h    {z2.h-z3.h}, pn8/z, [%[b_ptr], %[n], lsl #1]        \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                    \n"
      "   fmla    z10.h, z0.h, z4.h[0]                                \n"
      "   fmla    z12.h, z0.h, z4.h[1]                                \n"
      "   fmla    z14.h, z0.h, z4.h[2]                                \n"
      "   fmla    z16.h, z0.h, z4.h[3]                                \n"
      "   fmla    z20.h, z0.h, z4.h[4]                                \n"
      "   fmla    z22.h, z0.h, z4.h[5]                                \n"
      "   fmla    z24.h, z0.h, z4.h[6]                                \n"
      "   fmla    z26.h, z0.h, z4.h[7]                                \n"
      "   fmla    z11.h, z1.h, z4.h[0]                                \n"
      "   fmla    z13.h, z1.h, z4.h[1]                                \n"
      "   fmla    z15.h, z1.h, z4.h[2]                                \n"
      "   fmla    z17.h, z1.h, z4.h[3]                                \n"
      "   fmla    z21.h, z1.h, z4.h[4]                                \n"
      "   fmla    z23.h, z1.h, z4.h[5]                                \n"
      "   fmla    z25.h, z1.h, z4.h[6]                                \n"
      "   fmla    z27.h, z1.h, z4.h[7]                                \n"
      "   fmla    z10.h, z2.h, z5.h[0]                                \n"
      "   fmla    z12.h, z2.h, z5.h[1]                                \n"
      "   fmla    z14.h, z2.h, z5.h[2]                                \n"
      "   fmla    z16.h, z2.h, z5.h[3]                                \n"
      "   fmla    z20.h, z2.h, z5.h[4]                                \n"
      "   fmla    z22.h, z2.h, z5.h[5]                                \n"
      "   fmla    z24.h, z2.h, z5.h[6]                                \n"
      "   fmla    z26.h, z2.h, z5.h[7]                                \n"
      "   fmla    z11.h, z3.h, z5.h[0]                                \n"
      "   fmla    z13.h, z3.h, z5.h[1]                                \n"
      "   fmla    z15.h, z3.h, z5.h[2]                                \n"
      "   fmla    z17.h, z3.h, z5.h[3]                                \n"
      "   fmla    z21.h, z3.h, z5.h[4]                                \n"
      "   fmla    z23.h, z3.h, z5.h[5]                                \n"
      "   fmla    z25.h, z3.h, z5.h[6]                                \n"
      "   fmla    z27.h, z3.h, z5.h[7]                                \n"
      "   cmp     %[a_ptr], %[a_cnd]                                  \n"
      "   b.mi    3b                                                  \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #1                \n"
      "   st1h    {z10.h-z11.h}, pn8, [%[c_ptr]]                      \n"
      "   st1h    {z12.h-z13.h}, pn8, [%[c_ptr], %[off_1], lsl #1]    \n"
      "   st1h    {z14.h-z15.h}, pn8, [%[c_ptr], %[off_2], lsl #1]    \n"
      "   st1h    {z16.h-z17.h}, pn8, [%[c_ptr], %[off_3], lsl #1]    \n"
      "   add     %[c_ptr], %[c_ptr], %[n], lsl #3                    \n"
      "   st1h    {z20.h-z21.h}, pn8, [%[c_ptr]]                      \n"
      "   st1h    {z22.h-z23.h}, pn8, [%[c_ptr], %[off_1], lsl #1]    \n"
      "   st1h    {z24.h-z25.h}, pn8, [%[c_ptr], %[off_2], lsl #1]    \n"
      "   st1h    {z26.h-z27.h}, pn8, [%[c_ptr], %[off_3], lsl #1]    \n"

      // N loop tail
      "   inch    %[n_idx], all, mul #2                               \n"
      "   cmp     %[n_idx], %[n]                                      \n"
      "   b.mi    2b                                                  \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #4                    \n"
      "   add     %[m_idx], %[m_idx], #8                              \n"
      "   cmp     %[m_idx], %[m]                                      \n"
      "   b.mi    1b                                                  \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [c_ptr] "=&r"(c_ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [off_2] "r"(off_2), [off_3] "r"(off_3), [off_1] "r"(n), [a_src] "r"(a),
        [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z10", "z11", "z12", "z13", "z14",
        "z15", "z16", "z17", "z20", "z21", "z22", "z23", "z24", "z25", "z26",
        "z27", "p0", "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_204(struct loop_204_data *data)
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
  register uint64_t a_cnd = a + 2 * (m * k);

  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c0ptr;
  register uint64_t c1ptr;
  register uint64_t n_idx;
  register uint64_t m_idx;

  asm volatile(
      "   ptrue   p0.h                                                \n"

      // M loop head
      "   mov     %[m_idx], #0                                        \n"
      "1:                                                             \n"

      // N loop head
      "   mov     %[n_idx], #0                                        \n"
      "2:                                                             \n"

      // Accumulators
      "   mov     z10.h, #0                                           \n"
      "   mov     z11.h, #0                                           \n"
      "   mov     z12.h, #0                                           \n"
      "   mov     z13.h, #0                                           \n"
      "   mov     z14.h, #0                                           \n"
      "   mov     z15.h, #0                                           \n"
      "   mov     z16.h, #0                                           \n"
      "   mov     z17.h, #0                                           \n"
      "   mov     z20.h, #0                                           \n"
      "   mov     z21.h, #0                                           \n"
      "   mov     z22.h, #0                                           \n"
      "   mov     z23.h, #0                                           \n"
      "   mov     z24.h, #0                                           \n"
      "   mov     z25.h, #0                                           \n"
      "   mov     z26.h, #0                                           \n"
      "   mov     z27.h, #0                                           \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #1                \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #1                \n"
      "3:                                                             \n"
      "   ld1rqh  {z4.h}, p0/z, [%[a_ptr]]                            \n"
      "   ld1rqh  {z5.h}, p0/z, [%[a_ptr], %[m], lsl #1]              \n"
      "   add     %[a_ptr], %[a_ptr], %[m], lsl #2                    \n"
      "   ld1h    {z0.h}, p0/z, [%[b_ptr]]                            \n"
      "   ld1h    {z1.h}, p0/z, [%[b_ptr], #1, mul vl]                \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #1                    \n"
      "   ld1h    {z2.h}, p0/z, [%[b_ptr]]                            \n"
      "   ld1h    {z3.h}, p0/z, [%[b_ptr], #1, mul vl]                \n"
      "   add     %[b_ptr], %[b_ptr], %[n], lsl #1                    \n"
      "   fmla    z10.h, z0.h, z4.h[0]                                \n"
      "   fmla    z12.h, z0.h, z4.h[1]                                \n"
      "   fmla    z14.h, z0.h, z4.h[2]                                \n"
      "   fmla    z16.h, z0.h, z4.h[3]                                \n"
      "   fmla    z20.h, z0.h, z4.h[4]                                \n"
      "   fmla    z22.h, z0.h, z4.h[5]                                \n"
      "   fmla    z24.h, z0.h, z4.h[6]                                \n"
      "   fmla    z26.h, z0.h, z4.h[7]                                \n"
      "   fmla    z11.h, z1.h, z4.h[0]                                \n"
      "   fmla    z13.h, z1.h, z4.h[1]                                \n"
      "   fmla    z15.h, z1.h, z4.h[2]                                \n"
      "   fmla    z17.h, z1.h, z4.h[3]                                \n"
      "   fmla    z21.h, z1.h, z4.h[4]                                \n"
      "   fmla    z23.h, z1.h, z4.h[5]                                \n"
      "   fmla    z25.h, z1.h, z4.h[6]                                \n"
      "   fmla    z27.h, z1.h, z4.h[7]                                \n"
      "   fmla    z10.h, z2.h, z5.h[0]                                \n"
      "   fmla    z12.h, z2.h, z5.h[1]                                \n"
      "   fmla    z14.h, z2.h, z5.h[2]                                \n"
      "   fmla    z16.h, z2.h, z5.h[3]                                \n"
      "   fmla    z20.h, z2.h, z5.h[4]                                \n"
      "   fmla    z22.h, z2.h, z5.h[5]                                \n"
      "   fmla    z24.h, z2.h, z5.h[6]                                \n"
      "   fmla    z26.h, z2.h, z5.h[7]                                \n"
      "   fmla    z11.h, z3.h, z5.h[0]                                \n"
      "   fmla    z13.h, z3.h, z5.h[1]                                \n"
      "   fmla    z15.h, z3.h, z5.h[2]                                \n"
      "   fmla    z17.h, z3.h, z5.h[3]                                \n"
      "   fmla    z21.h, z3.h, z5.h[4]                                \n"
      "   fmla    z23.h, z3.h, z5.h[5]                                \n"
      "   fmla    z25.h, z3.h, z5.h[6]                                \n"
      "   fmla    z27.h, z3.h, z5.h[7]                                \n"
      "   cmp     %[a_ptr], %[a_cnd]                                  \n"
      "   b.mi    3b                                                  \n"

      // Store
      "   add     %[c0ptr], %[c_dst], %[n_idx], lsl #1                \n"
      "   addvl   %[c1ptr], %[c0ptr], #1                              \n"
      "   st1h    {z10.h}, p0, [%[c0ptr]]                             \n"
      "   st1h    {z11.h}, p0, [%[c1ptr]]                             \n"
      "   st1h    {z12.h}, p0, [%[c0ptr], %[off_1], lsl #1]           \n"
      "   st1h    {z13.h}, p0, [%[c1ptr], %[off_1], lsl #1]           \n"
      "   st1h    {z14.h}, p0, [%[c0ptr], %[off_2], lsl #1]           \n"
      "   st1h    {z15.h}, p0, [%[c1ptr], %[off_2], lsl #1]           \n"
      "   st1h    {z16.h}, p0, [%[c0ptr], %[off_3], lsl #1]           \n"
      "   st1h    {z17.h}, p0, [%[c1ptr], %[off_3], lsl #1]           \n"
      "   add     %[c0ptr], %[c0ptr], %[n], lsl #3                    \n"
      "   add     %[c1ptr], %[c1ptr], %[n], lsl #3                    \n"
      "   st1h    {z20.h}, p0, [%[c0ptr]]                             \n"
      "   st1h    {z21.h}, p0, [%[c1ptr]]                             \n"
      "   st1h    {z22.h}, p0, [%[c0ptr], %[off_1], lsl #1]           \n"
      "   st1h    {z23.h}, p0, [%[c1ptr], %[off_1], lsl #1]           \n"
      "   st1h    {z24.h}, p0, [%[c0ptr], %[off_2], lsl #1]           \n"
      "   st1h    {z25.h}, p0, [%[c1ptr], %[off_2], lsl #1]           \n"
      "   st1h    {z26.h}, p0, [%[c0ptr], %[off_3], lsl #1]           \n"
      "   st1h    {z27.h}, p0, [%[c1ptr], %[off_3], lsl #1]           \n"

      // N loop tail
      "   inch    %[n_idx], all, mul #2                               \n"
      "   cmp     %[n_idx], %[n]                                      \n"
      "   b.mi    2b                                                  \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #4                    \n"
      "   add     %[m_idx], %[m_idx], #8                              \n"
      "   cmp     %[m_idx], %[m]                                      \n"
      "   b.mi    1b                                                  \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [m_idx] "=&r"(m_idx),
        [c0ptr] "=&r"(c0ptr), [c1ptr] "=&r"(c1ptr), [n_idx] "=&r"(n_idx),
        [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [off_2] "r"(off_2), [off_3] "r"(off_3), [off_1] "r"(n), [a_src] "r"(a),
        [b_src] "r"(b)
      : "z0", "z1", "z2", "z3", "z4", "z5", "z10", "z11", "z12", "z13", "z14",
        "z15", "z16", "z17", "z20", "z21", "z22", "z23", "z24", "z25", "z26",
        "z27", "p0", "cc", "memory");
}

#elif defined(__ARM_NEON)

static void inner_loop_204(struct loop_204_data *data) {
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t)data->a;
  register uint64_t b = (uint64_t)data->b;
  register uint64_t c = (uint64_t)data->c;

  register uint64_t m_inc = m * 2;
  register uint64_t n_inc = n * 2;
  register uint64_t a_cnd = a + (m_inc * k);

  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;

  asm volatile(
      // M loop head
      "   mov     %[m_idx], #0                                \n"
      "1:                                                     \n"

      // N loop head
      "   mov     %[n_idx], #0                                \n"
      "2:                                                     \n"

      // Accumulators
      "   movi    v10.8h, #0                                  \n"
      "   movi    v11.8h, #0                                  \n"
      "   movi    v12.8h, #0                                  \n"
      "   movi    v13.8h, #0                                  \n"
      "   movi    v14.8h, #0                                  \n"
      "   movi    v15.8h, #0                                  \n"
      "   movi    v16.8h, #0                                  \n"
      "   movi    v17.8h, #0                                  \n"
      "   movi    v20.8h, #0                                  \n"
      "   movi    v21.8h, #0                                  \n"
      "   movi    v22.8h, #0                                  \n"
      "   movi    v23.8h, #0                                  \n"
      "   movi    v24.8h, #0                                  \n"
      "   movi    v25.8h, #0                                  \n"
      "   movi    v26.8h, #0                                  \n"
      "   movi    v27.8h, #0                                  \n"

      // K loop
      "   add     %[a_ptr], %[a_src], %[m_idx], lsl #1        \n"
      "   add     %[b_ptr], %[b_src], %[n_idx], lsl #1        \n"
      "3:                                                     \n"
      "   ld1     {v4.8h}, [%[a_ptr]], %[m_inc]               \n"
      "   ld1     {v5.8h}, [%[a_ptr]], %[m_inc]               \n"
      "   ld1     {v0.8h,v1.8h}, [%[b_ptr]], %[n_inc]         \n"
      "   ld1     {v2.8h,v3.8h}, [%[b_ptr]], %[n_inc]         \n"
      "   fmla    v10.8h, v0.8h, v4.h[0]                      \n"
      "   fmla    v12.8h, v0.8h, v4.h[1]                      \n"
      "   fmla    v14.8h, v0.8h, v4.h[2]                      \n"
      "   fmla    v16.8h, v0.8h, v4.h[3]                      \n"
      "   fmla    v20.8h, v0.8h, v4.h[4]                      \n"
      "   fmla    v22.8h, v0.8h, v4.h[5]                      \n"
      "   fmla    v24.8h, v0.8h, v4.h[6]                      \n"
      "   fmla    v26.8h, v0.8h, v4.h[7]                      \n"
      "   fmla    v11.8h, v1.8h, v4.h[0]                      \n"
      "   fmla    v13.8h, v1.8h, v4.h[1]                      \n"
      "   fmla    v15.8h, v1.8h, v4.h[2]                      \n"
      "   fmla    v17.8h, v1.8h, v4.h[3]                      \n"
      "   fmla    v21.8h, v1.8h, v4.h[4]                      \n"
      "   fmla    v23.8h, v1.8h, v4.h[5]                      \n"
      "   fmla    v25.8h, v1.8h, v4.h[6]                      \n"
      "   fmla    v27.8h, v1.8h, v4.h[7]                      \n"
      "   fmla    v10.8h, v2.8h, v5.h[0]                      \n"
      "   fmla    v12.8h, v2.8h, v5.h[1]                      \n"
      "   fmla    v14.8h, v2.8h, v5.h[2]                      \n"
      "   fmla    v16.8h, v2.8h, v5.h[3]                      \n"
      "   fmla    v20.8h, v2.8h, v5.h[4]                      \n"
      "   fmla    v22.8h, v2.8h, v5.h[5]                      \n"
      "   fmla    v24.8h, v2.8h, v5.h[6]                      \n"
      "   fmla    v26.8h, v2.8h, v5.h[7]                      \n"
      "   fmla    v11.8h, v3.8h, v5.h[0]                      \n"
      "   fmla    v13.8h, v3.8h, v5.h[1]                      \n"
      "   fmla    v15.8h, v3.8h, v5.h[2]                      \n"
      "   fmla    v17.8h, v3.8h, v5.h[3]                      \n"
      "   fmla    v21.8h, v3.8h, v5.h[4]                      \n"
      "   fmla    v23.8h, v3.8h, v5.h[5]                      \n"
      "   fmla    v25.8h, v3.8h, v5.h[6]                      \n"
      "   fmla    v27.8h, v3.8h, v5.h[7]                      \n"
      "   cmp     %[a_ptr], %[a_cnd]                          \n"
      "   b.mi    3b                                          \n"

      // Store
      "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #1        \n"
      "   st1     {v10.8h,v11.8h}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v12.8h,v13.8h}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v14.8h,v15.8h}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v16.8h,v17.8h}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v20.8h,v21.8h}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v22.8h,v23.8h}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v24.8h,v25.8h}, [%[c_ptr]], %[n_inc]       \n"
      "   st1     {v26.8h,v27.8h}, [%[c_ptr]], %[n_inc]       \n"

      // N loop tail
      "   add     %[n_idx], %[n_idx], #16                     \n"
      "   cmp     %[n_idx], %[n]                              \n"
      "   b.mi    2b                                          \n"

      // M loop tail
      "   add     %[c_dst], %[c_dst], %[n], lsl #4            \n"
      "   add     %[m_idx], %[m_idx], #8                      \n"
      "   cmp     %[m_idx], %[m]                              \n"
      "   b.mi    1b                                          \n"

      : [a_ptr] "=&r"(a_ptr), [b_ptr] "=&r"(b_ptr), [c_ptr] "=&r"(c_ptr),
        [m_idx] "=&r"(m_idx), [n_idx] "=&r"(n_idx), [c_dst] "+&r"(c)
      : [m] "r"(m), [n] "r"(n), [k] "r"(k), [a_cnd] "r"(a_cnd),
        [m_inc] "r"(m_inc), [n_inc] "r"(n_inc), [a_src] "r"(a), [b_src] "r"(b)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v10", "v11", "v12", "v13", "v14",
        "v15", "v16", "v17", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "cc", "memory");
}

#else

static void inner_loop_204(struct loop_204_data *data) {
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
// Default of 128KiB equates to original problem size (M=128, K=256, N=128)
#if (!defined(PROBLEM_SIZE_LIMIT_KIB) || PROBLEM_SIZE_LIMIT_KIB == 0)
#undef  PROBLEM_SIZE_LIMIT_KIB
#define PROBLEM_SIZE_LIMIT_KIB 128
#endif
#endif /* !HAVE_CANDIDATE */

// Actual input buffer memory footprint in bytes
#define PROBLEM_SIZE_ACTUAL(m, n, k) ((k) * ((m) + (n)) * sizeof(float16_t))

LOOP_DECL(204, OUTER_LOOP_ATTR)
{
  // Work out values for M, K and N to fit within problem size limit
  uint64_t M = 0;  // multiple of SVLh
  uint64_t N = 0;  // multiple of SVLb
  uint64_t K = 0;  // even

  // For this loop, N should remain as 2*M, M and K must be equal
  const uint64_t M_base = MAX_VL / 16;
  while (true) {
    uint64_t m = M + M_base;
    uint64_t n = 2 * m;
    uint64_t k = m;
    if (PROBLEM_SIZE_ACTUAL(m, n, k) <= PROBLEM_SIZE_LIMIT_KIB * 1024) {
      M = m;
      N = n;
      K = k;
    } else {
      break;
    }
  }

  struct loop_204_data data = { .m = M, .n = N, .k = K, };
  ALLOC_64B(data.a, M * K, "A matrix");
  ALLOC_64B(data.b, K * N, "B matrix");
  ALLOC_64B(data.c, M * N, "C matrix");

  fill_fp16(data.a, M * K);
  fill_fp16(data.b, K * N);

  inner_loops_204(iters, &data);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", K = %" PRIu64 ", N = %" PRIu64 "\n", M, K, N);
  printf("\t%" PRIu64 " x %" PRIu64 " * %" PRIu64 " x %" PRIu64 "\n", M, K, K, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M, N, K) / 1024.0f);
#endif

  int checksum = 0;
#define CHECK(x, y)                                             \
  {                                                             \
    FLOAT16_t c = fp16_to_native(data.c[(x) * N + (y)]);        \
    FLOAT16_t d = 0.0f;                                         \
    for (int k = 0; k < K; k++)                                 \
      d += FP16_MUL(data.a[k * M + (x)], data.b[k * N + (y)]);  \
    checksum += (int)!check_float(c, d, 0.04f);                 \
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
  FINALISE_LOOP_I(204, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
