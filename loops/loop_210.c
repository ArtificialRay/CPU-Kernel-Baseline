/*----------------------------------------------------------------------------
#
#   Loop 210: BF16-FP32 matrix-matrix multiply using MOPA / DOT
#
#   Purpose:
#     Use of bf16 to fp32 MOPA (or DOT) instructions.
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
    K: even

  Note: A and B matrices are considered to be re-arranged,
        as required by the BF16 -> FP32 matrix multiplication.
*/

struct loop_210_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  bfloat16_t *restrict a;
  bfloat16_t *restrict b;
  float *restrict c;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_210(struct loop_210_data *restrict data) {
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



static inline float bf16_mla(float c, bfloat16_t a, bfloat16_t b) {
  return ((bf16_to_f32(a) * bf16_to_f32(b)) + c);
}

#if !defined(HAVE_CANDIDATE)
#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_210(struct loop_210_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  bfloat16_t *restrict a = data->a;
  bfloat16_t *restrict b = data->b;
  float    *restrict c = data->c;
  for (uint64_t x = 0; x < m; x++) {
    for (uint64_t y = 0; y < n; y++) {
      float d = 0;
      for (uint64_t z = 0; z < k/2; z++){
        d = bf16_mla(d, a[z*m*2+2*x], b[z*n*2+2*y]);
        d = bf16_mla(d, a[z*m*2+2*x+1], b[z*n*2+2*y+1]);
      }
      c[x*n+y] = d;
    }
  }
}

#elif defined(HAVE_SME_INTRINSICS)

static void inner_loop_210(struct loop_210_data *data)
LOOP_ATTR
{

  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  bfloat16_t *a = (bfloat16_t *) data->a;
  bfloat16_t *b = (bfloat16_t *) data->b;
  float32_t  *c = (float32_t *)  data->c;

  bfloat16_t *ptr_a, *ptr_b;
  float32_t  *ptr_c;
  bfloat16_t *cnd_k = (bfloat16_t *) &data->a[m*k];
  uint64_t m_idx, n_idx, l_idx;
  uint64_t svl_s = svcntw();
  uint64_t l_cnd = svl_s * 4;
  uint64_t c_blk = svl_s * n;
  uint64_t c_off = c_blk + n;

  svcount_t c_all = svptrue_c16();
  svbool_t  p_all = svptrue_b16();

  svuint8x4_t    vec_c0, vec_c1;
  svbfloat16x2_t vec_a0, vec_a1, vec_b0, vec_b1;

  #define MOPA_TILE(t,x,i,j) \
    svmopa_za32_m(t, p_all, p_all, svget2(vec_a##x,i), svget2(vec_b##x,j))

  #define EXTR(x,i) svreinterpret_f32(svget4(vec_c##x,i))
  #define STORE_PAIR(x,i,j,o) \
    svst1(c_all, &ptr_c[o], svcreate2( EXTR(x,i), EXTR(x,j) ))

#if defined(__ARM_FEATURE_SME2p1)
  svzero_za();
#endif

  for (m_idx = 0; m_idx < m; m_idx += svl_s * 2) {
    for (n_idx = 0; n_idx < n; n_idx += svl_s * 2) {
#if !defined(__ARM_FEATURE_SME2p1)
      svzero_za();
#endif

      ptr_a = &a[m_idx << 1];
      ptr_b = &b[n_idx << 1];
      while (ptr_a < cnd_k) {
        vec_a0 = svld1_x2(c_all, &ptr_a[0]);
        vec_b0 = svld1_x2(c_all, &ptr_b[0]);
        vec_a1 = svld1_x2(c_all, &ptr_a[m << 1]);
        vec_b1 = svld1_x2(c_all, &ptr_b[n << 1]);

        MOPA_TILE(0,0,0,0);
        MOPA_TILE(1,0,0,1);
        MOPA_TILE(2,0,1,0);
        MOPA_TILE(3,0,1,1);
        MOPA_TILE(0,1,0,0);
        MOPA_TILE(1,1,0,1);
        MOPA_TILE(2,1,1,0);
        MOPA_TILE(3,1,1,1);

        ptr_a += m * 4;
        ptr_b += n * 4;
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
        STORE_PAIR(0,0,1,0);
        STORE_PAIR(1,0,1,n);
        STORE_PAIR(0,2,3,c_blk);
        STORE_PAIR(1,2,3,c_off);

        ptr_c += n * 2;
      }
    }
    c += c_blk * 2;
  }
}

#elif defined(HAVE_SVE_INTRINSICS)

static void inner_loop_210(struct loop_210_data *data)
LOOP_ATTR
{

  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  bfloat16_t *a = (bfloat16_t *) data->a;
  bfloat16_t *b = (bfloat16_t *) data->b;
  float32_t *c = (float32_t *) data->c;

  bfloat16_t *ptr_a, *ptr_b;
  float32_t  *ptr_c;
  bfloat16_t *cnd_k = (bfloat16_t *) &data->a[m*k];
  uint64_t svl_s = svcntw();

  uint64_t m_idx, n_idx;
  svbool_t p_all = svptrue_b16();

  svfloat32_t acc_00, acc_01, acc_02, acc_03;
  svfloat32_t acc_10, acc_11, acc_12, acc_13;
  svfloat32_t acc_20, acc_21, acc_22, acc_23;
  svfloat32_t acc_30, acc_31, acc_32, acc_33;
  svbfloat16_t vec_a0, vec_a1, vec_a2, vec_a3;
  svbfloat16_t vec_b0, vec_b1, vec_b2, vec_b3;

  #define ZERO(i,l) acc_##i##l = svdup_f32(0.0f)
  #define ZERO_GROUP(i) { ZERO(i,0); ZERO(i,1); ZERO(i,2); ZERO(i,3); }

  #define DOT(z,x,y,l) \
    acc_##z##l = svbfdot_lane(acc_##z##l, vec_b##y, vec_a##x, l)
  #define DOT_GROUP(z,x,y) \
    { DOT(z,x,y,0); DOT(z,x,y,1); DOT(z,x,y,2); DOT(z,x,y,3); }

  #define WRITE(r,l,i,o) svst1_vnum(p_all, &ptr_c[n*(r+l)], o, acc_##i##l)
  #define STORE(r,l,u,v) { WRITE(r,l,u,0); WRITE(r,l,v,1); }
  #define STORE_GROUP(r,u,v) \
    { STORE(r,0,u,v); STORE(r,1,u,v); STORE(r,2,u,v); STORE(r,3,u,v); }

  for (m_idx = 0; m_idx < 2*m; m_idx += 2*8) {
    for (n_idx = 0; n_idx < 2*n; n_idx += 2*2*svl_s) {
      ZERO_GROUP(0);
      ZERO_GROUP(1);
      ZERO_GROUP(2);
      ZERO_GROUP(3);

      ptr_a = &a[m_idx];
      ptr_b = &b[n_idx];
      while (ptr_a < cnd_k) {
        vec_a0 = svld1rq    (p_all, &ptr_a[0]);
        vec_a1 = svld1rq    (p_all, &ptr_a[8]);
        vec_a2 = svld1rq    (p_all, &ptr_a[2*m]);
        vec_a3 = svld1rq    (p_all, &ptr_a[2*m+8]);
        vec_b0 = svld1      (p_all, &ptr_b[0]);
        vec_b1 = svld1_vnum (p_all, &ptr_b[0], 1);
        vec_b2 = svld1      (p_all, &ptr_b[2*n]);
        vec_b3 = svld1_vnum (p_all, &ptr_b[2*n], 1);

        DOT_GROUP(0,0,0); DOT_GROUP(0,2,2);
        DOT_GROUP(1,0,1); DOT_GROUP(1,2,3);
        DOT_GROUP(2,1,0); DOT_GROUP(2,3,2);
        DOT_GROUP(3,1,1); DOT_GROUP(3,3,3);

        ptr_a += m * 4;
        ptr_b += n * 4;
      }

      ptr_c = &c[n_idx >> 1];
      STORE_GROUP(0,0,1);
      STORE_GROUP(4,2,3);
    }
    c += n * 8;
  }
}

#elif defined(__ARM_FEATURE_SME2)

static void inner_loop_210(struct loop_210_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t) data->a;
  register uint64_t b = (uint64_t) data->b;
  register uint64_t c = (uint64_t) data->c;

  register uint64_t svl_s;
  asm volatile( "cntw %[v]" : [v] "=&r" (svl_s) :: );

  register uint64_t a_cnd = (uint64_t) &data->a[m*k];
  register uint64_t l_cnd = svl_s * 4 - 8;
  register uint64_t c_blk = svl_s * n;
  register uint64_t c_off = c_blk + n;
  register uint64_t mx2   = m*2;
  register uint64_t nx2   = n*2;
  register uint64_t m_idx;
  register uint64_t n_idx;
  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c_ptr;
  // x12: slice index register for tile-to-vec mova

  asm volatile(
    "   ptrue   p0.h                                                      \n"
    "   ptrue   pn8.h                                                     \n"
#if defined(__ARM_FEATURE_SME2p1)
      "   zero    {za}                                                      \n"
#endif

    // M loop head
    "   mov     %[m_idx], #0                                              \n"
    "1:                                                                   \n"

    // N loop head
    "   mov     %[n_idx], #0                                              \n"
    "2:                                                                   \n"

    "   add     %[a_ptr], %[a_src], %[m_idx], lsl #1                      \n"
    "   add     %[b_ptr], %[b_src], %[n_idx], lsl #1                      \n"
    "   add     %[c_ptr], %[c_dst], %[n_idx], lsl #1                      \n"

    // K loop
    "   ld1h    { z2.h-z3.h }, pn8/z, [%[a_ptr]]                          \n"
#if !defined(__ARM_FEATURE_SME2p1)
    "   zero    {za}                                                      \n"
#endif
    "   ld1h    { z0.h-z1.h }, pn8/z, [%[b_ptr]]                          \n"
    "   bfmopa   za0.s, p0/m, p0/m, z2.h, z0.h                            \n"
    "   bfmopa   za1.s, p0/m, p0/m, z2.h, z1.h                            \n"
    "   ld1h    { z6.h-z7.h }, pn8/z, [%[a_ptr], %[mx2], lsl #1]          \n"
    "   ld1h    { z4.h-z5.h }, pn8/z, [%[b_ptr], %[nx2], lsl #1]          \n"
    "   bfmopa   za2.s, p0/m, p0/m, z3.h, z0.h                            \n"
    "   bfmopa   za3.s, p0/m, p0/m, z3.h, z1.h                            \n"
    "   add     %[a_ptr], %[a_ptr], %[mx2], lsl #2                        \n"
    "   add     %[b_ptr], %[b_ptr], %[nx2], lsl #2                        \n"
    "3:                                                                   \n"
    "   bfmopa   za0.s, p0/m, p0/m, z6.h, z4.h                            \n"
    "   bfmopa   za1.s, p0/m, p0/m, z6.h, z5.h                            \n"
    "   ld1h    { z2.h-z3.h }, pn8/z, [%[a_ptr]]                          \n"
    "   ld1h    { z0.h-z1.h }, pn8/z, [%[b_ptr]]                          \n"
    "   bfmopa   za2.s, p0/m, p0/m, z7.h, z4.h                            \n"
    "   bfmopa   za3.s, p0/m, p0/m, z7.h, z5.h                            \n"
    "   ld1h    { z6.h-z7.h }, pn8/z, [%[a_ptr], %[mx2], lsl #1]          \n"
    "   ld1H    { z4.h-z5.h }, pn8/z, [%[b_ptr], %[nx2], lsl #1]          \n"
    "   bfmopa   za0.s, p0/m, p0/m, z2.h, z0.h                            \n"
    "   bfmopa   za1.s, p0/m, p0/m, z2.h, z1.h                            \n"
    "   add     %[a_ptr], %[a_ptr], %[mx2], lsl #2                        \n"
    "   add     %[b_ptr], %[b_ptr], %[nx2], lsl #2                        \n"
    "   bfmopa   za2.s, p0/m, p0/m, z3.h, z0.h                            \n"
    "   bfmopa   za3.s, p0/m, p0/m, z3.h, z1.h                            \n"
    "   cmp     %[a_ptr], %[a_cnd]                                        \n"
    "   b.mi    3b                                                        \n"
    "   bfmopa   za0.s, p0/m, p0/m, z6.h, z4.h                            \n"
    "   bfmopa   za1.s, p0/m, p0/m, z6.h, z5.h                            \n"
    "   bfmopa   za2.s, p0/m, p0/m, z7.h, z4.h                            \n"
    "   bfmopa   za3.s, p0/m, p0/m, z7.h, z5.h                            \n"

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
    "   inch    %[n_idx], all, mul #2                                     \n"
    "   cmp     %[n_idx], %[nx2]                                          \n"
    "   b.mi    2b                                                        \n"

    // M loop tail
    "   add     %[c_dst], %[c_dst], %[c_blk], lsl #3                      \n"
    "   inch    %[m_idx], all, mul #2                                     \n"
    "   cmp     %[m_idx], %[mx2]                                          \n"
    "   b.mi    1b                                                        \n"

    : [a_ptr] "=&r" (a_ptr), [b_ptr] "=&r" (b_ptr), [c_ptr] "=&r" (c_ptr),
      [m_idx] "=&r" (m_idx), [n_idx] "=&r" (n_idx), [c_dst] "+&r" (c)
    : [m] "r" (m), [n] "r" (n), [k] "r" (k), [nx2] "r" (nx2),
      [mx2] "r" (mx2), [c_blk] "r" (c_blk), [c_off] "r" (c_off),
      [l_cnd] "r" (l_cnd), [a_cnd] "r" (a_cnd),
      [a_src] "r" (a), [b_src] "r" (b)
    : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "x12",
      "p0", "p8",
#ifdef __ARM_STATE_ZA
        "za",
#endif
        "cc", "memory");
}

#elif defined(__ARM_FEATURE_SVE2)

static void inner_loop_210(struct loop_210_data *data)
LOOP_ATTR
{
  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t) data->a;
  register uint64_t b = (uint64_t) data->b;
  register uint64_t c = (uint64_t) data->c;
  register uint64_t a_cnd = (uint64_t) &data->a[m*k];
  register uint64_t off_2 = n * 2;
  register uint64_t off_3 = n * 3;
  register uint64_t off_r = m + 4;

  register uint64_t a_ptr;
  register uint64_t b_ptr;
  register uint64_t c0ptr;
#if !defined(__ARM_FEATURE_SVE2p1)
  register uint64_t c1ptr;
#endif
  register uint64_t n_idx;
  register uint64_t m_idx;

  asm volatile(
    "   ptrue   p0.h                                              \n"
#if defined(__ARM_FEATURE_SVE2p1)
    "   ptrue   pn8.h                                             \n"
    "   ptrue   pn9.s                                             \n"
#endif
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
#if defined(__ARM_FEATURE_SVE2p1)
    "   ld1h    { z4.h-z5.h }, pn8/z, [%[b_ptr]]                  \n"
    "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                  \n"
    "   ld1h    { z6.h-z7.h }, pn8/z, [%[b_ptr]]                  \n"
    "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                  \n"
#else
    "   ld1h    {z4.h}, p0/z, [%[b_ptr]]                          \n"
    "   ld1h    {z5.h}, p0/z, [%[b_ptr], #1, mul vl]              \n"
    "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                  \n"
    "   ld1h    {z6.h}, p0/z, [%[b_ptr]]                          \n"
    "   ld1h    {z7.h}, p0/z, [%[b_ptr], #1, mul vl]              \n"
    "   add     %[b_ptr], %[b_ptr], %[n], lsl #2                  \n"
#endif
    "   bfdot    z10.s, z4.h, z0.h[0]                             \n"
    "   bfdot    z12.s, z4.h, z0.h[1]                             \n"
    "   bfdot    z14.s, z4.h, z0.h[2]                             \n"
    "   bfdot    z16.s, z4.h, z0.h[3]                             \n"
    "   bfdot    z11.s, z5.h, z0.h[0]                             \n"
    "   bfdot    z13.s, z5.h, z0.h[1]                             \n"
    "   bfdot    z15.s, z5.h, z0.h[2]                             \n"
    "   bfdot    z17.s, z5.h, z0.h[3]                             \n"
    "   bfdot    z20.s, z4.h, z1.h[0]                             \n"
    "   bfdot    z22.s, z4.h, z1.h[1]                             \n"
    "   bfdot    z24.s, z4.h, z1.h[2]                             \n"
    "   bfdot    z26.s, z4.h, z1.h[3]                             \n"
    "   bfdot    z21.s, z5.h, z1.h[0]                             \n"
    "   bfdot    z23.s, z5.h, z1.h[1]                             \n"
    "   bfdot    z25.s, z5.h, z1.h[2]                             \n"
    "   bfdot    z27.s, z5.h, z1.h[3]                             \n"
    "   bfdot    z10.s, z6.h, z2.h[0]                             \n"
    "   bfdot    z12.s, z6.h, z2.h[1]                             \n"
    "   bfdot    z14.s, z6.h, z2.h[2]                             \n"
    "   bfdot    z16.s, z6.h, z2.h[3]                             \n"
    "   bfdot    z11.s, z7.h, z2.h[0]                             \n"
    "   bfdot    z13.s, z7.h, z2.h[1]                             \n"
    "   bfdot    z15.s, z7.h, z2.h[2]                             \n"
    "   bfdot    z17.s, z7.h, z2.h[3]                             \n"
    "   bfdot    z20.s, z6.h, z3.h[0]                             \n"
    "   bfdot    z22.s, z6.h, z3.h[1]                             \n"
    "   bfdot    z24.s, z6.h, z3.h[2]                             \n"
    "   bfdot    z26.s, z6.h, z3.h[3]                             \n"
    "   bfdot    z21.s, z7.h, z3.h[0]                             \n"
    "   bfdot    z23.s, z7.h, z3.h[1]                             \n"
    "   bfdot    z25.s, z7.h, z3.h[2]                             \n"
    "   bfdot    z27.s, z7.h, z3.h[3]                             \n"
    "   cmp     %[a_ptr], %[a_cnd]                                \n"
    "   b.mi    3b                                                \n"

    // Store
#if defined(__ARM_FEATURE_SVE2p1)
    "   add     %[c0ptr], %[c_dst], %[n_idx], lsl #2              \n"
    "   st1w    { z10.s-z11.s }, pn9, [%[c0ptr]]                  \n"
    "   st1w    { z12.s-z13.s }, pn9, [%[c0ptr], %[off_1], lsl #2]\n"
    "   st1w    { z14.s-z15.s }, pn9, [%[c0ptr], %[off_2], lsl #2]\n"
    "   st1w    { z16.s-z17.s }, pn9, [%[c0ptr], %[off_3], lsl #2]\n"
    "   add     %[c0ptr], %[c0ptr], %[n], lsl #4                  \n"
    "   st1w    { z20.s-z21.s }, pn9, [%[c0ptr]]                  \n"
    "   st1w    { z22.s-z23.s }, pn9, [%[c0ptr], %[off_1], lsl #2]\n"
    "   st1w    { z24.s-z25.s }, pn9, [%[c0ptr], %[off_2], lsl #2]\n"
    "   st1w    { z26.s-z27.s }, pn9, [%[c0ptr], %[off_3], lsl #2]\n"
#else
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
#endif
    // N loop tail
    "   incw    %[n_idx], all, mul #2                             \n"
    "   cmp     %[n_idx], %[n]                                    \n"
    "   b.mi    2b                                                \n"

    // M loop tail
    "   add     %[c_dst], %[c_dst], %[n], lsl #5                  \n"
    "   add     %[m_idx], %[m_idx], #8                            \n"
    "   cmp     %[m_idx], %[m]                                    \n"
    "   b.mi    1b                                                \n"

    : [a_ptr] "=&r" (a_ptr), [b_ptr] "=&r" (b_ptr), [m_idx] "=&r" (m_idx),
      [c0ptr] "=&r" (c0ptr),
#if !defined(__ARM_FEATURE_SVE2p1)
      [c1ptr] "=&r" (c1ptr),
#endif
      [n_idx] "=&r" (n_idx), [c_dst] "+&r" (c)
    : [m] "r" (m), [n] "r" (n), [k] "r" (k),
      [a_cnd] "r" (a_cnd), [off_r] "r" (off_r), [off_l] "r" (m),
      [off_2] "r" (off_2), [off_3] "r" (off_3), [off_1] "r" (n),
      [a_src] "r" (a), [b_src] "r" (b)
    : "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",
      "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17",
      "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27",
      "p0",
#if defined(__ARM_FEATURE_SVE2p1)
      "p8", "p9",
#endif
      "cc", "memory"
  );
}

#elif defined(__ARM_NEON)

static void inner_loop_210(struct loop_210_data *data) {

  register uint64_t m = data->m;
  register uint64_t n = data->n;
  register uint64_t k = data->k;
  register uint64_t a = (uint64_t) data->a;
  register uint64_t b = (uint64_t) data->b;
  register uint64_t c = (uint64_t) data->c;
  register uint64_t a_cnd = (uint64_t) &data->a[m*k];
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
    "   bfdot   v10.4s, v4.8h, v0.2h[0]                      \n"
    "   bfdot   v12.4s, v4.8h, v0.2h[1]                      \n"
    "   bfdot   v14.4s, v4.8h, v0.2h[2]                      \n"
    "   bfdot   v16.4s, v4.8h, v0.2h[3]                      \n"
    "   bfdot   v20.4s, v4.8h, v1.2h[0]                      \n"
    "   bfdot   v22.4s, v4.8h, v1.2h[1]                      \n"
    "   bfdot   v24.4s, v4.8h, v1.2h[2]                      \n"
    "   bfdot   v26.4s, v4.8h, v1.2h[3]                      \n"
    "   bfdot   v11.4s, v5.8h, v0.2h[0]                      \n"
    "   bfdot   v13.4s, v5.8h, v0.2h[1]                      \n"
    "   bfdot   v15.4s, v5.8h, v0.2h[2]                      \n"
    "   bfdot   v17.4s, v5.8h, v0.2h[3]                      \n"
    "   bfdot   v21.4s, v5.8h, v1.2h[0]                      \n"
    "   bfdot   v23.4s, v5.8h, v1.2h[1]                      \n"
    "   bfdot   v25.4s, v5.8h, v1.2h[2]                      \n"
    "   bfdot   v27.4s, v5.8h, v1.2h[3]                      \n"
    "   bfdot   v10.4s, v6.8h, v2.2h[0]                      \n"
    "   bfdot   v12.4s, v6.8h, v2.2h[1]                      \n"
    "   bfdot   v14.4s, v6.8h, v2.2h[2]                      \n"
    "   bfdot   v16.4s, v6.8h, v2.2h[3]                      \n"
    "   bfdot   v20.4s, v6.8h, v3.2h[0]                      \n"
    "   bfdot   v22.4s, v6.8h, v3.2h[1]                      \n"
    "   bfdot   v24.4s, v6.8h, v3.2h[2]                      \n"
    "   bfdot   v26.4s, v6.8h, v3.2h[3]                      \n"
    "   bfdot   v11.4s, v7.8h, v2.2h[0]                      \n"
    "   bfdot   v13.4s, v7.8h, v2.2h[1]                      \n"
    "   bfdot   v15.4s, v7.8h, v2.2h[2]                      \n"
    "   bfdot   v17.4s, v7.8h, v2.2h[3]                      \n"
    "   bfdot   v21.4s, v7.8h, v3.2h[0]                      \n"
    "   bfdot   v23.4s, v7.8h, v3.2h[1]                      \n"
    "   bfdot   v25.4s, v7.8h, v3.2h[2]                      \n"
    "   bfdot   v27.4s, v7.8h, v3.2h[3]                      \n"
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

    : [a_ptr] "=&r" (a_ptr), [b_ptr] "=&r" (b_ptr), [c_ptr] "=&r" (c_ptr),
      [m_idx] "=&r" (m_idx), [n_idx] "=&r" (n_idx), [c_dst] "+&r" (c)
    : [m] "r" (m), [n] "r" (n), [k] "r" (k),
      [a_cnd] "r" (a_cnd), [m_inc] "r" (m_inc), [n_inc] "r" (n_inc),
      [a_src] "r" (a), [b_src] "r" (b)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
      "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
      "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
      "cc", "memory"
  );
}

#else

static void inner_loop_210(struct loop_210_data *data) {
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
#define PROBLEM_SIZE_ACTUAL(m,n,k) ((k)*((m)+(n))*sizeof(bfloat16_t))

LOOP_DECL(210, OUTER_LOOP_ATTR)
{
  // Work out values for M, K and N to fit within problem size limit
  uint64_t M = 0;  // multiple of SVLh
  uint64_t N = 0;  // multiple of SVLh
  uint64_t K = 0;  // even

  // For this loop, N should be equal to M, K should be 2*M
  const uint64_t M_base = MAX_VL / 16;
  while (true) {
    uint64_t m = M + M_base;
    uint64_t n = m ;      // Automatically a multiple of SVLh
    uint64_t k = m * 2;   // Automatically a multiple of 2
    if (PROBLEM_SIZE_ACTUAL(m,n,k) <= PROBLEM_SIZE_LIMIT_KIB*1024) {
      M = m;
      N = n;
      K = k;
    } else {
      break;
    }
  }

  struct loop_210_data data = { .m = M, .n = N, .k = K,};
  ALLOC_64B(data.a, M * K, "A matrix");
  ALLOC_64B(data.b, K * N, "B matrix");
  ALLOC_64B(data.c, M * N, "C vector");

  fill_bf16(data.a, M*K);
  fill_bf16(data.b, K*N);

  inner_loops_210(iters, &data);

#ifndef STANDALONE
  printf("Dimension sizes : M = %" PRIu64 ", K = %" PRIu64 ", N = %" PRIu64 "\n", M, K, N);
  printf("\t%" PRIu64 " x %" PRIu64 " * %" PRIu64 " x %" PRIu64 "\n", M, K, K, N);
  printf("\tTotal space used for inputs is approx. %.1f KiB\n",
         PROBLEM_SIZE_ACTUAL(M,N,K)/1024.0f);
#endif

  int checksum = 0;
#define CHECK(x,y)                                                 \
{                                                                  \
  float d = 0;                                                     \
  for (uint64_t k = 0; k < K/2; k++){                              \
    d = bf16_mla(d, data.a[k*M*2+2*(x)], data.b[k*N*2+2*(y)]);     \
    d = bf16_mla(d, data.a[k*M*2+2*(x)+1], data.b[k*N*2+2*(y)+1]); \
  }                                                                \
  checksum += (int)!check_float(d, data.c[(x) * N + (y)], 1e-3f);  \
}

#ifdef FULL_CHECK
  for (uint64_t m = 0; m < M; m++)
    for (uint64_t n = 0; n < N; n++) CHECK(m,n);
#else
  CHECK(0, 0);
  CHECK(M - 1, 0);
  CHECK(0, N - 1);
  CHECK(M - 1, N - 1);
  CHECK(M / 2, N / 2);
#endif

  bool passed = (checksum == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(210, passed, "%d", 0, checksum)
#endif
  return passed ? 0 : 1;
}
