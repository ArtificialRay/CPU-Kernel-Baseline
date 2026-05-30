/*----------------------------------------------------------------------------
#
#   Loop 123: Bitonic mergesort
#
#   Purpose:
#     Use of CMPGT with SEL instructions.
#
#   This file is part of the SIMD Loops project. For more information, visit:
#     https://gitlab.arm.com/architecture/simd-loops
#
#   Copyright (c) 2025, Arm Limited. All rights reserved.
#
#   SPDX-License-Identifier: BSD-3-Clause
#
----------------------------------------------------------------------------*/

#include "sort.h"


struct loop_123_data {
  uint32_t n;
  int32_t *restrict data;
  int32_t *restrict temp;
  uint32_t *restrict block_sizes;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_123(struct loop_123_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE) // Implementation

static void bitonic_merge(uint32_t n, int32_t *restrict a, int d) {
  if (n <= 1) return;
  uint32_t k = n / 2;
  for (uint32_t i = 0; i < k; i++) {
    if (d == (int)(a[i] > a[i + k])) swap_32(&a[i], &a[i + k]);
  }
  bitonic_merge(k, a, d);
  bitonic_merge(k, a + k, d);
}

static void bitonic_sort(uint32_t n, int32_t *restrict a, int d) {
  if (n <= 1) return;
  uint32_t k = n / 2;
  bitonic_sort(k, a, 1);
  bitonic_sort(k, a + k, 0);
  bitonic_merge(n, a, d);
}

static void NOINLINE do_sort(struct loop_123_data *restrict input) {
  uint32_t n = input->n;
  int32_t *data = input->data;

  bitonic_sort(n, data, 1);
} //Implementation

#elif defined(__ARM_FEATURE_SVE2) || defined(HAVE_SVE_INTRINSICS)  // Implementation
#include <limits.h>

static inline SWAP_DECL(swap_s32p, int32_t *);

// define shuffle patterns

// VL = 128b
static const uint8_t rearrange_8x1_4x2_lhs[4] = {0, 4, 2, 6};
static const uint8_t rearrange_8x1_4x2_rhs[4] = {5, 1, 7, 3};
static const uint8_t rearrange_4x2_2x4_lhs[4] = {0, 1, 4, 5};
static const uint8_t rearrange_4x2_2x4_rhs[4] = {7, 6, 3, 2};
static const uint8_t shuffle_4x2_2x4_stage1_lhs[4] = {0, 4, 2, 6};
static const uint8_t shuffle_4x2_2x4_stage1_rhs[4] = {1, 5, 3, 7};
static const uint8_t shuffle_4x2_2x4_stage2_lhs[4] = {5, 1, 7, 3};
static const uint8_t shuffle_4x2_2x4_stage2_rhs[4] = {4, 0, 6, 2};
static const uint8_t shuffle_2x4_1x8_stage1_lhs[4] = {0, 1, 4, 5};
static const uint8_t shuffle_2x4_1x8_stage1_rhs[4] = {2, 3, 6, 7};
static const uint8_t shuffle_2x4_1x8_stage2_lhs[4] = {0, 4, 2, 6};
static const uint8_t shuffle_2x4_1x8_stage2_rhs[4] = {1, 5, 3, 7};
static const uint8_t shuffle_2x4_1x8_stage3_lhs[4] = {7, 3, 6, 2};
static const uint8_t shuffle_2x4_1x8_stage3_rhs[4] = {5, 1, 4, 0};
// VL = 256b
static const uint8_t rearrange_16x1_8x2_lhs[8] = {0, 8, 2, 10, 4, 12, 6, 14};
static const uint8_t rearrange_16x1_8x2_rhs[8] = {9, 1, 11, 3, 13, 5, 15, 7};
static const uint8_t rearrange_8x2_4x4_lhs[8] = {0, 1, 8, 9, 4, 5, 12, 13};
static const uint8_t rearrange_8x2_4x4_rhs[8] = {11, 10, 3, 2, 15, 14, 7, 6};
static const uint8_t rearrange_4x4_2x8_lhs[8] = {0, 1, 2, 3, 8, 9, 10, 11};
static const uint8_t rearrange_4x4_2x8_rhs[8] = {15, 14, 13, 12, 7, 6, 5, 4};
static const uint8_t shuffle_8x2_4x4_stage1_lhs[8] = {0, 8,  2, 10,
                                                      4, 12, 6, 14};
static const uint8_t shuffle_8x2_4x4_stage1_rhs[8] = {1, 9,  3, 11,
                                                      5, 13, 7, 15};
static const uint8_t shuffle_8x2_4x4_stage2_lhs[8] = {9,  1, 11, 3,
                                                      13, 5, 15, 7};
static const uint8_t shuffle_8x2_4x4_stage2_rhs[8] = {8,  0, 10, 2,
                                                      12, 4, 14, 6};
static const uint8_t shuffle_4x4_2x8_stage1_lhs[8] = {0, 1, 8, 9, 4, 5, 12, 13};
static const uint8_t shuffle_4x4_2x8_stage1_rhs[8] = {2, 3, 10, 11,
                                                      6, 7, 14, 15};
static const uint8_t shuffle_4x4_2x8_stage2_lhs[8] = {0, 8,  2, 10,
                                                      4, 12, 6, 14};
static const uint8_t shuffle_4x4_2x8_stage2_rhs[8] = {1, 9,  3, 11,
                                                      5, 13, 7, 15};
static const uint8_t shuffle_4x4_2x8_stage3_lhs[8] = {11, 3, 10, 2,
                                                      15, 7, 14, 6};
static const uint8_t shuffle_4x4_2x8_stage3_rhs[8] = {9, 1, 8, 0, 13, 5, 12, 4};
static const uint8_t shuffle_2x8_1x16_stage1_lhs[8] = {0, 1, 2,  3,
                                                       8, 9, 10, 11};
static const uint8_t shuffle_2x8_1x16_stage1_rhs[8] = {4,  5,  6,  7,
                                                       12, 13, 14, 15};
static const uint8_t shuffle_2x8_1x16_stage2_lhs[8] = {0, 1, 8,  9,
                                                       4, 5, 12, 13};
static const uint8_t shuffle_2x8_1x16_stage2_rhs[8] = {2, 3, 10, 11,
                                                       6, 7, 14, 15};
static const uint8_t shuffle_2x8_1x16_stage3_lhs[8] = {0, 8,  2, 10,
                                                       4, 12, 6, 14};
static const uint8_t shuffle_2x8_1x16_stage3_rhs[8] = {1, 9,  3, 11,
                                                       5, 13, 7, 15};
static const uint8_t shuffle_2x8_1x16_stage4_lhs[8] = {15, 7, 14, 6,
                                                       13, 5, 12, 4};
static const uint8_t shuffle_2x8_1x16_stage4_rhs[8] = {11, 3, 10, 2,
                                                       9,  1, 8,  0};
// VL = 256b
static const uint8_t rearrange_32x1_16x2_lhs[16] = {
    0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30};
static const uint8_t rearrange_32x1_16x2_rhs[16] = {
    17, 1, 19, 3, 21, 5, 23, 7, 25, 9, 27, 11, 29, 13, 31, 15};
static const uint8_t rearrange_16x2_8x4_lhs[16] = {
    0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29};
static const uint8_t rearrange_16x2_8x4_rhs[16] = {
    19, 18, 3, 2, 23, 22, 7, 6, 27, 26, 11, 10, 31, 30, 15, 14};
static const uint8_t rearrange_8x4_4x8_lhs[16] = {0, 1, 2,  3,  16, 17, 18, 19,
                                                  8, 9, 10, 11, 24, 25, 26, 27};
static const uint8_t rearrange_8x4_4x8_rhs[16] = {
    23, 22, 21, 20, 7, 6, 5, 4, 31, 30, 29, 28, 15, 14, 13, 12};
static const uint8_t rearrange_4x8_2x16_lhs[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
static const uint8_t rearrange_4x8_2x16_rhs[16] = {
    31, 30, 29, 28, 27, 26, 25, 24, 15, 14, 13, 12, 11, 10, 9, 8};
static const uint8_t shuffle_16x2_8x4_stage1_lhs[16] = {
    0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30};
static const uint8_t shuffle_16x2_8x4_stage1_rhs[16] = {
    1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31};
static const uint8_t shuffle_16x2_8x4_stage2_lhs[16] = {
    17, 1, 19, 3, 21, 5, 23, 7, 25, 9, 27, 11, 29, 13, 31, 15};
static const uint8_t shuffle_16x2_8x4_stage2_rhs[16] = {
    16, 0, 18, 2, 20, 4, 22, 6, 24, 8, 26, 10, 28, 12, 30, 14};
static const uint8_t shuffle_8x4_4x8_stage1_lhs[16] = {
    0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29};
static const uint8_t shuffle_8x4_4x8_stage1_rhs[16] = {
    2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31};
static const uint8_t shuffle_8x4_4x8_stage2_lhs[16] = {
    0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30};
static const uint8_t shuffle_8x4_4x8_stage2_rhs[16] = {
    1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31};
static const uint8_t shuffle_8x4_4x8_stage3_lhs[16] = {
    19, 3, 18, 2, 23, 7, 22, 6, 27, 11, 26, 10, 31, 15, 30, 14};
static const uint8_t shuffle_8x4_4x8_stage3_rhs[16] = {
    17, 1, 16, 0, 21, 5, 20, 4, 25, 9, 24, 8, 29, 13, 28, 12};
static const uint8_t shuffle_4x8_2x16_stage1_lhs[16] = {
    0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27};
static const uint8_t shuffle_4x8_2x16_stage1_rhs[16] = {
    4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31};
static const uint8_t shuffle_4x8_2x16_stage2_lhs[16] = {
    0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29};
static const uint8_t shuffle_4x8_2x16_stage2_rhs[16] = {
    2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31};
static const uint8_t shuffle_4x8_2x16_stage3_lhs[16] = {
    0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30};
static const uint8_t shuffle_4x8_2x16_stage3_rhs[16] = {
    1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31};
static const uint8_t shuffle_4x8_2x16_stage4_lhs[16] = {
    23, 7, 22, 6, 21, 5, 20, 4, 31, 15, 30, 14, 29, 13, 28, 12};
static const uint8_t shuffle_4x8_2x16_stage4_rhs[16] = {
    19, 3, 18, 2, 17, 1, 16, 0, 27, 11, 26, 10, 25, 9, 24, 8};
static const uint8_t shuffle_2x16_1x32_stage1_lhs[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
static const uint8_t shuffle_2x16_1x32_stage1_rhs[16] = {
    8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
static const uint8_t shuffle_2x16_1x32_stage2_lhs[16] = {
    0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27};
static const uint8_t shuffle_2x16_1x32_stage2_rhs[16] = {
    4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31};
static const uint8_t shuffle_2x16_1x32_stage3_lhs[16] = {
    0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29};
static const uint8_t shuffle_2x16_1x32_stage3_rhs[16] = {
    2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31};
static const uint8_t shuffle_2x16_1x32_stage4_lhs[16] = {
    0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30};
static const uint8_t shuffle_2x16_1x32_stage4_rhs[16] = {
    1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31};
static const uint8_t shuffle_2x16_1x32_stage5_lhs[16] = {
    31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8};
static const uint8_t shuffle_2x16_1x32_stage5_rhs[16] = {
    23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0};

#define ARRANGE(N, S) rearrange_##N##_##S##hs
#define SHUFFLE(N, M, S) shuffle_##N##_stage##M##_##S##hs

#ifdef HAVE_SVE_INTRINSICS  // ACLE/SVE2

// shared sorting network snippets
static inline svint32x2_t bitonic_snip_1(const uint8_t *l, const uint8_t *r,
                                         svint32x2_t w, svbool_t p) {
  svuint32_t a = svld1ub_u32(p, l);
  svuint32_t b = svld1ub_u32(p, r);
  svint32_t u = svtbl2(w, a);
  svint32_t v = svtbl2(w, b);
  return svcreate2(u, v);
}
static inline svint32x2_t bitonic_snip_2(svint32x2_t w, svbool_t p) {
  svint32_t u = svget2(w, 0);
  svint32_t v = svget2(w, 1);
  svbool_t q = svcmpgt(p, u, v);
  svint32_t a = svsel(q, u, v);
  svint32_t b = svsel(q, v, u);
  return svcreate2(a, b);
}
static inline svint32x2_t bitonic_snip_3(const uint8_t *l, const uint8_t *r,
                                         svint32x2_t w, svbool_t p) {
  svuint32_t c = svld1ub_u32(p, l);
  svuint32_t d = svld1ub_u32(p, r);
  svint32_t u = svget2(w, 0);
  svint32_t v = svget2(w, 1);
  svbool_t q = svcmpgt(p, u, v);
  svint32_t a = svsel(q, u, v);
  svint32_t b = svsel(q, v, u);
  w = svcreate2(a, b);
  u = svtbl2(w, c);
  v = svtbl2(w, d);
  return svcreate2(u, v);
}

// shorthand macros
#define BITONIC_SNIP_1(N) bitonic_snip_1(ARRANGE(N, l), ARRANGE(N, r), w, p)
#define BITONIC_SNIP_2() bitonic_snip_2(w, p)
#define BITONIC_SNIP_3(N, M) \
  bitonic_snip_3(SHUFFLE(N, M, l), SHUFFLE(N, M, r), w, p)

// sorting network (2x) four 32-bit elements
static inline svint32x2_t bitonic_step_8x1_4x2(svint32x2_t w, svbool_t p) {
  return BITONIC_SNIP_2();
}
static inline svint32x2_t bitonic_step_4x2_2x4(svint32x2_t w, svbool_t p) {
  w = BITONIC_SNIP_3(4x2_2x4, 1);
  w = BITONIC_SNIP_3(4x2_2x4, 2);
  return w;
}
static inline svint32x2_t bitonic_step_2x4_1x8(svint32x2_t w, svbool_t p) {
  w = BITONIC_SNIP_3(2x4_1x8, 1);
  w = BITONIC_SNIP_3(2x4_1x8, 2);
  w = BITONIC_SNIP_3(2x4_1x8, 3);
  return w;
}
static svint32x2_t bitonic_sort_8x1_1x8(svint32x2_t w, svbool_t p) {
  w = bitonic_step_8x1_4x2(w, p);
  w = BITONIC_SNIP_1(8x1_4x2);
  w = bitonic_step_4x2_2x4(w, p);
  w = BITONIC_SNIP_1(4x2_2x4);
  w = bitonic_step_2x4_1x8(w, p);
  return w;
}

// sorting network (2x) eight 32-bit elements
static inline svint32x2_t bitonic_step_16x1_8x2(svint32x2_t w, svbool_t p) {
  return BITONIC_SNIP_2();
}
static inline svint32x2_t bitonic_step_8x2_4x4(svint32x2_t w, svbool_t p) {
  w = BITONIC_SNIP_3(8x2_4x4, 1);
  w = BITONIC_SNIP_3(8x2_4x4, 2);
  return w;
}
static inline svint32x2_t bitonic_step_4x4_2x8(svint32x2_t w, svbool_t p) {
  w = BITONIC_SNIP_3(4x4_2x8, 1);
  w = BITONIC_SNIP_3(4x4_2x8, 2);
  w = BITONIC_SNIP_3(4x4_2x8, 3);
  return w;
}
static inline svint32x2_t bitonic_step_2x8_1x16(svint32x2_t w, svbool_t p) {
  w = BITONIC_SNIP_3(2x8_1x16, 1);
  w = BITONIC_SNIP_3(2x8_1x16, 2);
  w = BITONIC_SNIP_3(2x8_1x16, 3);
  w = BITONIC_SNIP_3(2x8_1x16, 4);
  return w;
}
static svint32x2_t bitonic_sort_16x1_1x16(svint32x2_t w, svbool_t p) {
  w = bitonic_step_16x1_8x2(w, p);
  w = BITONIC_SNIP_1(16x1_8x2);
  w = bitonic_step_8x2_4x4(w, p);
  w = BITONIC_SNIP_1(8x2_4x4);
  w = bitonic_step_4x4_2x8(w, p);
  w = BITONIC_SNIP_1(4x4_2x8);
  w = bitonic_step_2x8_1x16(w, p);
  return w;
}

// sorting network (2x) sixteen 32-bit elements
static inline svint32x2_t bitonic_step_32x1_16x2(svint32x2_t w, svbool_t p) {
  return BITONIC_SNIP_2();
}
static inline svint32x2_t bitonic_step_16x2_8x4(svint32x2_t w, svbool_t p) {
  w = BITONIC_SNIP_3(16x2_8x4, 1);
  w = BITONIC_SNIP_3(16x2_8x4, 2);
  return w;
}

static inline svint32x2_t bitonic_step_8x4_4x8(svint32x2_t w, svbool_t p) {
  w = BITONIC_SNIP_3(8x4_4x8, 1);
  w = BITONIC_SNIP_3(8x4_4x8, 2);
  w = BITONIC_SNIP_3(8x4_4x8, 3);
  return w;
}
static inline svint32x2_t bitonic_step_4x8_2x16(svint32x2_t w, svbool_t p) {
  w = BITONIC_SNIP_3(4x8_2x16, 1);
  w = BITONIC_SNIP_3(4x8_2x16, 2);
  w = BITONIC_SNIP_3(4x8_2x16, 3);
  w = BITONIC_SNIP_3(4x8_2x16, 4);
  return w;
}
static inline svint32x2_t bitonic_step_2x16_1x32(svint32x2_t w, svbool_t p) {
  w = BITONIC_SNIP_3(2x16_1x32, 1);
  w = BITONIC_SNIP_3(2x16_1x32, 2);
  w = BITONIC_SNIP_3(2x16_1x32, 3);
  w = BITONIC_SNIP_3(2x16_1x32, 4);
  w = BITONIC_SNIP_3(2x16_1x32, 5);
  return w;
}
static svint32x2_t bitonic_sort_32x1_1x32(svint32x2_t w, svbool_t p) {
  w = bitonic_step_32x1_16x2(w, p);
  w = BITONIC_SNIP_1(32x1_16x2);
  w = bitonic_step_16x2_8x4(w, p);
  w = BITONIC_SNIP_1(16x2_8x4);
  w = bitonic_step_8x4_4x8(w, p);
  w = BITONIC_SNIP_1(8x4_4x8);
  w = bitonic_step_4x8_2x16(w, p);
  w = BITONIC_SNIP_1(4x8_2x16);
  w = bitonic_step_2x16_1x32(w, p);
  return w;
}

// merges two (already sorted) L/R blocks of tuples using sorting networks
static void bitonic_make_blocks(uint32_t n, int32_t *data) {
  typedef svint32x2_t (*sort_func)(svint32x2_t, svbool_t);
  static const sort_func bitonic_sorting_networks[5] = {
      NULL,
      NULL,
      bitonic_sort_8x1_1x8,
      bitonic_sort_16x1_1x16,
      bitonic_sort_32x1_1x32,
  };

  const uint32_t mvl = svcntw();
  const uint32_t m = n - (n % (2 * mvl));
  sort_func network_pass = bitonic_sorting_networks[mylog2_32(mvl)];

  svint32_t x, y, z;
  svint32x2_t w;
  uint32_t i;

#if defined(__ARM_FEATURE_SVE2p1)

  svboolx2_t q;
  svcount_t p;

  p = svptrue_c32();
  for (i = 0; i < m; i += 2 * svcntw()) {
    w = svld1_x2(p, data + i);
    w = network_pass(w, svptrue_b32());
    svst1(p, data + i, w);
  }

  if (i < n) {
    p = svwhilelt_c32((uint64_t)i, (uint64_t)n, 2);
    w = svld1_x2(p, data + i);
    x = svdup_s32(INT_MAX);
    q = svpext_lane_c32_x2(p, 0);
    y = svsel(svget2(q, 0), svget2(w, 0), x);
    z = svsel(svget2(q, 1), svget2(w, 1), x);
    w = network_pass(svcreate2(y, z), svget2(q, 0));
    svst1(p, data + i, w);
  }

#else

  svbool_t p, q;

  p = svptrue_b32();
  for (i = 0; i < m; i += 2 * svcntw()) {
    y = svld1_vnum(p, &data[i], 0);
    z = svld1_vnum(p, &data[i], 1);
    w = network_pass(svcreate2(y, z), p);
    svst1_vnum(p, &data[i], 0, svget2(w, 0));
    svst1_vnum(p, &data[i], 1, svget2(w, 1));
  }

  if (i < n) {
    p = svwhilelt_b32(i + mvl * 0, n);
    q = svwhilelt_b32(i + mvl * 1, n);
    x = svdup_s32(INT_MAX);
    y = svsel(p, svld1_vnum(p, &data[i], 0), x);
    z = svsel(q, svld1_vnum(q, &data[i], 1), x);
    w = network_pass(svcreate2(y, z), p);
    svst1_vnum(p, &data[i], 0, svget2(w, 0));
    svst1_vnum(q, &data[i], 1, svget2(w, 1));
  }

#endif
}

// takes an unsorted stream and sorts it into blocks of 2*mvl tuples
static void bitonic_merge_blocks(uint32_t n_lhs, uint32_t n_rhs,
                                 int32_t *data_lhs, int32_t *data_rhs,
                                 int32_t *data_out) {
  typedef svint32x2_t (*merge_func)(svint32x2_t, svbool_t);
  static const merge_func bitonic_merging_networks[5] = {
      NULL,
      NULL,
      bitonic_step_2x4_1x8,
      bitonic_step_2x8_1x16,
      bitonic_step_2x16_1x32,
  };

  const uint32_t mvl = svcntw();
  merge_func network_pass = bitonic_merging_networks[mylog2_32(mvl)];

  svbool_t p = svptrue_b32();
  svint32_t u, v;
  svint32x2_t w;
  uint32_t ptr_out = 0;
  uint32_t ptr_lhs = 0;
  uint32_t ptr_rhs = 0;

#define LOAD_SIDE(R, S)                     \
  R = svld1(p, &data_##S##hs[ptr_##S##hs]); \
  ptr_##S##hs += svcntw()
#define STORE_REG(R)               \
  svst1(p, &data_out[ptr_out], R); \
  ptr_out += svcntw()
#define NET_PASS              \
  w = svcreate2(u, svrev(v)); \
  w = network_pass(w, p);     \
  u = svget2(w, 0);           \
  v = svget2(w, 1);

  LOAD_SIDE(u, l);
  LOAD_SIDE(v, r);
  NET_PASS;
  STORE_REG(u);

  while (ptr_lhs < n_lhs && ptr_rhs < n_rhs) {
    if (data_lhs[ptr_lhs] < data_rhs[ptr_rhs]) {
      LOAD_SIDE(u, l);
    } else {
      LOAD_SIDE(u, r);
    }
    NET_PASS;
    STORE_REG(u);
  }

  while (ptr_lhs < n_lhs) {
    LOAD_SIDE(u, l);
    NET_PASS;
    STORE_REG(u);
  }

  while (ptr_rhs < n_rhs) {
    LOAD_SIDE(u, r);
    NET_PASS;
    STORE_REG(u);
  }

  STORE_REG(v);

#undef LOAD_SIDE
#undef STORE_REG
#undef NET_PASS

  // assert (ptr_out == (n_lhs + n_rhs))
}

#else  // ACLE/SVE2

// z0 and z1 are the key inputs a and b
// z2 and z3 are temporary registers
// z4 and z5 are temporary registers
// z9 and p1 are reserved
// p0 is the vector length

// shared sorting network snippets
static inline void bitonic_snip_1(const uint8_t *l, const uint8_t *r) {
  asm volatile(
      "ld1b   z4.s, p0/z, [%[idx_lhs]]    \n"
      "ld1b   z5.s, p0/z, [%[idx_rhs]]    \n"
      "tbl    z2.s, {z0.s, z1.s}, z4.s    \n"
      "tbl    z3.s, {z0.s, z1.s}, z5.s    \n"
      "mov    z0.d, z2.d                  \n"
      "mov    z1.d, z3.d                  \n"
      :
      : [idx_lhs] "r"(l), [idx_rhs] "r"(r)
      : "z0", "z1", "z2", "z3", "z4", "z5", "p0", "memory");
}
static inline void bitonic_snip_2(void) {
  asm volatile(
      "cmpgt  p2.s, p0/z, z0.s, z1.s  \n"
      "sel    z2.s, p2, z0.s, z1.s    \n"
      "sel    z3.s, p2, z1.s, z0.s    \n"
      "mov    z0.d, z2.d              \n"
      "mov    z1.d, z3.d              \n"
      ::
      : "z0", "z1", "z2", "z3", "p0", "p2");
}
static inline void bitonic_snip_3(const uint8_t *l, const uint8_t *r) {
  asm volatile(
      "ld1b   z4.s, p0/z, [%[idx_lhs]]    \n"
      "ld1b   z5.s, p0/z, [%[idx_rhs]]    \n"
      "cmpgt  p2.s, p0/z, z0.s, z1.s      \n"
      "sel    z2.s, p2, z0.s, z1.s        \n"
      "sel    z3.s, p2, z1.s, z0.s        \n"
      "tbl    z0.s, {z2.s, z3.s}, z4.s    \n"
      "tbl    z1.s, {z2.s, z3.s}, z5.s    \n"
      :
      : [idx_lhs] "r"(l), [idx_rhs] "r"(r)
      : "z0", "z1", "z2", "z3", "z4", "z5", "p0", "p2", "memory");
}

// shorthand macros
#define BITONIC_SNIP_1(N) bitonic_snip_1(ARRANGE(N, l), ARRANGE(N, r))
#define BITONIC_SNIP_2 bitonic_snip_2
#define BITONIC_SNIP_3(N, M) bitonic_snip_3(SHUFFLE(N, M, l), SHUFFLE(N, M, r))

// sorting network (2x) four 32-bit elements
static inline void bitonic_step_8x1_4x2(void) { BITONIC_SNIP_2(); }
static inline void bitonic_step_4x2_2x4(void) {
  BITONIC_SNIP_3(4x2_2x4, 1);
  BITONIC_SNIP_3(4x2_2x4, 2);
}
static inline void bitonic_step_2x4_1x8(void) {
  BITONIC_SNIP_3(2x4_1x8, 1);
  BITONIC_SNIP_3(2x4_1x8, 2);
  BITONIC_SNIP_3(2x4_1x8, 3);
}
static void bitonic_sort_8x1_1x8(void) {
  bitonic_step_8x1_4x2();
  BITONIC_SNIP_1(8x1_4x2);
  bitonic_step_4x2_2x4();
  BITONIC_SNIP_1(4x2_2x4);
  bitonic_step_2x4_1x8();
}

// sorting network (2x) eight 32-bit elements
static inline void bitonic_step_16x1_8x2(void) { BITONIC_SNIP_2(); }
static inline void bitonic_step_8x2_4x4(void) {
  BITONIC_SNIP_3(8x2_4x4, 1);
  BITONIC_SNIP_3(8x2_4x4, 2);
}
static inline void bitonic_step_4x4_2x8(void) {
  BITONIC_SNIP_3(4x4_2x8, 1);
  BITONIC_SNIP_3(4x4_2x8, 2);
  BITONIC_SNIP_3(4x4_2x8, 3);
}
static inline void bitonic_step_2x8_1x16(void) {
  BITONIC_SNIP_3(2x8_1x16, 1);
  BITONIC_SNIP_3(2x8_1x16, 2);
  BITONIC_SNIP_3(2x8_1x16, 3);
  BITONIC_SNIP_3(2x8_1x16, 4);
}
static void bitonic_sort_16x1_1x16(void) {
  bitonic_step_16x1_8x2();
  BITONIC_SNIP_1(16x1_8x2);
  bitonic_step_8x2_4x4();
  BITONIC_SNIP_1(8x2_4x4);
  bitonic_step_4x4_2x8();
  BITONIC_SNIP_1(4x4_2x8);
  bitonic_step_2x8_1x16();
}

// sorting network (2x) sixteen 32-bit elements
static inline void bitonic_step_32x1_16x2(void) { BITONIC_SNIP_2(); }
static inline void bitonic_step_16x2_8x4(void) {
  BITONIC_SNIP_3(16x2_8x4, 1);
  BITONIC_SNIP_3(16x2_8x4, 2);
}
static inline void bitonic_step_8x4_4x8(void) {
  BITONIC_SNIP_3(8x4_4x8, 1);
  BITONIC_SNIP_3(8x4_4x8, 2);
  BITONIC_SNIP_3(8x4_4x8, 3);
}
static inline void bitonic_step_4x8_2x16(void) {
  BITONIC_SNIP_3(4x8_2x16, 1);
  BITONIC_SNIP_3(4x8_2x16, 2);
  BITONIC_SNIP_3(4x8_2x16, 3);
  BITONIC_SNIP_3(4x8_2x16, 4);
}
static inline void bitonic_step_2x16_1x32(void) {
  BITONIC_SNIP_3(2x16_1x32, 1);
  BITONIC_SNIP_3(2x16_1x32, 2);
  BITONIC_SNIP_3(2x16_1x32, 3);
  BITONIC_SNIP_3(2x16_1x32, 4);
  BITONIC_SNIP_3(2x16_1x32, 5);
}
static void bitonic_sort_32x1_1x32(void) {
  bitonic_step_32x1_16x2();
  BITONIC_SNIP_1(32x1_16x2);
  bitonic_step_16x2_8x4();
  BITONIC_SNIP_1(16x2_8x4);
  bitonic_step_8x4_4x8();
  BITONIC_SNIP_1(8x4_4x8);
  bitonic_step_4x8_2x16();
  BITONIC_SNIP_1(4x8_2x16);
  bitonic_step_2x16_1x32();
}

// takes an unsorted stream and sorts it into blocks of 2*mvl tuples
static void bitonic_make_blocks(uint32_t n, int32_t *data) {
  typedef void (*sort_func)(void);
  static const sort_func bitonic_sorting_networks[5] = {
      NULL,
      NULL,
      bitonic_sort_8x1_1x8,
      bitonic_sort_16x1_1x16,
      bitonic_sort_32x1_1x32,
  };

  const uint32_t mvl = get_sve_vl() / 32;
  const uint32_t m = n - (n % (2 * mvl));
  sort_func network_pass = bitonic_sorting_networks[mylog2_32(mvl)];

  uint32_t i = 0;

#if defined(__ARM_FEATURE_SVE2p1)

  asm(
      // first iterate through standard case
      "   ptrue   pn8.s                                     \n"
      "1:                                                   \n"
      "   cmp     %w[i], %w[m]                              \n"
      "   b.ge    2f                                        \n"
      "   ld1w    {z0.s-z1.s}, pn8/z, [%[a], %x[i], lsl #2] \n"
      "   blr     %[f]                                      \n"
      "   st1w    {z0.s-z1.s}, pn8, [%[a], %x[i], lsl #2]   \n"
      "   inch    %x[i]                                     \n"
      "   b       1b                                        \n"
      "2:                                                   \n"
      // next clean up anything left
      "   mov     z9.s, #0x7FFFFFFF                         \n"
      "   whilelt pn8.s, %x[i], %x[n], vlx2                 \n"
      "   b.none  3f                                        \n"
      "   pext    {p0.s,p1.s}, pn8[0]                       \n"
      "   ld1w    {z0.s-z1.s}, pn8/z, [%[a], %x[i], lsl #2] \n"
      "   sel     z0.s, p0, z0.s, z9.s                      \n"
      "   sel     z1.s, p1, z1.s, z9.s                      \n"
      "   blr     %[f]                                      \n"
      "   st1w    {z0.s-z1.s}, pn8, [%[a], %x[i], lsl #2]   \n"
      "3:                                                   \n"
      : [i] "+&r"(i)
      : [a] "r"(data), [m] "r"(m), [n] "r"(n), [f] "r"(network_pass)
      : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
        "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x30",
        "z0", "z1", "z9", "p0", "p1", "p8", "cc", "memory");

#else

  int32_t *data2 = data + mvl;

  asm(
      // first iterate through standard case
      "   ptrue   p0.s                                    \n"
      "1:                                                 \n"
      "   cmp     %w[i], %w[m]                            \n"
      "   b.ge    2f                                      \n"
      "   ld1w    z0.s, p0/z, [%[a], %x[i], lsl #2]       \n"
      "   ld1w    z1.s, p0/z, [%[b], %x[i], lsl #2]       \n"
      "   blr     %[f]                                    \n"
      "   st1w    z0.s, p0, [%[a], %x[i], lsl #2]         \n"
      "   st1w    z1.s, p0, [%[b], %x[i], lsl #2]         \n"
      "   inch    %x[i]                                   \n"
      "   b       1b                                      \n"
      "2:                                                 \n"
      // next clean up anything left
      "   mov     z9.s, #0x7FFFFFFF                       \n"
      "   whilelt p0.s, %x[i], %x[n]                      \n"
      "   b.none  3f                                      \n"
      "   whilelt p1.s, %x[j], %x[n]                      \n"
      "   ld1w    z0.s, p0/z, [%[a], %x[i], lsl #2]       \n"
      "   ld1w    z1.s, p1/z, [%[b], %x[i], lsl #2]       \n"
      "   sel     z0.s, p0, z0.s, z9.s                    \n"
      "   sel     z1.s, p1, z1.s, z9.s                    \n"
      "   blr     %[f]                                    \n"
      "   st1w    z0.s, p0, [%[a], %x[i], lsl #2]         \n"
      "   st1w    z1.s, p1, [%[b], %x[i], lsl #2]         \n"
      "3:                                                 \n"
      : [i] "+&r"(i)
      : [a] "r"(data), [b] "r"(data2), [m] "r"(m), [n] "r"(n), [j] "r"(i + mvl),
        [f] "r"(network_pass)
      : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
        "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x30",
        "z0", "z1", "z9", "p0", "p1", "cc", "memory");

#endif
}

// merges two (already sorted) L/R blocks of tuples using sorting networks
static void bitonic_merge_blocks(uint32_t n_lhs, uint32_t n_rhs,
                                 int32_t *data_lhs, int32_t *data_rhs,
                                 int32_t *data_out) {
  typedef void (*merge_func)(void);
  static const merge_func bitonic_merging_networks[5] = {
      NULL,
      NULL,
      bitonic_step_2x4_1x8,
      bitonic_step_2x8_1x16,
      bitonic_step_2x16_1x32,
  };

  const uint32_t mvl = get_sve_vl() / 32;
  merge_func network_pass = bitonic_merging_networks[mylog2_32(mvl)];

  uint32_t ptr_out = 0;
  uint32_t ptr_lhs = 0;
  uint32_t ptr_rhs = 0;
  asm volatile("ptrue p0.s" ::: "p0");

  // first pass
  asm volatile(
      "ld1w   z0.s, p0/z, [%[data_lhs], %x[ptr_lhs], lsl #2]   \n"
      "ld1w   z1.s, p0/z, [%[data_rhs], %x[ptr_rhs], lsl #2]   \n"
      "incw   %x[ptr_lhs]                                      \n"
      "incw   %x[ptr_rhs]                                      \n"
      "rev    z1.s, z1.s                                       \n"
      : [ptr_lhs] "+&r"(ptr_lhs), [ptr_rhs] "+&r"(ptr_rhs)
      : [data_lhs] "r"(data_lhs), [data_rhs] "r"(data_rhs)
      : "z0", "z1", "p0", "memory");
  network_pass();
  asm volatile(
      "st1w   z0.s, p0  , [%[data_out], %x[ptr_out], lsl #2]   \n"
      "incw   %x[ptr_out]                                      \n"
      : [ptr_out] "+&r"(ptr_out)
      : [data_out] "r"(data_out)
      : "z0", "p0", "memory");

  // choose next block to compare against
  while (ptr_lhs < n_lhs && ptr_rhs < n_rhs) {
    if (data_lhs[ptr_lhs] < data_rhs[ptr_rhs])
      asm volatile(
          "ld1w   z0.s, p0/z, [%[data_lhs], %x[ptr_lhs], lsl #2]   \n"
          "incw   %x[ptr_lhs]                                      \n"
          : [ptr_lhs] "+&r"(ptr_lhs)
          : [data_lhs] "r"(data_lhs)
          : "z0", "p0", "memory");
    else
      asm volatile(
          "ld1w   z0.s, p0/z, [%[data_rhs], %x[ptr_rhs], lsl #2]   \n"
          "incw   %x[ptr_rhs]                                      \n"
          : [ptr_rhs] "+&r"(ptr_rhs)
          : [data_rhs] "r"(data_rhs)
          : "z0", "p0", "memory");
    asm volatile("rev z1.s, z1.s" ::: "z2");
    network_pass();
    asm volatile(
        "st1w   z0.s, p0  , [%[data_out], %x[ptr_out], lsl #2]   \n"
        "incw   %x[ptr_out]                                      \n"
        : [ptr_out] "+&r"(ptr_out)
        : [data_out] "r"(data_out)
        : "z0", "p0", "memory");
  }

  // now take values from just one block
  while (ptr_lhs < n_lhs) {
    asm volatile(
        "ld1w   z0.s, p0/z, [%[data_lhs], %x[ptr_lhs], lsl #2]   \n"
        "incw   %x[ptr_lhs]                                      \n"
        "rev    z1.s, z1.s                                       \n"
        : [ptr_lhs] "+&r"(ptr_lhs)
        : [data_lhs] "r"(data_lhs)
        : "z0", "z1", "p0", "memory");
    network_pass();
    asm volatile(
        "st1w   z0.s, p0  , [%[data_out], %x[ptr_out], lsl #2]   \n"
        "incw   %x[ptr_out]                                      \n"
        : [ptr_out] "+&r"(ptr_out)
        : [data_out] "r"(data_out)
        : "z0", "p0", "memory");
  }

  // and the other
  while (ptr_rhs < n_rhs) {
    asm volatile(
        "ld1w   z0.s, p0/z, [%[data_rhs], %x[ptr_rhs], lsl #2]   \n"
        "incw   %x[ptr_rhs]                                      \n"
        "rev    z1.s, z1.s                                       \n"
        : [ptr_rhs] "+&r"(ptr_rhs)
        : [data_rhs] "r"(data_rhs)
        : "z0", "z1", "p0", "memory");
    network_pass();
    asm volatile(
        "st1w   z0.s, p0  , [%[data_out], %x[ptr_out], lsl #2]   \n"
        "incw   %x[ptr_out]                                      \n"
        : [ptr_out] "+&r"(ptr_out)
        : [data_out] "r"(data_out)
        : "z0", "p0", "memory");
  }

  // final pass
  asm volatile(
      "st1w   z1.s, p0  , [%[data_out], %x[ptr_out], lsl #2]   \n"
      "incw   %x[ptr_out]                                      \n"
      : [ptr_out] "+&r"(ptr_out)
      : [data_out] "r"(data_out)
      : "z1", "p0", "memory");

  // assert (ptr_out == (n_lhs + n_rhs))
}

#endif  // ACLE/SVE2

// takes unsorted input
// creates sorted blocks of 2*mvl
// iteratively merges adjacent blocks until single block remaining
static void NOINLINE do_sort(struct loop_123_data *restrict input) {
  const uint32_t n = input->n;
  const uint32_t mvl = get_sve_vl() / 32;

  if (n % mvl) {
    printf(" - Error: buffer size must be a multiple of SVLs\n");
    return;
  }
  if (IS_NOT_POWER_OF_2(mvl) || mvl < 4 || mvl > 16) {
    printf("ABORT: disabled for VL > 64 bytes.\n");
    exit(2);
  }
  uint32_t num_blocks = (n + 2 * mvl - 1) / (2 * mvl);

  if (input->temp == NULL)
    ALLOC_64B(input->temp, n, "intermediate buffer");
  if (input->block_sizes == NULL)
    ALLOC_64B(input->block_sizes, num_blocks, "block size table");

  int32_t *data = input->data;
  int32_t *temp = input->temp;
  uint32_t *block_sizes = input->block_sizes;

  bitonic_make_blocks(n, data);

  int32_t *data_inp = data;
  int32_t *data_out = temp;
  uint32_t i;

  for (i = 0; i < num_blocks; i++)
    block_sizes[i] = min2_32u(2 * mvl, n - (i * 2 * mvl));

  while (num_blocks > 1) {
    uint32_t offset = 0;
    uint32_t num_blocks_temp = 0;

    for (i = 1; i < num_blocks; i += 2) {
      uint32_t n_lhs = block_sizes[i - 1];
      uint32_t n_rhs = block_sizes[i];
      uint32_t n_tot = n_lhs + n_rhs;

      // ((n_lhs % mvl) || (n_rhs % mvl))
      bitonic_merge_blocks(n_lhs, n_rhs, &data_inp[offset],
                           &data_inp[offset + n_lhs], &data_out[offset]);

      offset += n_tot;
      block_sizes[num_blocks_temp++] = n_tot;
    }

    // check for single dangling block that had no neighbour to merge with
    if (i == num_blocks) {
      uint32_t n_last = block_sizes[i - 1];
      memcpy(data_out + offset, data_inp + offset, sizeof(int32_t) * n_last);
      block_sizes[num_blocks_temp++] = n_last;
    }

    // switch in/out buffer pointers
    swap_s32p(&data_inp, &data_out);

    num_blocks = num_blocks_temp;
  }

  // copy across if necessary
  if (data_inp != data) {
    memcpy(data_out, data_inp, sizeof(int32_t) * n);
  }
}

#else
static void NOINLINE do_sort(struct loop_123_data *restrict input) {
  printf("ABORT: No implementations available for this targets.\n");
  exit(2);
}
#endif  // Implementation

#if !defined(HAVE_CANDIDATE)

static void inner_loop_123(struct loop_123_data *restrict input) {
  fill_int32(input->data, input->n);
  do_sort(input);
}
#endif /* !HAVE_CANDIDATE */

#ifndef SIZE
#define SIZE 256  // must be a multiple of SVLs
#endif

LOOP_DECL(123, NS_SVE_LOOP_ATTR)
{
  struct loop_123_data data = { .n = SIZE, .temp = NULL, .block_sizes = NULL, };

  ALLOC_64B(data.data, SIZE, "data array");

  inner_loops_123(iters, &data);

  int res = check_sorted(SIZE, data.data);
  bool passed = (res == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(123, passed, "%d", 0, res)
#endif
  return passed ? 0 : 1;
}
