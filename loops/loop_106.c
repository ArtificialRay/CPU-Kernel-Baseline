/*----------------------------------------------------------------------------
#
#   Loop 106: Sheep and goats
#
#   Purpose:
#     Use of BGRP instruction.
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


struct loop_106_data {
  uint32_t *restrict a;
  uint32_t *restrict b;
  uint32_t *restrict perm;
  int64_t n;
};

#define LOOP_ATTR SC_SVE_ATTR

static uint32_t permutation[5] = {0xaaaaaaaa, 0xcccccccc, 0x0f0f0f0f,
                                  0x0ff00ff0, 0x0ffff000};

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if __has_builtin(__builtin_popcount) || (__GNUC__ > 4)
#define popcount __builtin_popcount
#else
static uint32_t popcount(uint32_t x) {
  x = (x & 0x55555555) + ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = (x & 0x0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F);
  x = (x & 0x00FF00FF) + ((x >> 8) & 0x00FF00FF);
  x = (x & 0x0000FFFF) + ((x >> 16) & 0x0000FFFF);
  return x;
}
#endif

static __attribute__((unused)) uint32_t compress(uint32_t x, uint32_t m) {
  uint32_t mk, mp, mv, t;
  int i;

  x = x & m;     // Clear irrelevant bits.
  mk = ~m << 1;  // We will count 0's to right.

  for (i = 0; i < 5; i++) {
    mp = mk ^ (mk << 1);  // Parallel prefix.
    mp = mp ^ (mp << 2);
    mp = mp ^ (mp << 4);
    mp = mp ^ (mp << 8);
    mp = mp ^ (mp << 16);
    mv = mp & m;                      // Bits to move.
    m = (m ^ mv) | (mv >> (1 << i));  // Compress m.
    t = x & mv;
    x = (x ^ t) | (t >> (1 << i));    // Compress x.
    mk = mk & ~mp;
  }

  return x;
}

#define compress_left(x, m) (compress(x, m) << popcount(m))
#define sag(x, m) (compress_left(x, m) | compress(x, ~m))

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_106(struct loop_106_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static uint32_t permute(uint32_t x, uint32_t p[5]) {
  x = sag(x, p[0]);
  x = sag(x, p[1]);
  x = sag(x, p[2]);
  x = sag(x, p[3]);
  return sag(x, p[4]);
}

static void inner_loop_106(struct loop_106_data *restrict input) {
  uint32_t *restrict a = input->a;
  uint32_t *restrict b = input->b;
  uint32_t *restrict p = input->perm;
  int64_t n = input->n;

  for (int i = 0; i < n; i++) {
    b[i] = permute(a[i], p);
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) && defined(__ARM_FEATURE_SVE2_BITPERM)) || (defined(HAVE_SME_INTRINSICS) && defined(__ARM_FEATURE_SSVE_BITPERM))
static void inner_loop_106(struct loop_106_data *restrict input) {
  uint32_t *restrict a = input->a;
  uint32_t *restrict b = input->b;
  uint32_t *restrict perm = input->perm;
  int64_t n = input->n;

  svuint32_t pmt0 = svdup_u32_z(svptrue_b8(), ~perm[0]);
  svuint32_t pmt1 = svdup_u32_z(svptrue_b8(), ~perm[1]);
  svuint32_t pmt2 = svdup_u32_z(svptrue_b8(), ~perm[2]);
  svuint32_t pmt3 = svdup_u32_z(svptrue_b8(), ~perm[3]);
  svuint32_t pmt4 = svdup_u32_z(svptrue_b8(), ~perm[4]);

  svbool_t p;
  FOR_LOOP_32(int64_t, i, 0, n, p) {
    svuint32_t data = svld1(p, a + i);
    data = svbgrp(data, pmt0);
    data = svbgrp(data, pmt1);
    data = svbgrp(data, pmt2);
    data = svbgrp(data, pmt3);
    data = svbgrp(data, pmt4);
    svst1(p, b + i, data);
  }
}
#elif (defined(__ARM_FEATURE_SVE2) && defined(__ARM_FEATURE_SVE2_BITPERM)) || (defined(__ARM_FEATURE_SME) && defined(__ARM_FEATURE_SSVE_BITPERM))
static void inner_loop_106(struct loop_106_data *restrict input) LOOP_ATTR {
  uint32_t *restrict a = input->a;
  uint32_t *restrict b = input->b;
  uint32_t *restrict p = input->perm;
  int64_t n = input->n;

  int64_t i = 0;

  asm volatile(
      "   whilelo p0.s, %[i], %[n]                      \n"
      "   ld1rw   {z0.s}, p0/z, [%[p]]                  \n"
      "   ld1rw   {z1.s}, p0/z, [%[p], #4]              \n"
      "   ld1rw   {z2.s}, p0/z, [%[p], #8]              \n"
      "   ld1rw   {z3.s}, p0/z, [%[p], #12]             \n"
      "   ld1rw   {z4.s}, p0/z, [%[p], #16]             \n"
      "   not     z0.s, p0/m, z0.s                      \n"
      "   not     z1.s, p0/m, z1.s                      \n"
      "   not     z2.s, p0/m, z2.s                      \n"
      "   not     z3.s, p0/m, z3.s                      \n"
      "   not     z4.s, p0/m, z4.s                      \n"
      "1: ld1w    {z5.s}, p0/z, [%[a], %[i], lsl #2]    \n"
      "   bgrp    z5.s, z5.s, z0.s                      \n"
      "   bgrp    z5.s, z5.s, z1.s                      \n"
      "   bgrp    z5.s, z5.s, z2.s                      \n"
      "   bgrp    z5.s, z5.s, z3.s                      \n"
      "   bgrp    z5.s, z5.s, z4.s                      \n"
      "   st1w    {z5.s}, p0, [%[b], %[i], lsl #2]      \n"
      "   incw    %[i]                                  \n"
      "   whilelo p0.s, %[i], %[n]                      \n"
      "   b.any   1b                                    \n"
      // output operands, source operands, and clobber list
      : [i] "+&r"(i)
      : [a] "r"(a), [b] "r"(b), [n] "r"(n), [p] "r"(p)
      : "v0", "v1", "v2", "v3", "v4", "v5", "p0", "cc", "memory");
}
#else
static void inner_loop_106(struct loop_106_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 200
#endif

LOOP_DECL(106, SC_SVE_LOOP_ATTR)
{
  uint32_t p[5];
  struct loop_106_data data = { .perm = p, .n = SIZE, };

  ALLOC_64B(data.a, SIZE, "input data");
  ALLOC_64B(data.b, SIZE, "output buffer");

  fill_uint32(data.a, SIZE);
  fill_uint32(data.b, SIZE);

  p[0] = permutation[0];

  p[1] = sag(permutation[1], permutation[0]);
  p[2] = sag(permutation[2], permutation[0]);
  p[3] = sag(permutation[3], permutation[0]);
  p[4] = sag(permutation[4], permutation[0]);

  p[2] = sag(p[2], p[1]);
  p[3] = sag(p[3], p[1]);
  p[4] = sag(p[4], p[1]);

  p[3] = sag(p[3], p[2]);
  p[4] = sag(p[4], p[2]);

  p[4] = sag(p[4], p[3]);

  inner_loops_106(iters, &data);

  uint32_t checksum = 0;
  for (int i = 0; i < SIZE; i++) {
    checksum ^= data.b[i];
  }

  bool passed = checksum == 0x2bb9565a;
#ifndef STANDALONE
  FINALISE_LOOP_I(106, passed, "0x%08"PRIx32, 0x2bb9565a, checksum)
#endif
  return passed ? 0 : 1;
}
