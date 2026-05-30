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

#ifndef SORT_H_
#define SORT_H_

#include "helpers.h"
#include "loops.h"

#define IS_NOT_POWER_OF_2(v) (((v) & ((v)-1)) && (v))

static inline uint32_t mylog2_32(uint32_t val) {
  uint32_t ret = -1;
  while (val != 0) {
    val >>= 1;
    ret++;
  }
  return ret;
}

static inline uint32_t min2_32u(uint32_t a, uint32_t b) {
#ifdef __aarch64__
  uint32_t result;
  asm("cmp %w1, %w2 ; csel %w0, %w1, %w2, lo" : "=r"(result) : "r"(a), "r"(b));
  return result;
#else
  return a < b ? a : b;
#endif
}

static inline uint32_t max2_32u(uint32_t a, uint32_t b) {
#ifdef __aarch64__
  uint32_t result;
  asm("cmp %w1, %w2 ; csel %w0, %w1, %w2, hi" : "=r"(result) : "r"(a), "r"(b));
  return result;
#else
  return a > b ? a : b;
#endif
}

static inline uint32_t upper_power_of_two_32u(uint32_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

#define SWAP_DECL(NAME, TYPE)   \
  void NAME(TYPE *a, TYPE *b) { \
    TYPE c;                     \
    c = *a;                     \
    *a = *b;                    \
    *b = c;                     \
  }
static inline SWAP_DECL(swap_32, int32_t);

static inline int check_sorted(uint32_t n, int32_t *v) {
  int r = 0;
  for (uint32_t i = 1; i < n; i++) r += (int)(v[i] < v[i - 1]);
  return r;
}

// Shared sort functions
void com_sort_oet(uint32_t, int32_t *restrict, uint32_t);
void com_sort_insertion(uint32_t, int32_t *restrict);
void com_sort_radix(uint32_t, int32_t *restrict, int32_t *restrict);

typedef struct {
  uint32_t bits;
  int32_t offset;
} radix_t;
// (1) Subtract by minimum value to avoid dealing with -ve numbers
// (2) Find radix bits from shifted maximum
// (3) Add back offset to restore original values
radix_t find_radix_sub_offset(uint32_t, int32_t *restrict);
void post_add_offset(uint32_t, int32_t *restrict, int32_t);

#endif  // SORT_H_
