/*----------------------------------------------------------------------------
#
#   Loop 121: Quicksort
#
#   Purpose:
#     Use of CMPLT with COMPACT and CNTP instructions.
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


struct loop_121_data {
  uint32_t n;
  int32_t *restrict data;
  int32_t *restrict temp;
};

#if defined(__ARM_FEATURE_SVE)  // Helpers
#define VAR_UNUSED

// minimum partition length needed to use the median-of-3 optimisation, which
// becomes too expensive and less important as the partitions become smaller
#define MIN_N_MEDIAN3 8

// alternative value for vector implementation
#define THRESHOLD 4

static __attribute__((unused)) void tidy_step(uint32_t n, int32_t *restrict data) {
  com_sort_oet(n, data, THRESHOLD);
}
#else  // Helpers
#define VAR_UNUSED __attribute__((unused))

// known to be a good "all-purpose" value
#define THRESHOLD 8

static __attribute__((unused)) void tidy_step(uint32_t n, int32_t *restrict data) {
  com_sort_insertion(n, data);
}
#endif  // Helpers

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_121(struct loop_121_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE) || defined(HAVE_SVE_INTRINSICS) || defined(__ARM_FEATURE_SVE)
static inline uint32_t find_pivot(uint32_t n, const int32_t *restrict data) {
  struct tuple_32 {
    uint32_t idx;
    int32_t val;
  } t, candidates[3];

  candidates[0].idx = 0;
  candidates[1].idx = n - 1;
  candidates[2].idx = n / 2;

  candidates[0].val = data[candidates[0].idx];
  candidates[1].val = data[candidates[1].idx];
  candidates[2].val = data[candidates[2].idx];

  int i, j;
  for (i = 1; i < 3; i++) {
    t = candidates[i];
    for (j = i - 1; j >= 0 && candidates[j].val > t.val; j--)
      candidates[j + 1] = candidates[j];
    candidates[j + 1] = t;
  }

  return candidates[1].idx;
}
#endif

#if !defined(HAVE_CANDIDATE)

#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
// Quicksort
static void quicksort(uint32_t n, int32_t *restrict data, uint32_t threshold,
                      VAR_UNUSED int32_t *restrict temp) {
  if (n <= threshold) return;

  int32_t v = data[find_pivot(n, data)];

  uint32_t i = 0, j = n - 1;
  while (1) {
    while (i < j && data[i] <= v && data[j] >= v) {
      i++;
      j--;
    }
    while (data[i] < v) i++;
    while (data[j] > v) j--;
    if (i >= j) break;
    swap_32(&data[i], &data[j]);
  }

  quicksort(i, data, threshold, temp);
  quicksort(n - i, data + i, threshold, temp);
}
#elif defined(HAVE_SVE_INTRINSICS)  // Quicksort
static void quicksort(uint32_t n, int32_t *restrict data, uint32_t threshold,
                      VAR_UNUSED int32_t *restrict temp) {
  if (n <= threshold) return;

  if (n >= MIN_N_MEDIAN3) {
    uint32_t median_idx = find_pivot(n, data);
    if (median_idx != 0) swap_32(&data[0], &data[median_idx]);
  }
  svint32_t pivot = svdup_s32(data[0]);

  uint64_t ptr_lhs = 0, ptr_rhs = 0;
  svbool_t p;

  FOR_LOOP_32(uint32_t, i, 0, n, p) {
    svint32_t vec = svld1(p, data + i);

    svbool_t sel_lhs = svcmplt(p, vec, pivot);
    uint64_t inc_lhs = svcntp_b32(p, sel_lhs);
    svbool_t sto_lhs = svwhilelt_b32(0lu, inc_lhs);
    svint32_t vec_lhs = svcompact(sel_lhs, vec);
    svst1(sto_lhs, data + ptr_lhs, vec_lhs);
    ptr_lhs += inc_lhs;

    svbool_t sel_rhs = svnot_z(p, sel_lhs);
    uint64_t inc_rhs = svcntp_b32(p, sel_rhs);
    svbool_t sto_rhs = svwhilelt_b32(0lu, inc_rhs);
    svint32_t vec_rhs = svcompact(sel_rhs, vec);
    svst1(sto_rhs, temp + ptr_rhs, vec_rhs);
    ptr_rhs += inc_rhs;
  }

  FOR_LOOP_32(uint32_t, i, ptr_lhs, n, p)
  svst1(p, data + i, svld1(p, temp + (i - ptr_lhs)));

  quicksort(ptr_lhs, &data[0], threshold, temp);
  quicksort(ptr_rhs - 1, &data[ptr_lhs + 1], threshold, temp);
}
#elif defined(__ARM_FEATURE_SVE)  // Quicksort
static void quicksort(uint32_t n, int32_t *restrict data, uint32_t threshold,
                      VAR_UNUSED int32_t *restrict temp) {
  if (n <= threshold) return;

  if (n >= MIN_N_MEDIAN3) {
    uint32_t median_idx = find_pivot(n, data);
    if (median_idx != 0) swap_32(&data[0], &data[median_idx]);
  }
  int32_t pivot = data[0];

  uint64_t ptr_lhs = 0, ptr_rhs = 0;
  uint64_t inc_lhs = 0, inc_rhs = 0;

  uint32_t i = 0, j = 0;
  asm volatile(
      // broadcast pivot across a vector
      "   dup     z0.s, %w[pivot]                             \n"

      "   whilelt p1.s, %w[i], %w[n]                          \n"
      "   b.none  2f                                          \n"
      "1:                                                     \n"
      "   ld1w    z1.s, p1/z, [%[data], %x[i], lsl #2]        \n"
      "   incw    %x[i]                                       \n"

      // compress and store LHS
      "   cmplt   p2.s, p1/z, z1.s, z0.s                      \n"
      "   cntp    %[inc_lhs], p1, p2.s                        \n"
      "   compact z2.s, p2, z1.s                              \n"
      "   whilelt p4.s, xzr, %[inc_lhs]                       \n"
      "   st1w    z2.s, p4, [%[data], %[ptr_lhs], lsl #2]     \n"
      "   add     %[ptr_lhs], %[ptr_lhs], %[inc_lhs]          \n"

      // compress and store RHS
      "   not     p3.b, p1/z, p2.b                            \n"
      "   cntp    %[inc_rhs], p1, p3.s                        \n"
      "   compact z3.s, p3, z1.s                              \n"
      "   whilelt p4.s, xzr, %[inc_rhs]                       \n"
      "   st1w    z3.s, p4, [%[temp], %[ptr_rhs], lsl #2]     \n"
      "   add     %[ptr_rhs], %[ptr_rhs], %[inc_rhs]          \n"

      "   whilelt p1.s, %w[i], %w[n]                          \n"
      "   b.first 1b                                          \n"
      "2:                                                     \n"

      // copy buffers back to original
      // N.B. This may not be the most efficient way since we could
      // continue partitioning the values from the temporary buffers
      // however this would require some overhead
      // to track where the values currently are.
      // We would also need to modify the temp stores to
      // use the buffers+offset rather than buffers+0.
      "   mov     %x[i], %[ptr_lhs]                           \n"
      "   whilelt p1.s, %w[i], %w[n]                          \n"
      "   b.none  4f                                          \n"
      "3:                                                     \n"
      "   ld1w    z0.s, p1/z, [%[temp], %x[j], lsl #2]        \n"
      "   st1w    z0.s, p1  , [%[data], %x[i], lsl #2]        \n"
      "   incw    %x[i]                                       \n"
      "   incw    %x[j]                                       \n"
      "   whilelt p1.s, %w[i], %w[n]                          \n"
      "   b.first 3b                                          \n"
      "4:                                                     \n"

      : [inc_lhs] "+&r"(inc_lhs), [ptr_lhs] "+&r"(ptr_lhs),
        [inc_rhs] "+&r"(inc_rhs), [ptr_rhs] "+&r"(ptr_rhs), [i] "+&r"(i),
        [j] "+&r"(j)
      : [data] "r"(data), [temp] "r"(temp), [n] "r"(n), [pivot] "r"(pivot)
      : "z0", "z1", "z2", "z3", "z4", "p1", "p2", "p3", "p4", "cc", "memory");

  quicksort(ptr_lhs, &data[0], threshold, temp);
  quicksort(ptr_rhs - 1, &data[ptr_lhs + 1], threshold, temp);
}
#else

static void quicksort(uint32_t n, int32_t *restrict data, uint32_t threshold,
                      VAR_UNUSED int32_t *restrict temp) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif                   // Quicksort

static void NOINLINE do_sort(struct loop_121_data *input) {
  uint32_t n = input->n;
  int32_t *data = input->data;
  int32_t *temp = input->temp;

#ifdef SORT_NO_HYBRID
  quicksort(n, data, 1, temp);
#else
  quicksort(n, data, THRESHOLD, temp);
  tidy_step(n, data);
#endif
}

static void inner_loop_121(struct loop_121_data *input) {
  fill_int32(input->data, input->n);
  do_sort(input);
}
#endif /* !HAVE_CANDIDATE */

#ifndef SIZE
#define SIZE 256
#endif

LOOP_DECL(121, NS_SVE_LOOP_ATTR)
{
  struct loop_121_data data = { .n = SIZE, .temp = NULL, };

  ALLOC_64B(data.data, SIZE, "data array");
#ifdef __ARM_FEATURE_SVE
  ALLOC_64B(data.temp, SIZE, "intermediate buffer");
#endif

  inner_loops_121(iters, &data);

  int res = check_sorted(SIZE, data.data);
  bool passed = (res == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(121, passed, "%d", 0, res)
#endif
  return passed ? 0 : 1;
}
