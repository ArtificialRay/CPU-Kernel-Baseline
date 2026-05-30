/*----------------------------------------------------------------------------
#
#   Loop 120: Insertion sort
#
#   Purpose:
#     Use of CMPLT instruction.
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


struct loop_120_data {
  uint32_t n;
  int32_t *restrict data;
};


#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_120(struct loop_120_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

// Inner loop
static void NOINLINE do_sort(struct loop_120_data *restrict input) {
  uint32_t n = input->n;
  int32_t *data = input->data;

  com_sort_insertion(n, data);
}
#elif defined(HAVE_SVE_INTRINSICS)  // Inner loop
static void NOINLINE do_sort(struct loop_120_data *restrict input) {
  uint32_t n = input->n;
  int32_t *data = input->data;

  for (uint32_t i = n - 1; i > 0; i--) {
    int32_t tmp = data[i - 1];
    svint32_t u = svdup_s32(tmp);

    svbool_t p, q;
    uint32_t j, k;
    for (j = i; FOR_COND(p, 32, j, n);) {
      svint32_t v = svld1(p, data + j);
      q = svcmpge(p, v, u);
      if (!svptest_any(p, q)) {
        svst1(p, data + j - 1, v);
        j += svcntp_b32(p, p);
        continue;
      }
      q = svbrkb_z(p, q);
      svst1(q, data + j - 1, v);
      k = svcntp_b32(q, q);
      j += k;
      if (k == 0) break;
    }

    data[j - 1] = tmp;
  }
}
#elif defined(__ARM_FEATURE_SVE)  // Inner loop
static void NOINLINE do_sort(struct loop_120_data *restrict input) {
  uint32_t n = input->n;
  int32_t *data = input->data;

  for (uint32_t i = n - 1; i > 0; i--) {
    int32_t tmp = data[i - 1];

    uint32_t j = i, k;
    asm volatile(
        // broadcast current value across a vector
        "   dup     z0.s, %w[tmp]                           \n"

        "   whilelt p0.s, %w[j], %w[n]                      \n"
        "   b.none  99f                                     \n"
        "1:                                                 \n"
        "   ld1w    z1.s, p0/z, [%[data], %x[j], lsl #2]    \n"

        // shift all subsequent elements less than current value
        "   cmplt   p1.s, p0/z, z0.s, z1.s                  \n"
        "   b.any   2f                                      \n"

        "   st1w    {z1.s}, p0, [%[prev], %x[j], lsl #2]    \n"
        "   incp    %x[j], p0.s                             \n"
        "   b 3f                                            \n"
        "2:                                                 \n"
        "   brkb    p1.b, p0/z, p1.b                        \n"
        "   st1w    {z1.s}, p1, [%[prev], %x[j], lsl #2]    \n"

        // increment pointer and break if greater value encountered
        "   cntp    %x[k], p0, p1.s                         \n"
        "   add     %x[j], %x[j], %x[k]                     \n"
        "   cbz     %x[k], 99f                              \n"
        "3:                                                 \n"
        "   whilelt p0.s, %w[j], %w[n]                      \n"
        "   b.first 1b                                      \n"
        "99:                                                \n"
        : [j] "+&r"(j), [k] "=&r"(k)
        : [data] "r"(data), [prev] "r"((uint64_t)data - 4), [tmp] "r"(tmp),
          [n] "r"(n)
        : "z0", "z1", "p0", "p1", "cc", "memory");

    data[j - 1] = tmp;
  }
}
#else
static void NOINLINE do_sort(struct loop_120_data *restrict input) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif                   // Inner loop

#if !defined(HAVE_CANDIDATE)

static void inner_loop_120(struct loop_120_data *restrict input) {
  fill_int32(input->data, input->n);
  do_sort(input);
}
#endif /* !HAVE_CANDIDATE */

#ifndef SIZE
#define SIZE 256
#endif

LOOP_DECL(120, NS_SVE_LOOP_ATTR)
{
  struct loop_120_data data = { .n = SIZE };

  ALLOC_64B(data.data, SIZE, "data array");

  inner_loops_120(iters, &data);

  int res = check_sorted(SIZE, data.data);
  bool passed = (res == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(120, passed, "%d", 0, res)
#endif
  return passed ? 0 : 1;
}
