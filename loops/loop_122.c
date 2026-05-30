/*----------------------------------------------------------------------------
#
#   Loop 122: Odd-Even transposition sort
#
#   Purpose:
#     Use of CMPLT with SEL instructions.
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


struct loop_122_data {
  uint32_t n;
  int32_t *restrict data;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_122(struct loop_122_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static void NOINLINE do_sort(struct loop_122_data *restrict input) {
  uint32_t n = input->n;
  int32_t *data = input->data;

  uint32_t i;
  bool sorted;
  do {
    sorted = true;
    for (i = 1; i < n; i += 2) {
      if (data[i - 1] > data[i]) {
        swap_32(&data[i - 1], &data[i]);
        sorted = false;
      }
    }
    for (i = 1; i < n - 1; i += 2) {
      if (data[i + 1] < data[i]) {
        swap_32(&data[i + 1], &data[i]);
        sorted = false;
      }
    }
  } while (!sorted);
}
#elif defined(__ARM_FEATURE_SVE)
static void NOINLINE do_sort(struct loop_122_data *restrict input) {
  uint32_t n = input->n;
  int32_t *data = input->data;
  com_sort_oet(n, data, n);
}
#else

static void NOINLINE do_sort(struct loop_122_data *restrict input) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#if !defined(HAVE_CANDIDATE)

static void inner_loop_122(struct loop_122_data *restrict input) {
  fill_int32(input->data, input->n);
  do_sort(input);
}
#endif /* !HAVE_CANDIDATE */

#ifndef SIZE
#define SIZE 256
#endif

LOOP_DECL(122, NS_SVE_LOOP_ATTR)
{
  struct loop_122_data data = { .n = SIZE };

  ALLOC_64B(data.data, SIZE, "data array");

  inner_loops_122(iters, &data);

  int res = check_sorted(SIZE, data.data);
  bool passed = (res == 0);
#ifndef STANDALONE
  FINALISE_LOOP_I(122, passed, "%d", 0, res)
#endif
  return passed ? 0 : 1;
}
