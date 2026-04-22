/*----------------------------------------------------------------------------
#
#   Loop 034: Short string compares
#
#   Purpose:
#     Use of FFR for strcmp.
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


struct loop_034_data {
  uint8_t *a;
  uint8_t *b;
  uint8_t *lmt;
  uint32_t checksum;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_034(struct loop_034_data *restrict input) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)input;
}
// CANDIDATE_INJECT_END
#else
static void inner_loop_034(struct loop_034_data *restrict input) {
  uint8_t *p1 = input->a;
  uint8_t *p2 = input->b;
  uint8_t *lmt = input->lmt;

  uint32_t res = 0;
  uint32_t cnt = 0;
  int length = 13;
  while (p1 < lmt) {
    int64_t r = strcmp_opt(p1, p2);
    uint32_t cmp = 1;
    if (r > 0) cmp = 2;
    if (r < 0) cmp = 3;
    res += cnt * cmp;
    p1 += length;
    p2 += length;
    cnt++;
    length = 3 + (length + 11) % 43;
  }
  input->checksum = res;
}
#endif

#ifndef SIZE
#define SIZE 6000
#endif

LOOP_DECL(034, NS_SVE_LOOP_ATTR)
{
  if (SIZE >= sample_json_size) {
    printf("Size %d too big\n", SIZE);
    return 1;
  }

  struct loop_034_data data = { .checksum = 0 };

  ALLOC_64B(data.a, SIZE, "input string 1");
  ALLOC_64B(data.b, SIZE, "input string 2");

  memcpy(data.a, sample_json, SIZE);
  memcpy(data.b, sample_json, SIZE);

  int length = 13;
  int idx = length;
  int last = 0;
  while (idx < SIZE) {
    data.a[idx] = 0;
    data.b[idx] = 0;
    data.b[idx - 1] -= data.b[idx - 1] % 2;  // clear last bit
    last = idx;
    length = 3 + (length + 11) % 43;
    idx += length;
  }
  data.lmt = data.a + last;

  inner_loops_034(iters, &data);

  uint32_t checksum = data.checksum;
  uint32_t correct = 0x00007a8f;
  bool passed = checksum == correct;
#ifndef STANDALONE
  FINALISE_LOOP_I(34, passed, "0x%08"PRIx32, correct, checksum)
#endif
  return passed ? 0 : 1;
}
