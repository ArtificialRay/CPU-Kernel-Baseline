/*----------------------------------------------------------------------------
#
#   Loop 005: strlen short strings
#
#   Purpose:
#     Use of FF and NF loads instructions.
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


struct loop_005_data {
  uint8_t *p;
  uint8_t *lmt;
  uint32_t checksum;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_005(struct loop_005_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#else
static void inner_loop_005(struct loop_005_data *restrict data) {
  uint8_t *p = data->p;
  uint8_t *lmt = data->lmt;

  uint32_t res = 0;
  while (p < lmt) {
    uint32_t len = strlen_opt(p);
    p += len + 1;
    res += 1;
    res ^= (len % 0xffff) << 16;
  }
  data->checksum = res;
}
#endif

#ifndef SIZE
#define SIZE 4000
#endif

LOOP_DECL(005, NS_SVE_LOOP_ATTR)
{
  if (SIZE >= sample_json_size) {
    printf("Size %d too big\n", SIZE);
    return 1;
  }

  struct loop_005_data data = { .checksum = 0 };
  ALLOC_64B(data.p, SIZE, "string buffer");
  memcpy(data.p, sample_json, SIZE);

  int short_strings = 20;
  int idx = (rand_uint32() % short_strings) + 2;
  int last = 0;
  while (idx < SIZE) {
    data.p[idx] = 0;
    last = idx;
    idx += (rand_uint32() % short_strings) + 2;
  }
  data.lmt = data.p + last;

  inner_loops_005(iters, &data);

  uint32_t checksum = data.checksum;
  uint32_t correct = 0x00170161;
  bool passed = checksum == correct;
#ifndef STANDALONE
  FINALISE_LOOP_I(5, passed, "0x%08"PRIx32, correct, checksum)
#endif
  return passed ? 0 : 1;
}
