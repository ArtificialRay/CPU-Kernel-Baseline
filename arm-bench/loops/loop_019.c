/*----------------------------------------------------------------------------
#
#   Loop 019: Mark objects
#
#   Purpose:
#     Use of scatters store instruction.
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


typedef struct object {
  uint64_t payload;
  uint32_t mark;
  uint32_t payload2;
} object_t;

struct loop_019_data {
  object_t *restrict objects;
  uint32_t *restrict indexes;
  int64_t n;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_019(struct loop_019_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_019(struct loop_019_data *restrict data) {
  object_t *objects = data->objects;
  uint32_t *indexes = data->indexes;
  int64_t n = data->n;

  for (int i = 0; i < n; i++) {
    objects[indexes[i]].mark = 1;
  }
}
#elif defined(HAVE_SVE_INTRINSICS)

static void inner_loop_019(struct loop_019_data *restrict data) {
  object_t *objects = data->objects;
  uint32_t *indexes = data->indexes;
  int64_t n = data->n;

  svuint32_t ones = svdup_u32(1);

  svbool_t p;
  FOR_LOOP_32(int64_t, i, 0, n, p) {
    svuint32_t idx = svld1(p, indexes + i);
    idx = svlsl_x(p, idx, 2);
    idx = svadd_x(p, idx, 2);
    svst1_scatter_index(p, (uint32_t *)objects, idx, ones);
  }
}
#elif defined(__ARM_FEATURE_SVE)

static void inner_loop_019(struct loop_019_data *restrict data) {
  object_t *objects = data->objects;
  uint32_t *indexes = data->indexes;
  int64_t n = data->n;

  int64_t i = 0;

  asm volatile(
      "   mov     z1.s, #1                                    \n"
      "   b       2f                                          \n"
      "1: ld1w    {z0.s}, p0/z, [%[indexes], %[i], lsl #2]    \n"
      "   incw    %[i]                                        \n"
      "   lsl     z0.s, z0.s, #2                              \n"
      "   add     z0.s, z0.s, #2                              \n"
      "   st1w    {z1.s}, p0, [%[objects], z0.s, uxtw #2]     \n"
      "2: whilelo p0.s, %[i], %[n]                            \n"
      "   b.any   1b                                          \n"
      // output operands, source operands, and clobber list
      : [i] "+&r"(i)
      : [objects] "r"(objects), [indexes] "r"(indexes), [n] "r"(n)
      : "v0", "v1", "p0", "cc", "memory");
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)

// Scalar and Neon version (can't do better with Neon)
static void inner_loop_019(struct loop_019_data *restrict data) {
  object_t *objects = data->objects;
  uint32_t *indexes = data->indexes;
  int64_t n = data->n;

  uint64_t one = 1;
  uint64_t idx;
  object_t *objs = objects;
  uint32_t *lmt = indexes + n;

  asm volatile(
      "1:   ldr   %w[idx], [%[indexes]], #4         \n"
      "     add   %[idx], %[objs], %[idx], lsl #4   \n"
      "     str   %w[one], [%[idx], #8]             \n"
      "     cmp   %[indexes], %[lmt]                \n"
      "     b.ne  1b                                \n"
      // output operands, source operands, and clobber list
      : [idx] "=&r"(idx), [indexes] "+&r"(indexes)
      : [one] "r"(one), [lmt] "r"(lmt), [objs] "r"(objs)
      : "memory", "cc");
}
#else
static void inner_loop_019(struct loop_019_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 8000
#endif

LOOP_DECL(019, NS_SVE_LOOP_ATTR)
{
  struct loop_019_data data = { .n = SIZE };

  ALLOC_64B(data.objects, SIZE, "object array");
  ALLOC_64B(data.indexes, SIZE, "index buffer");

  memset(data.objects, 0, SIZE * sizeof(object_t));
  for (int i = 0; i < SIZE; i++) {
    data.indexes[i] = i;
  }

  // shuffle indexes evenly
  for (uint32_t i = 0; i < SIZE - 1; i++) {
    uint32_t other = (i + 1) + rand_uint32() % (SIZE - (i + 1));
    uint32_t tmp = data.indexes[i];
    data.indexes[i] = data.indexes[other];
    data.indexes[other] = tmp;
  }

  inner_loops_019(iters, &data);

  uint32_t res = 0;
  for (int i = 0; i < SIZE; i++) {
    res += data.objects[i].mark;
    res += data.objects[i].payload << 16;
  }

  bool passed = res == 0x00001f40;
#ifndef STANDALONE
  FINALISE_LOOP_I(19, passed, "0x%08"PRIx32, 0x00001f40, res)
#endif
  return passed ? 0 : 1;
}
