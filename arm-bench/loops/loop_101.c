/*----------------------------------------------------------------------------
#
#   Loop 101: Upscale filter
#
#   Purpose:
#     Use of top/bottom instructions.
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


struct loop_101_data {
  uint8_t *restrict a;
  uint8_t *restrict b;
  int n;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_101(struct loop_101_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static void inner_loop_101(struct loop_101_data *restrict input) {
  uint8_t *restrict a = input->a;
  uint8_t *restrict b = input->b;
  int n = input->n;

  for (int i = 0; i < n - 1; i++) {
    uint16_t s1 = b[i];
    uint16_t s2 = b[i + 1];
    a[2 * i] = (3 * s1 + s2 + 2) >> 2;
    a[2 * i + 1] = (3 * s2 + s1 + 2) >> 2;
  }
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))

static void inner_loop_101(struct loop_101_data *restrict input)
LOOP_ATTR
{
  uint8_t *restrict a = input->a;
  uint8_t *restrict b = input->b;
  int n = input->n;

  svuint8_t three = svdup_u8(3);
  svbool_t p;
  FOR_LOOP_8(int, i, 0, n - 1, p) {
    svuint8_t b0 = svld1(p, b + i);
    svuint8_t b1 = svld1(p, b + 1 + i);

    svuint16_t b0_u16 = svreinterpret_u16(b0);
    svuint16_t b0_ext = svextb_x(p, b0_u16);
    svuint16_t b0_evn = svmlalb(b0_ext, b1, three);
    svuint16_t b0_ls8 = svlsr_x(p, b0_u16, 8);
    svuint16_t b0_odd = svmlalt(b0_ls8, b1, three);

    svuint16_t b1_u16 = svreinterpret_u16(b1);
    svuint16_t b1_ext = svextb_x(p, b1_u16);
    svuint16_t b1_evn = svmlalb(b1_ext, b0, three);
    svuint16_t b1_ls8 = svlsr_x(p, b1_u16, 8);
    svuint16_t b1_odd = svmlalt(b1_ls8, b0, three);

    svuint8_t a0 = svrshrnt(svrshrnb(b1_evn, 2), b1_odd, 2);
    svuint8_t a1 = svrshrnt(svrshrnb(b0_evn, 2), b0_odd, 2);
    svst2(p, a, svcreate2(a0, a1));

    a += svcntb() * 2;
  }
}
#elif (defined(__ARM_FEATURE_SVE2) || defined(__ARM_FEATURE_SME))

static void inner_loop_101(struct loop_101_data *restrict input)
LOOP_ATTR
{
  uint8_t *restrict a = input->a;
  uint8_t *restrict b = input->b;
  int n = input->n;

  uint64_t i = 0;
  uint64_t lmt = (uint64_t)n - 1;
  uint8_t *b0 = b;
  uint8_t *b1 = b + 1;
  uint8_t *dst = a;

  asm volatile(
      "        whilelo p0.b, %[i], %[lmt]             \n"
      "        mov     z3.b, #3                       \n"
      "1:      ld1b    {z1.b}, p0/z, [%[b1], %[i]]    \n"
      "        ld1b    {z0.b}, p0/z, [%[b0], %[i]]    \n"
      "        incb    %[i]                           \n"
      "        movprfx z4, z1                         \n"
      "        uxtb    z4.h, p0/m, z1.h               \n"
      "        lsr     z5.h, z1.h, #8                 \n"
      "        movprfx z6, z0                         \n"
      "        uxtb    z6.h, p0/m, z0.h               \n"
      "        lsr     z7.h, z0.h, #8                 \n"
      "        umlalb  z4.h, z0.b, z3.b               \n"
      "        umlalt  z5.h, z0.b, z3.b               \n"
      "        umlalb  z6.h, z1.b, z3.b               \n"
      "        umlalt  z7.h, z1.b, z3.b               \n"
      "        rshrnb  z0.b, z4.h, #2                 \n"
      "        rshrnt  z0.b, z5.h, #2                 \n"
      "        rshrnb  z1.b, z6.h, #2                 \n"
      "        rshrnt  z1.b, z7.h, #2                 \n"
      "        st2b    {z0.b, z1.b}, p0, [%[dst]]     \n"
      "        incb    %[dst], all, mul #2            \n"
      "        whilelo p0.b, %[i], %[lmt]             \n"
      "        b.any   1b                             \n"  // loop back
      // output operands, source operands, and clobber list
      : [i] "+&r"(i), [dst] "+&r"(dst)
      : [lmt] "r"(lmt), [b0] "r"(b0), [b1] "r"(b1)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "p0", "cc", "memory");
}
#elif defined(__ARM_NEON)

static void inner_loop_101(struct loop_101_data *restrict input) {
  uint8_t *restrict a = input->a;
  uint8_t *restrict b = input->b;
  int n = input->n;

  uint64_t i = 0;
  uint64_t lmt = (uint64_t)n - 1;
  lmt -= lmt % 16;
  uint8_t *b0 = b;
  uint8_t *b1 = b + 1;
  uint8_t *dst = a;

  asm volatile(
      "       movi    v3.16b, #0x3                    \n"
      "       b       2f                              \n"
      "1:     ldr     q0, [%[b0], %[i]]               \n"
      "       ldr     q1, [%[b1], %[i]]               \n"
      "       add     %[i], %[i], #16                 \n"
      "       uxtl    v4.8h, v1.8b                    \n"
      "       uxtl2   v5.8h, v1.16b                   \n"
      "       uxtl    v6.8h, v0.8b                    \n"
      "       uxtl2   v7.8h, v0.16b                   \n"
      "       umlal   v4.8h, v0.8b, v3.8b             \n"
      "       umlal2  v5.8h, v0.16b, v3.16b           \n"
      "       umlal   v6.8h, v1.8b, v3.8b             \n"
      "       umlal2  v7.8h, v1.16b, v3.16b           \n"
      "       rshrn   v0.8b, v4.8h, #2                \n"
      "       rshrn2  v0.16b, v5.8h, #2               \n"
      "       rshrn   v1.8b, v6.8h, #2                \n"
      "       rshrn2  v1.16b, v7.8h, #2               \n"
      "       st2     {v0.16b, v1.16b}, [%[dst]], #32 \n"
      "2:     cmp     %[i], %[lmt]                    \n"
      "       b.lt    1b                              \n"  // loop back
      // output operands, source operands, and clobber list
      : [i] "+&r"(i), [dst] "+&r"(dst)
      : [lmt] "r"(lmt), [b0] "r"(b0), [b1] "r"(b1)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "cc", "memory");

  for (; i < n - 1; i++) {
    uint16_t s1 = b[i];
    uint16_t s2 = b[i + 1];
    a[2 * i] = (3 * s1 + s2 + 2) >> 2;
    a[2 * i + 1] = (3 * s2 + s1 + 2) >> 2;
  }
}
#else

static void inner_loop_101(struct loop_101_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#ifndef SIZE
#define SIZE 10000
#endif

LOOP_DECL(101, SC_SVE_LOOP_ATTR)
{
  struct loop_101_data data = { .n = SIZE };

  ALLOC_64B(data.a, SIZE * 2, "output buffer");
  ALLOC_64B(data.b, SIZE, "input data");

  fill_uint8(data.a, SIZE * 2);
  fill_uint8(data.b, SIZE);

  inner_loops_101(iters, &data);

  uint32_t checksum = 0;
  for (int i = 0; i < 2 * SIZE; i++) {
    checksum ^= data.a[i] << (8 * (i % 4));
  }

  bool passed = checksum == 0xfb82fb61;
#ifndef STANDALONE
  FINALISE_LOOP_I(101, passed, "0x%08"PRIx32, 0xfb82fb61, checksum)
#endif
  return passed ? 0 : 1;
}
