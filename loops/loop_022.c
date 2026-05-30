/*----------------------------------------------------------------------------
#
#   Loop 022: TCP checksum
#
#   Purpose:
#     Use of simd instructions for misaligned accesses.
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


struct loop_022_data {
  uint8_t *p;
  uint8_t *lmt;
  uint32_t checksum;
};

#define LOOP_ATTR SC_SVE_ATTR

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_022(struct loop_022_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)

static uint16_t NOINLINE tcp_checksum(uint8_t *data, uint16_t len) {
  uint64_t sum = 0;
  uint8_t *lmt = data + len;

  for (uint8_t *p = data; p < lmt; p += 2) {
    uint16_t word = *(uint16_t *)p;
    sum += word;
  }

  // only need one folding step since the number of accumulation steps cannot
  // exceed the range of a 32-bit integer.
  return ~((sum & 0xffff) + (sum >> 16));
}
#elif (defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS))

static uint16_t NOINLINE tcp_checksum(uint8_t *data, uint16_t len)
LOOP_ATTR
{
  svuint16_t ones = svdup_u16(1);
  svuint64_t acc = svdup_u64(0);

  svbool_t p;
  uint64_t n = len;
  FOR_LOOP_8(uint64_t, i, 0, n, p) {
    svuint16_t d = svreinterpret_u16(svld1(p, data + i));
    acc = svdot(acc, d, ones);
  }
  uint64_t sum = svaddv(svptrue_b64(), acc);

  // only need one folding step since the number of accumulation steps cannot
  // exceed the range of a 32-bit integer.
  return ~((sum & 0xffff) + (sum >> 16));
}
#elif (defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SME))

static uint16_t NOINLINE tcp_checksum(uint8_t *data, uint16_t len)
LOOP_ATTR
{
  uint8_t *lmt = data + len;
  uint8_t *p = data;
  uint64_t sum;

  asm volatile(
      "       ptrue   p0.b                              \n"
      "       mov     z0.h, #1                          \n"
      "       mov     z10.d, #0                         \n"
      "       whilelo p1.b, %[p], %[lmt]                \n"
      "1:     ld1h    {z1.h}, p1/z, [%[p]]              \n"
      "       udot    z10.d, z1.h, z0.h                 \n"
      "       incb    %[p]                              \n"
      "       whilelo p1.b, %[p], %[lmt]                \n"
      "       b.any   1b                                \n"
      // reduce to a single accumulator
      "       uaddv   d0, p0, z10.d                     \n"
      "       fmov    %[sum], d0                        \n"
      // output operands, source operands, and clobber list
      : [sum] "=&r"(sum), [p] "+&r"(p)
      : [lmt] "r"(lmt)
      : "x10", "v0", "v1", "v2", "v3", "v4", "v10", "v11", "v12", "v13",
        "p0", "p1", "cc", "memory");

  // only need one folding step since the number of accumulation steps cannot
  // exceed the range of a 32-bit integer.
  return ~((sum & 0xffff) + (sum >> 16));
}
#elif defined(__ARM_NEON)

static uint16_t NOINLINE tcp_checksum(uint8_t *data, uint16_t len) {
  uint8_t *lmt = data + len;
  uint8_t *p = data;
  uint32_t sum;
  asm volatile(
      "        movi v1.4s, #0                  \n"
      "        movi v2.4s, #0                  \n"
      "        add %[p], %[p], #14             \n"
      "        cmp %[p], %[lmt]                \n"
      "        b.ge 2f                         \n"
      "1:                                      \n"
      "        ldr q0, [%[p], #-14]            \n"
      "        uaddw v1.4s, v1.4s, v0.4h       \n"
      "        uaddw2 v2.4s, v2.4s, v0.8h      \n"
      "        add %[p], %[p], #16             \n"
      "        cmp %[p], %[lmt]                \n"
      "        b.lt 1b                         \n"
      "2:                                      \n"
      "        sub %[p], %[p], #8              \n"
      "        cmp %[p], %[lmt]                \n"
      "        b.ge 2f                         \n"
      "1:                                      \n"
      "        ldr d0, [%[p], #-6]             \n"
      "        uaddw v1.4s, v1.4s, v0.4h       \n"
      "        add %[p], %[p], #8              \n"
      "        cmp %[p], %[lmt]                \n"
      "        b.lt 1b                         \n"
      "2:                                      \n"
      "        sub %[p], %[p], #6              \n"
      "        add v1.4s, v1.4s, v2.4s         \n"
      "        addv s0, v1.4s                  \n"
      "        fmov %w[sum], s0                \n"
      "        cmp %[p], %[lmt]                \n"
      "        b.ge 2f                         \n"
      "1:                                      \n"
      "        ldrh w0, [%[p]], #2             \n"
      "        add %w[sum], %w[sum], w0        \n"
      "        cmp %[p], %[lmt]                \n"
      "        b.lt 1b                         \n"
      "2:                                      \n"
      : [sum] "=&r"(sum), [p] "+&r"(p)
      : [lmt] "r"(lmt)
      : "w0", "v0", "v1", "v2", "cc", "memory");
  // only need one folding step since the number of accumulation steps cannot
  // exceed the range of a 32-bit integer.
  return ~((sum & 0xffff) + (sum >> 16));
}
#elif defined(__aarch64__) && !defined(HAVE_AUTOVEC)

static uint16_t NOINLINE tcp_checksum(uint8_t *data, uint16_t len) {
  uint64_t sum = 0;
  uint64_t sum2 = 0;
  uint8_t *p = data;
  uint8_t *lmt = data + (len - (len % 32));
  uint16_t res;
  uint64_t len64 = len;

  asm volatile(
      "       cmp     %[len], #32                  \n"
      "       b.lt    2f                           \n"
      "1:     ldp     x10, x11, [%[p]]             \n"
      "       ldp     x12, x13, [%[p], #16]        \n"
      "       adds    %[sum], %[sum], x10          \n"
      "       adcs    %[sum], %[sum], x11          \n"
      "       adc     %[sum], %[sum], xzr          \n"
      "       adds    %[sum2], %[sum2], x12        \n"
      "       adcs    %[sum2], %[sum2], x13        \n"
      "       adc     %[sum2], %[sum2], xzr        \n"
      "       add     %[p], %[p], #32              \n"
      "       cmp     %[p], %[lmt]                 \n"
      "       b.lt    1b                           \n"
      "       and     %[len], %[len], 0x1f         \n"
      "       adds    %[sum], %[sum], %[sum2]      \n"
      "       adc     %[sum], %[sum], xzr          \n"
      "2:     cmp     %[len], #16                  \n"
      "       b.lt    3f                           \n"
      "       ldp     x10, x11, [%[p]]             \n"
      "       adds    %[sum], %[sum], x10          \n"
      "       adcs    %[sum], %[sum], x11          \n"
      "       adc     %[sum], %[sum], xzr          \n"
      "       sub     %[len], %[len], #16          \n"
      "       add     %[p], %[p], #16              \n"
      "3:     cmp     %[len], #8                   \n"
      "       b.lt    4f                           \n"
      "       ldr     x10, [%[p]]                  \n"
      "       adds    %[sum], %[sum], x10          \n"
      "       adc     %[sum], %[sum], xzr          \n"
      "       sub     %[len], %[len], #8           \n"
      "       add     %[p], %[p], #8               \n"
      "4:     cmp     %[len], #4                   \n"
      "       b.lt    5f                           \n"
      "       ldr     w10, [%[p]]                  \n"
      "       adds    %[sum], %[sum], x10          \n"
      "       adc     %[sum], %[sum], xzr          \n"
      "       sub     %[len], %[len], #4           \n"
      "       add     %[p], %[p], #4               \n"
      "5:     cmp     %[len], #2                   \n"
      "       b.lt    6f                           \n"
      "       ldrh    w10, [%[p]]                  \n"
      "       adds    %[sum], %[sum], x10          \n"
      "       adc     %[sum], %[sum], xzr          \n"
      "       sub     %[len], %[len], #2           \n"
      "       add     %[p], %[p], #2               \n"
      "6:     cbz     %[len], 7f                   \n"
      "       ldrb    w10, [%[p]]                  \n"
      "       adds    %[sum], %[sum], x10          \n"
      "       adc     %[sum], %[sum], xzr          \n"
      "7:     lsr     x10, %[sum], 32              \n"  // Fold 64-bit into 16
      "       adds    %w[sum], %w[sum], w10        \n"
      "       adc     %w[sum], %w[sum], wzr        \n"
      "       lsr     x10, %[sum], 16              \n"
      "       and     %[sum], %[sum], 0xffff       \n"  // 1st time may
      "       add     %w[sum], %w[sum], w10        \n"  // overflow into 17 bits
      "       lsr     x10, %[sum], 16              \n"
      "       and     %[sum], %[sum], 0xffff       \n"
      "       add     %w[sum], %w[sum], w10        \n"  // can't overflow
      "       mvn     %w[sum], %w[sum]             \n"
      "       and     %w[res], %w[sum], 0xffff     \n"
      // output operands, source operands, and clobber list
      : [res] "=&r"(res), [sum] "+&r"(sum), [sum2] "+&r"(sum2), [len] "+&r"(len64),
        [p] "+&r"(p)
      : [lmt] "r"(lmt)
      : "x10", "x11", "x12", "x13", "cc", "memory");

  return res;
}
#else
static void inner_loop_022(struct loop_022_data *data) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#if !defined(HAVE_CANDIDATE)

static void inner_loop_022(struct loop_022_data *restrict data)
LOOP_ATTR
{
  uint8_t *p = data->p;
  uint8_t *lmt = data->lmt;

  uint32_t res = 0;
  while (p < lmt) {
    uint16_t *plength = (void *)(p + 1);
    uint16_t length = *plength & 0xfe;
    uint16_t checksum = tcp_checksum(p, length);
    p += *plength;
    res += 1;
    res ^= checksum << 16;
  }
  data->checksum = res;
}
#endif /* !HAVE_CANDIDATE */

#ifndef SIZE
#define SIZE 20000
#endif

LOOP_DECL(022, SC_SVE_LOOP_ATTR)
{
  struct loop_022_data data = { .checksum = 0 };
  ALLOC_64B(data.p, SIZE, "string buffer");
  fill_uint8(data.p, SIZE);

  int min_packet = 55;
  int max_packet = 255;
  int range = max_packet - min_packet;
  int start = 0;
  int end = (rand_uint32() % range) + min_packet;
  int last = 0;
  while (end < SIZE) {
    uint16_t *plength = (void *)(data.p + start + 1);
    *plength = end - start;
    last = end;
    start = end;
    end = start + (rand_uint32() % range) + min_packet;
  }
  data.lmt = data.p + last;

  inner_loops_022(iters, &data);

  uint32_t checksum = data.checksum;
  uint32_t correct = 0x0b1a0081;
  bool passed = checksum == correct;
#ifndef STANDALONE
  FINALISE_LOOP_I(22, passed, "0x%08"PRIx32, correct, checksum)
#endif
  return passed ? 0 : 1;
}
