/*----------------------------------------------------------------------------
#
#   Loop 103: whitespace
#
#   Purpose:
#     Use of MATCH and NMATCH instructions.
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


struct loop_103_data {
  uint8_t *p;
  uint8_t *end;
  int checksum;
};

#if defined(HAVE_CANDIDATE)
// CANDIDATE_INJECT_START
static void inner_loop_103(struct loop_103_data *restrict data) {
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}
// CANDIDATE_INJECT_END
#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static uint8_t *NOINLINE skip_whitespace(uint8_t *p, uint8_t *end) {
  while (p != end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) {
    p++;
  }
  return p;
}
#elif defined(HAVE_SVE_INTRINSICS)
static uint8_t *NOINLINE skip_whitespace(uint8_t *p, uint8_t *end) {
  svuint8_t pat = svreinterpret_u8(svdup_u32(0x200A0D09));

  uint64_t ptr;
  svbool_t pg;
  FOR_LOOP_8(, ptr, (uint64_t)p, (uint64_t)end, pg) {
    svuint8_t c = svld1(pg, (uint8_t *)ptr);
    svbool_t pm = svnmatch(pg, c, pat);
    if (svptest_any(svptrue_b8(), pm)) {
      pm = svbrkb_z(pg, pm);
      ptr += svcntp_b8(pm, pm);
      break;
    }
  }

  return ptr < (uint64_t)end ? (uint8_t *)ptr : end;
}
#elif defined(__ARM_FEATURE_SVE2)
static uint8_t *NOINLINE skip_whitespace(uint8_t *p, uint8_t *end) {
  uint32_t pattern = 0x200A0D09;  // pattern ' \n\r\t'
  asm volatile(
      "        dup     z1.s, %w[pattern]          \n"
      "        whilelt p0.b, %[p], %[end]         \n"
      "1:      ld1b    z0.b, p0/z, [%[p]]         \n"
      "        nmatch  p1.b, p0/z, z0.b, z1.b     \n"
      "        b.any   2f                         \n"  // any match?
      "        incb    %[p]                       \n"
      "        whilelt p0.b, %[p], %[end]         \n"
      "        b.first 1b                         \n"  // loop back
      "        mov     %[p], %[end]               \n"  // point p to end
      "        b       3f                         \n"  // end of buffer, exit
      "2:      brkb    p1.b, p0/z, p1.b           \n"
      "        incp    %[p], p1.b                 \n"
      "3:                                         \n"
      // output operands, source operands, and clobber list
      : [p] "+&r"(p)
      : [end] "r"(end), [pattern] "r"(pattern)
      : "v0", "v1", "p0", "p1", "cc", "memory");

  return p;
}
#elif defined(__ARM_FEATURE_SVE)
static uint8_t *NOINLINE skip_whitespace(uint8_t *p, uint8_t *end) {
  asm volatile(
      "        dup     z4.b, 0x20               \n"  // space
      "        whilelt p0.b, %[p], %[end]       \n"
      "1:      ld1b    z0.b, p0/z, [%[p]]       \n"
      "        cmpne   p1.b, p0/z, z0.b, 0x09   \n"  // == tab?
      "        cmpne   p1.b, p1/z, z0.b, 0x0A   \n"  // == new line?
      "        cmpne   p1.b, p1/z, z0.b, 0x0D   \n"  // == return?
      "        cmpne   p1.b, p1/z, z0.b, z4.b   \n"
      "        b.any   2f                       \n"  // any match?
      "        incb    %[p]                     \n"
      "        whilelt p0.b, %[p], %[end]       \n"
      "        b.first 1b                       \n"  // loop back
      "        mov     %[p], %[end]             \n"  // point p to end
      "        b       3f                       \n"  // end of buffer, exit
      "2:      brkb    p1.b, p0/z, p1.b         \n"
      "        incp    %[p], p1.b               \n"
      "3:                                       \n"
      // output operands, source operands, and clobber list
      : [p] "+&r"(p)
      : [end] "r"(end)
      : "v0", "v4", "p0", "p1", "cc", "memory");

  return p;
}
#elif defined(__ARM_NEON)

static uint8_t *NOINLINE skip_whitespace(uint8_t *p, uint8_t *end) {
  uint8_t *lmt = end - 16;
  uint8_t *rslt;
  uint64_t found = 0;
  uint64_t low, high;

  asm volatile(
      "        movi    v10.16b, #0x20                   \n"
      "        movi    v11.16b, #0x0a                   \n"
      "        movi    v12.16b, #0x0d                   \n"
      "        movi    v13.16b, #0x09                   \n"
      "        b       2f                               \n"
      "1:      ldr     q4, [%[p]], #16                  \n"
      "        cmeq    v0.16b, v4.16b, v10.16b          \n"
      "        cmeq    v1.16b, v4.16b, v11.16b          \n"
      "        cmeq    v2.16b, v4.16b, v12.16b          \n"
      "        cmeq    v3.16b, v4.16b, v13.16b          \n"
      "        orr     v0.16b, v0.16b, v3.16b           \n"
      "        orr     v0.16b, v0.16b, v2.16b           \n"
      "        orr     v0.16b, v0.16b, v1.16b           \n"
      "        not     v0.16b, v0.16b                   \n"
      "        mov     %[low], v0.d[0]                  \n"
      "        mov     %[high], v0.d[1]                 \n"
      "        cbnz    %[low], 3f                       \n"
      "        cbnz    %[high], 4f                      \n"
      "2:      cmp     %[p], %[lmt]                     \n"
      "        b.lt    1b                               \n"  // loop back
      "        b       5f                               \n"
      "3:      mov     %[found], #1                     \n"
      "        rev64   %[low], %[low]                   \n"
      "        clz     %[low], %[low]                   \n"
      "        asr     %w[low], %w[low], #3             \n"
      "        sub     %[p], %[p], #16                  \n"
      "        add     %[rslt], %[p], %w[low], sxtw     \n"
      "        b       5f                               \n"
      "4:      mov     %[found], #1                     \n"
      "        rev64   %[high], %[high]                 \n"
      "        clz     %[high], %[high]                 \n"
      "        asr     %w[high], %w[high], #3           \n"
      "        add     %[rslt], %[p], %w[high], sxtw    \n"
      "        sub     %[rslt], %[rslt], #8             \n"
      "5:                                               \n"  // exit
      // output operands, source operands, and clobber list
      : [p] "+&r"(p), [found] "+&r"(found), [rslt] "=&r"(rslt), [low] "=&r"(low),
        [high] "=&r"(high)
      : [lmt] "r"(lmt)
      : "v0", "v1", "v2", "v3", "v4", "v10", "v11", "v12", "v13",
        "cc", "memory");

  if (found) {
    return rslt;
  }

  // Tail for the last 16 bytes
  while (p != end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) {
    p++;
  }
  return p;
}
#else
static uint8_t *NOINLINE skip_whitespace(uint8_t *p, uint8_t *end) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

#if !defined(HAVE_CANDIDATE)

#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
static uint8_t *NOINLINE skip_word(uint8_t *p, uint8_t *end) {
  while (p != end && *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t') {
    p++;
  }
  return p;
}
#elif defined(HAVE_SVE_INTRINSICS)
static uint8_t *NOINLINE skip_word(uint8_t *p, uint8_t *end) {
  svuint8_t pat = svreinterpret_u8(svdup_u32(0x200A0D09));

  uint64_t ptr;
  svbool_t pg;
  FOR_LOOP_8(, ptr, (uint64_t)p, (uint64_t)end, pg) {
    svuint8_t c = svld1(pg, (uint8_t *)ptr);
    svbool_t pm = svmatch(pg, c, pat);
    if (svptest_any(svptrue_b8(), pm)) {
      pm = svbrkb_z(pg, pm);
      ptr += svcntp_b8(pm, pm);
      break;
    }
  }

  return ptr < (uint64_t)end ? (uint8_t *)ptr : end;
}
#elif defined(__ARM_FEATURE_SVE2)
static uint8_t *NOINLINE skip_word(uint8_t *p, uint8_t *end) {
  uint32_t pattern = 0x200A0D09;  // pattern ' \n\r\t'
  asm volatile(
      "        dup     z1.s, %w[pattern]          \n"
      "        whilelt p0.b, %[p], %[end]         \n"
      "1:      ld1b    z0.b, p0/z, [%[p]]         \n"
      "        match   p1.b, p0/z, z0.b, z1.b     \n"
      "        b.any   2f                         \n"  // any match?
      "        incb    %[p]                       \n"
      "        whilelt p0.b, %[p], %[end]         \n"
      "        b.first 1b                         \n"  // loop back
      "        mov     %[p], %[end]               \n"  // point p to end
      "        b       3f                         \n"  // end of buffer, exit
      "2:      brkb    p1.b, p0/z, p1.b           \n"
      "        incp    %[p], p1.b                 \n"
      "3:                                         \n"
      // output operands, source operands, and clobber list
      : [p] "+&r"(p)
      : [end] "r"(end), [pattern] "r"(pattern)
      : "v0", "v1", "p0", "p1", "cc", "memory");

  return p;
}
#elif defined(__ARM_FEATURE_SVE)
static uint8_t *NOINLINE skip_word(uint8_t *p, uint8_t *end) {
  asm volatile(
      "        dup     z4.b, 0x20               \n"  // space
      "        whilelt p0.b, %[p], %[end]       \n"
      "1:      ld1b    z0.b, p0/z, [%[p]]       \n"
      "        cmpeq   p1.b, p0/z, z0.b, 0x09   \n"  // == tab?
      "        cmpeq   p2.b, p0/z, z0.b, 0x0A   \n"  // == new line?
      "        cmpeq   p3.b, p0/z, z0.b, 0x0D   \n"  // == return?
      "        cmpeq   p4.b, p0/z, z0.b, z4.b   \n"
      "        orr     p1.b, p0/z, p1.b, p2.b   \n"
      "        orr     p1.b, p0/z, p1.b, p3.b   \n"
      "        orrs    p1.b, p0/z, p1.b, p4.b   \n"
      "        b.any   2f                       \n"  // any match?
      "        incb    %[p]                     \n"
      "        whilelt p0.b, %[p], %[end]       \n"
      "        b.first 1b                       \n"  // loop back
      "        mov     %[p], %[end]             \n"  // point p to end
      "        b       3f                       \n"  // end of buffer, exit
      "2:      brkb    p1.b, p0/z, p1.b         \n"
      "        incp    %[p], p1.b               \n"
      "3:                                       \n"
      // output operands, source operands, and clobber list
      : [p] "+&r"(p)
      : [end] "r"(end)
      : "v0", "v4", "p0", "p1", "cc", "memory");

  return p;
}
#elif defined(__ARM_NEON)
static uint8_t *NOINLINE skip_word(uint8_t *p, uint8_t *end) {
  uint8_t *lmt = end - 16;
  uint8_t *rslt;
  uint64_t found = 0;
  uint64_t low, high;

  asm volatile(
      "        movi    v10.16b, #0x20                   \n"
      "        movi    v11.16b, #0x0a                   \n"
      "        movi    v12.16b, #0x0d                   \n"
      "        movi    v13.16b, #0x09                   \n"
      "        b       2f                               \n"
      "1:      ldr     q4, [%[p]], #16                  \n"
      "        cmeq    v0.16b, v4.16b, v10.16b          \n"
      "        cmeq    v1.16b, v4.16b, v11.16b          \n"
      "        cmeq    v2.16b, v4.16b, v12.16b          \n"
      "        cmeq    v3.16b, v4.16b, v13.16b          \n"
      "        orr     v0.16b, v0.16b, v3.16b           \n"
      "        orr     v0.16b, v0.16b, v2.16b           \n"
      "        orr     v0.16b, v0.16b, v1.16b           \n"
      "        mov     %[low], v0.d[0]                  \n"
      "        mov     %[high], v0.d[1]                 \n"
      "        cbnz    %[low], 3f                       \n"
      "        cbnz    %[high], 4f                      \n"
      "2:      cmp     %[p], %[lmt]                     \n"
      "        b.lt    1b                               \n"  // loop back
      "        b       5f                               \n"
      "3:      mov     %[found], #1                     \n"
      "        rev64   %[low], %[low]                   \n"
      "        clz     %[low], %[low]                   \n"
      "        asr     %w[low], %w[low], #3             \n"
      "        sub     %[p], %[p], #16                  \n"
      "        add     %[rslt], %[p], %w[low], sxtw     \n"
      "        b       5f                               \n"
      "4:      mov     %[found], #1                     \n"
      "        rev64   %[high], %[high]                 \n"
      "        clz     %[high], %[high]                 \n"
      "        asr     %w[high], %w[high], #3           \n"
      "        add     %[rslt], %[p], %w[high], sxtw    \n"
      "        sub     %[rslt], %[rslt], #8             \n"
      "5:                                               \n"  // exit
      // output operands, source operands, and clobber list
      : [p] "+&r"(p), [found] "+&r"(found), [rslt] "=&r"(rslt), [low] "=&r"(low),
        [high] "=&r"(high)
      : [lmt] "r"(lmt)
      : "v0", "v1", "v2", "v3", "v4", "v10", "v11", "v12", "v13", "cc");

  if (found) {
    return rslt;
  }

  // Tail for the last 16 bytes
  while (p != end && *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t') {
    p++;
  }
  return p;
}
#else
static uint8_t *NOINLINE skip_word(uint8_t *p, uint8_t *end) {
  printf("ABORT: No implementations available for this target.\n");
  exit(2);
}
#endif

// Count words
static void inner_loop_103(struct loop_103_data *restrict input) {
  uint8_t *p = input->p;
  uint8_t *end = input->end;

  int count = 0;

  p = skip_whitespace(p, end);
  while (p != end) {
    count++;
    p = skip_word(p, end);
    p = skip_whitespace(p, end);
  }

  input->checksum = count;
}
#endif /* !HAVE_CANDIDATE */

LOOP_DECL(103, NS_SVE_LOOP_ATTR)
{
  struct loop_103_data data = {
    .p = sample_json,
    .end = (sample_json + sample_json_size),
    .checksum = 0,
  };
  inner_loops_103(iters, &data);

  int checksum = data.checksum;
  bool passed = checksum == 0x00000396;
#ifndef STANDALONE
  FINALISE_LOOP_I(103, passed, "0x%08"PRIx32, 0x00000396, checksum)
#endif
  return passed ? 0 : 1;
}
