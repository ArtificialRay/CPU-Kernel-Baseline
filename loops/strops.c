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

#include "helpers.h"
#include "loops.h"

#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
uint64_t strlen_opt(uint8_t *s) { return strlen((void *)s); }
int64_t strcmp_opt(uint8_t *restrict s1, uint8_t *restrict s2) {
  return strcmp((void *)s1, (void *)s2);
}
#elif defined(HAVE_SVE_INTRINSICS)
uint64_t strlen_opt(uint8_t *s) {
  svbool_t all = svptrue_b8();
  svbool_t pf, pz;
  svuint8_t data;

  uint64_t i = 0;
  do {
    // pre-alignment optional?
#ifndef STRLEN_NO_ALIGN
    uint64_t align = 16 - ((uint64_t)s % 16);
    pf = svwhilele_b8((uint64_t)0, align);
    data = svld1(pf, s);
    pz = svcmpeq(pf, data, 0);
    if (svptest_any(pz, pz)) break;
    i += align;
#endif

    // svcntp(pX, pX) should optimise to incp, according to ACLE spec
    svsetffr();
    while (1) {
      data = svldff1(all, s + i);
      pf = svrdffr_z(all);
      pz = svcmpeq(pf, data, 0);
      if (svptest_any(pz, pz)) {
        break;
      } else if (svptest_last(all, pf)) {
        i += svcntb();
      } else {
        svsetffr();
        i += svcntp_b8(pf, pf);
      }
    }
  } while (0);
  pz = svbrkb_z(all, pz);
  i += svcntp_b8(pz, pz);

  return i;
}

int64_t strcmp_opt(uint8_t *restrict s1, uint8_t *restrict s2) {
  svbool_t all = svptrue_b8();
  svbool_t pf, pz, pc;
  svuint8_t d1, d2;

  svsetffr();
  uint64_t i = 0;
  do {
    d1 = svldff1(all, s1 + i);
    d2 = svldff1(all, s2 + i);
    pf = svrdffr_z(all);
    if (svptest_last(all, pf)) {
      pz = svcmpeq(all, d1, 0);
      pc = svcmpne(all, d1, d2);
      i += svcntb();
    } else {
      svsetffr();
      pz = svcmpeq(pf, d1, 0);
      pc = svcmpne(pf, d1, d2);
      i += svcntp_b8(pf, pf);
    }
    pc = svorr_z(all, pz, pc);
  } while (!svptest_any(all, pc));

  pc = svbrkb_z(all, pc);
  int64_t c1 = svlasta(pc, d1);
  int64_t c2 = svlasta(pc, d2);
  return c1 - c2;
}
#elif defined(__ARM_FEATURE_SVE)
uint64_t strlen_opt(uint8_t *s) {
  uint64_t align;
  uint64_t res;
  uint8_t *p;

  asm volatile(
      "   and     %[align], %[s], 0xf                 \n"
      "   sub     %[align], %[sixteen], %[align]      \n"
      "   whilele p0.b, xzr, %[align]                 \n"
      // Process up to 16 bytes to get in alignment
      "   ld1b    {z0.b}, p0/z, [%[s]]                \n"
      "   cmpeq   p1.b, p0/z, z0.b, #0                \n"
      "   b.none  1f                                  \n"  // cont
      "   brkb    p1.b, p0/z, p1.b                    \n"
      "   cntp    %[res], p0, p1.b                    \n"
      "   b       6f                                  \n"  // exit
      // Enter unrolled loop
      "1:                                             \n"  // cont
      "   setffr                                      \n"
      "   add     %[p], %[s], %[align]                \n"
      "   ptrue   p0.b                                \n"
      "   decb    %[p], all, mul #2                   \n"
      "2:                                             \n"  // loop
      "   incb    %[p], all, mul #2                   \n"
      "   ldff1b  {z0.b}, p0/z, [%[p]]                \n"
      "   ldnf1b  {z1.b}, p0/z, [%[p], #1, mul vl]    \n"
      "   rdffrs  p2.b, p0/z                          \n"
      "   b.nlast 4f                                  \n"  // fault
      "   cmpeq   p1.b, p0/z, z0.b, #0                \n"
      "   b.any   3f                                  \n"  // found
      "   cmpeq   p1.b, p0/z, z1.b, #0                \n"
      "   b.none  2b                                  \n"  // loop
      // Fallthrough into found
      "   incb    %[p]                                \n"
      "3:                                             \n"  // found
      "   brkb    p1.b, p0/z, p1.b                    \n"
      "   incp    %[p], p1.b                          \n"
      "   sub     %[res], %[p], %[s]                  \n"
      "   b       6f                                  \n"  // exit
      "4:                                             \n"  // fault
      "   setffr                                      \n"
      "   cmpeq   p1.b, p2/z, z0.b, #0                \n"
      "   b.any   3b                                  \n"  // found
      // Fallthrough into slow_loop
      "5:                                             \n"  // slow_loop
      "   incp    %[p], p2.b                          \n"
      "   ldff1b  {z0.b}, p0/z, [%[p]]                \n"
      "   rdffr   p2.b, p0/z                          \n"
      "   cmpeq   p1.b, p2/z, z0.b, #0                \n"
      "   b.none  5b                                  \n"  // slow_loop
      "   b       3b                                  \n"  // found
      "6:                                             \n"  // exit
      : [p] "=&r"(p), [align] "=&r"(align), [res] "=&r"(res)
      : [s] "r"(s), [sixteen] "r"(16L)
      : "v0", "v1", "p0", "p1", "p2", "cc", "memory");

  return res;
}

int64_t strcmp_opt(uint8_t *restrict s1, uint8_t *restrict s2) {
  int64_t res, c1, c2;
  uint64_t i = 0;

  asm volatile(
      "   ptrue p5.b                       \n"
      "   setffr                           \n"

      "1: ldff1b z0.b, p5/z, [%[s1], %[i]] \n"  // loop
      "   ldff1b z1.b, p5/z, [%[s2], %[i]] \n"
      "   rdffrs p7.b, p5/z                \n"
      "   b.nlast 3f                       \n"  // fault?

      "   incb %[i]                        \n"
      "   cmpeq p0.b, p5/z, z0.b, #0       \n"
      "   cmpne p1.b, p5/z, z0.b, z1.b     \n"
      "2: orrs p4.b, p5/z, p0.b, p1.b      \n"  // test
      "   b.none 1b                        \n"

      "   brkb p4.b, p5/z, p4.b            \n"  // return
      "   lasta %w[c1], p4, z0.b           \n"
      "   lasta %w[c2], p4, z1.b           \n"
      "   sub %w[res], %w[c1], %w[c2]      \n"
      "   b 4f                             \n"

      "3: incp %[i], p7.b                  \n"  // fault
      "   setffr                           \n"
      "   cmpeq p0.b, p7/z, z0.b, #0       \n"
      "   cmpne p1.b, p7/z, z0.b, z1.b     \n"
      "   b 2b                             \n"
      "4:                                  \n"
      : [c1] "=&r"(c1), [c2] "=&r"(c2), [i] "+r"(i), [res] "=&r"(res)
      : [s1] "r"(s1), [s2] "r"(s2)
      : "v0", "v1", "p0", "p1", "p4", "p5", "p7", "cc", "memory");

  return res;
}
#elif defined(__aarch64__) && \
    !defined(HAVE_AUTOVEC)
uint64_t strlen_opt(uint8_t *srcin) {
  uint64_t len;
  uint64_t src;
  uint64_t data1;
  uint64_t data2;
  uint64_t data2a;
  uint64_t has_nul1;
  uint64_t has_nul2;
  uint64_t tmp1;
  uint64_t tmp2;
  uint64_t tmp3;
  uint64_t tmp4;
  uint64_t zeroones;
  uint64_t pos;

  asm volatile(
      "   mov     %[zeroones], 0x0101010101010101             \n"
      "   bic     %[src], %[srcin], #0xf                      \n"
      "   ands    %[tmp1], %[srcin], #0xf                     \n"
      "   b.ne    4f                                          \n"  // misaligned

      "1:                                                     \n"  // loop
      "    ldp    %[data1] , %[data2], [%[src]], #16          \n"
      "2:                                                     \n"  // realigned
      "    sub    %[tmp1], %[data1], %[zeroones]              \n"
      "    orr    %[tmp2], %[data1], #0x7f7f7f7f7f7f7f7f      \n"
      "    sub    %[tmp3], %[data2], %[zeroones]              \n"
      "    orr    %[tmp4], %[data2], #0x7f7f7f7f7f7f7f7f      \n"
      "    bic    %[has_nul1], %[tmp1], %[tmp2]               \n"
      "    bics   %[has_nul2], %[tmp3], %[tmp4]               \n"
      "    ccmp   %[has_nul1], #0, #0, eq                     \n"
      "    b.eq   1b                                          \n"  // loop

      "    sub    %[len], %[src], %[srcin]                    \n"
      "    cbz    %[has_nul1], 3f                             \n"  // nul_in_data2
      "    sub    %[len], %[len], #8                          \n"
      "    mov    %[has_nul2], %[has_nul1]                    \n"

      "3:                                                     \n"  // nul_in_data2
      "    sub    %[len], %[len], #8                          \n"
      "    rev    %[has_nul2], %[has_nul2]                    \n"
      "    clz    %[pos], %[has_nul2]                         \n"
      "    add    %[len], %[len], %[pos], lsr #3              \n"
      "    b      5f                                          \n"  // exit

      "4:                                                     \n"  // misaligned
      "    cmp    %[tmp1], #8                                 \n"
      "    neg    %[tmp1], %[tmp1]                            \n"
      "    ldp    %[data1], %[data2], [%[src]], #16           \n"
      "    lsl    %[tmp1], %[tmp1], #3                        \n"
      "    mov    %[tmp2], #~0                                \n"
      "    lsr    %[tmp2], %[tmp2], %[tmp1]                   \n"
      "    orr    %[data1], %[data1], %[tmp2]                 \n"
      "    orr    %[data2a], %[data2], %[tmp2]                \n"
      "    csinv  %[data1], %[data1], xzr, le                 \n"
      "    csel   %[data2], %[data2], %[data2a], le           \n"
      "    b      2b                                          \n"  // realigned
      "5:                                                     \n"  // exit
      : [len] "=&r"(len), [src] "=&r"(src), [data1] "=&r"(data1),
        [data2] "=&r"(data2), [data2a] "=&r"(data2a),
        [has_nul1] "=&r"(has_nul1), [has_nul2] "=&r"(has_nul2),
        [tmp1] "=&r"(tmp1), [tmp2] "=&r"(tmp2), [tmp3] "=&r"(tmp3),
        [tmp4] "=&r"(tmp4), [zeroones] "=&r"(zeroones), [pos] "=&r"(pos)
      : [srcin] "r"(srcin)
      : "cc", "memory");

  return len;
}

int64_t strcmp_opt(uint8_t *restrict src1, uint8_t *restrict src2) {
  uint64_t result;
  uint64_t data1;
  uint64_t data2;
  uint64_t has_nul;
  uint64_t diff;
  uint64_t syndrome;
  uint64_t tmp1;
  uint64_t tmp2;
  uint64_t zeroones;
  uint64_t pos;

  asm volatile(
      "   eor     %[tmp1], %[src1], %[src2]                       \n"
      "   mov     %[zeroones], 0x0101010101010101                 \n"
      "   tst     %[tmp1], #7                                     \n"
      "   b.ne    5f                                              \n"
      "   ands    %[tmp1], %[src1], #7                            \n"
      "   b.ne    4f                                              \n"

      "1:                                                         \n"  // loop_aligned
      "   ldr    %[data1], [%[src1]], #8                          \n"
      "   ldr    %[data2], [%[src2]], #8                          \n"
      "2:                                                         \n"  // start_realigned
      "   sub    %[tmp1], %[data1], %[zeroones]                   \n"
      "   orr    %[tmp2], %[data1], #0x7f7f7f7f7f7f7f7f           \n"
      "   eor    %[diff], %[data1], %[data2]                      \n"
      "   bic    %[has_nul], %[tmp1], %[tmp2]                     \n"
      "   orr    %[syndrome], %[diff], %[has_nul]                 \n"
      "   cbz    %[syndrome], 1b                                  \n"

      "3:                                                         \n"  // end
      "   rev    %[syndrome], %[syndrome]                         \n"
      "   rev    %[data1], %[data1]                               \n"
      "   clz    %[pos], %[syndrome]                              \n"
      "   rev    %[data2], %[data2]                               \n"
      "   lsl    %[data1], %[data1], %[pos]                       \n"
      "   lsl    %[data2], %[data2], %[pos]                       \n"
      "   lsr    %[data1], %[data1], #56                          \n"
      "   sub    %[result], %[data1], %[data2], lsr #56           \n"
      "   b      9f                                               \n"

      "4:                                                         \n"  // mutual_align
      "   bic    %[src1], %[src1], #7                             \n"
      "   bic    %[src2], %[src2], #7                             \n"
      "   lsl    %[tmp1], %[tmp1], #3                             \n"
      "   ldr    %[data1], [%[src1]], #8                          \n"
      "   neg    %[tmp1], %[tmp1]                                 \n"
      "   ldr    %[data2], [%[src2]], #8                          \n"
      "   mov    %[tmp2], #~0                                     \n"
      "   lsr    %[tmp2], %[tmp2], %[tmp1]                        \n"
      "   orr    %[data1], %[data1], %[tmp2]                      \n"
      "   orr    %[data2], %[data2], %[tmp2]                      \n"
      "   b      2b                                               \n"

      "5:                                                         \n"  // misaligned8
      "   tst    %[src1], #7                                      \n"
      "   b.eq   7f                                               \n"
      "6:                                                         \n"  // do_misaligned
      "   ldrb   %w[data1], [%[src1]], #1                         \n"
      "   ldrb   %w[data2], [%[src2]], #1                         \n"
      "   cmp    %w[data1], #1                                    \n"
      "   ccmp   %w[data1], %w[data2], #0, cs                     \n"
      "   b.ne   8f                                               \n"
      "   tst    %[src1], #7                                      \n"
      "   b.ne   6b                                               \n"

      "7:                                                         \n"  // loop_misaligned
      "   and    %[tmp1], %[src2], #0xff8                         \n"
      "   eor    %[tmp1], %[tmp1], #0xff8                         \n"
      "   cbz    %[tmp1], 6b                                      \n"
      "   ldr    %[data1], [%[src1]], #8                          \n"
      "   ldr    %[data2], [%[src2]], #8                          \n"
      "   sub    %[tmp1], %[data1], %[zeroones]                   \n"
      "   orr    %[tmp2], %[data1], #0x7f7f7f7f7f7f7f7f           \n"
      "   eor    %[diff], %[data1], %[data2]                      \n"
      "   bic    %[has_nul], %[tmp1], %[tmp2]                     \n"
      "   orr    %[syndrome], %[diff], %[has_nul]                 \n"
      "   cbz    %[syndrome], 7b                                  \n"

      "   b      3b                                               \n"
      "8:                                                         \n"  // done
      "   sub    %[result], %[data1], %[data2]                    \n"
      "9:                                                         \n"  // exit
      : [result] "=&r"(result), [data1] "=&r"(data1), [data2] "=&r"(data2),
        [has_nul] "=&r"(has_nul), [diff] "=&r"(diff),
        [syndrome] "=&r"(syndrome), [tmp1] "=&r"(tmp1), [tmp2] "=&r"(tmp2),
        [zeroones] "=&r"(zeroones), [pos] "=&r"(pos), [src1] "+r"(src1),
        [src2] "+r"(src2)
      :
      : "cc", "memory");

  return result;
}
#else
uint64_t strlen_opt(uint8_t *s) { return strlen((void *)s); }
int64_t strcmp_opt(uint8_t *restrict s1, uint8_t *restrict s2) {
  return strcmp((void *)s1, (void *)s2);
}
#endif
