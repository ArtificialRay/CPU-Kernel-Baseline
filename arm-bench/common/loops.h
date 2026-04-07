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

#ifndef LOOPS_H_
#define LOOPS_H_

#ifdef __ARM_FEATURE_SME
// Compiler should have SME ACLE feature available when compiling
// SME code. Streaming modes are managed by the SME ABI.
#include <arm_sme.h>
// Caller function sme attributes:
//   * Streaming SVE loop (SSVE)
#define SC_SVE_LOOP_ATTR __arm_locally_streaming
//   * Streaming loop (SSVE + SME)
#define S_LOOP_ATTR              \
        __arm_locally_streaming  \
        __arm_new("za", "zt0")
//   * Non-streaming loop (SVE only)
#define NS_SVE_LOOP_ATTR

// Callee function sme attributes:
// Streaming compatible SVE code with no shared SME states
#ifdef __ARM_FEATURE_SVE
#define SC_SVE_ATTR __arm_streaming_compatible
#else
#define SC_SVE_ATTR __arm_streaming
#endif
// Streaming SVE code with no shared SME states
#define S_SVE_ATTR __arm_streaming
// Streaming compatible SVE code with ZT0 shared state only
#define SC_SVE_ZT0_ATTR            \
        __arm_streaming_compatible \
        __arm_inout("zt0")         \
        __arm_preserves("za")
// Streaming code with ZA & ZT0 shared states
#define SME_ZA_ZT0_ATTR          \
        __arm_streaming          \
        __arm_inout("za", "zt0")
// Streaming code with ZA shared state only
#define SME_ZA_ATTR              \
        __arm_streaming          \
        __arm_inout("za")        \
        __arm_preserves("zt0")
#else
#define SC_SVE_LOOP_ATTR
#define S_LOOP_ATTR
#define NS_SVE_LOOP_ATTR
#define SC_SVE_ATTR
#define S_SVE_ATTR
#define SC_SVE_ZT0_ATTR
#define SME_ZA_ZT0_ATTR
#define SME_ZA_ATTR
#endif  // __ARM_FEATURE_SME

#if defined(HAVE_SVE_INTRINSICS) || defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif  // HAVE_SVE_INTRINSICS

#if (defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_SME))
static inline uint32_t get_sve_vl(void)
SC_SVE_ATTR
{
  uint64_t vl = 0;
  asm volatile("incb %[vl], all, mul #8" : [vl] "+r"(vl) : :);
  return (uint32_t)vl;
}
#endif // __ARM_FEATURE_SVE

#if !defined(__ARM_FEATURE_SME) && !defined(__ARM_FEATURE_SVE) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__ARM_FEATURE_SME)
static inline uint32_t get_sme_vl(void){
  uint64_t svl = 0;
  asm volatile("rdsvl    %[svl], #8\n" : [svl] "+r"(svl) : :);
  return (uint32_t)svl;
}
#endif

static inline uint32_t get_vl(void){
#if defined(__ARM_FEATURE_SME)    // SME vl
  return get_sme_vl();
#elif defined(__ARM_FEATURE_SVE)  // SVE vl
  return get_sve_vl();
#else                    // NEON vl
  return 128;
#endif
}

// for-loop shorthand
#if defined(HAVE_SVE_INTRINSICS) || defined(HAVE_SME_INTRINSICS)
#define FOR_COND(P, S, I, N) \
  svptest_first(svptrue_b##S(), P = svwhilelt_b##S(I, N))
#define FOR_LOOP(T, I, M, N, P, S, W) \
  for (T I = M; FOR_COND(P, S, I, N); I += svcnt##W())
#define FOR_LOOP_8(T, I, M, N, P) FOR_LOOP(T, I, M, N, P, 8, b)
#define FOR_LOOP_16(T, I, M, N, P) FOR_LOOP(T, I, M, N, P, 16, h)
#define FOR_LOOP_32(T, I, M, N, P) FOR_LOOP(T, I, M, N, P, 32, w)
#define FOR_LOOP_64(T, I, M, N, P) FOR_LOOP(T, I, M, N, P, 64, d)
#endif

// Loop information
typedef int (*loop_function_t)(int);
#define LOOP_INIT(n)                              \
  int loop_##n (int);                             \
  extern loop_function_t __ptr_loop_##n;          \
  __attribute__((constructor))                    \
  static inline void loop_##n##_init(void) { __ptr_loop_##n = loop_##n; }

#ifndef LOOP_START
#define LOOP_START()
#endif

#ifndef LOOP_STOP
#define LOOP_STOP()
#endif

// Loop declaration
#define LOOP_DECL(n, sme_attr)                                              \
  LOOP_INIT(n)                                                              \
  sme_attr                                                                  \
  static void inner_loops_##n(int l, struct loop_##n##_data *restrict d)  { \
    LOOP_START();                                                           \
    for (int i = 0; i < l; ++i) inner_loop_##n(d);                          \
    LOOP_STOP();                                                            \
  }                                                                         \
  int loop_##n(int iters)

#ifndef FINALISE_LOOP_I
#define FINALISE_LOOP_I(loop_n, result, format_str, exp_val, obt_val)        \
{                                                                            \
  printf("LOOP_RESULT: " format_str "\n", (obt_val));                       \
  if (result) {                                                              \
    printf(" - Checksum correct.\n");                                        \
  } else {                                                                   \
    printf(" - Checksum wrong (got " format_str ", expected "                \
           format_str " ).\n", obt_val, exp_val);                            \
  }                                                                          \
}
#endif

#ifndef FINALISE_LOOP_F
#define FINALISE_LOOP_F(loop_n, result, format_str, exp_val, range, obt_val) \
{                                                                            \
  printf("LOOP_RESULT: " format_str "\n", (obt_val));                       \
  if (result) {                                                              \
    printf(" - Checksum correct.\n");                                        \
  } else {                                                                   \
    printf(" - Checksum wrong (got " format_str ", expected "                \
           format_str " +/- " format_str " ).\n", obt_val, range, obt_val);  \
  }                                                                          \
}
#endif

#ifndef FINALISE_LOOP_FP64
#define FINALISE_LOOP_FP64(loop_n, result, b_out, b_exp, format_str, exp_out, exp_exp, obt_out, obt_exp) \
{                                                                                                        \
  printf("LOOP_RESULT: " format_str " " format_str "\n", (obt_out), (obt_exp));                        \
  if (result) {                                                                                          \
    printf(" - Checksum correct.\n");                                                                    \
  } else {                                                                                               \
    if (!b_out) {                                                                                        \
      printf(" - Output checksum wrong (got " format_str ", expected " format_str ").\n",               \
             obt_out, exp_out);                                                                          \
    }                                                                                                    \
    if (!b_exp) {                                                                                        \
      printf(" - Exponent checksum wrong (got " format_str ", expected " format_str ").\n",             \
             obt_exp, exp_exp);                                                                          \
    }                                                                                                    \
  }                                                                                                      \
}
#endif

#ifndef FINALISE_LOOP_U256
#define FINALISE_LOOP_U256(loop_n, result, exp_val, obt_val)                 \
{                                                                            \
  if (result) {                                                              \
    printf(" - Checksum correct.\n");                                        \
  } else {                                                                   \
    printf(" - Checksum wrong:\n got ");                                     \
    print_hex256(obt_val);                                                   \
    printf(" - expected ");                                                  \
    print_hex256(exp_val);                                                   \
    printf("\n");                                                            \
  }                                                                          \
}
#endif

#endif  // LOOPS_H_
