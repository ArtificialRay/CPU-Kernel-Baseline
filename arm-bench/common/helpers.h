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

#ifndef HELPERS_H_
#define HELPERS_H_

#include <assert.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 64-bit aligned global allocator
void *alloc_64b(uint64_t size, const char *name);
#define ALLOC_64B(P, S, N) P = alloc_64b((S) * sizeof((P)[0]), N)

// Standard floating-point types
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#ifdef __ARM_FEATURE_BF16
#include <arm_bf16.h>
#endif
#elif defined(__aarch64__)
/* AArch64 scalar build (-U__ARM_NEON): fp16 as built-in, bf16 as raw bits */
typedef __fp16   float16_t;
typedef uint16_t bfloat16_t;  /* store raw bf16 bits; use bf16_to_f32() to convert */
typedef float    float32_t;
typedef double   float64_t;
#else
typedef uint16_t float16_t;
typedef uint16_t bfloat16_t;
typedef float float32_t;
typedef double float64_t;
#endif

// PRNG
uint32_t rand_uint32(void);
void fill_uint8(uint8_t *a, int n);
void fill_int8_mask(int8_t *a, int n, uint8_t mask);
void fill_uint16(uint16_t *a, int n);
void fill_uint32(uint32_t *a, int n);
void fill_uint64(uint64_t *a, int n);
void fill_int8(int8_t *a, int n);
void fill_int16(int16_t *a, int n);
void fill_int32(int32_t *a, int n);
void fill_int64(int64_t *a, int n);
void fill_fp16(float16_t *a, int n);
void fill_bf16(bfloat16_t *a, int n);
void fill_float(float *a, int n);
void fill_double(double *a, int n);
void fill_int64_range(int64_t *a, int n, int64_t min, int64_t max);
void fill_float_range(float *a, int n, float min, float max);
void fill_double_range(double *a, int n, double min, double max);
bool check_float(float n, float check, float epsilon);
bool check_exact_float(float n, uint32_t check);
bool check_scale_float(float n, float check, float err_abs, float err_rel);
bool check_double(double n, double check, double epsilon);
bool check_exact_double(double n, uint64_t check);

// FP16 support
#ifndef FP16_EMULATED
#if defined(__FLT16_MANT_DIG__)  // see <float.h> on GCC/LLVM
#define FP16_EMULATED 0
#else
#define FP16_EMULATED 1
#endif
#endif

#if FP16_EMULATED
typedef float32_t FLOAT16_t;
FLOAT16_t fp16_to_native(float16_t x);
float16_t native_to_fp16(FLOAT16_t x);
#else
typedef _Float16 FLOAT16_t;
typedef union {
  float16_t v;
  FLOAT16_t u;
} float16_cast_t;
static inline FLOAT16_t fp16_to_native(float16_t x) {
  float16_cast_t cast = {.v = x};
  return cast.u;
}
static inline float16_t native_to_fp16(FLOAT16_t x) {
  float16_cast_t cast = {.u = x};
  return cast.v;
}
#endif  // FP16_EMUlATED

// BF16 support
#if defined(__aarch64__)
#ifdef __ARM_NEON
typedef __bf16 BFLOAT16_t;

static inline float bf16_to_f32(bfloat16_t a) {
  BFLOAT16_t b;
  memcpy(&b, &a, sizeof(b));
  return vcvtah_f32_bf16(b);
}

static inline bfloat16_t f32_to_bf16(float n) {
  BFLOAT16_t a = vcvth_bf16_f32(n);
  bfloat16_t b;
  memcpy(&b, &a, sizeof(b));
  return b;
}
#else
/* AArch64 scalar build: bf16 ↔ f32 via bit manipulation (bf16 = upper 16b of f32) */
static inline float bf16_to_f32(bfloat16_t a) {
  uint32_t bits = (uint32_t)a << 16;
  float res;
  memcpy(&res, &bits, sizeof(res));
  return res;
}

static inline bfloat16_t f32_to_bf16(float n) {
  uint32_t bits;
  memcpy(&bits, &n, sizeof(bits));
  return (bfloat16_t)(bits >> 16);
}
#endif /* __ARM_NEON */
#else
float bf16_to_f32(bfloat16_t a);
bfloat16_t f32_to_bf16(float n);
#endif  // BF16 support (__aarch64__)

uint64_t strlen_opt(uint8_t *s);
int64_t strcmp_opt(uint8_t *restrict s1, uint8_t *restrict s2);

extern uint32_t static_rand_u32_length;
extern uint32_t static_rand_u32_values[];
extern uint32_t static_rand_flt_length;
extern uint32_t static_rand_flt_values[];
extern uint32_t static_rand_dbl_length;
extern uint64_t static_rand_dbl_values[];

extern uint8_t sample_json[];
extern uint32_t sample_json_size;

// Prevent GCC and LLVM from inlining function bodies and optimising away the
// entire loop, especially in cases where auto-vectorization is enabled.
// Additionally, avoid gcc discovering that input parameters are constants,
// since in some cases this can improve code generation unfairly (for instance
// using immediate forms of instructions like SSRA rather than a separate
// LSR+ADD).
#ifndef __clang__
#define NOINLINE __attribute__((noipa, noinline))
#else
#define NOINLINE __attribute__((noinline))
#endif

#endif  // HELPERS_H_
