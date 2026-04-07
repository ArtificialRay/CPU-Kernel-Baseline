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

#define MEMORY_SIZE (MAX_GLOBAL_MEMORY_KIB * 1024)
#define MEMORY_TAIL 128
static uint8_t *__global_memory_buffer = NULL;
static uint64_t __global_memory_offset = 0;
static uint64_t __global_memory_zeroes = 0;

void *alloc_64b(uint64_t size, const char *name) {
  if (__global_memory_offset + size > MEMORY_SIZE) {
    fprintf(stderr, "Unable to allocate %" PRIu64 " bytes", size);
    if (name != NULL) fprintf(stderr, " for '%s'", name);
    fprintf(stderr, "\n");
    exit(1);
  }

  uint8_t *p = __global_memory_buffer;
  if (p == NULL) {
    p = (uint8_t *)aligned_alloc(64, MEMORY_SIZE + MEMORY_TAIL);
    if (p == NULL) {
      fprintf(stderr, "Unable to request global buffer of %d (+%d) bytes",
              MEMORY_SIZE, MEMORY_TAIL);
      exit(1);
    }
    __global_memory_buffer = p;
  }

  uint64_t i = __global_memory_offset;
  uint64_t j = __global_memory_zeroes;
  uint64_t k = (i + size + 63) & (~(63lu));
  uint64_t l = (k < MEMORY_SIZE ? k : MEMORY_SIZE) + MEMORY_TAIL;
  __global_memory_offset = k;
  __global_memory_zeroes = l;
  memset(&p[j], 0, l - j);

  return (void *)&p[i];
}

static uint32_t static_rand_u32_index = 0;
static uint32_t static_rand_flt_index = 0;
static uint32_t static_rand_dbl_index = 0;

static uint32_t static_rand_uint32(void) {
  uint32_t idx = static_rand_u32_index;
  static_rand_u32_index = (static_rand_u32_index + 1) % static_rand_u32_length;
  return static_rand_u32_values[idx];
}

uint32_t rand_uint32(void) { return static_rand_uint32(); }

uint64_t static_rand_uint64(void) {
  uint64_t res = static_rand_uint32();
  res <<= 32;
  res |= static_rand_uint32();
  return res;
}

void fill_uint8(uint8_t *a, int n) { fill_uint32((void *)a, n / 4); }

void fill_int8_mask(int8_t *a, int n, uint8_t mask){
  uint32_t *b = (void *)a;
  uint32_t u32_mask = mask & mask & mask & mask;

  fill_uint32((void *)b, n/4);

  for (int i = 0; i < n/4; i++) {
    b[i] = b[i] & u32_mask;
  }

}

void fill_uint16(uint16_t *a, int n) { fill_uint32((void *)a, n / 2); }

void fill_uint32(uint32_t *a, int n) {
  int rem = n;

  while (rem) {
    int cpy = rem;
    uint32_t idx = static_rand_u32_index;
    if ((cpy >= 0) &&
        (((uint32_t)cpy) > (static_rand_u32_length - static_rand_u32_index))) {
      cpy = static_rand_u32_length - static_rand_u32_index;
      static_rand_u32_index = 0;
    } else {
      static_rand_u32_index += cpy;
    }
    memcpy(a, static_rand_u32_values + idx, 4 * cpy);
    rem -= cpy;
    a += cpy;
  }
}

void fill_uint64(uint64_t *a, int n) { fill_uint32((void *)a, 2 * n); }

void fill_int8(int8_t *a, int n) { fill_uint32((void *)a, n / 4); }

void fill_int16(int16_t *a, int n) { fill_uint16((void *)a, n); }

void fill_int32(int32_t *a, int n) { fill_uint32((void *)a, n); }

void fill_int64(int64_t *a, int n) { fill_uint32((void *)a, 2 * n); }

void fill_fp16(float16_t *b, int n) {
  uint16_t *a = (void *)b;
  fill_uint16(a, n);
  // Make sure no invalid exponents are generated
  for (int i = 0; i < n; i++) {
    a[i] = (a[i] & 0x9FFF) | 0x2000;
  }
}

void fill_float(float *a, int n) {
  int rem = n;

  while (rem) {
    int cpy = rem;
    uint32_t idx = static_rand_flt_index;
    if ((cpy >= 0) &&
        (((uint32_t)cpy) > (static_rand_flt_length - static_rand_flt_index))) {
      cpy = static_rand_flt_length - static_rand_flt_index;
      static_rand_flt_index = 0;
    } else {
      static_rand_flt_index += cpy;
    }
    memcpy(a, static_rand_flt_values + idx, 4 * cpy);
    rem -= cpy;
    a += cpy;
  }
}

void fill_double(double *a, int n) {
  int rem = n;

  while (rem) {
    int cpy = rem;
    uint32_t idx = static_rand_dbl_index;
    if ((cpy >= 0) &&
        (((uint32_t)cpy) > (static_rand_dbl_length - static_rand_dbl_index))) {
      cpy = static_rand_dbl_length - static_rand_dbl_index;
      static_rand_dbl_index = 0;
    } else {
      static_rand_dbl_index += cpy;
    }
    memcpy(a, static_rand_dbl_values + idx, 8 * cpy);
    rem -= cpy;
    a += cpy;
  }
}

void fill_int64_range(int64_t *a, int n, int64_t min, int64_t max) {
  fill_int64(a, n);

  for (int i = 0; i < n; i++) {
    a[i] = __builtin_labs(a[i]) % (max - min + 1) + min;
  }
}

void fill_float_range(float *a, int n, float min, float max) {
  fill_float(a, n);

  for (int i = 0; i < n; i++) {
    a[i] = min + a[i] * (max - min);
  }
}

void fill_double_range(double *a, int n, double min, double max) {
  fill_double(a, n);

  for (int i = 0; i < n; i++) {
    a[i] = min + a[i] * (max - min);
  }
}

void fill_bf16(bfloat16_t *a, int n) { fill_fp16((float16_t *)a, n); }

bool check_float(float n, float check, float epsilon) {
  return (n + epsilon >= check) && (n - epsilon <= check);
}

bool check_exact_float(float n, uint32_t check) {
  union {
    uint32_t u;
    float f;
  } bits;
  bits.f = n;
  return bits.u == check;
}

bool check_scale_float(float n, float check, float err_abs, float err_rel) {
  float diff = fabsf(check) * err_rel;
  return fabsf(n - check) <= ((diff > err_abs) ? diff : err_abs);
}

bool check_double(double n, double check, double epsilon) {
  return (n + epsilon >= check) && (n - epsilon <= check);
}

bool check_exact_double(double n, uint64_t check) {
  union {
    uint64_t u;
    double d;
  } bits;
  bits.d = n;
  return bits.u == check;
}

typedef union {
  float16_t v;
  uint16_t raw;
  struct {
    unsigned int mantissa : 10;
    unsigned int exponent : 5;
    unsigned int sign : 1;
  } parts;
} float16_bits_t;

typedef union {
  uint16_t u16;
  struct {
    unsigned int mantissa : 7;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
} bfloat16_bits_t;

typedef union {
  float32_t v;
  struct {
    unsigned int mantissa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
} float32_bits_t;

// software emulation of FP16
#if FP16_EMULATED
_Static_assert(sizeof(FLOAT16_t) == sizeof(float32_t),
               "native FP16 type must be 32-bit");

static inline float32_bits_t b16_to_b32(float16_bits_t fp16) {
  float32_bits_t fp32;
  fp32.parts.sign = fp16.parts.sign;
  fp32.parts.exponent = fp16.parts.exponent + 0x70;
  fp32.parts.mantissa = fp16.parts.mantissa << 13;
  return fp32;
}
static inline float16_bits_t b32_to_b16(float32_bits_t fp32) {
  float16_bits_t fp16;
  fp16.parts.sign = fp32.parts.sign;
  fp16.parts.exponent = fp32.parts.exponent - 0x70;
  fp16.parts.mantissa = fp32.parts.mantissa >> 13;
  return fp16;
}

FLOAT16_t fp16_to_native(float16_t a) {
  if (a == 0) {
    return 0.0f;
  } else {
    float16_bits_t fp16 = {.v = a};
    return b16_to_b32(fp16).v;
  }
}
float16_t native_to_fp16(FLOAT16_t a) {
  if (fabsf(a) < 6.103615625e-5f) {
    return 0;  // smaller than FP16 eps
  } else {
    float32_bits_t fp32 = {.v = a};
    return b32_to_b16(fp32).v;
  }
}

#else
_Static_assert(sizeof(FLOAT16_t) == sizeof(uint16_t),
               "native FP16 type must be 16-bit");
#endif  // FP16

// software emulation of BF16
#ifndef __aarch64__
float bf16_to_f32(bfloat16_t a) {
  if (a == 0) {
    return 0;
  }

  bfloat16_bits_t bf16;
  bf16.u16 = a;
  float32_bits_t res;
  res.parts.sign = bf16.parts.sign;
  res.parts.exponent = bf16.parts.exponent;
  res.parts.mantissa = bf16.parts.mantissa << 16;

  return res.v;
}

bfloat16_t f32_to_bf16(float n) {
  if (n == 0) {
    return 0;
  }

  float32_bits_t cast;
  cast.v = n;

  bfloat16_bits_t res;
  res.parts.sign = cast.parts.sign;
  res.parts.exponent = cast.parts.exponent;
  res.parts.mantissa = cast.parts.mantissa >> 16;

  return res.u16;
}
#endif
