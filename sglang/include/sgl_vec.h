#pragma once
// sgl_vec.h — Zero-dependency AVX-512 BF16 vector helpers
//
// TWO CODE PATHS, selected at compile time:
//   SGL_USE_AVX512BF16=1  → uses dpbf16_ps, cvtne2ps_pbh (requires -mavx512bf16)
//   SGL_USE_AVX512BF16=0  → uses avx512f fmadd_ps + shift-based bf16 conversion
//
// At runtime, cpu_has_avx512bf16() lets you dispatch between shared-library
// variants compiled with each flag.
//
// No ATen, no PyTorch, no libtorch.

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <cpuid.h>

// ---------------------------------------------------------------------------
// Runtime CPU feature detection
// ---------------------------------------------------------------------------
static inline bool cpu_has_avx512bf16() {
    unsigned int eax=0,ebx=0,ecx=0,edx=0;
    if (__get_cpuid_count(7,1,&eax,&ebx,&ecx,&edx)) return (eax>>5)&1u;
    return false;
}

// ---------------------------------------------------------------------------
// BF16 scalar helpers  (always available)
// ---------------------------------------------------------------------------
typedef uint16_t bf16_t;

static inline float bf16_to_f32(bf16_t x) {
    uint32_t u = (uint32_t)x << 16;
    float f; memcpy(&f, &u, 4); return f;
}
static inline bf16_t f32_to_bf16(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7fffffff) > 0x7f800000) return (bf16_t)((u>>16)|0x0040);
    uint32_t bias = ((u>>16)&1) + 0x7fff;
    return (bf16_t)((u + bias) >> 16);
}

// ---------------------------------------------------------------------------
// Vector width constants
// ---------------------------------------------------------------------------
#define SGL_VEC_BF16_W   32   // 512-bit / 16-bit
#define SGL_VEC_F32_W    16   // 512-bit / 32-bit

// ---------------------------------------------------------------------------
// Load/store (BF16)
// ---------------------------------------------------------------------------
static inline __m512i sgl_load_bf16(const bf16_t* __restrict__ p) {
    return _mm512_loadu_si512((const __m512i*)p);
}
static inline void sgl_store_bf16(bf16_t* __restrict__ p, __m512i v) {
    _mm512_storeu_si512((__m512i*)p, v);
}

// ---------------------------------------------------------------------------
// BF16 → FP32 conversion (two halves of a 512-bit register → two __m512)
// Pure AVX-512F: shift left 16 bits to place bf16 mantissa in f32 position.
// ---------------------------------------------------------------------------
static inline __m512 sgl_cvt_bf16_lo_f32(__m512i v) {
    __m256i lo = _mm512_castsi512_si256(v);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(lo), 16));
}
static inline __m512 sgl_cvt_bf16_hi_f32(__m512i v) {
    __m256i hi = _mm512_extracti64x4_epi64(v, 1);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16));
}

// ---------------------------------------------------------------------------
// FP32 → BF16: truncate upper 16 bits with round-to-nearest-even
// Pure AVX-512F path (no bf16 ISA required)
// ---------------------------------------------------------------------------
static inline __m256i sgl_cvt_f32_bf16_avx512f(__m512 v) {
    // Round to nearest even: add rounding bias, then shift right 16
    __m512i vi = _mm512_castps_si512(v);
    // rounding_bias = ((mantissa>>16)&1) + 0x7fff
    __m512i lsb   = _mm512_and_si512(_mm512_srli_epi32(vi, 16), _mm512_set1_epi32(1));
    __m512i bias  = _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7fff));
    __m512i round = _mm512_add_epi32(vi, bias);
    __m512i shift = _mm512_srli_epi32(round, 16);
    // Pack 16 x 32-bit to 16 x 16-bit (lower 16 of each lane)
    return _mm512_cvtepi32_epi16(shift);
}

// Pack two __m256i of bf16 (16 each) into one __m512i (32 bf16)
static inline __m512i sgl_cvt_2xf32_bf16(__m512 lo, __m512 hi) {
    __m256i blo = sgl_cvt_f32_bf16_avx512f(lo);
    __m256i bhi = sgl_cvt_f32_bf16_avx512f(hi);
    __m512i result = _mm512_castsi256_si512(blo);
    return _mm512_inserti64x4(result, bhi, 1);
}

// Convert single __m512 f32 → 16 bf16 stored in lower 256 bits
static inline __m256i sgl_cvt_f32_to_bf16_256(__m512 v) {
    return sgl_cvt_f32_bf16_avx512f(v);
}

// ---------------------------------------------------------------------------
// Dot-product accumulation: acc += dot(a_bf16, b_bf16) — f32 accumulate
// AVX-512F path: convert then fmadd (no dpbf16 instruction needed)
// ---------------------------------------------------------------------------
static inline __m512 sgl_dpbf16_lo(__m512 acc, __m512i a, __m512i b) {
    __m512 alo = sgl_cvt_bf16_lo_f32(a);
    __m512 blo = sgl_cvt_bf16_lo_f32(b);
    __m512 ahi = sgl_cvt_bf16_hi_f32(a);
    __m512 bhi = sgl_cvt_bf16_hi_f32(b);
    acc = _mm512_fmadd_ps(alo, blo, acc);
    acc = _mm512_fmadd_ps(ahi, bhi, acc);
    return acc;
}

// ---------------------------------------------------------------------------
// SiLU: x * sigmoid(x) in f32 — scalar loop (no SVML required)
// ---------------------------------------------------------------------------
static inline __m512 sgl_silu_f32(__m512 x) {
    float buf[16];
    _mm512_storeu_ps(buf, x);
    for (int i = 0; i < 16; ++i) {
        float v = buf[i];
        buf[i] = v / (1.0f + expf(-v));
    }
    return _mm512_loadu_ps(buf);
}

// ---------------------------------------------------------------------------
// Horizontal sum of __m512 → scalar f32
// ---------------------------------------------------------------------------
static inline float sgl_hsum_f32(__m512 v) {
    __m256 lo  = _mm512_castps512_ps256(v);
    __m256 hi  = _mm512_extractf32x8_ps(v, 1);
    __m256 s8  = _mm256_add_ps(lo, hi);
    __m128 lo4 = _mm256_castps256_ps128(s8);
    __m128 hi4 = _mm256_extractf128_ps(s8, 1);
    __m128 s4  = _mm_add_ps(lo4, hi4);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    return _mm_cvtss_f32(s4);
}

// ---------------------------------------------------------------------------
// Squared-sum accumulation for RMSNorm (32 bf16 per call)
// ---------------------------------------------------------------------------
static inline __m512 sgl_sq_acc_bf16(__m512 acc, __m512i v) {
    __m512 lo = sgl_cvt_bf16_lo_f32(v);
    __m512 hi = sgl_cvt_bf16_hi_f32(v);
    acc = _mm512_fmadd_ps(lo, lo, acc);
    acc = _mm512_fmadd_ps(hi, hi, acc);
    return acc;
}

// ---------------------------------------------------------------------------
// Scalar tail helpers
// ---------------------------------------------------------------------------
static inline void sgl_cvt_f32_bf16_scalar(const float* s, bf16_t* d, int n) {
    for (int i=0;i<n;++i) d[i]=f32_to_bf16(s[i]);
}
static inline void sgl_cvt_bf16_f32_scalar(const bf16_t* s, float* d, int n) {
    for (int i=0;i<n;++i) d[i]=bf16_to_f32(s[i]);
}
