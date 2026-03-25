// Elementwise vector operations - arithmetic, scaling, and basic math
#pragma once

#include "ggml-impl.h"
#include "simd-mappings.h"
#include "ggml.h"
#include "ggml-cpu.h"

#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

typedef double ggml_float;

#define GGML_GELU_FP16
#define GGML_GELU_QUICK_FP16
#define GGML_SOFT_MAX_UNROLL 4
#define GGML_VEC_DOT_UNROLL  2
#define GGML_VEC_MAD_UNROLL  32

#ifdef __cplusplus
extern "C" {
#endif

inline static void ggml_vec_set_i8(const int n, int8_t * x, const int8_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_set_i16(const int n, int16_t * x, const int16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void ggml_vec_set_i32(const int n, int32_t * x, const int32_t   v) { for (int i = 0; i < n; ++i) x[i] = v;    }
inline static void ggml_vec_cpy_i32(const int n, int32_t * y, const int32_t * x) { for (int i = 0; i < n; ++i) y[i] = x[i]; }

inline static void ggml_vec_set_f16(const int n, ggml_fp16_t * x, const ggml_fp16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_set_bf16(const int n, ggml_bf16_t * x, const ggml_bf16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) {
    int i = 0;
#if defined(__AVX2__)
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vz = _mm256_add_ps(vx, vy);
        _mm256_storeu_ps(z + i, vz);
    }
#endif
    for (; i < n; ++i) {
        z[i] = x[i] + y[i];
    }
}

inline static void ggml_vec_add_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) + GGML_CPU_FP16_TO_FP32(y[i]));
    }
}
inline static void ggml_vec_add1_f32(const int n, float * z, const float * x, const float   v) { for (int i = 0; i < n; ++i) z[i]  = x[i] + v;    }
inline static void ggml_vec_acc_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i] += x[i];        }
inline static void ggml_vec_acc1_f32(const int n, float * y, const float   v)                  { for (int i = 0; i < n; ++i) y[i] += v;           }
inline static void ggml_vec_sub_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] - y[i]; }
inline static void ggml_vec_sub_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) - GGML_CPU_FP16_TO_FP32(y[i]));
    }
}
inline static void ggml_vec_set_f32 (const int n, float * x, const float   v)                  { for (int i = 0; i < n; ++i) x[i]  = v;           }
inline static void ggml_vec_cpy_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = x[i];        }
inline static void ggml_vec_neg_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = -x[i];       }
inline static void ggml_vec_neg_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(-GGML_CPU_FP16_TO_FP32(x[i]));
    }
}

inline static void ggml_vec_mul_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]*y[i];   }
inline static void ggml_vec_mul_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) * GGML_CPU_FP16_TO_FP32(y[i]));
    }
}
inline static void ggml_vec_div_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]/y[i];   }
inline static void ggml_vec_div_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) / GGML_CPU_FP16_TO_FP32(y[i]));
    }
}

// compute GGML_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes

inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) {
#if defined(GGML_USE_ACCELERATE)
    vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)
        const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
        const int ggml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
        const int ggml_f32_step = 2 * ggml_f32_epr;

        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);
        const int np = (n & ~(ggml_f32_step - 1));
        svfloat32_t ay1;
        svfloat32_t ay2;
        for (int i = 0; i < np; i += ggml_f32_step) {
            ay1 = GGML_F32_VEC_LOAD(y + i);
            ay1 = GGML_F32_VEC_MUL(ay1, vx);
            GGML_F32_VEC_STORE(y + i, ay1);

            ay2 = GGML_F32_VEC_LOAD(y + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_MUL(ay2, vx);
            GGML_F32_VEC_STORE(y + i + 1*ggml_f32_epr, ay2);
        }
        // leftovers
        // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmad on available elements only
        for (int i = np; i < n; i += ggml_f32_epr) {
            svbool_t pg = svwhilelt_b32(i, n);
            ay1 = svld1_f32(pg, y + i);
            ay1 = svmul_f32_m(pg, ay1, vx);
            svst1_f32(pg, y + i, ay1);
        }
    #elif defined(__riscv_v_intrinsic)
        for (int i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
            vfloat32m8_t ny = __riscv_vfmul_vf_f32m8(ay, v, avl);
            __riscv_vse32_v_f32m8(&y[i], ny, avl);
        }
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        for (int i = np; i < n; ++i) {
            y[i] *= v;
        }
    #endif
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
#endif
}

inline static void ggml_vec_scale_f16(const int n, ggml_fp16_t * y, const float v) {
#if defined(GGML_SIMD) && defined(__ARM_FEATURE_SVE)
    const int sve_register_length = svcntb() * 8;
    const int ggml_f16_epr = sve_register_length / 16;
    const int ggml_f16_step = 2 * ggml_f16_epr;

    GGML_F16x_VEC vx =  GGML_F16x_VEC_SET1(v);
    const int np = (n & ~(ggml_f16_step - 1));
    svfloat16_t ay1, ay2;

    for (int i = 0; i < np; i += ggml_f16_step) {
        ay1 = GGML_F16x_VEC_LOAD(y + i + 0*ggml_f16_epr, 0);
        ay1 = GGML_F16x_VEC_MUL(ay1, vx);
        GGML_F16x_VEC_STORE(y + i + 0*ggml_f16_epr, ay1, 0);

        ay2 = GGML_F16x_VEC_LOAD(y + i + 1*ggml_f16_epr, 1);
        ay2 = GGML_F16x_VEC_MUL(ay2, vx);
        GGML_F16x_VEC_STORE(y + i + 1*ggml_f16_epr, ay2, 1);
    }
    // leftovers
    // maximum number of leftover elements will be less that ggmlF_16x_epr. Apply predicated svmad on available elements only
    if (np < n) {
        svbool_t pg = svwhilelt_b16(np, n);
        svfloat16_t hy = svld1_f16(pg, (__fp16 *)(y + np));
        svfloat16_t out = svmul_f16_m(pg, hy, vx);
        svst1_f16(pg, (__fp16 *)(y + np), out);
    }
#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfh)
    const ggml_fp16_t s = GGML_CPU_FP32_TO_FP16(v);
    const _Float16 scale = *(const _Float16*)(&s);

    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m4();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (int i = 0; i < np; i += step) {
        vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((const _Float16*)y + i, epr);
        ay0 = __riscv_vfmul_vf_f16m4(ay0, scale, epr);
        __riscv_vse16_v_f16m4((_Float16*)y + i, ay0, epr);
        __asm__ __volatile__ ("" ::: "memory");

        vfloat16m4_t ay1 = __riscv_vle16_v_f16m4((const _Float16*)y + i + epr, epr);
        ay1 = __riscv_vfmul_vf_f16m4(ay1, scale, epr);
        __riscv_vse16_v_f16m4((_Float16*)y + i + epr, ay1, epr);
        __asm__ __volatile__ ("" ::: "memory");
    }

    // leftovers
    int vl;
    for (int i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m4(n - i);
        vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((const _Float16*)y + i, vl);
        ay0 = __riscv_vfmul_vf_f16m4(ay0, scale, vl);
        __riscv_vse16_v_f16m4((_Float16*)y + i, ay0, vl);
    }
#elif defined(GGML_SIMD)
    const int np = (n & ~(GGML_F16_STEP - 1));

    GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

    GGML_F16_VEC ay[GGML_F16_ARR];

    for (int i = 0; i < np; i += GGML_F16_STEP) {
        for (int j = 0; j < GGML_F16_ARR; j++) {
            ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_MUL(ay[j], vx);

            GGML_F16_VEC_STORE(y + i + j*GGML_F16_EPR, ay, j);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i])*v);
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i])*v);
    }
#endif
}

inline static void ggml_vec_norm_f32 (const int n, float * s, const float * x) { ggml_vec_dot_f32(n, s, 0, x, 0, x, 0, 1); *s = sqrtf(*s);   }
inline static void ggml_vec_sqr_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i]*x[i];   }
inline static void ggml_vec_sqr_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(v*v);
    }
}
inline static void ggml_vec_sqrt_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sqrtf(x[i]); }
inline static void ggml_vec_sqrt_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(sqrtf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_log_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = logf(x[i]);  }
inline static void ggml_vec_log_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(logf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_sin_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sinf(x[i]);  }
inline static void ggml_vec_sin_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(sinf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_cos_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = cosf(x[i]);  }
inline static void ggml_vec_cos_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(cosf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}

#ifdef __cplusplus
}
#endif
