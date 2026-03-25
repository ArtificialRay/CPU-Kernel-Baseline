// GEMV vector kernels - dot products and multiply-add operations
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

// global data
//

// precomputed gelu table for f16 (128 KB)
extern ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
extern ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

//
// fundamental operations
//

void ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc);
void ggml_vec_dot_bf16(int n, float * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, int nrc);
void ggml_vec_dot_f16(int n, float * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc);


inline static void ggml_vec_dot_f16_unroll(const int n, const int xs, float * GGML_RESTRICT s, void * GGML_RESTRICT xv, ggml_fp16_t * GGML_RESTRICT y) {
    ggml_float sumf[GGML_VEC_DOT_UNROLL] = { 0.0 };

    ggml_fp16_t * GGML_RESTRICT x[GGML_VEC_DOT_UNROLL];

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        x[i] = (ggml_fp16_t *) ((char *) xv + i*xs);
    }

#if defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)

        const int sve_register_length = svcntb() * 8;
        const int ggml_f16_epr = sve_register_length / 16; // running when 16
        const int ggml_f16_step = 8 * ggml_f16_epr; // choose 8 SVE registers

        const int np = (n & ~(ggml_f16_step - 1));

        svfloat16_t sum_00 = svdup_n_f16(0.0f);
        svfloat16_t sum_01 = svdup_n_f16(0.0f);
        svfloat16_t sum_02 = svdup_n_f16(0.0f);
        svfloat16_t sum_03 = svdup_n_f16(0.0f);

        svfloat16_t sum_10 = svdup_n_f16(0.0f);
        svfloat16_t sum_11 = svdup_n_f16(0.0f);
        svfloat16_t sum_12 = svdup_n_f16(0.0f);
        svfloat16_t sum_13 = svdup_n_f16(0.0f);

        svfloat16_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
        svfloat16_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;

        for (int i = 0; i < np; i += ggml_f16_step) {
            ay1 = GGML_F16x_VEC_LOAD(y + i + 0 * ggml_f16_epr, 0); // 8 elements

            ax1 = GGML_F16x_VEC_LOAD(x[0] + i + 0*ggml_f16_epr, 0); // 8 elements
            sum_00 = GGML_F16x_VEC_FMA(sum_00, ax1, ay1);     // sum_00 = sum_00+ax1*ay1
            ax1 = GGML_F16x_VEC_LOAD(x[1] + i + 0*ggml_f16_epr, 0); // 8 elements
            sum_10 = GGML_F16x_VEC_FMA(sum_10, ax1, ay1);

            ay2 = GGML_F16x_VEC_LOAD(y + i + 1 * ggml_f16_epr, 1); // next 8 elements

            ax2 = GGML_F16x_VEC_LOAD(x[0] + i + 1*ggml_f16_epr, 1); // next 8 elements
            sum_01 = GGML_F16x_VEC_FMA(sum_01, ax2, ay2);
            ax2 = GGML_F16x_VEC_LOAD(x[1] + i + 1*ggml_f16_epr, 1);
            sum_11 = GGML_F16x_VEC_FMA(sum_11, ax2, ay2);

            ay3 = GGML_F16x_VEC_LOAD(y + i + 2 * ggml_f16_epr, 2);

            ax3 = GGML_F16x_VEC_LOAD(x[0] + i + 2*ggml_f16_epr, 2);
            sum_02 = GGML_F16x_VEC_FMA(sum_02, ax3, ay3);
            ax3 = GGML_F16x_VEC_LOAD(x[1] + i + 2*ggml_f16_epr, 2);
            sum_12 = GGML_F16x_VEC_FMA(sum_12, ax3, ay3);

            ay4 = GGML_F16x_VEC_LOAD(y + i + 3 * ggml_f16_epr, 3);

            ax4 = GGML_F16x_VEC_LOAD(x[0] + i + 3*ggml_f16_epr, 3);
            sum_03 = GGML_F16x_VEC_FMA(sum_03, ax4, ay4);
            ax4 = GGML_F16x_VEC_LOAD(x[1] + i + 3*ggml_f16_epr, 3);
            sum_13 = GGML_F16x_VEC_FMA(sum_13, ax4, ay4);

            ay5 = GGML_F16x_VEC_LOAD(y + i + 4 * ggml_f16_epr, 4);

            ax5 = GGML_F16x_VEC_LOAD(x[0] + i + 4*ggml_f16_epr, 4);

            sum_00 = GGML_F16x_VEC_FMA(sum_00, ax5, ay5);
            ax5 = GGML_F16x_VEC_LOAD(x[1] + i + 4*ggml_f16_epr, 4);
            sum_10 = GGML_F16x_VEC_FMA(sum_10, ax5, ay5);

            ay6 = GGML_F16x_VEC_LOAD(y + i + 5 * ggml_f16_epr, 5);

            ax6 = GGML_F16x_VEC_LOAD(x[0] + i + 5*ggml_f16_epr, 5);

            sum_01 = GGML_F16x_VEC_FMA(sum_01, ax6, ay6);
            ax6 = GGML_F16x_VEC_LOAD(x[1] + i + 5*ggml_f16_epr, 5);
            sum_11 = GGML_F16x_VEC_FMA(sum_11, ax6, ay6);

            ay7 = GGML_F16x_VEC_LOAD(y + i + 6 * ggml_f16_epr, 6);

            ax7 = GGML_F16x_VEC_LOAD(x[0] + i + 6*ggml_f16_epr, 6);

            sum_02 = GGML_F16x_VEC_FMA(sum_02, ax7, ay7);
            ax7 = GGML_F16x_VEC_LOAD(x[1] + i + 6*ggml_f16_epr, 6);
            sum_12 = GGML_F16x_VEC_FMA(sum_12, ax7, ay7);

            ay8 = GGML_F16x_VEC_LOAD(y + i + 7 * ggml_f16_epr, 7);

            ax8 = GGML_F16x_VEC_LOAD(x[0] + i + 7*ggml_f16_epr, 7);

            sum_03 = GGML_F16x_VEC_FMA(sum_03, ax8, ay8);
            ax8 = GGML_F16x_VEC_LOAD(x[1] + i + 7*ggml_f16_epr, 7);
            sum_13 = GGML_F16x_VEC_FMA(sum_13, ax8, ay8);
        }

        const int np2 = (n & ~(ggml_f16_epr - 1));
        for (int k = np; k < np2; k += ggml_f16_epr) {
            svfloat16_t ry = GGML_F16x_VEC_LOAD(y + k, 0);

            svfloat16_t rx = GGML_F16x_VEC_LOAD(x[0] + k, 0);
            sum_00 = GGML_F16x_VEC_FMA(sum_00, rx, ry);
            rx = GGML_F16x_VEC_LOAD(x[1] + k, 0);
            sum_10 = GGML_F16x_VEC_FMA(sum_10, rx, ry);
        }

        if (np2 < n) {
            svbool_t pg = svwhilelt_b16(np2, n);
            svfloat16_t hx_0 = svld1_f16(pg, (const __fp16 *)(x[0] + np2));
            svfloat16_t hx_1 = svld1_f16(pg, (const __fp16 *)(x[1] + np2));
            svfloat16_t hy = svld1_f16(pg, (const __fp16 *)(y + np2));

            sum_00 = svmad_f16_x(pg, hx_0, hy, sum_00);
            sum_10 = svmad_f16_x(pg, hx_1, hy, sum_10);
        }
        GGML_F16x_VEC_REDUCE(sumf[0], sum_00, sum_01, sum_02, sum_03);
        GGML_F16x_VEC_REDUCE(sumf[1], sum_10, sum_11, sum_12, sum_13);

    #elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfh)
        size_t vl = __riscv_vsetvlmax_e32m4();

        // initialize accumulators to all zeroes
        vfloat32m4_t vsum0_0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t vsum0_1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t vsum1_0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t vsum1_1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

        // calculate step size
        const size_t epr = __riscv_vsetvlmax_e16m2();
        const size_t step = epr * 2;
        const int np = (n & ~(step - 1));

        // unroll by 2 along the row dimension
        for (int i = 0; i < np; i += step) {
            vfloat16m2_t ay0 = __riscv_vle16_v_f16m2((const _Float16 *)(y + i), epr);
            vfloat16m2_t ax0_0 = __riscv_vle16_v_f16m2((const _Float16 *)(x[0] + i), epr);
            vfloat16m2_t ax1_0 = __riscv_vle16_v_f16m2((const _Float16 *)(x[1] + i), epr);
            vsum0_0 = __riscv_vfwmacc_vv_f32m4(vsum0_0, ax0_0, ay0, epr);
            vsum1_0 = __riscv_vfwmacc_vv_f32m4(vsum1_0, ax1_0, ay0, epr);

            vfloat16m2_t ay1 = __riscv_vle16_v_f16m2((const _Float16 *)(y + i + epr), epr);
            vfloat16m2_t ax0_1 = __riscv_vle16_v_f16m2((const _Float16 *)(x[0] + i + epr), epr);
            vfloat16m2_t ax1_1 = __riscv_vle16_v_f16m2((const _Float16 *)(x[1] + i + epr), epr);
            vsum0_1 = __riscv_vfwmacc_vv_f32m4(vsum0_1, ax0_1, ay1, epr);
            vsum1_1 = __riscv_vfwmacc_vv_f32m4(vsum1_1, ax1_1, ay1, epr);
        }

        vfloat32m4_t vsum0 = __riscv_vfadd_vv_f32m4(vsum0_0, vsum0_1, vl);
        vfloat32m4_t vsum1 = __riscv_vfadd_vv_f32m4(vsum1_0, vsum1_1, vl);

        // leftovers
        for (int i = np; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m2(n - i);
            vfloat16m2_t ay = __riscv_vle16_v_f16m2((const _Float16 *)(y + i), vl);
            vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16 *)(x[0] + i), vl);
            vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((const _Float16 *)(x[1] + i), vl);

            vsum0 = __riscv_vfwmacc_vv_f32m4(vsum0, ax0, ay, vl);
            vsum1 = __riscv_vfwmacc_vv_f32m4(vsum1, ax1, ay, vl);
        }

        // reduce
        vl = __riscv_vsetvlmax_e32m2();
        vfloat32m2_t acc0_0 = __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(vsum0, 0),
                                    __riscv_vget_v_f32m4_f32m2(vsum0, 1), vl);
        vl = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t acc0_1 = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(acc0_0, 0),
        __riscv_vget_v_f32m2_f32m1(acc0_0, 1), vl);
        vfloat32m1_t redsum0 = __riscv_vfredusum_vs_f32m1_f32m1(
                                    acc0_1, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);

        vl = __riscv_vsetvlmax_e32m2();
        vfloat32m2_t acc1_0 = __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(vsum1, 0),
                                    __riscv_vget_v_f32m4_f32m2(vsum1, 1), vl);
        vl = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t acc1_1 = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(acc1_0, 0),
                                    __riscv_vget_v_f32m2_f32m1(acc1_0, 1), vl);
        vfloat32m1_t redsum1 = __riscv_vfredusum_vs_f32m1_f32m1(
                                    acc1_1, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
        sumf[0] = __riscv_vfmv_f_s_f32m1_f32(redsum0);
        sumf[1] = __riscv_vfmv_f_s_f32m1_f32(redsum1);

    #else
        const int np = (n & ~(GGML_F16_STEP - 1));

        GGML_F16_VEC sum[GGML_VEC_DOT_UNROLL][GGML_F16_ARR] = { { GGML_F16_VEC_ZERO } };

        GGML_F16_VEC ax[GGML_F16_ARR];
        GGML_F16_VEC ay[GGML_F16_ARR];

        for (int i = 0; i < np; i += GGML_F16_STEP) {
            for (int j = 0; j < GGML_F16_ARR; j++) {
                ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);

                for (int k = 0; k < GGML_VEC_DOT_UNROLL; ++k) {
                    ax[j] = GGML_F16_VEC_LOAD(x[k] + i + j*GGML_F16_EPR, j);

                    sum[k][j] = GGML_F16_VEC_FMA(sum[k][j], ax[j], ay[j]);
                }
            }
        }

        // reduce sum0..sum3 to sum0
        for (int k = 0; k < GGML_VEC_DOT_UNROLL; ++k) {
            GGML_F16_VEC_REDUCE(sumf[k], sum[k]);
        }

        // leftovers
        for (int i = np; i < n; ++i) {
            for (int j = 0; j < GGML_VEC_DOT_UNROLL; ++j) {
                sumf[j] += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[j][i])*GGML_CPU_FP16_TO_FP32(y[i]));
            }
        }
    #endif
#else
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < GGML_VEC_DOT_UNROLL; ++j) {
            sumf[j] += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[j][i])*GGML_CPU_FP16_TO_FP32(y[i]));
        }
    }
#endif

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        s[i] = (float)sumf[i];
    }
}

inline static void ggml_vec_mad_f32(const int n, float * GGML_RESTRICT y, const float * GGML_RESTRICT x, const float v) {
#if defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)

        const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
        const int ggml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
        const int ggml_f32_step = 8 * ggml_f32_epr; // choose 8 SVE registers
        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

        const int np = (n & ~(ggml_f32_step - 1));
        svfloat32_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
        svfloat32_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;
        for (int i = 0; i < np; i += ggml_f32_step) {

            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            ay1 = GGML_F32_VEC_FMA(ay1, ax1, vx);

            GGML_F32_VEC_STORE(y + i, ay1);

            ax2 = GGML_F32_VEC_LOAD(x + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_LOAD(y + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_FMA(ay2, ax2, vx);

            GGML_F32_VEC_STORE(y + i + 1*ggml_f32_epr, ay2);

            ax3 = GGML_F32_VEC_LOAD(x + i + 2*ggml_f32_epr);
            ay3 = GGML_F32_VEC_LOAD(y + i + 2*ggml_f32_epr);
            ay3 = GGML_F32_VEC_FMA(ay3, ax3, vx);

            GGML_F32_VEC_STORE(y + i + 2*ggml_f32_epr, ay3);

            ax4 = GGML_F32_VEC_LOAD(x + i + 3*ggml_f32_epr);
            ay4 = GGML_F32_VEC_LOAD(y + i + 3*ggml_f32_epr);
            ay4 = GGML_F32_VEC_FMA(ay4, ax4, vx);

            GGML_F32_VEC_STORE(y + i + 3*ggml_f32_epr, ay4);

            ax5 = GGML_F32_VEC_LOAD(x + i + 4*ggml_f32_epr);
            ay5 = GGML_F32_VEC_LOAD(y + i + 4*ggml_f32_epr);
            ay5 = GGML_F32_VEC_FMA(ay5, ax5, vx);

            GGML_F32_VEC_STORE(y + i + 4*ggml_f32_epr, ay5);

            ax6 = GGML_F32_VEC_LOAD(x + i + 5*ggml_f32_epr);
            ay6 = GGML_F32_VEC_LOAD(y + i + 5*ggml_f32_epr);
            ay6 = GGML_F32_VEC_FMA(ay6, ax6, vx);

            GGML_F32_VEC_STORE(y + i + 5*ggml_f32_epr, ay6);

            ax7 = GGML_F32_VEC_LOAD(x + i + 6*ggml_f32_epr);
            ay7 = GGML_F32_VEC_LOAD(y + i + 6*ggml_f32_epr);
            ay7 = GGML_F32_VEC_FMA(ay7, ax7, vx);

            GGML_F32_VEC_STORE(y + i + 6*ggml_f32_epr, ay7);

            ax8 = GGML_F32_VEC_LOAD(x + i + 7*ggml_f32_epr);
            ay8 = GGML_F32_VEC_LOAD(y + i + 7*ggml_f32_epr);
            ay8 = GGML_F32_VEC_FMA(ay8, ax8, vx);

            GGML_F32_VEC_STORE(y + i + 7*ggml_f32_epr, ay8);
        }
        // leftovers
        // Since 8 unrolls are done in above loop, leftovers lie in range [0, ggml_f32_step] which is handled in below loop
        const int np2 = (n & ~(ggml_f32_epr - 1));
        for (int i = np; i < np2; i += ggml_f32_epr) {
            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            ay1 = GGML_F32_VEC_FMA(ay1, ax1, vx);

            GGML_F32_VEC_STORE(y + i, ay1);
        }
        // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmad on available elements only
        if (np2 < n) {
            svbool_t pg =svwhilelt_b32(np2, n);
            ax1 = svld1_f32(pg, x + np2);
            ay1 = svld1_f32(pg, y + np2);
            ay1 = svmad_f32_m(pg, ax1, vx, ay1);

            svst1_f32(pg, y + np2, ay1);
        }
    #elif defined(__riscv_v_intrinsic)
        for (int i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ax = __riscv_vle32_v_f32m8(&x[i], avl);
            vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
            vfloat32m8_t ny = __riscv_vfmadd_vf_f32m8(ax, v, ay, avl);
            __riscv_vse32_v_f32m8(&y[i], ny, avl);
        }
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

        GGML_F32_VEC ax[GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_FMA(ay[j], ax[j], vx);

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        for (int i = np; i < n; ++i) {
            y[i] += x[i]*v;
        }
    #endif
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] += x[i]*v;
    }
#endif
}

inline static void ggml_vec_mad_f16(const int n, ggml_fp16_t * GGML_RESTRICT y, const ggml_fp16_t * GGML_RESTRICT x, const float v) {
#if defined(GGML_SIMD) && defined(__ARM_FEATURE_SVE)
    const int sve_register_length = svcntb() * 8;
    const int ggml_f16_epr = sve_register_length / 16;
    const int ggml_f16_step = 8 * ggml_f16_epr;

    GGML_F16x_VEC vx = GGML_F16x_VEC_SET1(v);

    int np = (n & ~(ggml_f16_step - 1));

    svfloat16_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
    svfloat16_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;
    for (int i = 0; i < np; i += ggml_f16_step) {
        ax1 = GGML_F16x_VEC_LOAD(x + i + 0 * ggml_f16_epr, 0);
        ay1 = GGML_F16x_VEC_LOAD(y + i + 0 * ggml_f16_epr, 0);
        ay1 = GGML_F16x_VEC_FMA(ay1, ax1, vx);

        GGML_F16x_VEC_STORE(y + i + 0 * ggml_f16_epr, ay1, 0);

        ax2 = GGML_F16x_VEC_LOAD(x + i + 1 * ggml_f16_epr, 1);
        ay2 = GGML_F16x_VEC_LOAD(y + i + 1 * ggml_f16_epr, 1);
        ay2 = GGML_F16x_VEC_FMA(ay2, ax2, vx);

        GGML_F16x_VEC_STORE(y + i + 1 * ggml_f16_epr, ay2, 1);

        ax3 = GGML_F16x_VEC_LOAD(x + i + 2 * ggml_f16_epr, 2);
        ay3 = GGML_F16x_VEC_LOAD(y + i + 2 * ggml_f16_epr, 2);
        ay3 = GGML_F16x_VEC_FMA(ay3, ax3, vx);

        GGML_F16x_VEC_STORE(y + i + 2 * ggml_f16_epr, ay3, 2);

        ax4 = GGML_F16x_VEC_LOAD(x + i + 3 * ggml_f16_epr, 3);
        ay4 = GGML_F16x_VEC_LOAD(y + i + 3 * ggml_f16_epr, 3);
        ay4 = GGML_F16x_VEC_FMA(ay4, ax4, vx);

        GGML_F16x_VEC_STORE(y + i + 3 * ggml_f16_epr, ay4, 3);

        ax5 = GGML_F16x_VEC_LOAD(x + i + 4 * ggml_f16_epr, 4);
        ay5 = GGML_F16x_VEC_LOAD(y + i + 4 * ggml_f16_epr, 4);
        ay5 = GGML_F16x_VEC_FMA(ay5, ax5, vx);

        GGML_F16x_VEC_STORE(y + i + 4 * ggml_f16_epr, ay5, 4);

        ax6 = GGML_F16x_VEC_LOAD(x + i + 5 * ggml_f16_epr, 5);
        ay6 = GGML_F16x_VEC_LOAD(y + i + 5 * ggml_f16_epr, 5);
        ay6 = GGML_F16x_VEC_FMA(ay6, ax6, vx);

        GGML_F16x_VEC_STORE(y + i + 5 * ggml_f16_epr, ay6, 5);

        ax7 = GGML_F16x_VEC_LOAD(x + i + 6 * ggml_f16_epr, 6);
        ay7 = GGML_F16x_VEC_LOAD(y + i + 6 * ggml_f16_epr, 6);
        ay7 = GGML_F16x_VEC_FMA(ay7, ax7, vx);

        GGML_F16x_VEC_STORE(y + i + 6 * ggml_f16_epr, ay7, 6);

        ax8 = GGML_F16x_VEC_LOAD(x + i + 7 * ggml_f16_epr, 7);
        ay8 = GGML_F16x_VEC_LOAD(y + i + 7 * ggml_f16_epr, 7);
        ay8 = GGML_F16x_VEC_FMA(ay8, ax8, vx);

        GGML_F16x_VEC_STORE(y + i + 7 * ggml_f16_epr, ay8, 7);
    }
    const int np2 = (n & ~(ggml_f16_epr - 1));
    for (int k = np; k < np2; k += ggml_f16_epr) {
        svfloat16_t rx = GGML_F16x_VEC_LOAD(x + k, 0);
        svfloat16_t ry = GGML_F16x_VEC_LOAD(y + k, 0);
        ry = GGML_F16x_VEC_FMA(ry, rx, vx);

        GGML_F16x_VEC_STORE(y + k, ry, 0);
    }

    if (np2 < n) {
        svbool_t pg = svwhilelt_b16(np2, n);
        svfloat16_t hx = svld1_f16(pg, (const __fp16 *)(x + np2));
        svfloat16_t hy = svld1_f16(pg, (const __fp16 *)(y + np2));
        hy = svmad_f16_x(pg, hx, vx, hy);
        svst1_f16(pg, (__fp16 *)(y + np2), hy);
    }
    np = n;
#elif defined(__riscv_zvfh) // implies __riscv_v_intrinsic
    const ggml_fp16_t s = GGML_CPU_FP32_TO_FP16(v);
    const _Float16 scale = *(const _Float16*)(&s);

    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m4();
    const int step = epr * 2;
    int np = (n & ~(step - 1));

    // unroll by 2
    for (int i = 0; i < np; i += step) {
        vfloat16m4_t ax0 = __riscv_vle16_v_f16m4((const _Float16*)x + i, epr);
        vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((const _Float16*)y + i, epr);
        ay0 = __riscv_vfmacc_vf_f16m4(ay0, scale, ax0, epr);
        __riscv_vse16_v_f16m4((_Float16*)y + i, ay0, epr);
        __asm__ __volatile__ ("" ::: "memory");

        vfloat16m4_t ax1 = __riscv_vle16_v_f16m4((const _Float16*)x + i + epr, epr);
        vfloat16m4_t ay1 = __riscv_vle16_v_f16m4((const _Float16*)y + i + epr, epr);
        ay1 = __riscv_vfmacc_vf_f16m4(ay1, scale, ax1, epr);
        __riscv_vse16_v_f16m4((_Float16*)y + i + epr, ay1, epr);
        __asm__ __volatile__ ("" ::: "memory");
    }

    // leftovers
    int vl;
    for (int i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m4(n - i);
        vfloat16m4_t ax0 = __riscv_vle16_v_f16m4((const _Float16*)x + i, vl);
        vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((const _Float16*)y + i, vl);
        ay0 = __riscv_vfmacc_vf_f16m4(ay0, scale, ax0, vl);
        __riscv_vse16_v_f16m4((_Float16*)y + i, ay0, vl);
    }
    np = n;
#elif defined(GGML_SIMD)
    const int np = (n & ~(GGML_F16_STEP - 1));

    GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

    GGML_F16_VEC ax[GGML_F16_ARR];
    GGML_F16_VEC ay[GGML_F16_ARR];

    for (int i = 0; i < np; i += GGML_F16_STEP) {
        for (int j = 0; j < GGML_F16_ARR; j++) {
            ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_FMA(ay[j], ax[j], vx);

            GGML_F16_VEC_STORE(y + i + j*GGML_F16_EPR, ay, j);
        }
    }
#else
    const int np = 0;
#endif

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i]) + GGML_CPU_FP16_TO_FP32(x[i])*v);
    }
}

// xs and vs are byte strides of x and v
inline static void ggml_vec_mad_f32_unroll(const int n, const int xs, const int vs, float * GGML_RESTRICT y, const float * GGML_RESTRICT xv, const float * GGML_RESTRICT vv) {

    const float * GGML_RESTRICT x[GGML_VEC_MAD_UNROLL];
    const float * GGML_RESTRICT v[GGML_VEC_MAD_UNROLL];

    for (int i = 0; i < GGML_VEC_MAD_UNROLL; ++i) {
        x[i] = (const float *) ((const char *) xv + i*xs);
        v[i] = (const float *) ((const char *) vv + i*vs);
    }

#if defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)
        // scalar Route to scalar implementation       //TODO: Write SVE code
        for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
            for (int i = 0; i < n; ++i) {
                y[i] += x[k][i]*v[k][0];
            }
        }
    #elif defined(__riscv_v_intrinsic)
        for (int i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
            for (int k = 0; k < GGML_VEC_MAD_UNROLL; k++) {
                vfloat32m8_t ax = __riscv_vle32_v_f32m8(&x[k][i], avl);
                ay = __riscv_vfmadd_vf_f32m8(ax, v[k][0], ay, avl);
            }
            __riscv_vse32_v_f32m8(&y[i], ay, avl);
        }
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vx[GGML_VEC_MAD_UNROLL];

        for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
            vx[k] = GGML_F32_VEC_SET1(v[k][0]);
        }

        GGML_F32_VEC ax[GGML_VEC_MAD_UNROLL][GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);

                for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
                    ax[k][j] = GGML_F32_VEC_LOAD(x[k] + i + j*GGML_F32_EPR);
                    ay[j] = GGML_F32_VEC_FMA(ay[j], ax[k][j], vx[k]);
                }

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
            for (int i = np; i < n; ++i) {
                y[i] += x[k][i]*v[k][0];
            }
        }
    #endif
#else
    // scalar
    for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
        for (int i = 0; i < n; ++i) {
            y[i] += x[k][i]*v[k][0];
        }
    }
#endif
}

inline static void ggml_vec_mad1_f32(const int n, float * y, const float * x, const float s, const float b) {
#if defined(GGML_USE_ACCELERATE)
    vDSP_vsmsa(x, 1, &s, &b, y, 1, n);
#elif defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)
        // scalar ; TODO: Write SVE code
        for (int i = 0; i < n; ++i) {
            y[i] = x[i]*s + b;
        }
    #elif defined(__riscv_v_intrinsic)
        for (int i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ax = __riscv_vle32_v_f32m8(&x[i], avl);
            vfloat32m8_t vb = __riscv_vfmv_v_f_f32m8(b, avl);
            vfloat32m8_t ny = __riscv_vfmadd_vf_f32m8(ax, s, vb, avl);
            __riscv_vse32_v_f32m8(&y[i], ny, avl);
        }
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vs = GGML_F32_VEC_SET1(s);
        GGML_F32_VEC vb = GGML_F32_VEC_SET1(b);

        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ay[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_FMA(vb, ay[j], vs);

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        for (int i = np; i < n; ++i) {
            y[i] = x[i]*s + b;
        }
    #endif
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = x[i]*s + b;
    }
#endif
}

//inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) { for (int i = 0; i < n; ++i) y[i] *= v;          }

#ifdef __cplusplus
}
#endif
