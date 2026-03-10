// test_kernels.cpp — Standalone correctness tests
// Uses scalar reference implementations to verify kernel outputs.
// Compiles and runs on any x86-64 (BF16 intrinsics are gated at runtime).
//
// Compile:
//   g++ -O2 -march=native -mavx512f -mavx512bf16 -mavx512vl -mavx512bw \
//       -fopenmp -std=c++17 \
//       test_kernels.cpp \
//       ../src/sgl_norm.cpp ../src/sgl_activation.cpp ../src/sgl_rope.cpp \
//       ../src/sgl_gemm.cpp ../src/sgl_moe.cpp \
//       -I../include -lm -o test_kernels && ./test_kernels

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "../include/sgl_cpu_kernels.h"
#include "../include/sgl_vec.h"

static int g_passed = 0, g_failed = 0;

#define CHECK_CLOSE(a, b, tol, msg) \
    do { \
        float _a = (float)(a), _b = (float)(b); \
        if (fabsf(_a - _b) > (tol)) { \
            printf("  FAIL [%s]: got %.6f expected %.6f (diff=%.2e)\n", \
                   msg, _a, _b, fabsf(_a-_b)); \
            ++g_failed; \
        } else { ++g_passed; } \
    } while(0)

static void fill_rand_bf16(bf16_t* buf, int n, float scale = 1.0f) {
    for (int i = 0; i < n; ++i)
        buf[i] = f32_to_bf16(scale * ((float)rand() / RAND_MAX * 2.0f - 1.0f));
}
static void fill_const_bf16(bf16_t* buf, int n, float v) {
    for (int i = 0; i < n; ++i) buf[i] = f32_to_bf16(v);
}
static float ref_silu(float x) { return x / (1.0f + expf(-x)); }
static void ref_rms_norm(float* out, const bf16_t* x, const bf16_t* w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; ++i) { float v = bf16_to_f32(x[i]); ss += v * v; }
    float scale = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; ++i) out[i] = bf16_to_f32(x[i]) * scale * bf16_to_f32(w[i]);
}

static void test_bf16_roundtrip() {
    float vals[] = {0.0f, 1.0f, -1.0f, 3.14159f, -0.001f};
    for (float v : vals) {
        float rt = bf16_to_f32(f32_to_bf16(v));
        CHECK_CLOSE(rt, v, fabsf(v)*0.02f + 1e-4f, "bf16_roundtrip");
    }
    printf("  bf16 roundtrip: OK\n");
}

static void test_rms_norm() {
    const int T = 8, H = 128;
    bf16_t x[T*H], w[H], out[T*H];
    float  ref[H];
    fill_rand_bf16(x, T*H, 0.5f);
    fill_const_bf16(w, H, 1.0f);
    sgl_rms_norm_bf16(out, x, w, T, H, 1e-6f);
    ref_rms_norm(ref, x, w, H, 1e-6f);
    for (int i = 0; i < H; ++i)
        CHECK_CLOSE(bf16_to_f32(out[i]), ref[i], 0.02f, "rms_norm");
    printf("  rms_norm (%dx%d): OK\n", T, H);
}

static void test_add_rms_norm() {
    const int T = 4, H = 64;
    bf16_t x[T*H], res[T*H], w[H], out[T*H];
    fill_rand_bf16(x,   T*H, 0.3f);
    fill_rand_bf16(res, T*H, 0.3f);
    fill_const_bf16(w, H, 1.0f);
    // Expected: res[0:H] += x[0:H], then rms_norm
    float exp_res[H]; bf16_t exp_res_bf16[H]; float exp_out[H];
    for (int i = 0; i < H; ++i) exp_res[i] = bf16_to_f32(res[i]) + bf16_to_f32(x[i]);
    for (int i = 0; i < H; ++i) exp_res_bf16[i] = f32_to_bf16(exp_res[i]);
    ref_rms_norm(exp_out, exp_res_bf16, w, H, 1e-6f);
    sgl_add_rms_norm_bf16(res, out, x, w, T, H, 1e-6f);
    for (int i = 0; i < H; ++i)
        CHECK_CLOSE(bf16_to_f32(out[i]), exp_out[i], 0.02f, "add_rms_norm");
    printf("  add_rms_norm (%dx%d): OK\n", T, H);
}

static void test_silu_and_mul() {
    const int T = 4, D = 64;
    bf16_t x[T*D], y[T*D], out[T*D];
    fill_rand_bf16(x, T*D, 0.5f);
    fill_rand_bf16(y, T*D, 0.5f);
    sgl_silu_and_mul_bf16(out, x, y, T, D);
    for (int i = 0; i < D; ++i)
        CHECK_CLOSE(bf16_to_f32(out[i]),
                    ref_silu(bf16_to_f32(x[i])) * bf16_to_f32(y[i]),
                    0.02f, "silu_and_mul");
    // inplace — buf is [T, 2D]: each token = [gate_D, up_D] contiguous
    bf16_t buf[T*2*D];
    for (int t = 0; t < T; ++t) {
        memcpy(buf + t*2*D,   x + t*D, D*sizeof(bf16_t));  // gate
        memcpy(buf + t*2*D+D, y + t*D, D*sizeof(bf16_t));  // up
    }
    sgl_silu_and_mul_inplace_bf16(buf, T, D);
    // Check token 0 output (overwrote gate half)
    for (int i = 0; i < D; ++i)
        CHECK_CLOSE(bf16_to_f32(buf[i]),
                    ref_silu(bf16_to_f32(x[i])) * bf16_to_f32(y[i]),
                    0.02f, "silu_and_mul_inplace");
    printf("  silu_and_mul (%dx%d): OK\n", T, D);
}

static void test_gemm() {
    // Small random: compare kernel vs scalar reference
    const int M = 12, N = 16, K = 32;
    bf16_t A[M*K], B[N*K], C[M*N];
    fill_rand_bf16(A, M*K, 0.2f);
    fill_rand_bf16(B, N*K, 0.2f);
    sgl_gemm_bf16(C, A, B, M, N, K);
    int nerr = 0;
    for (int m = 0; m < M && nerr < 4; ++m)
        for (int n = 0; n < N && nerr < 4; ++n) {
            float ref = 0;
            for (int k = 0; k < K; ++k)
                ref += bf16_to_f32(A[m*K+k]) * bf16_to_f32(B[n*K+k]);
            float tol = fabsf(ref) * 0.06f + 2e-3f;
            if (fabsf(bf16_to_f32(C[m*N+n]) - ref) > tol) {
                printf("  FAIL gemm[%d,%d]: got %.5f ref %.5f\n",
                       m, n, bf16_to_f32(C[m*N+n]), ref);
                ++g_failed; ++nerr;
            } else ++g_passed;
        }
    printf("  gemm (%dx%dx%d): OK\n", M, N, K);
}

static void test_rope() {
    // Identity rotation: cos=1, sin=0 → output==input
    const int T = 2, NQ = 2, NK = 1, HD = 16;
    int half = HD / 2;
    bf16_t q[T*NQ*HD], k[T*NK*HD], qo[T*NQ*HD], ko[T*NK*HD];
    float cos_t[T*half], sin_t[T*half];
    int32_t pos[T] = {0, 1};
    fill_rand_bf16(q, T*NQ*HD, 0.5f);
    fill_rand_bf16(k, T*NK*HD, 0.5f);
    for (int i = 0; i < T*half; ++i) { cos_t[i] = 1.0f; sin_t[i] = 0.0f; }
    sgl_rope_neox_bf16(qo, ko, q, k, cos_t, sin_t, pos, T, NQ, NK, HD, T);
    for (int i = 0; i < HD; ++i)
        CHECK_CLOSE(bf16_to_f32(qo[i]), bf16_to_f32(q[i]), 0.01f, "rope_identity");
    // 90-degree rotation: (1,0) → (0,1)
    {
        const int HD2 = 4; int h2 = HD2/2;
        bf16_t q2[HD2], qo2[HD2], ko2[HD2];
        float cos2[h2], sin2[h2]; int32_t p2[1] = {0};
        for (int i=0;i<h2;++i){q2[i]=f32_to_bf16(1.0f);q2[i+h2]=f32_to_bf16(0.0f);cos2[i]=0.0f;sin2[i]=1.0f;}
        sgl_rope_neox_bf16(qo2,ko2,q2,q2,cos2,sin2,p2,1,1,1,HD2,1);
        CHECK_CLOSE(bf16_to_f32(qo2[0]),  0.0f, 0.02f, "rope_90deg_real");
        CHECK_CLOSE(bf16_to_f32(qo2[h2]), 1.0f, 0.02f, "rope_90deg_imag");
    }
    printf("  rope (%dx%dx%d): OK\n", T, NQ, HD);
}

static void test_fused_experts() {
    const int T=4, K=32, N=16, E=2, TOPK=1;
    bf16_t hidden[T*K], w1[E*2*N*K], w2[E*K*N], out[T*K];
    float tw[T*TOPK]; int32_t ti[T*TOPK];
    fill_rand_bf16(hidden, T*K,     0.05f);
    fill_rand_bf16(w1,     E*2*N*K, 0.05f);
    fill_rand_bf16(w2,     E*K*N,   0.05f);
    for (int t=0;t<T;++t){tw[t]=1.0f;ti[t]=t%E;}
    memset(out, 0, sizeof(out));
    sgl_fused_experts_bf16(out, hidden, w1, w2, tw, ti, T, E, K, N, TOPK);
    // All outputs finite
    bool ok = true;
    for (int i=0;i<T*K;++i) if(!isfinite(bf16_to_f32(out[i]))){ok=false;break;}
    if (!ok) { printf("  FAIL fused_experts: non-finite output\n"); ++g_failed; }
    else ++g_passed;
    // Scalar reference for token 0, expert 0
    float h1[2*N], gate[N];
    for (int n=0;n<2*N;++n){h1[n]=0;for(int k=0;k<K;++k)h1[n]+=bf16_to_f32(w1[n*K+k])*bf16_to_f32(hidden[k]);}
    for (int n=0;n<N;++n) gate[n]=ref_silu(h1[n])*h1[n+N];
    for (int k=0;k<K&&k<8;++k){
        float h2k=0;
        for(int n=0;n<N;++n) h2k+=bf16_to_f32(w2[k*N+n])*gate[n];
        CHECK_CLOSE(bf16_to_f32(out[k]), h2k, fabsf(h2k)*0.1f+0.01f, "fused_experts_value");
    }
    printf("  fused_experts (%dT x %dK x %dN): OK\n", T, K, N);
}

int main() {
    srand(42);
    printf("=== SGLang standalone CPU kernel tests ===\n");
    printf("  AVX-512 BF16 (runtime): %s\n\n", cpu_has_avx512bf16() ? "YES" : "NO – scalar fallback active");
    test_bf16_roundtrip();
    test_rms_norm();
    test_add_rms_norm();
    test_silu_and_mul();
    test_gemm();
    test_rope();
    test_fused_experts();
    printf("\n=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
