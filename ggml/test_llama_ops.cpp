// =============================================================================
// test_llama_ops.cpp — Unit tests for llama_neon_ops.h
//
// Compile (ARM64):
//   aarch64-linux-gnu-g++ -O2 -std=c++11 -I../include -o test_llama_ops test_llama_ops.cpp
//
// Compile (ARM64 with dotprod):
//   aarch64-linux-gnu-g++ -O2 -std=c++11 -march=armv8.2-a+dotprod -I../include \
//       -o test_llama_ops test_llama_ops.cpp
//
// Compile (x86 host, scalar fallback for CI):
//   g++ -O2 -std=c++11 -DLLAMA_NO_NEON -I../include -o test_llama_ops test_llama_ops.cpp
// =============================================================================

#include "llama_neon_ops.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <functional>
#include <numeric>

using namespace llama_ops;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------
static int g_pass = 0, g_fail = 0;

static bool approx(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) <= tol + 1e-5f * std::fabs(b);
}
static bool vec_approx(const float* a, const float* b, int n, float tol = 1e-3f) {
    for (int i = 0; i < n; ++i)
        if (!approx(a[i], b[i], tol)) {
            std::printf("  [%d]  got %.6f  expected %.6f  diff %.2e\n",
                        i, a[i], b[i], std::fabs(a[i]-b[i]));
            return false;
        }
    return true;
}
#define TEST(name, expr) do {                                         \
    bool _ok = (expr);                                                \
    std::printf("[%s] %s\n", _ok ? "PASS" : "FAIL", name);           \
    if (_ok) ++g_pass; else ++g_fail;                                 \
} while(0)

// Simple deterministic fill
static void fill(float* p, int n, float seed = 1.f) {
    for (int i = 0; i < n; ++i)
        p[i] = (float)(((int)(i * 2654435769u + (int)(seed * 1000))) % 200 - 100) * 0.02f;
}

// ---------------------------------------------------------------------------
// 1. FP16 ↔ FP32 conversion
// ---------------------------------------------------------------------------
void test_fp16_convert() {
    // Round-trip: f32 → f16 → f32 should match to ~1/1000 relative error
    const int N = 256;
    std::vector<float>  orig(N), roundtrip(N);
    std::vector<fp16_t> buf(N);
    fill(orig.data(), N, 1.f);

    fp32_to_fp16_row(orig.data(), buf.data(), N);
    fp16_to_fp32_row(buf.data(), roundtrip.data(), N);

    TEST("fp16 round-trip (vector)", vec_approx(orig.data(), roundtrip.data(), N, 1e-2f));

    // Non-multiple-of-4 length
    const int M = 13;
    std::vector<float>  orig2(M), rt2(M);
    std::vector<fp16_t> buf2(M);
    fill(orig2.data(), M, 2.f);
    fp32_to_fp16_row(orig2.data(), buf2.data(), M);
    fp16_to_fp32_row(buf2.data(), rt2.data(), M);
    TEST("fp16 round-trip (n=13)",   vec_approx(orig2.data(), rt2.data(), M, 1e-2f));

    // Known value: 1.0f should survive
    fp16_t h1 = f32_to_f16(1.0f);
    float  r1 = f16_to_f32(h1);
    TEST("fp16 scalar 1.0f",         approx(r1, 1.0f, 1e-3f));

    // Zero
    TEST("fp16 scalar 0.0f",         approx(f16_to_f32(f32_to_f16(0.f)), 0.f));
}

// ---------------------------------------------------------------------------
// 2. Q8_0 quantise / dequantise
// ---------------------------------------------------------------------------
void test_q8_0_quant() {
    const int N = QK8_0;
    std::vector<float> orig(N), rec(N);
    fill(orig.data(), N, 3.f);
    // Scale up so range is representative
    for (int i = 0; i < N; ++i) orig[i] *= 5.f;

    block_q8_0 blk = quantize_q8_0_block(orig.data());
    dequantize_q8_0_block(blk, rec.data());

    // Q8_0 max absolute error ≤ scale/2 = amax/254.  Allow 2%.
    float amax8 = 0.f;
    for (int i = 0; i < N; ++i) amax8 = std::max(amax8, std::fabs(orig[i]));
    TEST("q8_0 round-trip",          vec_approx(orig.data(), rec.data(), N, amax8 / 127.f * 1.5f));

    // Scale stored in fp16 should be > 0
    TEST("q8_0 scale positive",      f16_to_f32(blk.d) > 0.f);
}

// ---------------------------------------------------------------------------
// 3. Q4_0 quantise / dequantise
// ---------------------------------------------------------------------------
void test_q4_0_quant() {
    const int N = QK4_0;
    std::vector<float> orig(N), rec(N);
    fill(orig.data(), N, 4.f);
    for (int i = 0; i < N; ++i) orig[i] *= 3.f;

    block_q4_0 blk = quantize_q4_0_block(orig.data());
    dequantize_q4_0_block(blk, rec.data());

    // Q4_0 max absolute error ≤ scale/2 = amax/14.  Allow 1.5×.
    float amax4 = 0.f;
    for (int i = 0; i < N; ++i) amax4 = std::max(amax4, std::fabs(orig[i]));
    TEST("q4_0 round-trip",          vec_approx(orig.data(), rec.data(), N, amax4 / 7.f * 1.5f));
    TEST("q4_0 scale positive",      f16_to_f32(blk.d) > 0.f);
}

// ---------------------------------------------------------------------------
// 4. vec_dot_q8_0_q8_0  (reference vs NEON)
// ---------------------------------------------------------------------------
void test_vec_dot_q8q8() {
    const int NB = 4;                  // 4 blocks × 32 = 128 elements
    const int N  = NB * QK8_0;
    std::vector<float> a(N), b(N);
    fill(a.data(), N, 5.f);
    fill(b.data(), N, 6.f);
    for (auto& v : a) v *= 2.f;
    for (auto& v : b) v *= 2.f;

    // Reference: plain float dot
    float ref = 0.f;
    for (int i = 0; i < N; ++i) ref += a[i] * b[i];

    // Quantise both sides
    std::vector<block_q8_0> qa(NB), qb(NB);
    for (int i = 0; i < NB; ++i) {
        qa[i] = quantize_q8_0_block(a.data() + i * QK8_0);
        qb[i] = quantize_q8_0_block(b.data() + i * QK8_0);
    }

    float got = vec_dot_q8_0_q8_0(qa.data(), qb.data(), NB);
    // Q8 introduces ~1% relative error
    TEST("vec_dot_q8q8 value",       approx(got, ref, std::fabs(ref) * 0.03f + 1e-3f));

    // Sign check: if we negate b, dot should negate too
    for (auto& v : b) v = -v;
    for (int i = 0; i < NB; ++i)
        qb[i] = quantize_q8_0_block(b.data() + i * QK8_0);
    float got_neg = vec_dot_q8_0_q8_0(qa.data(), qb.data(), NB);
    TEST("vec_dot_q8q8 sign flip",   approx(got_neg, -got, std::fabs(got) * 0.05f + 1e-3f));
}

// ---------------------------------------------------------------------------
// 5. vec_dot_q4_0_q8_0  (the decode-stage hot path)
// ---------------------------------------------------------------------------
void test_vec_dot_q4q8() {
    const int NB = 4;
    const int N  = NB * QK4_0;
    std::vector<float> a(N), b(N);
    fill(a.data(), N, 7.f);
    fill(b.data(), N, 8.f);
    for (auto& v : a) v *= 2.5f;
    for (auto& v : b) v *= 2.5f;

    float ref = 0.f;
    for (int i = 0; i < N; ++i) ref += a[i] * b[i];

    std::vector<block_q4_0> qa(NB);
    std::vector<block_q8_0> qb(NB);
    for (int i = 0; i < NB; ++i) {
        qa[i] = quantize_q4_0_block(a.data() + i * QK4_0);
        qb[i] = quantize_q8_0_block(b.data() + i * QK8_0);
    }

    float got = vec_dot_q4_0_q8_0(qa.data(), qb.data(), NB);
    // Q4_0 has coarser quantisation, allow ~15% relative tolerance
    TEST("vec_dot_q4q8 value",       approx(got, ref, std::fabs(ref) * 0.15f + 1e-2f));

    // Monotonicity: scaling activations should scale the result
    for (auto& v : b) v *= 2.f;
    for (int i = 0; i < NB; ++i)
        qb[i] = quantize_q8_0_block(b.data() + i * QK8_0);
    float got2 = vec_dot_q4_0_q8_0(qa.data(), qb.data(), NB);
    // got2 should be approximately 2× got
    TEST("vec_dot_q4q8 scale mono",  approx(got2, got * 2.f, std::fabs(got) * 0.20f + 1e-2f));
}

// ---------------------------------------------------------------------------
// 6. vec_dot_f32
// ---------------------------------------------------------------------------
void test_vec_dot_f32() {
    const int N = 128;
    std::vector<float> a(N), b(N);
    fill(a.data(), N, 9.f);
    fill(b.data(), N, 10.f);

    float ref = 0.f;
    for (int i = 0; i < N; ++i) ref += a[i] * b[i];
    float got = vec_dot_f32(a.data(), b.data(), N);
    TEST("vec_dot_f32",              approx(got, ref, 1e-4f));

    // n not div-4
    const int M = 17;
    fill(a.data(), M, 11.f); fill(b.data(), M, 12.f);
    float ref2 = 0.f;
    for (int i = 0; i < M; ++i) ref2 += a[i] * b[i];
    TEST("vec_dot_f32 n=17",         approx(vec_dot_f32(a.data(), b.data(), M), ref2, 1e-4f));
}

// ---------------------------------------------------------------------------
// 7. RMS Norm
// ---------------------------------------------------------------------------
static void ref_rms_norm(const float* x, const float* w, float* y, int n, float eps) {
    float ss = 0.f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    float scale = 1.f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * scale * (w ? w[i] : 1.f);
}

void test_rms_norm() {
    const int N = 128;
    std::vector<float> x(N), w(N), got(N), ref(N);
    fill(x.data(), N, 1.f);
    for (int i = 0; i < N; ++i) w[i] = 1.f + i * 0.01f;

    ref_rms_norm(x.data(), w.data(), ref.data(), N, 1e-5f);
    rms_norm(x.data(), w.data(), got.data(), N, 1e-5f);
    TEST("rms_norm with weight",     vec_approx(got.data(), ref.data(), N, 1e-4f));

    // No weight
    ref_rms_norm(x.data(), nullptr, ref.data(), N, 1e-5f);
    rms_norm(x.data(), nullptr, got.data(), N, 1e-5f);
    TEST("rms_norm no weight",       vec_approx(got.data(), ref.data(), N, 1e-4f));

    // n not div-4
    const int M = 11;
    std::vector<float> xm(M), wm(M), gm(M), rm(M);
    fill(xm.data(), M, 2.f);
    for (int i = 0; i < M; ++i) wm[i] = 1.f;
    ref_rms_norm(xm.data(), wm.data(), rm.data(), M, 1e-5f);
    rms_norm(xm.data(), wm.data(), gm.data(), M, 1e-5f);
    TEST("rms_norm n=11",            vec_approx(gm.data(), rm.data(), M, 1e-4f));

    // Output should have unit RMS (when weight=1)
    float ss = 0.f;
    for (int i = 0; i < N; ++i) ss += got[i] * got[i] / (float)N;
    // not exactly 1.0 because weight isn't uniform, but let's test the no-weight case
    std::vector<float> nw_out(N);
    rms_norm(x.data(), nullptr, nw_out.data(), N, 1e-5f);
    float rms_of_out = 0.f;
    for (int i = 0; i < N; ++i) rms_of_out += nw_out[i] * nw_out[i];
    rms_of_out = std::sqrt(rms_of_out / N);
    TEST("rms_norm output RMS≈1",    approx(rms_of_out, 1.f, 1e-4f));
}

// ---------------------------------------------------------------------------
// 8. Layer Norm
// ---------------------------------------------------------------------------
static void ref_layer_norm(const float* x, const float* w, const float* b, float* y, int n, float eps) {
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; ++i) { float d = x[i]-mean; var += d*d; }
    var = var/n + eps;
    float inv = 1.f/std::sqrt(var);
    for (int i = 0; i < n; ++i) {
        float v = (x[i]-mean)*inv;
        y[i] = (w ? w[i]*v : v) + (b ? b[i] : 0.f);
    }
}

void test_layer_norm() {
    const int N = 64;
    std::vector<float> x(N), w(N), b(N), got(N), ref(N);
    fill(x.data(), N, 1.f);
    for (int i = 0; i < N; ++i) { w[i] = 1.f + i*0.01f; b[i] = i*0.005f; }

    ref_layer_norm(x.data(), w.data(), b.data(), ref.data(), N, 1e-5f);
    layer_norm(x.data(), w.data(), b.data(), got.data(), N, 1e-5f);
    TEST("layer_norm with w+b",      vec_approx(got.data(), ref.data(), N, 1e-4f));

    ref_layer_norm(x.data(), nullptr, nullptr, ref.data(), N, 1e-5f);
    layer_norm(x.data(), nullptr, nullptr, got.data(), N, 1e-5f);
    TEST("layer_norm no w/b",        vec_approx(got.data(), ref.data(), N, 1e-4f));
}

// ---------------------------------------------------------------------------
// 9. Softmax
// ---------------------------------------------------------------------------
void test_softmax() {
    const int N = 32;
    std::vector<float> x(N), ref(N);
    fill(x.data(), N, 1.f);
    for (auto& v : x) v *= 3.f;

    // Reference
    float mx = *std::max_element(x.begin(), x.end());
    float s = 0.f;
    for (int i = 0; i < N; ++i) { ref[i] = std::exp(x[i] - mx); s += ref[i]; }
    for (int i = 0; i < N; ++i) ref[i] /= s;

    std::vector<float> got(x);
    softmax(got.data(), N);
    TEST("softmax values",           vec_approx(got.data(), ref.data(), N, 1e-4f));

    // Sum must equal 1
    float sum = 0.f;
    for (int i = 0; i < N; ++i) sum += got[i];
    TEST("softmax sum=1",            approx(sum, 1.f, 1e-5f));

    // All outputs must be positive
    bool all_pos = true;
    for (int i = 0; i < N; ++i) if (got[i] <= 0.f) { all_pos = false; break; }
    TEST("softmax all positive",     all_pos);

    // n=7 (not div-4)
    const int M = 7;
    std::vector<float> xm = {1.f, -1.f, 0.5f, 2.f, -0.3f, 0.1f, 1.5f};
    mx = *std::max_element(xm.begin(), xm.end()); s = 0.f;
    std::vector<float> rm(M);
    for (int i = 0; i < M; ++i) { rm[i] = std::exp(xm[i]-mx); s += rm[i]; }
    for (int i = 0; i < M; ++i) rm[i] /= s;
    softmax(xm.data(), M);
    TEST("softmax n=7",              vec_approx(xm.data(), rm.data(), M, 1e-5f));
}

// ---------------------------------------------------------------------------
// 10. SiLU
// ---------------------------------------------------------------------------
void test_silu() {
    const int N = 64;
    std::vector<float> x(N), ref(N);
    fill(x.data(), N, 1.f);
    for (int i = 0; i < N; ++i) ref[i] = x[i] / (1.f + std::exp(-x[i]));

    std::vector<float> got(x);
    silu(got.data(), N);
    TEST("silu values",              vec_approx(got.data(), ref.data(), N, 1e-3f));

    // SiLU(0) = 0
    std::vector<float> z = {0.f, 0.f, 0.f, 0.f};
    silu(z.data(), 4);
    TEST("silu(0)=0",                approx(z[0], 0.f, 1e-6f));

    // SiLU(large positive) ≈ identity
    std::vector<float> big = {10.f};
    silu(big.data(), 1);
    TEST("silu large pos ≈ x",       approx(big[0], 10.f * (1.f / (1.f + std::exp(-10.f))), 1e-4f));
}

// ---------------------------------------------------------------------------
// 11. ReLU
// ---------------------------------------------------------------------------
void test_relu() {
    const int N = 20;
    std::vector<float> x(N);
    fill(x.data(), N, 1.f);
    std::vector<float> ref(N);
    for (int i = 0; i < N; ++i) ref[i] = x[i] > 0.f ? x[i] : 0.f;
    relu(x.data(), N);
    TEST("relu values",              vec_approx(x.data(), ref.data(), N, 0.f));
    bool all_nn = true;
    for (int i = 0; i < N; ++i) if (x[i] < 0.f) { all_nn = false; break; }
    TEST("relu non-negative",        all_nn);
}

// ---------------------------------------------------------------------------
// 12. RoPE
// ---------------------------------------------------------------------------
void test_rope() {
    const int head_dim = 64;
    const int pos = 5;
    const float base = 10000.f;

    std::vector<float> x(head_dim);
    fill(x.data(), head_dim, 1.f);

    // Reference: scalar
    std::vector<float> ref(x);
    rope_f32(ref.data(), head_dim, pos, base);

    // NEON path
    std::vector<float> got(x);
    rope_f32_neon(got.data(), head_dim, pos, base);
    TEST("rope NEON vs scalar",      vec_approx(got.data(), ref.data(), head_dim, 1e-4f));

    // RoPE must preserve L2 norm (rotation is isometric)
    float norm_in  = vec_dot_f32(x.data(), x.data(), head_dim);
    float norm_out = vec_dot_f32(ref.data(), ref.data(), head_dim);
    TEST("rope preserves norm",      approx(norm_out, norm_in, 1e-3f));

    // pos=0 → cos(0)=1, sin(0)=0 → output equals input
    std::vector<float> x0(x);
    rope_f32(x0.data(), head_dim, 0, base);
    TEST("rope pos=0 identity",      vec_approx(x0.data(), x.data(), head_dim, 1e-5f));
}

// ---------------------------------------------------------------------------
// 13. Element-wise ops
// ---------------------------------------------------------------------------
void test_eltwise() {
    const int N = 17;
    std::vector<float> a(N), b(N), got(N), ref(N);
    fill(a.data(), N, 1.f); fill(b.data(), N, 2.f);

    // ADD
    for (int i = 0; i < N; ++i) ref[i] = a[i] + b[i];
    vec_add_f32(a.data(), b.data(), got.data(), N);
    TEST("vec_add_f32",              vec_approx(got.data(), ref.data(), N, 0.f));

    // MUL
    for (int i = 0; i < N; ++i) ref[i] = a[i] * b[i];
    vec_mul_f32(a.data(), b.data(), got.data(), N);
    TEST("vec_mul_f32",              vec_approx(got.data(), ref.data(), N, 0.f));

    // SCALE
    std::vector<float> s(a);
    const float sc = 3.14f;
    for (int i = 0; i < N; ++i) ref[i] = a[i] * sc;
    vec_scale_f32(s.data(), sc, N);
    TEST("vec_scale_f32",            vec_approx(s.data(), ref.data(), N, 1e-5f));
}

// ---------------------------------------------------------------------------
// 14. matmul_q4_q8  (multi-row quantized matmul)
// ---------------------------------------------------------------------------
void test_matmul_q4q8() {
    const int OUT  = 8;
    const int NB   = 2;       // 2 blocks × 32 = 64 input elements
    const int N    = NB * QK4_0;

    // Random weight matrix and input vector
    std::vector<float> W(OUT * N), x(N);
    fill(W.data(), OUT * N, 5.f);
    fill(x.data(), N, 6.f);
    for (auto& v : W) v *= 2.f;
    for (auto& v : x) v *= 2.f;

    // Float reference
    std::vector<float> ref(OUT, 0.f);
    for (int r = 0; r < OUT; ++r)
        for (int c = 0; c < N; ++c)
            ref[r] += W[r * N + c] * x[c];

    // Quantise
    std::vector<block_q4_0> Wq(OUT * NB);
    std::vector<block_q8_0> xq(NB);
    for (int r = 0; r < OUT; ++r)
        for (int i = 0; i < NB; ++i)
            Wq[r * NB + i] = quantize_q4_0_block(W.data() + r * N + i * QK4_0);
    for (int i = 0; i < NB; ++i)
        xq[i] = quantize_q8_0_block(x.data() + i * QK8_0);

    std::vector<float> got(OUT);
    matmul_q4_q8(Wq.data(), xq.data(), got.data(), OUT, NB);

    // Q4 tolerance ~15%
    bool ok = true;
    for (int r = 0; r < OUT; ++r)
        if (!approx(got[r], ref[r], std::fabs(ref[r]) * 0.18f + 0.02f)) { ok = false; break; }
    TEST("matmul_q4_q8",             ok);
}

// ---------------------------------------------------------------------------
int main() {
    std::printf("=== llama_neon_ops unit tests ===\n");
#if LLAMA_I8MM
    std::printf("Backend: ARM NEON + dotprod + i8mm\n\n");
#elif LLAMA_DOTPROD
    std::printf("Backend: ARM NEON + dotprod\n\n");
#elif LLAMA_NEON
    std::printf("Backend: ARM NEON (baseline)\n\n");
#else
    std::printf("Backend: scalar fallback (LLAMA_NO_NEON)\n\n");
#endif

    test_fp16_convert();
    test_q8_0_quant();
    test_q4_0_quant();
    test_vec_dot_f32();
    test_vec_dot_q8q8();
    test_vec_dot_q4q8();
    test_rms_norm();
    test_layer_norm();
    test_softmax();
    test_silu();
    test_relu();
    test_rope();
    test_eltwise();
    test_matmul_q4q8();

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
