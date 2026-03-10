// =============================================================================
// test_ops.cpp  —  Unit tests for arm_neon_ops.h
//
// Compile (ARM64 device / cross-compile):
//   aarch64-linux-gnu-g++ -O2 -std=c++11 -I../include -o test_ops test_ops.cpp
//
// Compile (x86 host, scalar fallback):
//   g++ -O2 -std=c++11 -DNCNN_NO_NEON -I../include -o test_ops test_ops.cpp
// =============================================================================

#include "arm_neon_ops.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <functional>

// ---------------------------------------------------------------------------
// Tiny test harness
// ---------------------------------------------------------------------------
static int g_passed = 0, g_failed = 0;

static bool approx_eq(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol + 1e-6f * std::fabs(b);
}

static bool vec_approx_eq(const float* a, const float* b, int N, float tol = 1e-4f) {
    for (int i = 0; i < N; ++i)
        if (!approx_eq(a[i], b[i], tol)) {
            std::printf("  mismatch at [%d]: got %.6f  expected %.6f\n", i, a[i], b[i]);
            return false;
        }
    return true;
}

#define TEST(name, expr) do {                              \
    bool _ok = (expr);                                     \
    std::printf("[%s] %s\n", _ok ? "PASS" : "FAIL", name);\
    if (_ok) ++g_passed; else ++g_failed;                  \
} while(0)

// ---------------------------------------------------------------------------
// Reference (plain C) implementations for golden values
// ---------------------------------------------------------------------------
static void ref_conv2d(
    const float* input, const float* weight, const float* bias, float* output,
    int IC, int IH, int IW, int OC, int KH, int KW)
{
    int OH = IH - KH + 1, OW = IW - KW + 1;
    for (int oc = 0; oc < OC; ++oc) {
        for (int oh = 0; oh < OH; ++oh) for (int ow = 0; ow < OW; ++ow) {
            float s = bias ? bias[oc] : 0.f;
            for (int ic = 0; ic < IC; ++ic)
                for (int kh = 0; kh < KH; ++kh)
                    for (int kw = 0; kw < KW; ++kw)
                        s += input[(ic*IH + oh+kh)*IW + ow+kw]
                           * weight[((oc*IC + ic)*KH + kh)*KW + kw];
            output[(oc*OH + oh)*OW + ow] = s;
        }
    }
}

static void ref_linear(
    const float* in, const float* w, const float* b, float* out,
    int N, int IC, int OC)
{
    for (int n = 0; n < N; ++n)
        for (int oc = 0; oc < OC; ++oc) {
            float s = b ? b[oc] : 0.f;
            for (int ic = 0; ic < IC; ++ic) s += in[n*IC+ic] * w[oc*IC+ic];
            out[n*OC+oc] = s;
        }
}

static void ref_batchnorm(
    const float* in, const float* mean, const float* var,
    const float* gamma, const float* beta, float* out,
    int N, int C, int H, int W, float eps)
{
    int HW = H * W;
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            float s = gamma[c] / std::sqrt(var[c] + eps);
            float b = beta[c] - s * mean[c];
            for (int i = 0; i < HW; ++i)
                out[(n*C + c)*HW + i] = s * in[(n*C + c)*HW + i] + b;
        }
}

static void ref_layernorm(
    const float* in, const float* g, const float* b, float* out,
    int N, int D, float eps)
{
    for (int n = 0; n < N; ++n) {
        float mean = 0.f, var = 0.f;
        for (int i = 0; i < D; ++i) mean += in[n*D + i];
        mean /= D;
        for (int i = 0; i < D; ++i) { float d = in[n*D+i]-mean; var += d*d; }
        var /= D;
        float inv = 1.f / std::sqrt(var + eps);
        for (int i = 0; i < D; ++i) {
            float v = (in[n*D+i] - mean) * inv;
            out[n*D+i] = g ? g[i]*v + (b ? b[i] : 0.f) : v;
        }
    }
}

static void ref_softmax(const float* in, float* out, int N, int D) {
    for (int n = 0; n < N; ++n) {
        float mx = in[n*D];
        for (int i = 1; i < D; ++i) mx = std::max(mx, in[n*D+i]);
        float s = 0.f;
        for (int i = 0; i < D; ++i) { out[n*D+i] = std::exp(in[n*D+i]-mx); s += out[n*D+i]; }
        for (int i = 0; i < D; ++i) out[n*D+i] /= s;
    }
}

// Simple deterministic fill
static void fill(float* p, int N, float seed = 1.f) {
    for (int i = 0; i < N; ++i) p[i] = (float)((i * 1234567 + (int)(seed*100)) % 97 - 48) * 0.05f;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_conv2d() {
    // IC=2, IH=6, IW=6, OC=3, KH=3, KW=3  => OH=OW=4
    const int IC=2, IH=6, IW=6, OC=3, KH=3, KW=3;
    const int OH=IH-KH+1, OW=IW-KW+1;
    std::vector<float> inp(IC*IH*IW), wt(OC*IC*KH*KW), bias(OC), out(OC*OH*OW), ref(OC*OH*OW);
    fill(inp.data(), inp.size(), 1.f);
    fill(wt.data(),  wt.size(),  2.f);
    fill(bias.data(),bias.size(),3.f);

    ref_conv2d(inp.data(), wt.data(), bias.data(), ref.data(), IC,IH,IW,OC,KH,KW);
    ops::conv2d_nchw(inp.data(), wt.data(), bias.data(), out.data(), IC,IH,IW,OC,KH,KW);
    TEST("conv2d with bias",     vec_approx_eq(out.data(), ref.data(), ref.size()));

    // no bias
    ref_conv2d(inp.data(), wt.data(), nullptr, ref.data(), IC,IH,IW,OC,KH,KW);
    ops::conv2d_nchw(inp.data(), wt.data(), nullptr, out.data(), IC,IH,IW,OC,KH,KW);
    TEST("conv2d no bias",       vec_approx_eq(out.data(), ref.data(), ref.size()));

    // 1x1 conv
    const int OC2=4, KH2=1, KW2=1;
    const int OH2=IH, OW2=IW;
    std::vector<float> wt2(OC2*IC*KH2*KW2), out2(OC2*OH2*OW2), ref2(OC2*OH2*OW2);
    fill(wt2.data(), wt2.size(), 4.f);
    ref_conv2d(inp.data(), wt2.data(), nullptr, ref2.data(), IC,IH,IW,OC2,KH2,KW2);
    ops::conv2d_nchw(inp.data(), wt2.data(), nullptr, out2.data(), IC,IH,IW,OC2,KH2,KW2);
    TEST("conv2d 1x1",           vec_approx_eq(out2.data(), ref2.data(), ref2.size()));
}

void test_depthwise_conv() {
    const int C=4, IH=8, IW=8, KH=3, KW=3;
    const int OH=IH-KH+1, OW=IW-KW+1;
    std::vector<float> inp(C*IH*IW), wt(C*KH*KW), bias(C), out(C*OH*OW), ref(C*OH*OW);
    fill(inp.data(), inp.size(), 1.f);
    fill(wt.data(),  wt.size(),  2.f);
    fill(bias.data(),bias.size(),3.f);

    // reference using conv2d with groups
    for (int c = 0; c < C; ++c) {
        const float* ic = inp.data() + c*IH*IW;
        const float* wc = wt.data()  + c*KH*KW;
        float*       oc = ref.data() + c*OH*OW;
        ref_conv2d(ic, wc, bias.data()+c, oc, 1,IH,IW,1,KH,KW);
    }
    ops::depthwise_conv2d_nchw(inp.data(), wt.data(), bias.data(), out.data(), C,IH,IW,KH,KW);
    TEST("depthwise_conv2d",     vec_approx_eq(out.data(), ref.data(), ref.size()));
}

void test_linear() {
    const int N=3, IC=16, OC=8;
    std::vector<float> inp(N*IC), wt(OC*IC), bias(OC), out(N*OC), ref(N*OC);
    fill(inp.data(), inp.size(), 1.f);
    fill(wt.data(),  wt.size(),  2.f);
    fill(bias.data(),bias.size(),3.f);

    ref_linear(inp.data(), wt.data(), bias.data(), ref.data(), N, IC, OC);
    ops::linear(inp.data(), wt.data(), bias.data(), out.data(), N, IC, OC);
    TEST("linear with bias",     vec_approx_eq(out.data(), ref.data(), ref.size()));

    ref_linear(inp.data(), wt.data(), nullptr, ref.data(), N, IC, OC);
    ops::linear(inp.data(), wt.data(), nullptr, out.data(), N, IC, OC);
    TEST("linear no bias",       vec_approx_eq(out.data(), ref.data(), ref.size()));

    // Non-multiple-of-4 IC
    const int IC2=13;
    std::vector<float> wt2(OC*IC2), inp2(N*IC2), out2(N*OC), ref2(N*OC);
    fill(inp2.data(), inp2.size(), 5.f);
    fill(wt2.data(),  wt2.size(),  6.f);
    ref_linear(inp2.data(), wt2.data(), nullptr, ref2.data(), N, IC2, OC);
    ops::linear(inp2.data(), wt2.data(), nullptr, out2.data(), N, IC2, OC);
    TEST("linear IC not div-4",  vec_approx_eq(out2.data(), ref2.data(), ref2.size()));
}

void test_batchnorm() {
    const int N=2, C=4, H=5, W=5;
    std::vector<float> inp(N*C*H*W), mean(C), var(C), gamma(C), beta(C);
    std::vector<float> out(N*C*H*W), ref(N*C*H*W);
    fill(inp.data(),   inp.size(),   1.f);
    for (int c = 0; c < C; ++c) {
        mean[c]  = (float)c * 0.1f;
        var[c]   = (float)(c + 1) * 0.5f;
        gamma[c] = 1.f + c * 0.2f;
        beta[c]  = (float)c * 0.05f;
    }
    ref_batchnorm(inp.data(), mean.data(), var.data(), gamma.data(), beta.data(),
                  ref.data(), N, C, H, W, 1e-5f);
    ops::batchnorm_nchw(inp.data(), mean.data(), var.data(), gamma.data(), beta.data(),
                        out.data(), N, C, H, W, 1e-5f);
    TEST("batchnorm",            vec_approx_eq(out.data(), ref.data(), ref.size()));
}

void test_layernorm() {
    const int N=4, D=16;
    std::vector<float> inp(N*D), gamma(D), beta(D), out(N*D), ref(N*D);
    fill(inp.data(),   inp.size(),   1.f);
    fill(gamma.data(), gamma.size(), 2.f);
    fill(beta.data(),  beta.size(),  3.f);
    // normalise gamma to be close to 1 so differences aren't huge
    for (int i = 0; i < D; ++i) gamma[i] = 1.f + gamma[i]*0.1f;

    ref_layernorm(inp.data(), gamma.data(), beta.data(), ref.data(), N, D, 1e-5f);
    ops::layernorm(inp.data(), gamma.data(), beta.data(), out.data(), N, D, 1e-5f);
    TEST("layernorm with gamma/beta", vec_approx_eq(out.data(), ref.data(), ref.size()));

    // No gamma/beta
    ref_layernorm(inp.data(), nullptr, nullptr, ref.data(), N, D, 1e-5f);
    ops::layernorm(inp.data(), nullptr, nullptr, out.data(), N, D, 1e-5f);
    TEST("layernorm no affine",       vec_approx_eq(out.data(), ref.data(), ref.size()));

    // D not a multiple of 4
    const int D2 = 11;
    std::vector<float> inp2(N*D2), g2(D2), b2(D2), out2(N*D2), ref2(N*D2);
    fill(inp2.data(), inp2.size(), 7.f);
    for (int i = 0; i < D2; ++i) { g2[i] = 1.f + i*0.05f; b2[i] = i*0.02f; }
    ref_layernorm(inp2.data(), g2.data(), b2.data(), ref2.data(), N, D2, 1e-5f);
    ops::layernorm(inp2.data(), g2.data(), b2.data(), out2.data(), N, D2, 1e-5f);
    TEST("layernorm D not div-4",     vec_approx_eq(out2.data(), ref2.data(), ref2.size()));
}

void test_softmax() {
    const int N=3, D=8;
    std::vector<float> inp(N*D), out(N*D), ref(N*D);
    fill(inp.data(), inp.size(), 1.f);
    ref_softmax(inp.data(), ref.data(), N, D);
    ops::softmax(inp.data(), out.data(), N, D);
    TEST("softmax",              vec_approx_eq(out.data(), ref.data(), ref.size()));

    // Check probabilities sum to 1
    bool sum_ok = true;
    for (int n = 0; n < N; ++n) {
        float s = 0.f;
        for (int i = 0; i < D; ++i) s += out[n*D+i];
        if (std::fabs(s - 1.f) > 1e-5f) { sum_ok = false; break; }
    }
    TEST("softmax sums to 1",    sum_ok);
}

void test_relu() {
    const int N = 20;
    std::vector<float> x(N), ref(N);
    fill(x.data(), N, 1.f);
    for (int i = 0; i < N; ++i) ref[i] = x[i] > 0.f ? x[i] : 0.f;
    ops::relu(x.data(), N);
    TEST("relu",                 vec_approx_eq(x.data(), ref.data(), N));
    bool nonneg = true;
    for (int i = 0; i < N; ++i) if (x[i] < 0.f) { nonneg = false; break; }
    TEST("relu non-negative",    nonneg);
}

void test_eltwise() {
    const int N = 13;
    std::vector<float> a(N), b(N), out(N), ref_add(N), ref_mul(N);
    fill(a.data(), N, 1.f);
    fill(b.data(), N, 2.f);
    for (int i = 0; i < N; ++i) { ref_add[i] = a[i]+b[i]; ref_mul[i] = a[i]*b[i]; }
    ops::eltwise_add(a.data(), b.data(), out.data(), N);
    TEST("eltwise_add",          vec_approx_eq(out.data(), ref_add.data(), N));
    ops::eltwise_mul(a.data(), b.data(), out.data(), N);
    TEST("eltwise_mul",          vec_approx_eq(out.data(), ref_mul.data(), N));
}

void test_gru_cell() {
    // Small GRU: input_size=4, hidden_size=4
    const int I=4, H=4;
    std::vector<float> x(I), h(H, 0.f), h_ref(H, 0.f);
    std::vector<float> Wi(3*H*I), Wh(3*H*H), bi(3*H), bh(3*H);
    std::vector<float> ws(6*H);  // workspace

    fill(x.data(),  I,    1.f);
    fill(Wi.data(), 3*H*I, 2.f);
    fill(Wh.data(), 3*H*H, 3.f);
    fill(bi.data(), 3*H,   4.f);
    fill(bh.data(), 3*H,   5.f);

    // Reference: compute manually
    {
        std::vector<float> gates_i(3*H), gates_h(3*H), h0(H, 0.f);
        ref_linear(x.data(),  Wi.data(), bi.data(), gates_i.data(), 1, I, 3*H);
        ref_linear(h0.data(), Wh.data(), bh.data(), gates_h.data(), 1, H, 3*H);
        // r, z gates (sigmoid)
        std::vector<float> r(H), z(H);
        for (int i=0;i<H;++i) {
            r[i] = 1.f/(1.f+std::exp(-(gates_i[i]     + gates_h[i])));
            z[i] = 1.f/(1.f+std::exp(-(gates_i[H+i]   + gates_h[H+i])));
        }
        // n gate (tanh)
        for (int i=0;i<H;++i)
            h_ref[i] = (1.f-z[i]) * std::tanh(gates_i[2*H+i] + r[i]*gates_h[2*H+i])
                       + z[i] * h0[i];
    }

    ops::gru_cell(x.data(), h.data(), Wi.data(), Wh.data(),
                  bi.data(), bh.data(), ws.data(), I, H);
    TEST("gru_cell",             vec_approx_eq(h.data(), h_ref.data(), H, 1e-4f));
}

void test_attention() {
    // Single-head attention: Tq=3, Tk=4, D=8, Dv=8
    const int Tq=3, Tk=4, D=8, Dv=8;
    std::vector<float> Q(Tq*D), K(Tk*D), V(Tk*Dv);
    std::vector<float> out(Tq*Dv), ref(Tq*Dv), attn(Tq*Tk), attn_ref(Tq*Tk);

    fill(Q.data(), Q.size(), 1.f);
    fill(K.data(), K.size(), 2.f);
    fill(V.data(), V.size(), 3.f);

    // Reference
    float scale = 1.f / std::sqrt((float)D);
    for (int q=0;q<Tq;++q) for (int k=0;k<Tk;++k) {
        float dot=0.f;
        for (int d=0;d<D;++d) dot += Q[q*D+d]*K[k*D+d];
        attn_ref[q*Tk+k] = dot * scale;
    }
    ref_softmax(attn_ref.data(), attn_ref.data(), Tq, Tk);
    for (int q=0;q<Tq;++q) for (int dv=0;dv<Dv;++dv) {
        float s=0.f;
        for (int k=0;k<Tk;++k) s += attn_ref[q*Tk+k]*V[k*Dv+dv];
        ref[q*Dv+dv] = s;
    }

    ops::scaled_dot_product_attention(Q.data(), K.data(), V.data(),
                                      out.data(), attn.data(), Tq, Tk, D, Dv);
    TEST("scaled_dot_product_attention", vec_approx_eq(out.data(), ref.data(), Tq*Dv));
}

// ---------------------------------------------------------------------------
int main() {
    std::printf("=== arm_neon_ops unit tests ===\n");
#if HAS_NEON
    std::printf("Backend: ARM NEON intrinsics\n\n");
#else
    std::printf("Backend: scalar fallback (NCNN_NO_NEON or no __ARM_NEON)\n\n");
#endif

    test_conv2d();
    test_depthwise_conv();
    test_linear();
    test_batchnorm();
    test_layernorm();
    test_softmax();
    test_relu();
    test_eltwise();
    test_gru_cell();
    test_attention();

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
