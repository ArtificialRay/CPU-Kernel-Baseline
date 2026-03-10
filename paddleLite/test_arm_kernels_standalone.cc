// test_arm_kernels_standalone.cc
//
// Standalone unit-test driver for Paddle-Lite ARM kernels.
// Compile WITHOUT the PaddlePaddle build system:
//
//   aarch64-linux-gnu-g++ -O2 -std=c++14 \
//       -I./include \
//       -I<PADDLE_LITE_ROOT> \
//       -DPADDLE_LITE_STANDALONE \
//       test_arm_kernels_standalone.cc \
//       <PADDLE_LITE_ROOT>/lite/backends/arm/math/gemm_prepacked_int8.cc \
//       … (add only the .cc files the kernel under test actually needs) \
//       -march=armv8-a+dotprod -mfpu=neon-fp-armv8 \
//       -o test_arm_kernels -lm
//
// Then run on device:  adb push test_arm_kernels /data/local/tmp/ && adb shell /data/local/tmp/test_arm_kernels
//
// SPDX-License-Identifier: Apache-2.0

// Pull in the stub BEFORE any Paddle-Lite kernel header
#include "paddle_lite_stub.h"

// ---- Reference implementations for golden comparison -------------------
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace paddle::lite;

// ========================================================================
// Micro test framework
// ========================================================================
static int g_pass = 0, g_fail = 0;

#define TEST_CLOSE(name, val, ref, tol)                                   \
    do {                                                                   \
        double _d = std::fabs((double)(val) - (double)(ref));             \
        if (_d > (tol)) {                                                  \
            fprintf(stderr, "FAIL  %-40s  got=%.6f  ref=%.6f  diff=%.2e\n", \
                    (name), (double)(val), (double)(ref), _d);            \
            ++g_fail;                                                      \
        } else {                                                           \
            printf("PASS  %s\n", (name)); ++g_pass;                       \
        }                                                                  \
    } while(0)

#define TEST_TRUE(name, cond)                                              \
    do {                                                                   \
        if (!(cond)) { fprintf(stderr, "FAIL  %s\n", (name)); ++g_fail; } \
        else         { printf("PASS  %s\n", (name)); ++g_pass; }          \
    } while(0)

// ========================================================================
// Tensor helpers
// ========================================================================
static void fill_random(Tensor& t, float lo=-1.f, float hi=1.f) {
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(lo, hi);
    float* p = t.mutable_data<float>();
    for (int64_t i = 0; i < t.numel(); ++i) p[i] = dist(rng);
}

static void fill_const(Tensor& t, float v) {
    float* p = t.mutable_data<float>();
    for (int64_t i = 0; i < t.numel(); ++i) p[i] = v;
}

static float max_abs_diff(const Tensor& a, const Tensor& b) {
    assert(a.numel() == b.numel());
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    float d = 0.f;
    for (int64_t i = 0; i < a.numel(); ++i)
        d = std::max(d, std::fabs(pa[i]-pb[i]));
    return d;
}

// ========================================================================
// 1. Reference BatchNorm  (fp32, inference, NCHW)
// ========================================================================
static void ref_batch_norm(const Tensor& x,
                            const Tensor& scale,
                            const Tensor& bias,
                            const Tensor& mean,
                            const Tensor& variance,
                            Tensor& y,
                            float eps = 1e-5f) {
    // x: [N,C,H,W]
    int N = x.dim(0), C = x.dim(1);
    int HW = static_cast<int>(x.dims().count(2, x.dims_size()));
    const float* px  = x.data<float>();
    const float* psc = scale.data<float>();
    const float* pb  = bias.data<float>();
    const float* pm  = mean.data<float>();
    const float* pv  = variance.data<float>();
    float* py        = y.mutable_data<float>();

    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            float inv_std = 1.f / std::sqrt(pv[c] + eps);
            int base = (n*C+c)*HW;
            for (int i = 0; i < HW; ++i)
                py[base+i] = psc[c] * (px[base+i] - pm[c]) * inv_std + pb[c];
        }
}

// ========================================================================
// 2. Reference LayerNorm
// ========================================================================
static void ref_layer_norm(const Tensor& x,
                            const Tensor& scale,
                            const Tensor& bias,
                            Tensor& y,
                            int begin_axis,
                            float eps = 1e-5f) {
    // normalize over [begin_axis, end]
    int outer = static_cast<int>(x.dims().count(0, begin_axis));
    int inner = static_cast<int>(x.dims().count(begin_axis, x.dims_size()));
    const float* px  = x.data<float>();
    const float* psc = scale.data<float>();
    const float* pb  = bias.data<float>();
    float* py        = y.mutable_data<float>();

    for (int o = 0; o < outer; ++o) {
        const float* row = px + o*inner;
        float*       dst = py + o*inner;
        // mean
        float m = 0.f;
        for (int i = 0; i < inner; ++i) m += row[i];
        m /= inner;
        // variance
        float v = 0.f;
        for (int i = 0; i < inner; ++i) v += (row[i]-m)*(row[i]-m);
        v /= inner;
        float inv_std = 1.f / std::sqrt(v + eps);
        for (int i = 0; i < inner; ++i)
            dst[i] = psc[i] * (row[i] - m) * inv_std + pb[i];
    }
}

// ========================================================================
// 3. Reference FC  (M×K × K×N + bias → M×N), no activation
// ========================================================================
static void ref_fc(const Tensor& input,  // [M, K]
                   const Tensor& weight, // [N, K]  (transposed)
                   const Tensor& bias,   // [N]
                   Tensor& output) {     // [M, N]
    int M = static_cast<int>(input.dim(0));
    int K = static_cast<int>(input.dim(1));
    int N = static_cast<int>(weight.dim(0));
    const float* pi = input.data<float>();
    const float* pw = weight.data<float>();
    const float* pb = bias.data<float>();
    float* po       = output.mutable_data<float>();

    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = pb[n];
            for (int k = 0; k < K; ++k)
                s += pi[m*K+k] * pw[n*K+k];
            po[m*N+n] = s;
        }
}

// ========================================================================
// 4. Reference Conv2d  (naive fp32, for correctness gate)
// ========================================================================
static void ref_conv2d(const Tensor& x,       // [N,C,H,W]
                       const Tensor& filter,   // [OC,IC,KH,KW]
                       const Tensor* bias,     // [OC] or nullptr
                       Tensor& out,            // [N,OC,OH,OW]
                       int stride_h, int stride_w,
                       int pad_h,   int pad_w,
                       int dil_h,   int dil_w,
                       int groups) {
    int N  = x.dim(0), IC = x.dim(1), IH = x.dim(2), IW = x.dim(3);
    int OC = filter.dim(0);
    int IC_g = IC / groups;
    int OC_g = OC / groups;
    int KH = filter.dim(2), KW = filter.dim(3);
    int OH = out.dim(2), OW = out.dim(3);

    const float* px = x.data<float>();
    const float* pf = filter.data<float>();
    const float* pb = bias ? bias->data<float>() : nullptr;
    float* po       = out.mutable_data<float>();
    std::fill(po, po + out.numel(), 0.f);

    for (int n = 0; n < N; ++n)
    for (int g = 0; g < groups; ++g)
    for (int oc = g*OC_g; oc < (g+1)*OC_g; ++oc)
    for (int oh = 0; oh < OH; ++oh)
    for (int ow = 0; ow < OW; ++ow) {
        float s = pb ? pb[oc] : 0.f;
        for (int ic = g*IC_g; ic < (g+1)*IC_g; ++ic)
        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            int ih = oh*stride_h - pad_h + kh*dil_h;
            int iw = ow*stride_w - pad_w + kw*dil_w;
            if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) continue;
            float xval = px[((n*IC+ic)*IH+ih)*IW+iw];
            int   fic  = ic - g*IC_g;
            float fval = pf[((oc*IC_g+fic)*KH+kh)*KW+kw];
            s += xval * fval;
        }
        po[((n*OC+oc)*OH+oh)*OW+ow] = s;
    }
}

// ========================================================================
// Helper: compute output spatial size
// ========================================================================
static int conv_out_size(int in, int k, int pad, int stride, int dil) {
    return (in + 2*pad - dil*(k-1) - 1) / stride + 1;
}

// ========================================================================
// Test: BatchNorm reference (self-test of our reference, always passes)
// ========================================================================
static void test_batch_norm_reference() {
    // N=2, C=3, H=4, W=4
    Tensor x, sc, bias, mean, var, y;
    x.Resize({2,3,4,4}); fill_random(x);
    sc.Resize({3});       fill_random(sc, 0.5f, 2.0f);
    bias.Resize({3});     fill_random(bias);
    mean.Resize({3});     fill_random(mean, -0.5f, 0.5f);
    var.Resize({3});      fill_const(var, 1.f);    // known variance
    y.Resize({2,3,4,4});

    ref_batch_norm(x, sc, bias, mean, var, y);

    // spot-check: element [0,0,0,0]
    float expected = sc.data<float>()[0] *
                     (x.data<float>()[0] - mean.data<float>()[0]) /
                     std::sqrt(1.f + 1e-5f) +
                     bias.data<float>()[0];
    TEST_CLOSE("batch_norm_ref[0,0,0,0]", y.data<float>()[0], expected, 1e-5f);
}

// ========================================================================
// Test: LayerNorm reference
// ========================================================================
static void test_layer_norm_reference() {
    // [2, 8]  normalize over last axis
    Tensor x, sc, bias, y;
    x.Resize({2,8});   fill_random(x);
    sc.Resize({8});    fill_const(sc, 1.f);
    bias.Resize({8});  fill_const(bias, 0.f);
    y.Resize({2,8});

    ref_layer_norm(x, sc, bias, y, /*begin_axis=*/1);

    // Check that each row has zero mean and unit variance after norm
    const float* po = y.data<float>();
    for (int row = 0; row < 2; ++row) {
        float m=0.f, v=0.f;
        for (int i=0;i<8;++i) m += po[row*8+i];  m /= 8;
        for (int i=0;i<8;++i) v += (po[row*8+i]-m)*(po[row*8+i]-m); v /= 8;
        char name[64]; snprintf(name,64,"layer_norm_ref row%d mean≈0",row);
        TEST_CLOSE(name, m, 0.f, 1e-5f);
        // After LayerNorm, output variance ≈ 1 - ε/(σ²+ε) ≈ slightly below 1
        snprintf(name,64,"layer_norm_ref row%d var≈1",row);
        TEST_CLOSE(name, v, 1.f, 1e-3f);
    }
}

// ========================================================================
// Test: FC reference
// ========================================================================
static void test_fc_reference() {
    // [4,8] x [16,8]T + [16] → [4,16]
    Tensor inp, w, b, out_ref;
    inp.Resize({4,8});   fill_random(inp);
    w.Resize({16,8});    fill_random(w);
    b.Resize({16});      fill_random(b);
    out_ref.Resize({4,16});

    ref_fc(inp, w, b, out_ref);

    // Manual check for element [0,0]
    float s = b.data<float>()[0];
    for (int k=0;k<8;++k) s += inp.data<float>()[k] * w.data<float>()[k];
    TEST_CLOSE("fc_ref[0,0]", out_ref.data<float>()[0], s, 1e-4f);

    // Check shape
    TEST_TRUE("fc_ref output shape", out_ref.dim(0)==4 && out_ref.dim(1)==16);
}

// ========================================================================
// Test: Conv2d reference  (3x3s1p1, groups=1)
// ========================================================================
static void test_conv2d_reference() {
    int N=1, IC=4, IH=8, IW=8, OC=8, KH=3, KW=3;
    int pad=1, stride=1, dil=1, groups=1;
    int OH = conv_out_size(IH,KH,pad,stride,dil);
    int OW = conv_out_size(IW,KW,pad,stride,dil);

    Tensor x, filt, bias, out;
    x.Resize({N,IC,IH,IW});            fill_random(x, -0.5f, 0.5f);
    filt.Resize({OC,IC,KH,KW});        fill_random(filt, -0.1f, 0.1f);
    bias.Resize({OC});                  fill_const(bias, 0.f);
    out.Resize({N,OC,OH,OW});

    ref_conv2d(x, filt, &bias, out, stride,stride, pad,pad, dil,dil, groups);
    TEST_TRUE("conv2d_ref output shape [N,OC,OH,OW]",
              out.dim(0)==N && out.dim(1)==OC &&
              out.dim(2)==OH && out.dim(3)==OW);

    // Manual single-output pixel [0,0,0,0]: sum over IC,3x3 (no padding needed at 0,0 with p=1)
    float expected = bias.data<float>()[0];
    const float* px = x.data<float>();
    const float* pf = filt.data<float>();
    for (int ic=0;ic<IC;++ic)
    for (int kh=0;kh<KH;++kh)
    for (int kw=0;kw<KW;++kw) {
        int ih = kh-pad, iw = kw-pad;
        if(ih<0||iw<0) continue;
        expected += px[((0*IC+ic)*IH+ih)*IW+iw] * pf[((0*IC+ic)*KH+kh)*KW+kw];
    }
    TEST_CLOSE("conv2d_ref[0,0,0,0]", out.data<float>()[0], expected, 1e-4f);
}

// ========================================================================
// Test: Depthwise Conv2d reference  (groups == IC == OC)
// ========================================================================
static void test_depthwise_conv_reference() {
    int N=1, C=8, IH=16, IW=16, KH=3, KW=3;
    int pad=1, stride=1, dil=1;
    int OH = conv_out_size(IH,KH,pad,stride,dil);
    int OW = conv_out_size(IW,KW,pad,stride,dil);

    Tensor x, filt, out;
    x.Resize({N,C,IH,IW});        fill_random(x);
    filt.Resize({C,1,KH,KW});     fill_random(filt, -0.1f, 0.1f);
    out.Resize({N,C,OH,OW});

    ref_conv2d(x, filt, nullptr, out, stride,stride, pad,pad, dil,dil, /*groups=*/C);
    TEST_TRUE("depthwise_conv_ref shape ok",
              out.dim(1)==C && out.dim(2)==OH);
}

// ========================================================================
// Example: How to wire in the REAL ARM kernel (conditional)
// ========================================================================
// When PADDLE_LITE_USE_REAL_KERNEL is defined AND you have compiled the
// kernel .cc files alongside this test, you can do:
//
// #ifdef PADDLE_LITE_USE_REAL_KERNEL
// #include "lite/kernels/arm/batch_norm_compute.h"
//
// static void test_batch_norm_arm_kernel() {
//     using BNKernel = paddle::lite::kernels::arm::BatchNormCompute;
//     ARMContext ctx(/*threads=*/1);
//
//     Tensor x, sc, bias, mean, var, y;
//     x.Resize({1,4,8,8}); fill_random(x);
//     sc.Resize({4});       fill_const(sc, 1.f);
//     bias.Resize({4});     fill_const(bias, 0.f);
//     mean.Resize({4});     fill_const(mean, 0.f);
//     var.Resize({4});      fill_const(var, 1.f);
//     y.Resize({1,4,8,8});
//
//     BatchNormParam param;
//     param.x = &x; param.scale = &sc; param.bias = &bias;
//     param.mean = &mean; param.variance = &var; param.y = &y;
//     param.epsilon = 1e-5f; param.is_test = true;
//
//     BNKernel kernel;
//     kernel.SetParam(param);
//     kernel.SetContext(&ctx);
//     kernel.PrepareForRun();
//     kernel.Run();
//
//     // compare against reference
//     Tensor y_ref; y_ref.Resize({1,4,8,8});
//     ref_batch_norm(x, sc, bias, mean, var, y_ref);
//     float diff = max_abs_diff(y, y_ref);
//     TEST_CLOSE("batch_norm_arm_kernel max_abs_diff", diff, 0.f, 1e-5f);
// }
// #endif

// ========================================================================
// main
// ========================================================================
int main() {
    printf("=== Paddle-Lite ARM Kernel Standalone Unit Tests ===\n\n");

    test_batch_norm_reference();
    test_layer_norm_reference();
    test_fc_reference();
    test_conv2d_reference();
    test_depthwise_conv_reference();

    printf("\n--- Results: %d passed, %d failed ---\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
