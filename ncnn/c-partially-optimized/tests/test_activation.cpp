// test_activation.cpp
// Tests for activation/: absval, relu, leakyrelu, elu, gelu, sigmoid, tanh,
//   swish, hardswish, hardsigmoid, selu, clip, mish, softplus, prelu, bnll,
//   celu, erf, exp, log, power, shrink, threshold
//
// Section 1: reference-only tests (no ncnn link needed)
// Section 2: real ncnn kernel tests (linked via activation_impl + ncnn_stub)

#include "test_utils.h"

// ── Real ncnn kernel headers ──────────────────────────────────────────────────
#include "ncnn_helpers.h"
#include "../activation/relu.h"
#include "../activation/sigmoid.h"
#include "../activation/gelu.h"
#include "../activation/clip.h"
#include "../activation/tanh.h"
#include "../activation/absval.h"
#include "../activation/swish.h"

// ─── Reference implementations ───────────────────────────────────────────────

static float ref_absval(float x)      { return fabsf(x); }
static float ref_relu(float x)        { return x > 0.f ? x : 0.f; }
static float ref_leakyrelu(float x, float slope) { return x > 0.f ? x : slope * x; }
static float ref_elu(float x, float alpha) { return x >= 0.f ? x : alpha * (expf(x) - 1.f); }

// fast_gelu = 0 (exact)
static float ref_gelu(float x) {
    return 0.5f * x * (1.f + erff(x * 0.70710678f));
}
// fast_gelu = 1 (tanh approximation)
static float ref_fast_gelu(float x) {
    return 0.5f * x * (1.f + tanhf(0.79788452f * (x + 0.044715f * x * x * x)));
}

static float ref_sigmoid(float x)     { return 1.f / (1.f + expf(-x)); }
static float ref_tanh(float x)        { return tanhf(x); }
static float ref_swish(float x)       { return x * ref_sigmoid(x); }

static float ref_hardswish(float x) {
    if (x <= -3.f) return 0.f;
    if (x >= 3.f)  return x;
    return x * (x + 3.f) / 6.f;
}
static float ref_hardsigmoid(float x, float alpha = 1.f/6.f, float beta = 0.5f) {
    float v = alpha * x + beta;
    if (v < 0.f) return 0.f;
    if (v > 1.f) return 1.f;
    return v;
}

static float ref_selu(float x) {
    const float alpha  = 1.6732632423543772f;
    const float lambda = 1.0507009873554805f;
    return lambda * (x > 0.f ? x : alpha * (expf(x) - 1.f));
}
static float ref_celu(float x, float alpha = 1.f) {
    return (x > 0.f ? x : 0.f) + std::min(0.f, alpha * (expf(x / alpha) - 1.f));
}

static float ref_clip(float x, float lo, float hi) {
    return std::max(lo, std::min(hi, x));
}

static float ref_mish(float x) {
    return x * tanhf(logf(1.f + expf(x)));
}
static float ref_softplus(float x) { return logf(1.f + expf(x)); }

// PReLU: per-channel slope; here tested with scalar for simplicity
static float ref_prelu(float x, float slope) { return x > 0.f ? x : slope * x; }

// BNLL: log(1 + exp(x)), clamped for large positives
static float ref_bnll(float x) { return x > 0.f ? x + logf(1.f + expf(-x)) : logf(1.f + expf(x)); }

// Erf activation: output = erf(x)
static float ref_erf(float x)   { return erff(x); }
static float ref_exp(float x, float base, float scale, float shift) {
    return (base == -1.f) ? expf(shift + scale * x) : powf(base, shift + scale * x);
}
static float ref_log(float x, float base, float scale, float shift) {
    float val = shift + scale * x;
    return (base == -1.f) ? logf(val) : logf(val) / logf(base);
}
static float ref_power(float x, float power_val, float scale, float shift) {
    return powf(shift + scale * x, power_val);
}
static float ref_shrink(float x, float bias, float lambd) {
    if (x > lambd)  return x - bias;
    if (x < -lambd) return x + bias;
    return 0.f;
}
static float ref_threshold(float x, float thresh) { return x > thresh ? 1.f : 0.f; }

// ─── Test cases ──────────────────────────────────────────────────────────────

static const float kInputs[] = { -3.f, -1.f, -0.5f, 0.f, 0.5f, 1.f, 3.f };
static const int N = 7;
static const float kTol = 1e-5f;

void test_absval() {
    float expected[] = { 3.f, 1.f, 0.5f, 0.f, 0.5f, 1.f, 3.f };
    for (int i = 0; i < N; ++i) ASSERT_NEAR(ref_absval(kInputs[i]), expected[i], kTol);
}

void test_relu() {
    float expected[] = { 0.f, 0.f, 0.f, 0.f, 0.5f, 1.f, 3.f };
    for (int i = 0; i < N; ++i) ASSERT_NEAR(ref_relu(kInputs[i]), expected[i], kTol);
}

void test_leakyrelu() {
    float slope = 0.1f;
    float expected[] = { -0.3f, -0.1f, -0.05f, 0.f, 0.5f, 1.f, 3.f };
    for (int i = 0; i < N; ++i) ASSERT_NEAR(ref_leakyrelu(kInputs[i], slope), expected[i], kTol);
}

void test_elu() {
    float alpha = 1.f;
    // positive unchanged; negative: alpha*(e^x - 1)
    for (int i = 0; i < N; ++i) {
        float x = kInputs[i];
        float ref = x >= 0.f ? x : alpha * (expf(x) - 1.f);
        ASSERT_NEAR(ref_elu(x, alpha), ref, kTol);
    }
    ASSERT_NEAR(ref_elu(0.f, 1.f), 0.f, kTol);
    ASSERT_NEAR(ref_elu(2.f, 1.f), 2.f, kTol);
    ASSERT_TRUE(ref_elu(-1.f, 1.f) < 0.f);   // negative output for negative input
}

void test_gelu_exact() {
    // gelu(0) = 0
    ASSERT_NEAR(ref_gelu(0.f), 0.f, kTol);
    // gelu is monotonically increasing for large x: gelu(3) ≈ 3
    ASSERT_NEAR(ref_gelu(3.f), 3.f, 0.01f);
    // gelu(-3) ≈ 0 (near-zero for very negative)
    ASSERT_NEAR(ref_gelu(-3.f), 0.f, 0.01f);
}

void test_gelu_fast() {
    // fast gelu and exact gelu should agree to within ~0.01
    float xs[] = { -2.f, -1.f, 0.f, 1.f, 2.f };
    for (float x : xs) ASSERT_NEAR(ref_fast_gelu(x), ref_gelu(x), 0.02f);
}

void test_sigmoid() {
    ASSERT_NEAR(ref_sigmoid(0.f), 0.5f, kTol);
    ASSERT_NEAR(ref_sigmoid(1.f), 0.7310586f, kTol);
    ASSERT_NEAR(ref_sigmoid(-1.f), 0.2689414f, kTol);
    // range (0, 1) for all inputs
    for (int i = 0; i < N; ++i) {
        float v = ref_sigmoid(kInputs[i]);
        ASSERT_TRUE(v > 0.f && v < 1.f);
    }
}

void test_tanh() {
    ASSERT_NEAR(ref_tanh(0.f), 0.f, kTol);
    ASSERT_NEAR(ref_tanh(1.f), 0.7615942f, kTol);
    ASSERT_NEAR(ref_tanh(-1.f), -0.7615942f, kTol);
    // anti-symmetric
    ASSERT_NEAR(ref_tanh(-0.5f), -ref_tanh(0.5f), kTol);
}

void test_swish() {
    // swish(0) = 0 * sigmoid(0) = 0
    ASSERT_NEAR(ref_swish(0.f), 0.f, kTol);
    // large positive: swish(x) ≈ x
    ASSERT_NEAR(ref_swish(5.f), 5.f * ref_sigmoid(5.f), kTol);
    // slightly negative allowed for small negatives
    ASSERT_TRUE(ref_swish(-0.2f) < 0.f);
}

void test_hardswish() {
    ASSERT_NEAR(ref_hardswish(-4.f), 0.f, kTol);
    ASSERT_NEAR(ref_hardswish(4.f), 4.f, kTol);
    ASSERT_NEAR(ref_hardswish(0.f), 0.f, kTol);
    ASSERT_NEAR(ref_hardswish(1.f), 1.f * 4.f / 6.f, kTol);  // x*(x+3)/6 at x=1 = 4/6
}

void test_hardsigmoid() {
    ASSERT_NEAR(ref_hardsigmoid(-10.f), 0.f, kTol);
    ASSERT_NEAR(ref_hardsigmoid(10.f), 1.f, kTol);
    ASSERT_NEAR(ref_hardsigmoid(0.f), 0.5f, kTol);
}

void test_selu() {
    ASSERT_NEAR(ref_selu(0.f), 0.f, kTol);
    ASSERT_NEAR(ref_selu(1.f), 1.0507009873554805f, kTol);
    ASSERT_TRUE(ref_selu(-1.f) < 0.f);
}

void test_celu() {
    ASSERT_NEAR(ref_celu(0.f), 0.f, kTol);
    ASSERT_NEAR(ref_celu(1.f), 1.f, kTol);  // positive unchanged
    ASSERT_TRUE(ref_celu(-1.f) < 0.f);
}

void test_clip() {
    ASSERT_NEAR(ref_clip(-5.f, -2.f, 2.f), -2.f, kTol);
    ASSERT_NEAR(ref_clip(5.f, -2.f, 2.f), 2.f, kTol);
    ASSERT_NEAR(ref_clip(1.f, -2.f, 2.f), 1.f, kTol);
    ASSERT_NEAR(ref_clip(0.f, 0.f, 0.f), 0.f, kTol);
}

void test_mish() {
    ASSERT_NEAR(ref_mish(0.f), 0.f, kTol);
    // mish(1) = 1 * tanh(softplus(1)) = 1 * tanh(ln(1+e)) ≈ 0.8651
    ASSERT_NEAR(ref_mish(1.f), 0.8651993f, 2e-4f);
    ASSERT_TRUE(ref_mish(-0.5f) < 0.f);
}

void test_softplus() {
    // softplus(0) = ln(2) ≈ 0.6931
    ASSERT_NEAR(ref_softplus(0.f), logf(2.f), kTol);
    // softplus(x) ≈ x for large x
    ASSERT_NEAR(ref_softplus(10.f), 10.f, 0.01f);
    // softplus is always positive
    for (int i = 0; i < N; ++i) ASSERT_TRUE(ref_softplus(kInputs[i]) > 0.f);
}

void test_prelu() {
    float slope = 0.25f;
    ASSERT_NEAR(ref_prelu(2.f, slope), 2.f, kTol);
    ASSERT_NEAR(ref_prelu(-2.f, slope), -0.5f, kTol);
    ASSERT_NEAR(ref_prelu(0.f, slope), 0.f, kTol);
}

void test_bnll() {
    // bnll(0) = log(2)
    ASSERT_NEAR(ref_bnll(0.f), logf(2.f), kTol);
    // always positive
    for (int i = 0; i < N; ++i) ASSERT_TRUE(ref_bnll(kInputs[i]) > 0.f);
}

void test_erf_activation() {
    ASSERT_NEAR(ref_erf(0.f), 0.f, kTol);
    ASSERT_NEAR(ref_erf(1.f), 0.8427007f, kTol);
    ASSERT_NEAR(ref_erf(-1.f), -0.8427007f, kTol);  // anti-symmetric
}

void test_exp_activation() {
    // base=-1 (natural): exp(shift + scale*x) = exp(1 * x) at scale=1, shift=0
    ASSERT_NEAR(ref_exp(1.f, -1.f, 1.f, 0.f), expf(1.f), kTol);
    // base=2: 2^x
    ASSERT_NEAR(ref_exp(3.f, 2.f, 1.f, 0.f), 8.f, kTol);
}

void test_log_activation() {
    // natural log at x=e
    ASSERT_NEAR(ref_log(expf(1.f), -1.f, 1.f, 0.f), 1.f, kTol);
    // base 10: log10(100) = 2
    ASSERT_NEAR(ref_log(100.f, 10.f, 1.f, 0.f), 2.f, 1e-4f);
}

void test_power_activation() {
    // (1*x)^2 = x^2
    ASSERT_NEAR(ref_power(3.f, 2.f, 1.f, 0.f), 9.f, kTol);
    // (x + 1)^2 at x=2: (2+1)^2 = 9
    ASSERT_NEAR(ref_power(2.f, 2.f, 1.f, 1.f), 9.f, kTol);
}

void test_shrink() {
    float bias = 0.f, lambd = 0.5f;
    ASSERT_NEAR(ref_shrink(1.f,  bias, lambd), 1.f, kTol);
    ASSERT_NEAR(ref_shrink(-1.f, bias, lambd), -1.f, kTol);
    ASSERT_NEAR(ref_shrink(0.3f, bias, lambd), 0.f, kTol);  // within lambda
    ASSERT_NEAR(ref_shrink(-0.3f, bias, lambd), 0.f, kTol);
}

void test_threshold() {
    ASSERT_NEAR(ref_threshold(0.5f, 0.3f), 1.f, kTol);
    ASSERT_NEAR(ref_threshold(0.1f, 0.3f), 0.f, kTol);
    ASSERT_NEAR(ref_threshold(0.3f, 0.3f), 0.f, kTol); // not strictly greater
}

// ─── Batch / vectorized tests ────────────────────────────────────────────────

void test_relu_batch() {
    std::vector<float> in = { -2.f, -1.f, 0.f, 1.f, 2.f };
    std::vector<float> out(in.size());
    for (size_t i = 0; i < in.size(); ++i) out[i] = ref_relu(in[i]);
    float expected[] = { 0.f, 0.f, 0.f, 1.f, 2.f };
    ASSERT_VEC_NEAR(out, expected, in.size(), kTol);
}

void test_sigmoid_symmetry() {
    // sigmoid(x) + sigmoid(-x) == 1 for all x
    float xs[] = { -3.f, -1.f, 0.f, 1.f, 3.f };
    for (float x : xs) ASSERT_NEAR(ref_sigmoid(x) + ref_sigmoid(-x), 1.f, kTol);
}

// ─── Real ncnn kernel tests ───────────────────────────────────────────────────

void test_relu_ncnn()
{
    // slope=0 → standard ReLU
    std::vector<float> vals = { -3.f, -1.f, 0.f, 1.f, 3.f };
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::ReLU relu; relu.slope = 0.f;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(relu.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    float expected[] = { 0.f, 0.f, 0.f, 1.f, 3.f };
    ASSERT_VEC_NEAR(out, expected, 5, 1e-6f);
}

void test_leakyrelu_ncnn()
{
    std::vector<float> vals = { -2.f, -1.f, 0.f, 1.f };
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::ReLU relu; relu.slope = 0.1f;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(relu.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    ASSERT_NEAR(out[0], -0.2f, 1e-5f);
    ASSERT_NEAR(out[1], -0.1f, 1e-5f);
    ASSERT_NEAR(out[2],  0.f,  1e-5f);
    ASSERT_NEAR(out[3],  1.f,  1e-5f);
}

void test_sigmoid_ncnn()
{
    std::vector<float> vals = { -1.f, 0.f, 1.f };
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::Sigmoid sig;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(sig.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    ASSERT_NEAR(out[0], 1.f / (1.f + expf(1.f)),  1e-5f);
    ASSERT_NEAR(out[1], 0.5f,                      1e-5f);
    ASSERT_NEAR(out[2], 1.f / (1.f + expf(-1.f)), 1e-5f);
}

void test_gelu_ncnn()
{
    // fast_gelu=0 → exact GELU; compare against ref_gelu()
    float xs[] = { -1.f, 0.f, 1.f, 2.f };
    std::vector<float> vals(xs, xs + 4);
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::GELU gelu; gelu.fast_gelu = 0;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(gelu.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    for (int i = 0; i < 4; ++i)
        ASSERT_NEAR(out[i], ref_gelu(xs[i]), 1e-4f);
}

void test_gelu_fast_ncnn()
{
    // fast_gelu=1 → tanh approximation; should be close to exact
    float xs[] = { -1.f, 0.f, 1.f, 2.f };
    std::vector<float> vals(xs, xs + 4);
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::GELU gelu; gelu.fast_gelu = 1;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(gelu.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    for (int i = 0; i < 4; ++i)
        ASSERT_NEAR(out[i], ref_gelu(xs[i]), 0.02f);
}

void test_clip_ncnn()
{
    std::vector<float> vals = { -5.f, -1.f, 0.f, 1.f, 5.f };
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::Clip clip; clip.min = -2.f; clip.max = 2.f;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(clip.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    float expected[] = { -2.f, -1.f, 0.f, 1.f, 2.f };
    ASSERT_VEC_NEAR(out, expected, 5, 1e-6f);
}

void test_tanh_ncnn()
{
    float xs[] = { -1.f, 0.f, 1.f };
    std::vector<float> vals(xs, xs + 3);
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::TanH tanh_layer;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(tanh_layer.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    for (int i = 0; i < 3; ++i)
        ASSERT_NEAR(out[i], tanhf(xs[i]), 1e-5f);
}

void test_absval_ncnn()
{
    std::vector<float> vals = { -3.f, 0.f, 2.f };
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::AbsVal absval;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(absval.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    ASSERT_NEAR(out[0], 3.f, 1e-6f);
    ASSERT_NEAR(out[1], 0.f, 1e-6f);
    ASSERT_NEAR(out[2], 2.f, 1e-6f);
}

void test_swish_ncnn()
{
    float xs[] = { -1.f, 0.f, 1.f };
    std::vector<float> vals(xs, xs + 3);
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::Swish swish;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(swish.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    for (int i = 0; i < 3; ++i)
        ASSERT_NEAR(out[i], xs[i] / (1.f + expf(-xs[i])), 1e-5f);
}

void test_relu_multichannel_ncnn()
{
    // 2 channels × 3 elements each — tests the per-channel loop
    std::vector<float> flat = { -1.f, 2.f, -3.f,   4.f, -5.f, 6.f };
    ncnn::Mat m = make_mat(3, 1, 2, flat);
    ncnn::ReLU relu; relu.slope = 0.f;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(relu.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    float expected[] = { 0.f, 2.f, 0.f,   4.f, 0.f, 6.f };
    ASSERT_VEC_NEAR(out, expected, 6, 1e-6f);
}

int main() {
    printf("=== test_activation ===\n");
    printf("\n-- Reference tests --\n");
    RUN_TEST(test_absval);
    RUN_TEST(test_relu);
    RUN_TEST(test_leakyrelu);
    RUN_TEST(test_elu);
    RUN_TEST(test_gelu_exact);
    RUN_TEST(test_gelu_fast);
    RUN_TEST(test_sigmoid);
    RUN_TEST(test_tanh);
    RUN_TEST(test_swish);
    RUN_TEST(test_hardswish);
    RUN_TEST(test_hardsigmoid);
    RUN_TEST(test_selu);
    RUN_TEST(test_celu);
    RUN_TEST(test_clip);
    RUN_TEST(test_mish);
    RUN_TEST(test_softplus);
    RUN_TEST(test_prelu);
    RUN_TEST(test_bnll);
    RUN_TEST(test_erf_activation);
    RUN_TEST(test_exp_activation);
    RUN_TEST(test_log_activation);
    RUN_TEST(test_power_activation);
    RUN_TEST(test_shrink);
    RUN_TEST(test_threshold);
    RUN_TEST(test_relu_batch);
    RUN_TEST(test_sigmoid_symmetry);

    printf("\n-- Real ncnn::ReLU / Sigmoid / GELU / Clip / TanH / AbsVal / Swish --\n");
    RUN_TEST(test_relu_ncnn);
    RUN_TEST(test_leakyrelu_ncnn);
    RUN_TEST(test_sigmoid_ncnn);
    RUN_TEST(test_gelu_ncnn);
    RUN_TEST(test_gelu_fast_ncnn);
    RUN_TEST(test_clip_ncnn);
    RUN_TEST(test_tanh_ncnn);
    RUN_TEST(test_absval_ncnn);
    RUN_TEST(test_swish_ncnn);
    RUN_TEST(test_relu_multichannel_ncnn);

    print_summary("activation");
    return g_failed > 0 ? 1 : 0;
}
