// test_norm.cpp — ARM norm kernel tests
// Tests LayerNorm_arm, RMSNorm_arm, BatchNorm_arm, InstanceNorm_arm, GroupNorm_arm

#include "test_utils.h"
#include "ncnn_helpers.h"

#include "../norm/layernorm_arm.h"
#include "../norm/rmsnorm_arm.h"
#include "../norm/batchnorm_arm.h"
#include "../norm/instancenorm_arm.h"
#include "../norm/groupnorm_arm.h"

// ─── Reference implementations ───────────────────────────────────────────────

static void ref_layernorm(float* p, int n, float eps,
                           const float* gamma = nullptr,
                           const float* beta  = nullptr) {
    float mean = 0.f;
    for (int i = 0; i < n; ++i) mean += p[i];
    mean /= n;
    float var = 0.f;
    for (int i = 0; i < n; ++i) var += (p[i] - mean) * (p[i] - mean);
    var /= n;
    float inv = 1.f / sqrtf(var + eps);
    for (int i = 0; i < n; ++i) {
        p[i] = (p[i] - mean) * inv;
        if (gamma) p[i] *= gamma[i];
        if (beta)  p[i] += beta[i];
    }
}

static void ref_rmsnorm(float* p, int n, float eps,
                         const float* gamma = nullptr) {
    float rms = 0.f;
    for (int i = 0; i < n; ++i) rms += p[i] * p[i];
    rms = sqrtf(rms / n + eps);
    for (int i = 0; i < n; ++i) {
        p[i] /= rms;
        if (gamma) p[i] *= gamma[i];
    }
}

static void ref_batchnorm(float* p, int n_spatial, int channels,
                           const float* slope, const float* mean,
                           const float* var, const float* bias, float eps) {
    for (int c = 0; c < channels; ++c) {
        float b = slope[c] / sqrtf(var[c] + eps);
        float a = bias[c] - slope[c] * mean[c] / sqrtf(var[c] + eps);
        float* ch = p + c * n_spatial;
        for (int i = 0; i < n_spatial; ++i) ch[i] = b * ch[i] + a;
    }
}

static void ref_instancenorm(float* p, int n, float eps,
                              float gamma = 1.f, float beta = 0.f) {
    float mean = 0.f;
    for (int i = 0; i < n; ++i) mean += p[i];
    mean /= n;
    float var = 0.f;
    for (int i = 0; i < n; ++i) var += (p[i] - mean) * (p[i] - mean);
    var /= n;
    float inv = 1.f / sqrtf(var + eps);
    for (int i = 0; i < n; ++i) p[i] = (p[i] - mean) * inv * gamma + beta;
}

static void ref_groupnorm(float* p, int c, int h, int w, int groups,
                           float eps,
                           const float* gamma = nullptr,
                           const float* beta  = nullptr) {
    int gc = c / groups;
    int spatial = h * w;
    for (int g = 0; g < groups; ++g) {
        int n = gc * spatial;
        float* gp = p + g * gc * spatial;
        float mean = 0.f;
        for (int i = 0; i < n; ++i) mean += gp[i];
        mean /= n;
        float var = 0.f;
        for (int i = 0; i < n; ++i) var += (gp[i] - mean) * (gp[i] - mean);
        var /= n;
        float inv = 1.f / sqrtf(var + eps);
        for (int i = 0; i < n; ++i) {
            int ch = g * gc + i / spatial;
            gp[i] = (gp[i] - mean) * inv;
            if (gamma) gp[i] *= gamma[ch];
            if (beta)  gp[i] += beta[ch];
        }
    }
}

// ─── Reference-only test cases ────────────────────────────────────────────────

void test_layernorm_basic() {
    float p[] = { 1.f, 2.f, 3.f, 4.f };
    ref_layernorm(p, 4, 1e-5f);
    float mean_out = 0.f, var_out = 0.f;
    for (float v : p) mean_out += v;
    mean_out /= 4;
    for (float v : p) var_out += (v - mean_out) * (v - mean_out);
    var_out /= 4;
    ASSERT_NEAR(mean_out, 0.f, 1e-5f);
    ASSERT_NEAR(var_out, 1.f, 1e-4f);
}

void test_layernorm_affine() {
    float p[] = { 1.f, 2.f, 3.f, 4.f };
    float gamma[] = { 2.f, 2.f, 2.f, 2.f };
    float beta[]  = { 1.f, 1.f, 1.f, 1.f };
    ref_layernorm(p, 4, 1e-5f, gamma, beta);
    ASSERT_TRUE(p[0] < p[1] && p[1] < p[2] && p[2] < p[3]);
}

void test_layernorm_constant_input() {
    float p[] = { 5.f, 5.f, 5.f, 5.f };
    ref_layernorm(p, 4, 1e-5f);
    for (float v : p) ASSERT_NEAR(v, 0.f, 1e-4f);
}

void test_rmsnorm_basic() {
    float p[] = { 3.f, 4.f };
    float expected_rms = sqrtf(12.5f + 1e-5f);
    float expected[] = { 3.f / expected_rms, 4.f / expected_rms };
    ref_rmsnorm(p, 2, 1e-5f);
    ASSERT_VEC_NEAR(p, expected, 2, 1e-4f);
}

void test_rmsnorm_with_gamma() {
    float p[] = { 1.f, 1.f, 1.f, 1.f };
    float gamma[] = { 1.f, 2.f, 3.f, 4.f };
    ref_rmsnorm(p, 4, 1e-5f, gamma);
    ASSERT_VEC_NEAR(p, gamma, 4, 1e-4f);
}

void test_batchnorm_basic() {
    float data[] = { 1.f, 2.f, 3.f,   4.f, 5.f, 6.f };
    float slope[] = { 1.f, 1.f };
    float mean[]  = { 2.f, 5.f };
    float var2[]  = { 1.f, 1.f };
    float bias[]  = { 0.f, 0.f };
    ref_batchnorm(data, 3, 2, slope, mean, var2, bias, 1e-5f);
    ASSERT_NEAR(data[0], -1.f, 1e-4f);
    ASSERT_NEAR(data[1],  0.f, 1e-4f);
    ASSERT_NEAR(data[2],  1.f, 1e-4f);
    ASSERT_NEAR(data[3], -1.f, 1e-4f);
    ASSERT_NEAR(data[4],  0.f, 1e-4f);
    ASSERT_NEAR(data[5],  1.f, 1e-4f);
}

void test_instancenorm() {
    float p[] = { 1.f, 3.f, 5.f, 7.f };
    ref_instancenorm(p, 4, 1e-5f);
    float m = 0.f;
    for (float v : p) m += v;
    m /= 4;
    ASSERT_NEAR(m, 0.f, 1e-5f);
}

void test_groupnorm_basic() {
    float p[] = { 1.f, 3.f,   5.f, 7.f };
    ref_groupnorm(p, 4, 1, 1, 2, 1e-5f);
    ASSERT_NEAR(p[0], -1.f, 1e-4f);
    ASSERT_NEAR(p[1],  1.f, 1e-4f);
    ASSERT_NEAR(p[2], -1.f, 1e-4f);
    ASSERT_NEAR(p[3],  1.f, 1e-4f);
}

// ─── Real ncnn ARM kernel tests ───────────────────────────────────────────────

void test_layernorm_arm_basic()
{
    std::vector<float> vals = { 1.f, 2.f, 3.f, 4.f };
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::LayerNorm_arm ln;
    ln.affine_size = 4;
    ln.eps         = 1e-5f;
    ln.affine      = 0;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(ln.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    std::vector<float> ref = vals;
    layernorm_inplace(ref.data(), 4, 1e-5f);
    ASSERT_VEC_NEAR(out, ref.data(), 4, 1e-3f);
}

void test_layernorm_arm_affine()
{
    std::vector<float> vals  = { 1.f, 2.f, 3.f, 4.f };
    std::vector<float> gamma = { 2.f, 2.f, 2.f, 2.f };
    std::vector<float> beta  = { 1.f, 1.f, 1.f, 1.f };
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::LayerNorm_arm ln;
    ln.affine_size = 4;
    ln.eps         = 1e-5f;
    ln.affine      = 1;
    ln.gamma_data  = make_weight(gamma);
    ln.beta_data   = make_weight(beta);
    ncnn::Option opt = make_opt();
    ASSERT_EQ(ln.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    std::vector<float> ref = vals;
    layernorm_inplace(ref.data(), 4, 1e-5f, gamma.data(), beta.data());
    ASSERT_VEC_NEAR(out, ref.data(), 4, 1e-3f);
}

void test_rmsnorm_arm()
{
    std::vector<float> vals  = { 3.f, 4.f };
    std::vector<float> gamma = { 1.f, 1.f };
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::RMSNorm_arm rms;
    rms.affine_size = 2;
    rms.eps         = 1e-5f;
    rms.affine      = 1;
    rms.gamma_data  = make_weight(gamma);
    ncnn::Option opt = make_opt();
    ASSERT_EQ(rms.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    float expected_rms = sqrtf(12.5f + 1e-5f);
    ASSERT_NEAR(out[0], 3.f / expected_rms, 1e-3f);
    ASSERT_NEAR(out[1], 4.f / expected_rms, 1e-3f);
}

void test_batchnorm_arm()
{
    int channels = 2;
    std::vector<float> flat = { 1.f, 2.f, 3.f,   4.f, 5.f, 6.f };
    ncnn::Mat m = make_mat(3, 1, 2, flat);
    ncnn::BatchNorm_arm bn;
    bn.channels = channels;
    bn.eps      = 0.f;
    bn.slope_data = make_weight({ 1.f, 1.f });
    bn.mean_data  = make_weight({ 2.f, 5.f });
    bn.var_data   = make_weight({ 1.f, 1.f });
    bn.bias_data  = make_weight({ 0.f, 0.f });
    // Compute a_data/b_data as load_model() normally would
    bn.a_data = make_weight({ -2.f, -5.f });
    bn.b_data = make_weight({ 1.f, 1.f });
    ncnn::Option opt = make_opt();
    ASSERT_EQ(bn.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    ASSERT_NEAR(out[0], -1.f, 1e-3f);
    ASSERT_NEAR(out[1],  0.f, 1e-3f);
    ASSERT_NEAR(out[2],  1.f, 1e-3f);
    ASSERT_NEAR(out[3], -1.f, 1e-3f);
    ASSERT_NEAR(out[4],  0.f, 1e-3f);
    ASSERT_NEAR(out[5],  1.f, 1e-3f);
}

void test_instancenorm_arm()
{
    std::vector<float> vals = { 1.f, 3.f, 5.f, 7.f };
    ncnn::Mat m = make_mat(4, 1, 1, vals);
    ncnn::InstanceNorm_arm in_layer;
    in_layer.channels = 1;
    in_layer.eps      = 1e-5f;
    in_layer.affine   = 0;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(in_layer.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    float mean = 0.f;
    for (float v : out) mean += v;
    mean /= 4;
    ASSERT_NEAR(mean, 0.f, 1e-3f);
}

void test_groupnorm_arm()
{
    std::vector<float> flat = { 1.f, 3.f, 5.f, 7.f };
    ncnn::Mat m = make_mat(1, 1, 4, flat);
    ncnn::GroupNorm_arm gn;
    gn.group      = 2;
    gn.channels   = 4;
    gn.eps        = 1e-5f;
    gn.affine     = 0;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(gn.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    ASSERT_NEAR(out[0], -1.f, 1e-3f);
    ASSERT_NEAR(out[1],  1.f, 1e-3f);
    ASSERT_NEAR(out[2], -1.f, 1e-3f);
    ASSERT_NEAR(out[3],  1.f, 1e-3f);
}

int main() {
    printf("=== test_norm (ARM) ===\n");
    printf("\n-- Reference tests --\n");
    RUN_TEST(test_layernorm_basic);
    RUN_TEST(test_layernorm_affine);
    RUN_TEST(test_layernorm_constant_input);
    RUN_TEST(test_rmsnorm_basic);
    RUN_TEST(test_rmsnorm_with_gamma);
    RUN_TEST(test_batchnorm_basic);
    RUN_TEST(test_instancenorm);
    RUN_TEST(test_groupnorm_basic);

    printf("\n-- Real ARM norm kernels --\n");
    RUN_TEST(test_layernorm_arm_basic);
    RUN_TEST(test_layernorm_arm_affine);
    RUN_TEST(test_rmsnorm_arm);
    RUN_TEST(test_batchnorm_arm);
    RUN_TEST(test_instancenorm_arm);
    RUN_TEST(test_groupnorm_arm);

    print_summary("norm_arm");
    return g_failed > 0 ? 1 : 0;
}
