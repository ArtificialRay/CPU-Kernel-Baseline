// test_norm.cpp
// Tests for norm/:
//   layernorm, batchnorm, groupnorm, instancenorm, rmsnorm, lrn, mvn, normalize
//
// Section 1: reference-only tests
// Section 2: real ncnn kernel tests (linked via norm_impl + ncnn_stub)

#include "test_utils.h"

// ── Real ncnn kernel headers ──────────────────────────────────────────────────
#include "ncnn_helpers.h"
#include "../norm/layernorm.h"
#include "../norm/rmsnorm.h"
#include "../norm/batchnorm.h"
#include "../norm/instancenorm.h"
#include "../norm/groupnorm.h"

// ─── Reference implementations ───────────────────────────────────────────────

// Layer norm: normalize over last dimension
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

// RMS norm: normalize by root-mean-square (no mean subtraction)
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

// Batch norm (inference): out = b * in + a, where
//   b = slope / sqrt(var + eps),  a = bias - slope * mean / sqrt(var + eps)
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

// Instance norm (per-sample per-channel normalization)
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

// Group norm: split channels into groups, normalize within each group
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

// LRN (Local Response Normalization): cross-channel
static void ref_lrn(const float* in, float* out, int c, int spatial,
                     int local_size, float alpha, float beta, float bias = 1.f) {
    int half = local_size / 2;
    for (int ci = 0; ci < c; ++ci) {
        for (int s = 0; s < spatial; ++s) {
            float sum = 0.f;
            for (int j = ci - half; j <= ci + half; ++j) {
                if (j >= 0 && j < c) {
                    float v = in[j * spatial + s];
                    sum += v * v;
                }
            }
            float scale = powf(bias + alpha * sum, beta);
            out[ci * spatial + s] = in[ci * spatial + s] / scale;
        }
    }
}

// MVN: Mean-Variance Normalization
static void ref_mvn(float* p, int n, float eps,
                    bool normalize_variance = true, bool across_channels = false) {
    // Simple single-segment version
    float mean = 0.f;
    for (int i = 0; i < n; ++i) mean += p[i];
    mean /= n;
    for (int i = 0; i < n; ++i) p[i] -= mean;
    if (normalize_variance) {
        float var = 0.f;
        for (int i = 0; i < n; ++i) var += p[i] * p[i];
        var = sqrtf(var / n + eps);
        for (int i = 0; i < n; ++i) p[i] /= var;
    }
}

// L2 Normalize over spatial
static void ref_normalize_l2(float* p, int n, float eps) {
    float norm = 0.f;
    for (int i = 0; i < n; ++i) norm += p[i] * p[i];
    norm = sqrtf(norm + eps);
    for (int i = 0; i < n; ++i) p[i] /= norm;
}

// ─── Test cases ──────────────────────────────────────────────────────────────

void test_layernorm_basic() {
    float p[] = { 1.f, 2.f, 3.f, 4.f };
    ref_layernorm(p, 4, 1e-5f);
    // After normalizing: sum of squares = 1 (unit variance), mean=0
    float mean_out = 0.f, var_out = 0.f;
    for (float v : p) mean_out += v;
    mean_out /= 4;
    for (float v : p) var_out += (v - mean_out) * (v - mean_out);
    var_out /= 4;
    ASSERT_NEAR(mean_out, 0.f, 1e-5f);
    ASSERT_NEAR(var_out, 1.f, 1e-4f);  // unit variance
}

void test_layernorm_affine() {
    float p[] = { 1.f, 2.f, 3.f, 4.f };
    float gamma[] = { 2.f, 2.f, 2.f, 2.f };
    float beta[]  = { 1.f, 1.f, 1.f, 1.f };
    ref_layernorm(p, 4, 1e-5f, gamma, beta);
    // gamma=2, beta=1 → scaled & shifted normalized
    // normalized[0] should be most negative, normalized[3] most positive
    ASSERT_TRUE(p[0] < p[1] && p[1] < p[2] && p[2] < p[3]);
}

void test_layernorm_constant_input() {
    // All-same input → mean=that value, var=0, output should be 0 (clamped by eps)
    float p[] = { 5.f, 5.f, 5.f, 5.f };
    ref_layernorm(p, 4, 1e-5f);
    for (float v : p) ASSERT_NEAR(v, 0.f, 1e-4f);
}

void test_rmsnorm_basic() {
    float p[] = { 3.f, 4.f };  // rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
    float expected_rms = sqrtf(12.5f + 1e-5f);
    float expected[] = { 3.f / expected_rms, 4.f / expected_rms };
    ref_rmsnorm(p, 2, 1e-5f);
    ASSERT_VEC_NEAR(p, expected, 2, 1e-4f);
}

void test_rmsnorm_with_gamma() {
    float p[] = { 1.f, 1.f, 1.f, 1.f };
    float gamma[] = { 1.f, 2.f, 3.f, 4.f };
    ref_rmsnorm(p, 4, 1e-5f, gamma);
    // All inputs equal → rms=1, normalized=1 → output = gamma
    ASSERT_VEC_NEAR(p, gamma, 4, 1e-4f);
}

void test_batchnorm_basic() {
    // 2 channels, 3 spatial positions each
    float data[] = { 1.f, 2.f, 3.f,   // ch0
                     4.f, 5.f, 6.f };  // ch1
    float slope[] = { 1.f, 1.f };
    float mean[]  = { 2.f, 5.f };      // mean of ch0 and ch1
    float var[]   = { 0.f, 0.f };      // zero variance (constant channels skipped in tests)
    // Use non-trivial mean/var
    float var2[]  = { 1.f, 1.f };
    float bias[]  = { 0.f, 0.f };
    ref_batchnorm(data, 3, 2, slope, mean, var2, bias, 1e-5f);
    // ch0: (x - 2) / sqrt(1 + eps) ≈ [-1, 0, 1]
    ASSERT_NEAR(data[0], -1.f, 1e-4f);
    ASSERT_NEAR(data[1],  0.f, 1e-4f);
    ASSERT_NEAR(data[2],  1.f, 1e-4f);
    // ch1: (x - 5) / sqrt(1) ≈ [-1, 0, 1]
    ASSERT_NEAR(data[3], -1.f, 1e-4f);
    ASSERT_NEAR(data[4],  0.f, 1e-4f);
    ASSERT_NEAR(data[5],  1.f, 1e-4f);
}

void test_batchnorm_scale_bias() {
    float data[] = { 2.f };
    float slope[] = { 2.f };
    float mean[]  = { 1.f };
    float var[]   = { 1.f };
    float bias[]  = { 3.f };
    ref_batchnorm(data, 1, 1, slope, mean, var, bias, 0.f);
    // b = 2/sqrt(1) = 2, a = 3 - 2*1 = 1  → out = 2*2 + 1 = 5
    ASSERT_NEAR(data[0], 5.f, 1e-5f);
}

void test_instancenorm() {
    float p[] = { 1.f, 3.f, 5.f, 7.f };  // mean=4, var=5
    ref_instancenorm(p, 4, 1e-5f);
    // After normalization: mean≈0, variance≈1
    float m = 0.f;
    for (float v : p) m += v;
    m /= 4;
    ASSERT_NEAR(m, 0.f, 1e-5f);
}

void test_groupnorm_basic() {
    // 4 channels, 1x1 spatial, 2 groups (2 channels per group)
    float p[] = { 1.f, 3.f,   // group0
                  5.f, 7.f };  // group1
    ref_groupnorm(p, 4, 1, 1, 2, 1e-5f);
    // group0: mean=2, var=1 → [-1, 1]
    ASSERT_NEAR(p[0], -1.f, 1e-4f);
    ASSERT_NEAR(p[1],  1.f, 1e-4f);
    // group1: mean=6, var=1 → [-1, 1]
    ASSERT_NEAR(p[2], -1.f, 1e-4f);
    ASSERT_NEAR(p[3],  1.f, 1e-4f);
}

void test_lrn_basic() {
    // 3 channels, 1 spatial, local_size=3, alpha=1, beta=0.75
    float in[] = { 1.f, 2.f, 3.f };  // c=3, spatial=1
    float out[3];
    ref_lrn(in, out, 3, 1, 3, 1.f, 0.75f, 1.f);
    // c=0: sum=(1^2+2^2)=5 → scale=(1+5)^0.75 ≈ 3.834  out≈0.261
    // c=1: sum=(1^2+2^2+3^2)=14 → scale=(1+14)^0.75 ≈ 7.622  out≈0.263
    // c=2: sum=(2^2+3^2)=13 → scale=(1+13)^0.75 ≈ 7.217  out≈0.415
    ASSERT_TRUE(out[0] > 0.f && out[0] < 1.f);
    ASSERT_TRUE(out[1] > 0.f && out[1] < 1.f);
    ASSERT_TRUE(out[2] > 0.f && out[2] < 3.f);
}

void test_mvn_mean_only() {
    float p[] = { 1.f, 3.f, 5.f, 7.f };
    ref_mvn(p, 4, 1e-5f, false);
    // mean = 4, output = x - 4
    ASSERT_NEAR(p[0], -3.f, 1e-5f);
    ASSERT_NEAR(p[3],  3.f, 1e-5f);
}

void test_mvn_full() {
    float p[] = { 2.f, 4.f, 4.f, 4.f, 5.f, 5.f, 7.f, 9.f };  // classic example: mean=5, std=2
    ref_mvn(p, 8, 1e-5f, true);
    // After MVN: approximately unit std
    float var = 0.f;
    for (int i = 0; i < 8; ++i) var += p[i] * p[i];
    var /= 8;
    ASSERT_NEAR(sqrtf(var), 1.f, 0.01f);
}

void test_normalize_l2() {
    float p[] = { 3.f, 4.f };  // norm = 5
    ref_normalize_l2(p, 2, 1e-5f);
    ASSERT_NEAR(p[0], 0.6f, 1e-4f);
    ASSERT_NEAR(p[1], 0.8f, 1e-4f);
    // After L2 norm: p·p ≈ 1
    float norm = p[0]*p[0] + p[1]*p[1];
    ASSERT_NEAR(norm, 1.f, 1e-4f);
}

// ─── Real ncnn kernel tests ───────────────────────────────────────────────────

void test_layernorm_ncnn()
{
    // affine_size=4, no gamma/beta: output should have mean≈0, var≈1
    std::vector<float> vals = { 1.f, 2.f, 3.f, 4.f };
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::LayerNorm ln;
    ln.affine_size = 4;
    ln.eps         = 1e-5f;
    ln.affine      = 0;  // no gamma/beta
    ncnn::Option opt = make_opt();
    ASSERT_EQ(ln.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    // Compare with ref
    std::vector<float> ref = vals;
    layernorm_inplace(ref.data(), 4, 1e-5f);
    ASSERT_VEC_NEAR(out, ref.data(), 4, 1e-4f);
}

void test_layernorm_affine_ncnn()
{
    // affine=1 with gamma=2, beta=1
    std::vector<float> vals = { 1.f, 2.f, 3.f, 4.f };
    std::vector<float> gamma = { 2.f, 2.f, 2.f, 2.f };
    std::vector<float> beta  = { 1.f, 1.f, 1.f, 1.f };

    ncnn::Mat m = make_mat_1d(vals);
    ncnn::LayerNorm ln;
    ln.affine_size = 4;
    ln.eps         = 1e-5f;
    ln.affine      = 1;
    ln.gamma_data  = make_weight(gamma);
    ln.beta_data   = make_weight(beta);
    ncnn::Option opt = make_opt();
    ASSERT_EQ(ln.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    // reference
    std::vector<float> ref = vals;
    layernorm_inplace(ref.data(), 4, 1e-5f, gamma.data(), beta.data());
    ASSERT_VEC_NEAR(out, ref.data(), 4, 1e-4f);
}

void test_rmsnorm_ncnn()
{
    std::vector<float> vals  = { 3.f, 4.f };
    std::vector<float> gamma = { 1.f, 1.f };
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::RMSNorm rms;
    rms.affine_size = 2;
    rms.eps         = 1e-5f;
    rms.affine      = 1;
    rms.gamma_data  = make_weight(gamma);
    ncnn::Option opt = make_opt();
    ASSERT_EQ(rms.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    // rms = sqrt((9+16)/2 + eps) ≈ sqrt(12.5)
    float expected_rms = sqrtf(12.5f + 1e-5f);
    ASSERT_NEAR(out[0], 3.f / expected_rms, 1e-4f);
    ASSERT_NEAR(out[1], 4.f / expected_rms, 1e-4f);
}

void test_batchnorm_ncnn()
{
    // 2 channels, 3 spatial positions: (x - mean) / sqrt(var + eps) * slope + bias
    // Use slope=1, mean=2/5, var=1, bias=0 → (x-mean)/1
    int channels = 2;
    std::vector<float> flat = { 1.f, 2.f, 3.f,   4.f, 5.f, 6.f };  // [c=2, h=1, w=3]
    ncnn::Mat m = make_mat(3, 1, 2, flat);
    ncnn::BatchNorm bn;
    bn.channels = channels;
    bn.eps      = 0.f;
    bn.slope_data = make_weight({ 1.f, 1.f });
    bn.mean_data  = make_weight({ 2.f, 5.f });
    bn.var_data   = make_weight({ 1.f, 1.f });
    bn.bias_data  = make_weight({ 0.f, 0.f });
    // BatchNorm::forward_inplace uses a_data/b_data pre-computed by load_model:
    //   b[i] = slope[i] / sqrt(var[i] + eps)
    //   a[i] = bias[i] - slope[i] * mean[i] / sqrt(var[i] + eps)
    bn.b_data = make_weight({ 1.f, 1.f });          // 1/sqrt(1) = 1
    bn.a_data = make_weight({ -2.f, -5.f });        // 0 - 1*mean/1
    ncnn::Option opt = make_opt();
    ASSERT_EQ(bn.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    // ch0: (x-2)/1 = [-1, 0, 1]
    ASSERT_NEAR(out[0], -1.f, 1e-4f);
    ASSERT_NEAR(out[1],  0.f, 1e-4f);
    ASSERT_NEAR(out[2],  1.f, 1e-4f);
    // ch1: (x-5)/1 = [-1, 0, 1]
    ASSERT_NEAR(out[3], -1.f, 1e-4f);
    ASSERT_NEAR(out[4],  0.f, 1e-4f);
    ASSERT_NEAR(out[5],  1.f, 1e-4f);
}

void test_instancenorm_ncnn()
{
    // 1 channel, 4 spatial, affine=0
    std::vector<float> vals = { 1.f, 3.f, 5.f, 7.f };
    ncnn::Mat m = make_mat(4, 1, 1, vals);
    ncnn::InstanceNorm in_layer;
    in_layer.channels = 1;
    in_layer.eps      = 1e-5f;
    in_layer.affine   = 0;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(in_layer.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    float mean = 0.f;
    for (float v : out) mean += v;
    mean /= 4;
    ASSERT_NEAR(mean, 0.f, 1e-4f);
}

void test_groupnorm_ncnn()
{
    // 4 channels, 1×1 spatial, 2 groups → pairs normalize independently
    std::vector<float> flat = { 1.f, 3.f, 5.f, 7.f };  // [c=4, h=1, w=1]
    ncnn::Mat m = make_mat(1, 1, 4, flat);
    ncnn::GroupNorm gn;
    gn.group      = 2;
    gn.channels   = 4;
    gn.eps        = 1e-5f;
    gn.affine     = 0;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(gn.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    // group0 (channels 0,1): mean=2, var=1 → normalized [-1, 1]
    ASSERT_NEAR(out[0], -1.f, 1e-4f);
    ASSERT_NEAR(out[1],  1.f, 1e-4f);
    // group1 (channels 2,3): mean=6, var=1 → normalized [-1, 1]
    ASSERT_NEAR(out[2], -1.f, 1e-4f);
    ASSERT_NEAR(out[3],  1.f, 1e-4f);
}

int main() {
    printf("=== test_norm ===\n");
    printf("\n-- Reference tests --\n");
    RUN_TEST(test_layernorm_basic);
    RUN_TEST(test_layernorm_affine);
    RUN_TEST(test_layernorm_constant_input);
    RUN_TEST(test_rmsnorm_basic);
    RUN_TEST(test_rmsnorm_with_gamma);
    RUN_TEST(test_batchnorm_basic);
    RUN_TEST(test_batchnorm_scale_bias);
    RUN_TEST(test_instancenorm);
    RUN_TEST(test_groupnorm_basic);
    RUN_TEST(test_lrn_basic);
    RUN_TEST(test_mvn_mean_only);
    RUN_TEST(test_mvn_full);
    RUN_TEST(test_normalize_l2);

    printf("\n-- Real ncnn::LayerNorm / RMSNorm / BatchNorm / InstanceNorm / GroupNorm --\n");
    RUN_TEST(test_layernorm_ncnn);
    RUN_TEST(test_layernorm_affine_ncnn);
    RUN_TEST(test_rmsnorm_ncnn);
    RUN_TEST(test_batchnorm_ncnn);
    RUN_TEST(test_instancenorm_ncnn);
    RUN_TEST(test_groupnorm_ncnn);

    print_summary("norm");
    return g_failed > 0 ? 1 : 0;
}
