// test_mapped_ip.cpp
// Tests for mapped InnerProduct and Requantize kernel implementations
// (both base c-partially-optimized and ARM variants).

#include "test_utils.h"
#include "ncnn_helpers.h"

#include "../mapped/innerproduct/innerproduct.h"
#include "../mapped/innerproduct/innerproduct_arm.h"
#include "../mapped/requantize/requantize.h"
#include "../mapped/requantize/requantize_arm.h"

// ═══════════════════════════════════════════════════════════════════
// ── Reference implementations ────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════

// Fully-connected (inner product) reference
// Weight layout: [out, in] flat
static std::vector<float> ref_innerproduct(const std::vector<float>& in,
                                            const std::vector<float>& weight,
                                            const std::vector<float>& bias,
                                            int in_size, int out_size)
{
    std::vector<float> out(out_size);
    for (int o = 0; o < out_size; ++o) {
        float sum = bias.empty() ? 0.f : bias[o];
        for (int i = 0; i < in_size; ++i)
            sum += in[i] * weight[o * in_size + i];
        out[o] = sum;
    }
    return out;
}

// Requantize reference: int32 → int8
// scale_in: per-channel or single, scale_out: per-channel or single
static std::vector<int8_t> ref_requantize(const std::vector<int32_t>& in_int32,
                                           const std::vector<float>& scale_in,
                                           const std::vector<float>& scale_out,
                                           const std::vector<float>& bias,
                                           int channels, int wh)
{
    std::vector<int8_t> out(channels * wh);
    for (int c = 0; c < channels; ++c) {
        float si = (scale_in.size() == 1) ? scale_in[0] : scale_in[c];
        float so = (scale_out.size() == 1) ? scale_out[0] : scale_out[c];
        float b  = bias.empty() ? 0.f : bias[c];
        for (int i = 0; i < wh; ++i) {
            float v = in_int32[c * wh + i] * si + b;
            v *= so;
            int iv = (int)roundf(v);
            if (iv >  127) iv =  127;
            if (iv < -127) iv = -127;
            out[c * wh + i] = (int8_t)iv;
        }
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════
// ── Kernel runner helpers ─────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════

static bool run_innerproduct(int in_size, int out_size, bool with_bias = false)
{
    std::vector<float> weight = make_weights(out_size * in_size);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_size); for (int i = 0; i < out_size; ++i) bias[i] = i * 0.05f; }

    std::vector<float> in_flat(in_size);
    for (int i = 0; i < in_size; ++i) in_flat[i] = (i + 1) * 0.1f;

    ncnn::Mat bottom = make_mat_1d(in_flat);
    ncnn::Mat top;

    ncnn::InnerProduct ip;
    ip.num_output       = out_size;
    ip.bias_term        = with_bias ? 1 : 0;
    ip.weight_data_size = out_size * in_size;
    ip.int8_scale_term  = 0;
    ip.activation_type  = 0;
    ip.weight_data      = make_weight(weight);
    if (with_bias) ip.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    int ret = ip.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  InnerProduct::forward failed %d\n", ret); g_failed++; return false; }

    std::vector<float> ref = ref_innerproduct(in_flat, weight, bias, in_size, out_size);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data(), out_size, 1e-3f);
    return g_failed == before;
}

static bool run_innerproduct_arm(int in_size, int out_size, bool with_bias = false)
{
    std::vector<float> weight = make_weights(out_size * in_size);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_size); for (int i = 0; i < out_size; ++i) bias[i] = i * 0.05f; }

    std::vector<float> in_flat(in_size);
    for (int i = 0; i < in_size; ++i) in_flat[i] = (i + 1) * 0.1f;

    ncnn::Mat bottom = make_mat_1d(in_flat);
    ncnn::Mat top;

    ncnn::InnerProduct_arm ip;
    ip.num_output       = out_size;
    ip.bias_term        = with_bias ? 1 : 0;
    ip.weight_data_size = out_size * in_size;
    ip.int8_scale_term  = 0;
    ip.activation_type  = 0;
    ip.weight_data      = make_weight(weight);
    if (with_bias) ip.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    if (ip.create_pipeline(opt) != 0) {
        fprintf(stderr, "  InnerProduct_arm::create_pipeline failed\n");
        g_failed++; return false;
    }
    int ret = ip.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  InnerProduct_arm::forward failed %d\n", ret); g_failed++; return false; }

    std::vector<float> ref = ref_innerproduct(in_flat, weight, bias, in_size, out_size);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data(), out_size, 1e-3f);
    return g_failed == before;
}

// Helper: make int32 Mat for requantize input
static ncnn::Mat make_mat_int32(int w_, int h_, int c_, const std::vector<int32_t>& flat)
{
    ncnn::Mat m;
    m.create(w_, h_, c_, (size_t)4u, (ncnn::Allocator*)0);
    for (int cc = 0; cc < c_; ++cc)
        for (int hh = 0; hh < h_; ++hh) {
            int* dst = (int*)m.channel(cc).row(hh);
            const int* src = flat.data() + cc * h_ * w_ + hh * w_;
            memcpy(dst, src, w_ * sizeof(int));
        }
    return m;
}

static bool run_requantize(int channels, int wh,
                             bool per_channel_scale_in = false,
                             bool per_channel_scale_out = false,
                             bool with_bias = false)
{
    // Build int32 input
    std::vector<int32_t> in_int32(channels * wh);
    for (int i = 0; i < (int)in_int32.size(); ++i) in_int32[i] = (i % 256) - 128;

    int si_size = per_channel_scale_in  ? channels : 1;
    int so_size = per_channel_scale_out ? channels : 1;
    std::vector<float> scale_in(si_size), scale_out(so_size);
    for (int i = 0; i < si_size; ++i) scale_in[i]  = 0.01f + i * 0.001f;
    for (int i = 0; i < so_size; ++i) scale_out[i] = 1.0f  + i * 0.05f;

    std::vector<float> bias;
    if (with_bias) { bias.resize(channels); for (int i = 0; i < channels; ++i) bias[i] = i * 0.1f; }

    // Build ncnn Mats
    ncnn::Mat bottom = make_mat_int32(wh, 1, channels, in_int32);
    ncnn::Mat top;

    ncnn::Requantize req;
    req.scale_in_data_size  = si_size;
    req.scale_out_data_size = so_size;
    req.bias_data_size      = with_bias ? channels : 0;
    req.activation_type     = 0;
    req.scale_in_data       = make_weight(scale_in);
    req.scale_out_data      = make_weight(scale_out);
    if (with_bias) req.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    int ret = req.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Requantize::forward failed %d\n", ret); g_failed++; return false; }

    std::vector<int8_t> ref = ref_requantize(in_int32, scale_in, scale_out, bias, channels, wh);
    std::vector<int8_t> got; read_mat_int8(top, got);
    int before = g_failed;
    for (int i = 0; i < (int)ref.size(); ++i) {
        if (got[i] != ref[i] && abs((int)got[i] - (int)ref[i]) > 1) {
            fprintf(stderr, "  FAIL requantize[%d]: got=%d ref=%d\n", i, (int)got[i], (int)ref[i]);
            g_failed++;
            break;
        }
    }
    return g_failed == before;
}

static bool run_requantize_arm(int channels, int wh,
                                bool per_channel_scale_in = false,
                                bool per_channel_scale_out = false,
                                bool with_bias = false)
{
    std::vector<int32_t> in_int32(channels * wh);
    for (int i = 0; i < (int)in_int32.size(); ++i) in_int32[i] = (i % 256) - 128;

    int si_size = per_channel_scale_in  ? channels : 1;
    int so_size = per_channel_scale_out ? channels : 1;
    std::vector<float> scale_in(si_size), scale_out(so_size);
    for (int i = 0; i < si_size; ++i) scale_in[i]  = 0.01f + i * 0.001f;
    for (int i = 0; i < so_size; ++i) scale_out[i] = 1.0f  + i * 0.05f;

    std::vector<float> bias;
    if (with_bias) { bias.resize(channels); for (int i = 0; i < channels; ++i) bias[i] = i * 0.1f; }

    ncnn::Mat bottom = make_mat_int32(wh, 1, channels, in_int32);
    ncnn::Mat top;

    ncnn::Requantize_arm req;
    req.scale_in_data_size  = si_size;
    req.scale_out_data_size = so_size;
    req.bias_data_size      = with_bias ? channels : 0;
    req.activation_type     = 0;
    req.scale_in_data       = make_weight(scale_in);
    req.scale_out_data      = make_weight(scale_out);
    if (with_bias) req.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    if (req.create_pipeline(opt) != 0) {
        fprintf(stderr, "  Requantize_arm::create_pipeline failed\n");
        g_failed++; return false;
    }
    int ret = req.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Requantize_arm::forward failed %d\n", ret); g_failed++; return false; }

    std::vector<int8_t> ref = ref_requantize(in_int32, scale_in, scale_out, bias, channels, wh);
    std::vector<int8_t> got; read_mat_int8(top, got);
    int before = g_failed;
    for (int i = 0; i < (int)ref.size(); ++i) {
        if (got[i] != ref[i] && abs((int)got[i] - (int)ref[i]) > 1) {
            fprintf(stderr, "  FAIL requantize_arm[%d]: got=%d ref=%d\n", i, (int)got[i], (int)ref[i]);
            g_failed++;
            break;
        }
    }
    return g_failed == before;
}

// ═══════════════════════════════════════════════════════════════════
// ── Test cases ────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════

void test_ip_base_small()    { ASSERT_TRUE(run_innerproduct(4, 4)); }
void test_ip_base_medium()   { ASSERT_TRUE(run_innerproduct(16, 8)); ASSERT_TRUE(run_innerproduct(64, 32)); }
void test_ip_base_large()    { ASSERT_TRUE(run_innerproduct(128, 64)); }
void test_ip_base_bias()     { ASSERT_TRUE(run_innerproduct(16, 8, true)); }

void test_ip_arm_small()     { ASSERT_TRUE(run_innerproduct_arm(4, 4)); }
void test_ip_arm_medium()    { ASSERT_TRUE(run_innerproduct_arm(16, 8)); ASSERT_TRUE(run_innerproduct_arm(64, 32)); }
void test_ip_arm_large()     { ASSERT_TRUE(run_innerproduct_arm(128, 64)); }
void test_ip_arm_bias()      { ASSERT_TRUE(run_innerproduct_arm(16, 8, true)); }

void test_req_base_simple()  { ASSERT_TRUE(run_requantize(4, 8)); }
void test_req_base_per_ch()  { ASSERT_TRUE(run_requantize(8, 4, true, true)); }
void test_req_base_bias()    { ASSERT_TRUE(run_requantize(4, 8, false, false, true)); }
void test_req_base_sizes()   {
    ASSERT_TRUE(run_requantize(1,  16));
    ASSERT_TRUE(run_requantize(16,  1));
    ASSERT_TRUE(run_requantize(8,   8, true, false));
}

void test_req_arm_simple()   { ASSERT_TRUE(run_requantize_arm(4, 8)); }
void test_req_arm_per_ch()   { ASSERT_TRUE(run_requantize_arm(8, 4, true, true)); }
void test_req_arm_bias()     { ASSERT_TRUE(run_requantize_arm(4, 8, false, false, true)); }
void test_req_arm_sizes()    {
    ASSERT_TRUE(run_requantize_arm(1,  16));
    ASSERT_TRUE(run_requantize_arm(16,  1));
    ASSERT_TRUE(run_requantize_arm(8,   8, true, false));
}

int main()
{
    printf("=== test_mapped_ip ===\n");

    printf("\n-- InnerProduct (base) --\n");
    RUN_TEST(test_ip_base_small);
    RUN_TEST(test_ip_base_medium);
    RUN_TEST(test_ip_base_large);
    RUN_TEST(test_ip_base_bias);

    printf("\n-- InnerProduct_arm --\n");
    RUN_TEST(test_ip_arm_small);
    RUN_TEST(test_ip_arm_medium);
    RUN_TEST(test_ip_arm_large);
    RUN_TEST(test_ip_arm_bias);

    printf("\n-- Requantize (base) --\n");
    RUN_TEST(test_req_base_simple);
    RUN_TEST(test_req_base_per_ch);
    RUN_TEST(test_req_base_bias);
    RUN_TEST(test_req_base_sizes);

    printf("\n-- Requantize_arm --\n");
    RUN_TEST(test_req_arm_simple);
    RUN_TEST(test_req_arm_per_ch);
    RUN_TEST(test_req_arm_bias);
    RUN_TEST(test_req_arm_sizes);

    print_summary("mapped_ip");
    return g_failed > 0 ? 1 : 0;
}
