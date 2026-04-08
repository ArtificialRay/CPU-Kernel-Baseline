#include "test_utils.h"
#include "ncnn_helpers.h"
#include "ref_conv.h"
// ── Base class headers ────────────────────────────────────────────────────────
#include "../../ncnn/mapped/convolution/convolution.h"

// ── ARM class headers ─────────────────────────────────────────────────────────
#include "../../ncnn/mapped/convolution/convolution_arm.h"


// CANDIDATE_INJECT_START
static bool run_conv2d(int in_c, int out_c, int in_h, int in_w,
                        int kh, int kw, int stride_h, int stride_w,
                        int pad_top, int pad_left,
                        int dil_h = 1, int dil_w = 1,
                        bool with_bias = false)
{
    int wsize = out_c * in_c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.5f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.1f; }

    TestMat in(in_w, in_h, in_c); in.fill_range();
    ncnn::Mat bottom = make_mat(in.w, in.h, in.c, in.data);
    ncnn::Mat top;

    ncnn::Convolution conv;
    conv.num_output       = out_c;
    conv.kernel_w         = kw;    conv.kernel_h  = kh;
    conv.dilation_w       = dil_w; conv.dilation_h = dil_h;
    conv.stride_w         = stride_w; conv.stride_h = stride_h;
    conv.pad_left         = pad_left; conv.pad_right  = pad_left;
    conv.pad_top          = pad_top;  conv.pad_bottom = pad_top;
    conv.pad_value        = 0.f;
    conv.bias_term        = with_bias ? 1 : 0;
    conv.weight_data_size = wsize;
    conv.int8_scale_term  = 0;
    conv.activation_type  = 0;
    conv.dynamic_weight   = 0;
    conv.weight_data      = make_weight(weight);
    if (with_bias) conv.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    int ret = conv.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Convolution::forward failed %d\n", ret); g_failed++; return false; }

    TestMat ref = ref_conv2d(in, weight, bias, out_c, kh, kw, stride_h, stride_w, pad_top, pad_left, dil_h, dil_w);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data.data(), ref.total(), 1e-3f);
    return g_failed == before;
}
// CANDIDATE_INJECT_END

// BASELINE_INJECT_START
// Generic runner for Convolution_arm
static bool run_conv2d_arm(int in_c, int out_c, int in_h, int in_w,
                             int kh, int kw, int stride_h, int stride_w,
                             int pad_top, int pad_left,
                             int dil_h = 1, int dil_w = 1,
                             bool with_bias = false)
{
    int wsize = out_c * in_c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.5f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.1f; }

    TestMat in(in_w, in_h, in_c); in.fill_range();
    ncnn::Mat bottom = make_mat(in.w, in.h, in.c, in.data);
    ncnn::Mat top;

    ncnn::Convolution_arm conv;
    conv.num_output       = out_c;
    conv.kernel_w         = kw;    conv.kernel_h  = kh;
    conv.dilation_w       = dil_w; conv.dilation_h = dil_h;
    conv.stride_w         = stride_w; conv.stride_h = stride_h;
    conv.pad_left         = pad_left; conv.pad_right  = pad_left;
    conv.pad_top          = pad_top;  conv.pad_bottom = pad_top;
    conv.pad_value        = 0.f;
    conv.bias_term        = with_bias ? 1 : 0;
    conv.weight_data_size = wsize;
    conv.int8_scale_term  = 0;
    conv.activation_type  = 0;
    conv.dynamic_weight   = 0;
    conv.weight_data      = make_weight(weight);
    if (with_bias) conv.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    if (conv.create_pipeline(opt) != 0) {
        fprintf(stderr, "  Convolution_arm::create_pipeline failed\n");
        g_failed++; return false;
    }
    int ret = conv.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Convolution_arm::forward failed %d\n", ret); g_failed++; return false; }

    TestMat ref = ref_conv2d(in, weight, bias, out_c, kh, kw, stride_h, stride_w, pad_top, pad_left, dil_h, dil_w);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data.data(), ref.total(), 1e-3f);
    return g_failed == before;
}
// BASELINE_INJECT_END


// CANDIDATE_TESTCASE_START
// ── Convolution (base) ────────────────────────────────────────────
void test_conv_base_1x1_s1() {
    ASSERT_TRUE(run_conv2d(1, 1, 4, 4, 1, 1, 1, 1, 0, 0));
    ASSERT_TRUE(run_conv2d(3, 4, 5, 5, 1, 1, 1, 1, 0, 0));
    ASSERT_TRUE(run_conv2d(8, 8, 7, 7, 1, 1, 1, 1, 0, 0));
}

void test_conv_base_3x3_s1() {
    ASSERT_TRUE(run_conv2d(1, 1, 5, 5, 3, 3, 1, 1, 0, 0));
    ASSERT_TRUE(run_conv2d(3, 4, 8, 8, 3, 3, 1, 1, 1, 1));
    ASSERT_TRUE(run_conv2d(4, 8, 10, 10, 3, 3, 1, 1, 1, 1));
}

void test_conv_base_3x3_s2() {
    ASSERT_TRUE(run_conv2d(1, 1, 8, 8, 3, 3, 2, 2, 0, 0));
    ASSERT_TRUE(run_conv2d(4, 8, 12, 12, 3, 3, 2, 2, 1, 1));
}

void test_conv_base_5x5() {
    ASSERT_TRUE(run_conv2d(3, 4, 10, 10, 5, 5, 1, 1, 2, 2));
}

void test_conv_base_dilation() {
    ASSERT_TRUE(run_conv2d(2, 4, 8, 8, 3, 3, 1, 1, 0, 0, 2, 2));
}

void test_conv_base_bias() {
    ASSERT_TRUE(run_conv2d(4, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true));
}
// CANDIDATE_TESTCASE_END

// BASELINE_TESTCASE_START
// ── Convolution_arm ───────────────────────────────────────────────
void test_conv_arm_1x1_s1() {
    ASSERT_TRUE(run_conv2d_arm(1, 1, 4, 4, 1, 1, 1, 1, 0, 0));
    ASSERT_TRUE(run_conv2d_arm(3, 4, 5, 5, 1, 1, 1, 1, 0, 0));
    ASSERT_TRUE(run_conv2d_arm(8, 8, 7, 7, 1, 1, 1, 1, 0, 0));
}

void test_conv_arm_3x3_s1() {
    ASSERT_TRUE(run_conv2d_arm(1, 1, 5, 5, 3, 3, 1, 1, 0, 0));
    ASSERT_TRUE(run_conv2d_arm(3, 4, 8, 8, 3, 3, 1, 1, 1, 1));
    ASSERT_TRUE(run_conv2d_arm(4, 8, 10, 10, 3, 3, 1, 1, 1, 1));
}

void test_conv_arm_3x3_s2() {
    ASSERT_TRUE(run_conv2d_arm(1, 1, 8, 8, 3, 3, 2, 2, 0, 0));
    ASSERT_TRUE(run_conv2d_arm(4, 8, 12, 12, 3, 3, 2, 2, 1, 1));
}

void test_conv_arm_5x5() {
    ASSERT_TRUE(run_conv2d_arm(3, 4, 10, 10, 5, 5, 1, 1, 2, 2));
}

void test_conv_arm_bias() {
    ASSERT_TRUE(run_conv2d_arm(4, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true));
}
// BASELINE_TESTCASE_END