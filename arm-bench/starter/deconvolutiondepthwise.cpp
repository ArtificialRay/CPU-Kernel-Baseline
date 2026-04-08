#include "test_utils.h"
#include "ncnn_helpers.h"
#include "ref_conv.h"
#include "../../ncnn/mapped/deconvolutiondepthwise/deconvolutiondepthwise.h"
#include "../../ncnn/mapped/deconvolutiondepthwise/deconvolutiondepthwise_arm.h"

// CANDIDATE_INJECT_START
// Generic runner for DeconvolutionDepthWise (base)
static bool run_depthwise_deconv2d(int c, int in_h, int in_w,
                                    int kh, int kw, int stride_h, int stride_w,
                                    bool with_bias = false)
{
    int wsize = c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(c); for (int i = 0; i < c; ++i) bias[i] = i * 0.1f; }

    TestMat in(in_w, in_h, c); in.fill_range();
    ncnn::Mat bottom = make_mat(in.w, in.h, in.c, in.data);
    ncnn::Mat top;

    ncnn::DeconvolutionDepthWise ddw;
    ddw.num_output         = c;
    ddw.kernel_w           = kw;    ddw.kernel_h  = kh;
    ddw.dilation_w         = 1;     ddw.dilation_h = 1;
    ddw.stride_w           = stride_w; ddw.stride_h = stride_h;
    ddw.pad_left           = 0; ddw.pad_right  = 0;
    ddw.pad_top            = 0; ddw.pad_bottom = 0;
    ddw.output_pad_right   = 0; ddw.output_pad_bottom = 0;
    ddw.output_w           = 0; ddw.output_h = 0;
    ddw.bias_term          = with_bias ? 1 : 0;
    ddw.weight_data_size   = wsize;
    ddw.group              = c;
    ddw.activation_type    = 0;
    ddw.dynamic_weight     = 0;
    ddw.weight_data        = make_weight(weight);
    if (with_bias) ddw.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    int ret = ddw.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  DeconvolutionDepthWise::forward failed %d\n", ret); g_failed++; return false; }

    TestMat ref = ref_depthwise_deconv2d(in, weight, bias, kh, kw, stride_h, stride_w);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data.data(), ref.total(), 1e-3f);
    return g_failed == before;
}
// CANDIDATE_INJECT_END

// Generic runner for DeconvolutionDepthWise_arm
static bool run_depthwise_deconv2d_arm(int c, int in_h, int in_w,
                                        int kh, int kw, int stride_h, int stride_w,
                                        bool with_bias = false)
{
    int wsize = c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(c); for (int i = 0; i < c; ++i) bias[i] = i * 0.1f; }

    TestMat in(in_w, in_h, c); in.fill_range();
    ncnn::Mat bottom = make_mat(in.w, in.h, in.c, in.data);
    ncnn::Mat top;

    ncnn::DeconvolutionDepthWise_arm ddw;
    ddw.num_output         = c;
    ddw.kernel_w           = kw;    ddw.kernel_h  = kh;
    ddw.dilation_w         = 1;     ddw.dilation_h = 1;
    ddw.stride_w           = stride_w; ddw.stride_h = stride_h;
    ddw.pad_left           = 0; ddw.pad_right  = 0;
    ddw.pad_top            = 0; ddw.pad_bottom = 0;
    ddw.output_pad_right   = 0; ddw.output_pad_bottom = 0;
    ddw.output_w           = 0; ddw.output_h = 0;
    ddw.bias_term          = with_bias ? 1 : 0;
    ddw.weight_data_size   = wsize;
    ddw.group              = c;
    ddw.activation_type    = 0;
    ddw.dynamic_weight     = 0;
    ddw.weight_data        = make_weight(weight);
    if (with_bias) ddw.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    if (ddw.create_pipeline(opt) != 0) {
        fprintf(stderr, "  DeconvolutionDepthWise_arm::create_pipeline failed\n");
        g_failed++; return false;
    }
    int ret = ddw.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  DeconvolutionDepthWise_arm::forward failed %d\n", ret); g_failed++; return false; }

    TestMat ref = ref_depthwise_deconv2d(in, weight, bias, kh, kw, stride_h, stride_w);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data.data(), ref.total(), 1e-3f);
    return g_failed == before;
}
// CANDIDATE_TESTCASE_START
// ── DeconvolutionDepthWise (base) ─────────────────────────────────
void test_dw_deconv_base_2x2_s2() {
    ASSERT_TRUE(run_depthwise_deconv2d(2, 3, 3, 2, 2, 2, 2));
    ASSERT_TRUE(run_depthwise_deconv2d(4, 4, 4, 2, 2, 2, 2));
}

void test_dw_deconv_base_3x3_s1() {
    ASSERT_TRUE(run_depthwise_deconv2d(2, 4, 4, 3, 3, 1, 1));
    ASSERT_TRUE(run_depthwise_deconv2d(4, 5, 5, 3, 3, 1, 1));
}
// CANDIDATE_TESTCASE_END

// BASELINE_TESTCASE_START
// ── DeconvolutionDepthWise_arm ────────────────────────────────────
void test_dw_deconv_arm_2x2_s2() {
    ASSERT_TRUE(run_depthwise_deconv2d_arm(2, 3, 3, 2, 2, 2, 2));
    ASSERT_TRUE(run_depthwise_deconv2d_arm(4, 4, 4, 2, 2, 2, 2));
}

void test_dw_deconv_arm_3x3_s1() {
    ASSERT_TRUE(run_depthwise_deconv2d_arm(2, 4, 4, 3, 3, 1, 1));
    ASSERT_TRUE(run_depthwise_deconv2d_arm(4, 5, 5, 3, 3, 1, 1));
}
// BASELINE_TESTCASE_END