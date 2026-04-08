#include "test_utils.h"
#include "ncnn_helpers.h"

#include "../../ncnn/mapped/deconvolution/deconvolution.h"
#include "../../ncnn/mapped/deconvolution/deconvolution_arm.h


// CANDIDATE_INJECT_START
// Generic runner for Deconvolution (base)
// Weight layout for deconvolution in ncnn: [in_c, out_c, kh, kw]
static bool run_deconv2d(int in_c, int out_c, int in_h, int in_w,
                          int kh, int kw, int stride_h, int stride_w,
                          bool with_bias = false)
{
    int wsize = in_c * out_c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.1f; }

    TestMat in(in_w, in_h, in_c); in.fill_range();
    ncnn::Mat bottom = make_mat(in.w, in.h, in.c, in.data);
    ncnn::Mat top;

    ncnn::Deconvolution deconv;
    deconv.num_output         = out_c;
    deconv.kernel_w           = kw;    deconv.kernel_h  = kh;
    deconv.dilation_w         = 1;     deconv.dilation_h = 1;
    deconv.stride_w           = stride_w; deconv.stride_h = stride_h;
    deconv.pad_left           = 0; deconv.pad_right  = 0;
    deconv.pad_top            = 0; deconv.pad_bottom = 0;
    deconv.output_pad_right   = 0; deconv.output_pad_bottom = 0;
    deconv.output_w           = 0; deconv.output_h = 0;
    deconv.bias_term          = with_bias ? 1 : 0;
    deconv.weight_data_size   = wsize;
    deconv.activation_type    = 0;
    deconv.dynamic_weight     = 0;
    deconv.weight_data        = make_weight(weight);
    if (with_bias) deconv.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    int ret = deconv.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Deconvolution::forward failed %d\n", ret); g_failed++; return false; }

    TestMat ref = ref_deconv2d(in, weight, bias, in_c, out_c, kh, kw, stride_h, stride_w);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data.data(), ref.total(), 1e-3f);
    return g_failed == before;
}
// CANDIDATE_INJECT_END

// Generic runner for Deconvolution_arm
static bool run_deconv2d_arm(int in_c, int out_c, int in_h, int in_w,
                               int kh, int kw, int stride_h, int stride_w,
                               bool with_bias = false)
{
    int wsize = in_c * out_c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.1f; }

    TestMat in(in_w, in_h, in_c); in.fill_range();
    ncnn::Mat bottom = make_mat(in.w, in.h, in.c, in.data);
    ncnn::Mat top;

    ncnn::Deconvolution_arm deconv;
    deconv.num_output         = out_c;
    deconv.kernel_w           = kw;    deconv.kernel_h  = kh;
    deconv.dilation_w         = 1;     deconv.dilation_h = 1;
    deconv.stride_w           = stride_w; deconv.stride_h = stride_h;
    deconv.pad_left           = 0; deconv.pad_right  = 0;
    deconv.pad_top            = 0; deconv.pad_bottom = 0;
    deconv.output_pad_right   = 0; deconv.output_pad_bottom = 0;
    deconv.output_w           = 0; deconv.output_h = 0;
    deconv.bias_term          = with_bias ? 1 : 0;
    deconv.weight_data_size   = wsize;
    deconv.activation_type    = 0;
    deconv.dynamic_weight     = 0;
    deconv.weight_data        = make_weight(weight);
    if (with_bias) deconv.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    if (deconv.create_pipeline(opt) != 0) {
        fprintf(stderr, "  Deconvolution_arm::create_pipeline failed\n");
        g_failed++; return false;
    }
    int ret = deconv.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Deconvolution_arm::forward failed %d\n", ret); g_failed++; return false; }

    TestMat ref = ref_deconv2d(in, weight, bias, in_c, out_c, kh, kw, stride_h, stride_w);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data.data(), ref.total(), 1e-3f);
    return g_failed == before;
}

// CANDIDATE_TESTCASE_START
// ── Deconvolution (base) ──────────────────────────────────────────
void test_deconv_base_2x2_s2() {
    ASSERT_TRUE(run_deconv2d(1, 1, 3, 3, 2, 2, 2, 2));
    ASSERT_TRUE(run_deconv2d(2, 4, 4, 4, 2, 2, 2, 2));
}

void test_deconv_base_3x3_s1() {
    ASSERT_TRUE(run_deconv2d(1, 1, 4, 4, 3, 3, 1, 1));
    ASSERT_TRUE(run_deconv2d(3, 4, 5, 5, 3, 3, 1, 1));
}

void test_deconv_base_bias() {
    ASSERT_TRUE(run_deconv2d(2, 4, 4, 4, 3, 3, 1, 1, true));
}
// CANDIDATE_TESTCASE_END
// BASELINE_TESTCASE_START
// ── Deconvolution_arm ─────────────────────────────────────────────
void test_deconv_arm_2x2_s2() {
    ASSERT_TRUE(run_deconv2d_arm(1, 1, 3, 3, 2, 2, 2, 2));
    ASSERT_TRUE(run_deconv2d_arm(2, 4, 4, 4, 2, 2, 2, 2));
}

void test_deconv_arm_3x3_s1() {
    ASSERT_TRUE(run_deconv2d_arm(1, 1, 4, 4, 3, 3, 1, 1));
    ASSERT_TRUE(run_deconv2d_arm(3, 4, 5, 5, 3, 3, 1, 1));
}

void test_deconv_arm_bias() {
    ASSERT_TRUE(run_deconv2d_arm(2, 4, 4, 4, 3, 3, 1, 1, true));
}
// BASELINE_TESTCASE_END