// test_conv.cpp — ARM conv kernel tests
// Tests Convolution_arm, ConvolutionDepthWise_arm, Deconvolution_arm

#include "test_utils.h"
#include "ncnn_helpers.h"

#include "../conv/convolution_arm.h"
#include "../conv/convolutiondepthwise_arm.h"
#include "../conv/deconvolution_arm.h"

// ─── Reference 2-D convolution ───────────────────────────────────────────────

static TestMat ref_conv2d(const TestMat& in,
                           const std::vector<float>& weight,
                           const std::vector<float>& bias,
                           int out_c, int kh, int kw,
                           int stride_h, int stride_w,
                           int pad_top, int pad_left,
                           int dil_h = 1, int dil_w = 1) {
    int in_c = in.c, in_h = in.h, in_w = in.w;
    int out_h = (in_h + 2 * pad_top  - dil_h * (kh - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * pad_left - dil_w * (kw - 1) - 1) / stride_w + 1;
    TestMat out(out_w, out_h, out_c);
    for (int oc = 0; oc < out_c; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float sum = bias.empty() ? 0.f : bias[oc];
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int khi = 0; khi < kh; ++khi) {
                        for (int kwi = 0; kwi < kw; ++kwi) {
                            int ih = oh * stride_h - pad_top  + khi * dil_h;
                            int iw = ow * stride_w - pad_left + kwi * dil_w;
                            float px = (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                                       ? in.at(iw, ih, ic) : 0.f;
                            int widx = ((oc * in_c + ic) * kh + khi) * kw + kwi;
                            sum += px * weight[widx];
                        }
                    }
                }
                out.at(ow, oh, oc) = sum;
            }
        }
    }
    return out;
}

static TestMat ref_depthwise_conv2d(const TestMat& in,
                                     const std::vector<float>& weight,
                                     const std::vector<float>& bias,
                                     int kh, int kw, int stride_h, int stride_w,
                                     int pad_top, int pad_left) {
    int c = in.c;
    int out_h = (in.h + 2 * pad_top  - kh) / stride_h + 1;
    int out_w = (in.w + 2 * pad_left - kw) / stride_w + 1;
    TestMat out(out_w, out_h, c);
    for (int ch = 0; ch < c; ++ch) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float sum = bias.empty() ? 0.f : bias[ch];
                for (int khi = 0; khi < kh; ++khi) {
                    for (int kwi = 0; kwi < kw; ++kwi) {
                        int ih = oh * stride_h - pad_top  + khi;
                        int iw = ow * stride_w - pad_left + kwi;
                        float px = (ih >= 0 && ih < in.h && iw >= 0 && iw < in.w)
                                   ? in.at(iw, ih, ch) : 0.f;
                        sum += px * weight[(ch * kh + khi) * kw + kwi];
                    }
                }
                out.at(ow, oh, ch) = sum;
            }
        }
    }
    return out;
}

// ─── Reference-only test cases ────────────────────────────────────────────────

void test_conv2d_1x1() {
    TestMat in(3, 3, 1);
    in.fill_range();
    std::vector<float> weight = { 2.f };
    std::vector<float> bias   = { 1.f };
    TestMat out = ref_conv2d(in, weight, bias, 1, 1, 1, 1, 1, 0, 0);
    ASSERT_EQ(out.c, 1); ASSERT_EQ(out.h, 3); ASSERT_EQ(out.w, 3);
    for (int i = 0; i < 9; ++i)
        ASSERT_NEAR(out.data[i], (float)(i + 1) * 2.f + 1.f, 1e-5f);
}

void test_conv2d_3x3_no_pad() {
    TestMat in(4, 4, 1);
    in.fill_range();
    std::vector<float> weight(9, 1.f);
    std::vector<float> bias;
    TestMat out = ref_conv2d(in, weight, bias, 1, 3, 3, 1, 1, 0, 0);
    ASSERT_EQ(out.h, 2); ASSERT_EQ(out.w, 2);
    ASSERT_NEAR(out.at(0, 0, 0), 54.f, 1e-4f);
    ASSERT_NEAR(out.at(1, 0, 0), 63.f, 1e-4f);
}

void test_conv2d_stride2() {
    TestMat in(4, 4, 1);
    in.fill_range();
    std::vector<float> weight(4, 1.f);
    std::vector<float> bias;
    TestMat out = ref_conv2d(in, weight, bias, 1, 2, 2, 2, 2, 0, 0);
    ASSERT_EQ(out.h, 2); ASSERT_EQ(out.w, 2);
    ASSERT_NEAR(out.at(0, 0, 0), 14.f, 1e-5f);
    ASSERT_NEAR(out.at(1, 0, 0), 22.f, 1e-5f);
}

void test_depthwise_conv2d() {
    TestMat in(4, 4, 2);
    for (int i = 0; i < 16; ++i) in.data[i]    = (float)(i + 1);
    for (int i = 0; i < 16; ++i) in.data[16+i] = (float)(i + 1) * 2.f;
    std::vector<float> weight(2 * 9, 1.f);
    std::vector<float> bias;
    TestMat out = ref_depthwise_conv2d(in, weight, bias, 3, 3, 1, 1, 0, 0);
    ASSERT_EQ(out.c, 2); ASSERT_EQ(out.h, 2); ASSERT_EQ(out.w, 2);
    ASSERT_NEAR(out.at(0, 0, 0), 54.f, 1e-4f);
    ASSERT_NEAR(out.at(0, 0, 1), 108.f, 1e-4f);
}

// ─── Real ARM kernel tests ────────────────────────────────────────────────────

void test_conv2d_arm_1x1()
{
    TestMat in(3, 3, 1);
    in.fill_range();
    std::vector<float> weight = { 2.f };
    std::vector<float> bias   = { 1.f };

    ncnn::Mat bottom = make_mat(in.w, in.h, in.c, in.data);
    ncnn::Mat top;

    ncnn::Convolution_arm conv;
    conv.num_output       = 1;
    conv.kernel_w         = 1; conv.kernel_h = 1;
    conv.dilation_w       = 1; conv.dilation_h = 1;
    conv.stride_w         = 1; conv.stride_h = 1;
    conv.pad_left         = 0; conv.pad_right = 0;
    conv.pad_top          = 0; conv.pad_bottom = 0;
    conv.pad_value        = 0.f;
    conv.bias_term        = 1;
    conv.weight_data_size = 1;
    conv.int8_scale_term  = 0;
    conv.activation_type  = 0;
    conv.dynamic_weight   = 0;
    conv.weight_data      = make_weight(weight);
    conv.bias_data        = make_weight(bias);

    ncnn::Option opt = make_opt();
    int cret = conv.create_pipeline(opt);
    if (cret != 0) {
        printf("  (Convolution_arm create_pipeline returned %d)\n", cret);
        g_failed++;
        return;
    }
    ASSERT_EQ(conv.forward(bottom, top, opt), 0);

    TestMat ref_out = ref_conv2d(in, weight, bias, 1, 1, 1, 1, 1, 0, 0);
    std::vector<float> got; read_mat(top, got);
    ASSERT_VEC_NEAR(got, ref_out.data.data(), ref_out.total(), 1e-3f);
}

void test_conv2d_arm_3x3()
{
    TestMat in(4, 4, 1);
    in.fill_range();
    std::vector<float> weight(9, 1.f);
    std::vector<float> bias;

    ncnn::Mat bottom = make_mat(in.w, in.h, in.c, in.data);
    ncnn::Mat top;

    ncnn::Convolution_arm conv;
    conv.num_output       = 1;
    conv.kernel_w         = 3; conv.kernel_h = 3;
    conv.dilation_w       = 1; conv.dilation_h = 1;
    conv.stride_w         = 1; conv.stride_h = 1;
    conv.pad_left         = 0; conv.pad_right = 0;
    conv.pad_top          = 0; conv.pad_bottom = 0;
    conv.pad_value        = 0.f;
    conv.bias_term        = 0;
    conv.weight_data_size = 9;
    conv.int8_scale_term  = 0;
    conv.activation_type  = 0;
    conv.dynamic_weight   = 0;
    conv.weight_data      = make_weight(weight);

    ncnn::Option opt = make_opt();
    int cret = conv.create_pipeline(opt);
    if (cret != 0) {
        printf("  (Convolution_arm create_pipeline returned %d)\n", cret);
        g_failed++;
        return;
    }
    ASSERT_EQ(conv.forward(bottom, top, opt), 0);

    TestMat ref_out = ref_conv2d(in, weight, bias, 1, 3, 3, 1, 1, 0, 0);
    std::vector<float> got; read_mat(top, got);
    ASSERT_VEC_NEAR(got, ref_out.data.data(), ref_out.total(), 1e-3f);
}

void test_depthwise_conv2d_arm()
{
    TestMat in(4, 4, 2);
    in.fill_range();
    std::vector<float> weight(2 * 9, 1.f);
    std::vector<float> bias;

    ncnn::Mat bottom = make_mat(in.w, in.h, in.c, in.data);
    ncnn::Mat top;

    ncnn::ConvolutionDepthWise_arm dw;
    dw.num_output       = 2;
    dw.kernel_w         = 3; dw.kernel_h = 3;
    dw.dilation_w       = 1; dw.dilation_h = 1;
    dw.stride_w         = 1; dw.stride_h = 1;
    dw.pad_left         = 0; dw.pad_right = 0;
    dw.pad_top          = 0; dw.pad_bottom = 0;
    dw.pad_value        = 0.f;
    dw.bias_term        = 0;
    dw.weight_data_size = 2 * 9;
    dw.group            = 2;
    dw.int8_scale_term  = 0;
    dw.activation_type  = 0;
    dw.dynamic_weight   = 0;
    dw.weight_data      = make_weight(weight);

    ncnn::Option opt = make_opt();
    int cret = dw.create_pipeline(opt);
    if (cret != 0) {
        printf("  (ConvolutionDepthWise_arm create_pipeline returned %d)\n", cret);
        g_failed++;
        return;
    }
    ASSERT_EQ(dw.forward(bottom, top, opt), 0);

    ASSERT_EQ(top.c, 2);
    ASSERT_EQ(top.h, 2);
    ASSERT_EQ(top.w, 2);

    // ch0 top-left 3x3: 1+2+3+5+6+7+9+10+11 = 54
    float v00 = top.channel(0).row(0)[0];
    ASSERT_NEAR(v00, 54.f, 1e-3f);
    // ch1: starts at 17: 17+18+19+21+22+23+25+26+27 = 198
    float v10 = top.channel(1).row(0)[0];
    ASSERT_NEAR(v10, 198.f, 1e-3f);
}

int main() {
    printf("=== test_conv (ARM) ===\n");
    printf("\n-- Reference tests --\n");
    RUN_TEST(test_conv2d_1x1);
    RUN_TEST(test_conv2d_3x3_no_pad);
    RUN_TEST(test_conv2d_stride2);
    RUN_TEST(test_depthwise_conv2d);

    printf("\n-- Real ARM Convolution / ConvolutionDepthWise --\n");
    RUN_TEST(test_conv2d_arm_1x1);
    RUN_TEST(test_conv2d_arm_3x3);
    RUN_TEST(test_depthwise_conv2d_arm);

    print_summary("conv_arm");
    return g_failed > 0 ? 1 : 0;
}
