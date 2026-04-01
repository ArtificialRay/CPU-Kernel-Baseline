// test_conv.cpp
// Tests for conv/:
//   convolution (2D), convolutiondepthwise, deconvolution (transpose conv),
//   convolution1d, convolution3d, deconvolutiondepthwise, deformableconv2d
//
// Section 1: reference-only tests
// Section 2: real ncnn kernel tests (linked via conv_impl + ncnn_stub)

#include "test_utils.h"

// ── Real ncnn kernel headers ──────────────────────────────────────────────────
#include "ncnn_helpers.h"
#include "../conv/convolution.h"
#include "../conv/convolutiondepthwise.h"
#include "../conv/deconvolution.h"

// ─── Reference 2-D convolution (NCHW, no batch dim, float) ──────────────────
//   input:   [in_c, in_h, in_w]
//   weight:  [out_c, in_c, kh, kw]
//   output:  [out_c, out_h, out_w]
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

// ─── Reference depth-wise convolution ────────────────────────────────────────
//   weight: [in_c, 1, kh, kw]  (one kernel per channel)
static TestMat ref_depthwise_conv2d(const TestMat& in,
                                     const std::vector<float>& weight,
                                     const std::vector<float>& bias,
                                     int kh, int kw,
                                     int stride_h, int stride_w,
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

// ─── Reference transpose (deconvolution) ─────────────────────────────────────
//   weight: [in_c, out_c, kh, kw]
static TestMat ref_deconv2d(const TestMat& in,
                              const std::vector<float>& weight,
                              const std::vector<float>& bias,
                              int out_c, int kh, int kw,
                              int stride_h, int stride_w) {
    int in_c = in.c;
    int out_h = (in.h - 1) * stride_h + kh;
    int out_w = (in.w - 1) * stride_w + kw;
    TestMat out(out_w, out_h, out_c);
    if (!bias.empty())
        for (int oc = 0; oc < out_c; ++oc)
            for (int i = 0; i < out_h * out_w; ++i)
                out.channel_ptr(oc)[i] = bias[oc];

    for (int ic = 0; ic < in_c; ++ic) {
        for (int ih = 0; ih < in.h; ++ih) {
            for (int iw = 0; iw < in.w; ++iw) {
                float pv = in.at(iw, ih, ic);
                for (int oc = 0; oc < out_c; ++oc) {
                    for (int khi = 0; khi < kh; ++khi) {
                        for (int kwi = 0; kwi < kw; ++kwi) {
                            int oh = ih * stride_h + khi;
                            int ow = iw * stride_w + kwi;
                            int widx = ((ic * out_c + oc) * kh + khi) * kw + kwi;
                            out.at(ow, oh, oc) += pv * weight[widx];
                        }
                    }
                }
            }
        }
    }
    return out;
}

// ─── Reference 1-D convolution ───────────────────────────────────────────────
static TestMat ref_conv1d(const TestMat& in,          // [in_c, 1, in_w]
                            const std::vector<float>& weight,
                            const std::vector<float>& bias,
                            int out_c, int kw, int stride = 1, int pad = 0) {
    int in_c = in.c, in_w = in.w;
    int out_w = (in_w + 2 * pad - kw) / stride + 1;
    TestMat out(out_w, 1, out_c);
    for (int oc = 0; oc < out_c; ++oc) {
        for (int ow = 0; ow < out_w; ++ow) {
            float sum = bias.empty() ? 0.f : bias[oc];
            for (int ic = 0; ic < in_c; ++ic) {
                for (int kwi = 0; kwi < kw; ++kwi) {
                    int iw = ow * stride - pad + kwi;
                    float px = (iw >= 0 && iw < in_w) ? in.at(iw, 0, ic) : 0.f;
                    sum += px * weight[(oc * in_c + ic) * kw + kwi];
                }
            }
            out.at(ow, 0, oc) = sum;
        }
    }
    return out;
}

// ─── Test cases ──────────────────────────────────────────────────────────────

void test_conv2d_1x1() {
    // 1-channel 3x3 input, 1x1 kernel, 1 output channel: basically scalar mult
    TestMat in(3, 3, 1);
    in.fill_range(); // 1..9
    std::vector<float> weight = { 2.f };  // single weight
    std::vector<float> bias   = { 1.f };
    TestMat out = ref_conv2d(in, weight, bias, 1, 1, 1, 1, 1, 0, 0);
    ASSERT_EQ(out.c, 1); ASSERT_EQ(out.h, 3); ASSERT_EQ(out.w, 3);
    // each output = input * 2 + 1
    for (int i = 0; i < 9; ++i)
        ASSERT_NEAR(out.data[i], (float)(i + 1) * 2.f + 1.f, 1e-5f);
}

void test_conv2d_3x3_no_pad() {
    // 1-channel 4x4 input, 3x3 kernel → 2x2 output
    TestMat in(4, 4, 1);
    in.fill_range();  // 1..16
    // all-ones kernel: output[i,j] = sum of 3x3 patch
    std::vector<float> weight(9, 1.f);
    std::vector<float> bias;
    TestMat out = ref_conv2d(in, weight, bias, 1, 3, 3, 1, 1, 0, 0);
    ASSERT_EQ(out.h, 2); ASSERT_EQ(out.w, 2);
    // top-left 3x3: 1+2+3+5+6+7+9+10+11 = 54
    ASSERT_NEAR(out.at(0, 0, 0), 54.f, 1e-4f);
    // top-right 3x3: 2+3+4+6+7+8+10+11+12 = 63
    ASSERT_NEAR(out.at(1, 0, 0), 63.f, 1e-4f);
}

void test_conv2d_with_bias() {
    TestMat in(2, 2, 1);
    in.data = { 1.f, 2.f, 3.f, 4.f };
    std::vector<float> weight = { 1.f, 0.f, 0.f, 1.f }; // diagonal 2x2 kernel
    std::vector<float> bias = { 5.f };
    TestMat out = ref_conv2d(in, weight, bias, 1, 2, 2, 1, 1, 0, 0);
    ASSERT_EQ(out.h, 1); ASSERT_EQ(out.w, 1);
    // 1*1 + 2*0 + 3*0 + 4*1 + 5 = 10
    ASSERT_NEAR(out.at(0, 0, 0), 10.f, 1e-5f);
}

void test_conv2d_stride2() {
    TestMat in(4, 4, 1);
    in.fill_range();
    std::vector<float> weight(4, 1.f);  // 2x2 all-ones
    std::vector<float> bias;
    TestMat out = ref_conv2d(in, weight, bias, 1, 2, 2, 2, 2, 0, 0);
    ASSERT_EQ(out.h, 2); ASSERT_EQ(out.w, 2);
    // top-left 2x2: 1+2+5+6 = 14
    ASSERT_NEAR(out.at(0, 0, 0), 14.f, 1e-5f);
    // top-right 2x2: 3+4+7+8 = 22
    ASSERT_NEAR(out.at(1, 0, 0), 22.f, 1e-5f);
}

void test_conv2d_dilation() {
    // 5x5 input, 2x2 dilated(2) kernel → no padding → valid output 3x3
    TestMat in(5, 5, 1);
    in.fill_range();  // 1..25
    std::vector<float> weight = { 1.f, 0.f, 0.f, 1.f }; // identity in diag
    std::vector<float> bias;
    TestMat out = ref_conv2d(in, weight, bias, 1, 2, 2, 1, 1, 0, 0, 2, 2);
    ASSERT_EQ(out.h, 3); ASSERT_EQ(out.w, 3);
    // out[0,0] = in[0,0]*1 + in[0,2]*0 + in[2,0]*0 + in[2,2]*1 = 1 + 13 = 14
    ASSERT_NEAR(out.at(0, 0, 0), 1.f + 13.f, 1e-5f);
}

void test_conv2d_multi_channel() {
    // 2-channel 2x2 → 1 output channel
    TestMat in(2, 2, 2);
    // ch0: [1,2,3,4], ch1: [5,6,7,8]
    in.data = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };
    // weight [out_c=1, in_c=2, kh=1, kw=1]: w_ch0=1, w_ch1=-1
    std::vector<float> weight = { 1.f, -1.f };
    std::vector<float> bias;
    TestMat out = ref_conv2d(in, weight, bias, 1, 1, 1, 1, 1, 0, 0);
    ASSERT_EQ(out.c, 1); ASSERT_EQ(out.h, 2); ASSERT_EQ(out.w, 2);
    // out[y,x] = in_ch0[y,x] - in_ch1[y,x]
    float expected[] = { 1-5.f, 2-6.f, 3-7.f, 4-8.f };
    ASSERT_VEC_NEAR(out.data, expected, 4, 1e-5f);
}

void test_depthwise_conv2d() {
    TestMat in(4, 4, 2);
    for (int i = 0; i < 16; ++i) in.data[i]    = (float)(i + 1);  // ch0: 1..16
    for (int i = 0; i < 16; ++i) in.data[16+i] = (float)(i + 1) * 2.f; // ch1: 2..32
    // 3x3 all-ones kernels for each channel
    std::vector<float> weight(2 * 9, 1.f);
    std::vector<float> bias;
    TestMat out = ref_depthwise_conv2d(in, weight, bias, 3, 3, 1, 1, 0, 0);
    ASSERT_EQ(out.c, 2); ASSERT_EQ(out.h, 2); ASSERT_EQ(out.w, 2);
    // ch0 top-left 3x3: 1+2+3+5+6+7+9+10+11 = 54
    ASSERT_NEAR(out.at(0, 0, 0), 54.f, 1e-4f);
    // ch1 is doubled: 108
    ASSERT_NEAR(out.at(0, 0, 1), 108.f, 1e-4f);
}

void test_deconv2d_stride2() {
    // 2x2 input, stride=2 → 4x4 output
    TestMat in(2, 2, 1);
    in.data = { 1.f, 2.f, 3.f, 4.f };
    std::vector<float> weight(4, 1.f);  // 2x2 all-ones, 1→1 channel
    std::vector<float> bias;
    TestMat out = ref_deconv2d(in, weight, bias, 1, 2, 2, 2, 2);
    ASSERT_EQ(out.h, 4); ASSERT_EQ(out.w, 4);
    // Each input scatters into a 2x2 block with stride 2 (no overlap)
    ASSERT_NEAR(out.at(0, 0, 0), 1.f, 1e-5f);  // top-left corner from input[0,0]
    ASSERT_NEAR(out.at(1, 0, 0), 1.f, 1e-5f);
    ASSERT_NEAR(out.at(2, 0, 0), 2.f, 1e-5f);  // from input[0,1]
}

void test_conv1d_basic() {
    TestMat in(5, 1, 1);
    in.data = { 1.f, 2.f, 3.f, 4.f, 5.f };
    std::vector<float> weight = { 1.f, 2.f, 1.f };  // simple filter
    std::vector<float> bias;
    TestMat out = ref_conv1d(in, weight, bias, 1, 3);
    ASSERT_EQ(out.w, 3);
    // out[0] = 1*1 + 2*2 + 3*1 = 8
    ASSERT_NEAR(out.at(0), 8.f, 1e-5f);
    // out[1] = 2*1 + 3*2 + 4*1 = 12
    ASSERT_NEAR(out.at(1), 12.f, 1e-5f);
}

void test_conv2d_padding_same() {
    // With pad=1, a 3x3 convolution on a 3x3 input yields 3x3 output ("same")
    TestMat in(3, 3, 1);
    in.fill_range();
    std::vector<float> weight(9, 1.f);
    std::vector<float> bias;
    TestMat out = ref_conv2d(in, weight, bias, 1, 3, 3, 1, 1, 1, 1);
    ASSERT_EQ(out.h, 3); ASSERT_EQ(out.w, 3);
    // center pixel should see full 3x3 → sum 1..9 = 45
    ASSERT_NEAR(out.at(1, 1, 0), 45.f, 1e-4f);
    // corner pixel sees 2x2 sub-window → 1+2+4+5 = 12
    ASSERT_NEAR(out.at(0, 0, 0), 12.f, 1e-4f);
}

void test_conv2d_output_shape() {
    // Verify output size formula: out = (in + 2*pad - dil*(k-1) - 1) / stride + 1
    struct Case { int in, k, stride, pad, dil, expected_out; };
    Case cases[] = {
        { 5, 3, 1, 0, 1, 3 },  // standard
        { 5, 3, 2, 0, 1, 2 },  // strided
        { 5, 3, 1, 1, 1, 5 },  // same-ish
        { 9, 3, 1, 0, 2, 5 },  // dilated
    };
    for (auto& c : cases) {
        int out = (c.in + 2 * c.pad - c.dil * (c.k - 1) - 1) / c.stride + 1;
        ASSERT_EQ(out, c.expected_out);
    }
}

// ─── Real ncnn kernel tests ───────────────────────────────────────────────────

// Helper: set up and run ncnn::Convolution with given parameters.
// input: [in_c, in_h, in_w], weight: [out_c, in_c, kh, kw], bias: [out_c]
static bool run_conv2d_ncnn(const TestMat& in_ref,
                              const std::vector<float>& weight,
                              const std::vector<float>& bias,
                              int out_c, int kh, int kw,
                              int stride_h, int stride_w,
                              int pad_top, int pad_left,
                              TestMat& out_ref_result,
                              std::vector<float>& ncnn_out,
                              float tol = 1e-4f)
{
    int in_c = in_ref.c;
    // ── reference ──
    out_ref_result = ref_conv2d(in_ref, weight, bias, out_c, kh, kw,
                                 stride_h, stride_w, pad_top, pad_left);
    // ── ncnn::Convolution ──
    std::vector<float> flat_in(in_ref.data.begin(), in_ref.data.end());
    ncnn::Mat bottom = make_mat(in_ref.w, in_ref.h, in_c, flat_in);
    ncnn::Mat top;

    ncnn::Convolution conv;
    conv.num_output       = out_c;
    conv.kernel_w         = kw;
    conv.kernel_h         = kh;
    conv.dilation_w       = 1;
    conv.dilation_h       = 1;
    conv.stride_w         = stride_w;
    conv.stride_h         = stride_h;
    conv.pad_left         = pad_left;
    conv.pad_right        = pad_left;
    conv.pad_top          = pad_top;
    conv.pad_bottom       = pad_top;
    conv.pad_value        = 0.f;
    conv.bias_term        = bias.empty() ? 0 : 1;
    conv.weight_data_size = out_c * in_c * kh * kw;
    conv.int8_scale_term  = 0;
    conv.activation_type  = 0;
    conv.dynamic_weight   = 0;
    conv.weight_data      = make_weight(weight);
    if (!bias.empty()) conv.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    if (conv.forward(bottom, top, opt) != 0) return false;
    read_mat(top, ncnn_out);
    return true;
}

void test_conv2d_ncnn_1x1()
{
    TestMat in(3, 3, 1);
    in.fill_range();
    std::vector<float> weight = { 2.f };
    std::vector<float> bias   = { 1.f };
    TestMat ref_out;
    std::vector<float> got;
    ASSERT_TRUE(run_conv2d_ncnn(in, weight, bias, 1, 1, 1, 1, 1, 0, 0, ref_out, got));
    ASSERT_VEC_NEAR(got, ref_out.data.data(), ref_out.total(), 1e-4f);
}

void test_conv2d_ncnn_3x3()
{
    TestMat in(4, 4, 1);
    in.fill_range();
    std::vector<float> weight(9, 1.f);  // all-ones
    std::vector<float> bias;
    TestMat ref_out;
    std::vector<float> got;
    ASSERT_TRUE(run_conv2d_ncnn(in, weight, bias, 1, 3, 3, 1, 1, 0, 0, ref_out, got));
    ASSERT_VEC_NEAR(got, ref_out.data.data(), ref_out.total(), 1e-4f);
}

void test_conv2d_ncnn_multichannel()
{
    // 2-ch input, 1-ch output, 1×1 kernel
    TestMat in(2, 2, 2);
    in.data = { 1.f,2.f,3.f,4.f,  5.f,6.f,7.f,8.f };
    std::vector<float> weight = { 1.f, -1.f };  // [out=1, in=2, k=1,1]
    std::vector<float> bias;
    TestMat ref_out;
    std::vector<float> got;
    ASSERT_TRUE(run_conv2d_ncnn(in, weight, bias, 1, 1, 1, 1, 1, 0, 0, ref_out, got));
    ASSERT_VEC_NEAR(got, ref_out.data.data(), ref_out.total(), 1e-4f);
}

void test_conv2d_ncnn_stride2()
{
    TestMat in(4, 4, 1);
    in.fill_range();
    std::vector<float> weight(4, 1.f);  // 2×2 all-ones
    std::vector<float> bias;
    TestMat ref_out;
    std::vector<float> got;
    ASSERT_TRUE(run_conv2d_ncnn(in, weight, bias, 1, 2, 2, 2, 2, 0, 0, ref_out, got));
    ASSERT_VEC_NEAR(got, ref_out.data.data(), ref_out.total(), 1e-4f);
}

void test_depthwise_conv2d_ncnn()
{
    // 2 channels, each convolved independently with its own 3×3 kernel
    TestMat in(4, 4, 2);
    in.fill_range();  // ch0: 1..16, ch1: 17..32
    // all-ones weight: [in_c=2, 1, 3, 3]
    std::vector<float> weight(2 * 9, 1.f);
    std::vector<float> bias;

    ncnn::Mat bottom = make_mat(in.w, in.h, in.c, in.data);
    ncnn::Mat top;

    ncnn::ConvolutionDepthWise dw;
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
    ASSERT_EQ(dw.forward(bottom, top, opt), 0);

    // Output: 2-ch, 2×2 (valid, 4-3+1=2)
    ASSERT_EQ(top.c, 2);
    ASSERT_EQ(top.h, 2);
    ASSERT_EQ(top.w, 2);

    // ch0 top-left 3×3: 1+2+3+5+6+7+9+10+11 = 54
    float v00 = top.channel(0).row(0)[0];
    ASSERT_NEAR(v00, 54.f, 1e-3f);
    // ch1: same spatial but ch starts at 17: 17+18+19+21+22+23+25+26+27 = 198
    float v10 = top.channel(1).row(0)[0];
    ASSERT_NEAR(v10, 198.f, 1e-3f);
}

void test_conv2d_ncnn_with_bias()
{
    TestMat in(2, 2, 1);
    in.data = { 1.f, 2.f, 3.f, 4.f };
    std::vector<float> weight = { 1.f, 0.f, 0.f, 1.f };
    std::vector<float> bias   = { 5.f };
    TestMat ref_out;
    std::vector<float> got;
    ASSERT_TRUE(run_conv2d_ncnn(in, weight, bias, 1, 2, 2, 1, 1, 0, 0, ref_out, got));
    ASSERT_NEAR(got[0], 10.f, 1e-4f);  // 1+4+5
}

int main() {
    printf("=== test_conv ===\n");
    printf("\n-- Reference tests --\n");
    RUN_TEST(test_conv2d_1x1);
    RUN_TEST(test_conv2d_3x3_no_pad);
    RUN_TEST(test_conv2d_with_bias);
    RUN_TEST(test_conv2d_stride2);
    RUN_TEST(test_conv2d_dilation);
    RUN_TEST(test_conv2d_multi_channel);
    RUN_TEST(test_depthwise_conv2d);
    RUN_TEST(test_deconv2d_stride2);
    RUN_TEST(test_conv1d_basic);
    RUN_TEST(test_conv2d_padding_same);
    RUN_TEST(test_conv2d_output_shape);

    printf("\n-- Real ncnn::Convolution / ConvolutionDepthWise --\n");
    RUN_TEST(test_conv2d_ncnn_1x1);
    RUN_TEST(test_conv2d_ncnn_3x3);
    RUN_TEST(test_conv2d_ncnn_multichannel);
    RUN_TEST(test_conv2d_ncnn_stride2);
    RUN_TEST(test_conv2d_ncnn_with_bias);
    RUN_TEST(test_depthwise_conv2d_ncnn);

    print_summary("conv");
    return g_failed > 0 ? 1 : 0;
}
