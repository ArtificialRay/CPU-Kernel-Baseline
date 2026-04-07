// test_mapped_conv.cpp
// Tests for mapped conv kernel implementations:
//   Convolution, ConvolutionDepthWise, Deconvolution, DeconvolutionDepthWise,
//   Convolution1D — both base (c-partially-optimized) and ARM variants.
// Reference implementations are pure C with no ncnn dependency.

#include "test_utils.h"
#include "ncnn_helpers.h"

// ── Base class headers ────────────────────────────────────────────────────────
#include "../mapped/convolution/convolution.h"
#include "../mapped/convolutiondepthwise/convolutiondepthwise.h"
#include "../mapped/deconvolution/deconvolution.h"
#include "../mapped/deconvolutiondepthwise/deconvolutiondepthwise.h"
#include "../mapped/convolution1d/convolution1d.h"

// ── ARM class headers ─────────────────────────────────────────────────────────
#include "../mapped/convolution/convolution_arm.h"
#include "../mapped/convolutiondepthwise/convolutiondepthwise_arm.h"
#include "../mapped/deconvolution/deconvolution_arm.h"
#include "../mapped/deconvolutiondepthwise/deconvolutiondepthwise_arm.h"
#include "../mapped/convolution1d/convolution1d_arm.h"

// ═══════════════════════════════════════════════════════════════════
// ── Pure-C reference implementations ─────────────────────────────
// ═══════════════════════════════════════════════════════════════════

// 2-D convolution reference
static TestMat ref_conv2d(const TestMat& in,
                           const std::vector<float>& weight,
                           const std::vector<float>& bias,
                           int out_c, int kh, int kw,
                           int stride_h, int stride_w,
                           int pad_top, int pad_left,
                           int dil_h = 1, int dil_w = 1)
{
    int in_c = in.c, in_h = in.h, in_w = in.w;
    int out_h = (in_h + 2 * pad_top  - dil_h * (kh - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * pad_left - dil_w * (kw - 1) - 1) / stride_w + 1;
    TestMat out(out_w, out_h, out_c);
    for (int oc = 0; oc < out_c; ++oc)
    for (int oh = 0; oh < out_h; ++oh)
    for (int ow = 0; ow < out_w; ++ow) {
        float sum = bias.empty() ? 0.f : bias[oc];
        for (int ic = 0; ic < in_c; ++ic)
        for (int khi = 0; khi < kh; ++khi)
        for (int kwi = 0; kwi < kw; ++kwi) {
            int ih = oh * stride_h - pad_top  + khi * dil_h;
            int iw = ow * stride_w - pad_left + kwi * dil_w;
            float px = (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                       ? in.at(iw, ih, ic) : 0.f;
            int widx = ((oc * in_c + ic) * kh + khi) * kw + kwi;
            sum += px * weight[widx];
        }
        out.at(ow, oh, oc) = sum;
    }
    return out;
}

// Depthwise 2-D convolution reference (group == channels)
static TestMat ref_depthwise_conv2d(const TestMat& in,
                                     const std::vector<float>& weight,
                                     const std::vector<float>& bias,
                                     int kh, int kw,
                                     int stride_h, int stride_w,
                                     int pad_top, int pad_left,
                                     int dil_h = 1, int dil_w = 1)
{
    int c = in.c;
    int out_h = (in.h + 2 * pad_top  - dil_h * (kh - 1) - 1) / stride_h + 1;
    int out_w = (in.w + 2 * pad_left - dil_w * (kw - 1) - 1) / stride_w + 1;
    TestMat out(out_w, out_h, c);
    for (int ch = 0; ch < c; ++ch)
    for (int oh = 0; oh < out_h; ++oh)
    for (int ow = 0; ow < out_w; ++ow) {
        float sum = bias.empty() ? 0.f : bias[ch];
        for (int khi = 0; khi < kh; ++khi)
        for (int kwi = 0; kwi < kw; ++kwi) {
            int ih = oh * stride_h - pad_top  + khi * dil_h;
            int iw = ow * stride_w - pad_left + kwi * dil_w;
            float px = (ih >= 0 && ih < in.h && iw >= 0 && iw < in.w)
                       ? in.at(iw, ih, ch) : 0.f;
            sum += px * weight[(ch * kh + khi) * kw + kwi];
        }
        out.at(ow, oh, ch) = sum;
    }
    return out;
}

// 2-D transposed convolution (deconvolution) reference
// Weight layout: [out_c, in_c, kh, kw] (ncnn layout: per-outch block of in_c*kh*kw)
static TestMat ref_deconv2d(const TestMat& in,
                              const std::vector<float>& weight,
                              const std::vector<float>& bias,
                              int in_c, int out_c,
                              int kh, int kw,
                              int stride_h, int stride_w,
                              int dil_h = 1, int dil_w = 1)
{
    int w = in.w, h = in.h;
    int ke_h = dil_h * (kh - 1) + 1;
    int ke_w = dil_w * (kw - 1) + 1;
    int out_h = (h - 1) * stride_h + ke_h;
    int out_w = (w - 1) * stride_w + ke_w;
    TestMat out(out_w, out_h, out_c, std::vector<float>(out_w * out_h * out_c, 0.f));
    // fill bias
    if (!bias.empty())
        for (int oc = 0; oc < out_c; ++oc)
            for (int i = 0; i < out_h * out_w; ++i)
                out.at(i % out_w, i / out_w, oc) = bias[oc];

    for (int oc = 0; oc < out_c; ++oc)
    for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j) {
        for (int ic = 0; ic < in_c; ++ic) {
            float val = in.at(j, i, ic);
            for (int ki = 0; ki < kh; ++ki)
            for (int kj = 0; kj < kw; ++kj) {
                int oy = i * stride_h + ki * dil_h;
                int ox = j * stride_w + kj * dil_w;
                // Weight layout: [out_c, in_c, kh, kw] matching ncnn deconv forward
                float wt = weight[((oc * in_c + ic) * kh + ki) * kw + kj];
                out.at(ox, oy, oc) += val * wt;
            }
        }
    }
    return out;
}

// Depthwise 2-D transposed convolution reference (group == channels)
static TestMat ref_depthwise_deconv2d(const TestMat& in,
                                       const std::vector<float>& weight,
                                       const std::vector<float>& bias,
                                       int kh, int kw,
                                       int stride_h, int stride_w,
                                       int dil_h = 1, int dil_w = 1)
{
    int c = in.c, h = in.h, w = in.w;
    int ke_h = dil_h * (kh - 1) + 1;
    int ke_w = dil_w * (kw - 1) + 1;
    int out_h = (h - 1) * stride_h + ke_h;
    int out_w = (w - 1) * stride_w + ke_w;
    TestMat out(out_w, out_h, c, std::vector<float>(out_w * out_h * c, 0.f));
    if (!bias.empty())
        for (int ch = 0; ch < c; ++ch)
            for (int i = 0; i < out_h * out_w; ++i)
                out.at(i % out_w, i / out_w, ch) = bias[ch];

    for (int ch = 0; ch < c; ++ch)
    for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j) {
        float val = in.at(j, i, ch);
        for (int ki = 0; ki < kh; ++ki)
        for (int kj = 0; kj < kw; ++kj) {
            int oy = i * stride_h + ki * dil_h;
            int ox = j * stride_w + kj * dil_w;
            out.at(ox, oy, ch) += val * weight[(ch * kh + ki) * kw + kj];
        }
    }
    return out;
}

// 1-D convolution reference
static TestMat ref_conv1d(const TestMat& in,
                           const std::vector<float>& weight,
                           const std::vector<float>& bias,
                           int out_c, int kw,
                           int stride_w, int pad_left,
                           int dil_w = 1)
{
    int in_c = in.h;   // Convolution1D: h=channels, w=sequence_length
    int in_w = in.w;
    int out_w = (in_w + 2 * pad_left - dil_w * (kw - 1) - 1) / stride_w + 1;
    TestMat out(out_w, out_c, 1);
    for (int oc = 0; oc < out_c; ++oc)
    for (int ow = 0; ow < out_w; ++ow) {
        float sum = bias.empty() ? 0.f : bias[oc];
        for (int ic = 0; ic < in_c; ++ic)
        for (int kwi = 0; kwi < kw; ++kwi) {
            int iw = ow * stride_w - pad_left + kwi * dil_w;
            float px = (iw >= 0 && iw < in_w) ? in.at(iw, ic) : 0.f;
            sum += px * weight[(oc * in_c + ic) * kw + kwi];
        }
        out.at(ow, oc) = sum;
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════
// ── Helpers to run real kernel implementations ────────────────────
// ═══════════════════════════════════════════════════════════════════

// Generic runner for Convolution (base class)
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

// Generic runner for ConvolutionDepthWise (base)
static bool run_depthwise_conv2d(int c, int in_h, int in_w,
                                  int kh, int kw, int stride_h, int stride_w,
                                  int pad_top, int pad_left,
                                  int dil_h = 1, int dil_w = 1,
                                  bool with_bias = false)
{
    int wsize = c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.5f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(c); for (int i = 0; i < c; ++i) bias[i] = i * 0.1f; }

    TestMat in(in_w, in_h, c); in.fill_range();
    ncnn::Mat bottom = make_mat(in.w, in.h, in.c, in.data);
    ncnn::Mat top;

    ncnn::ConvolutionDepthWise dw;
    dw.num_output       = c;
    dw.kernel_w         = kw;    dw.kernel_h  = kh;
    dw.dilation_w       = dil_w; dw.dilation_h = dil_h;
    dw.stride_w         = stride_w; dw.stride_h = stride_h;
    dw.pad_left         = pad_left; dw.pad_right  = pad_left;
    dw.pad_top          = pad_top;  dw.pad_bottom = pad_top;
    dw.pad_value        = 0.f;
    dw.bias_term        = with_bias ? 1 : 0;
    dw.weight_data_size = wsize;
    dw.group            = c;
    dw.int8_scale_term  = 0;
    dw.activation_type  = 0;
    dw.dynamic_weight   = 0;
    dw.weight_data      = make_weight(weight);
    if (with_bias) dw.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    int ret = dw.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  ConvolutionDepthWise::forward failed %d\n", ret); g_failed++; return false; }

    TestMat ref = ref_depthwise_conv2d(in, weight, bias, kh, kw, stride_h, stride_w, pad_top, pad_left, dil_h, dil_w);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data.data(), ref.total(), 1e-3f);
    return g_failed == before;
}

// Generic runner for ConvolutionDepthWise_arm
static bool run_depthwise_conv2d_arm(int c, int in_h, int in_w,
                                      int kh, int kw, int stride_h, int stride_w,
                                      int pad_top, int pad_left,
                                      int dil_h = 1, int dil_w = 1,
                                      bool with_bias = false)
{
    int wsize = c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.5f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(c); for (int i = 0; i < c; ++i) bias[i] = i * 0.1f; }

    TestMat in(in_w, in_h, c); in.fill_range();
    ncnn::Mat bottom = make_mat(in.w, in.h, in.c, in.data);
    ncnn::Mat top;

    ncnn::ConvolutionDepthWise_arm dw;
    dw.num_output       = c;
    dw.kernel_w         = kw;    dw.kernel_h  = kh;
    dw.dilation_w       = dil_w; dw.dilation_h = dil_h;
    dw.stride_w         = stride_w; dw.stride_h = stride_h;
    dw.pad_left         = pad_left; dw.pad_right  = pad_left;
    dw.pad_top          = pad_top;  dw.pad_bottom = pad_top;
    dw.pad_value        = 0.f;
    dw.bias_term        = with_bias ? 1 : 0;
    dw.weight_data_size = wsize;
    dw.group            = c;
    dw.int8_scale_term  = 0;
    dw.activation_type  = 0;
    dw.dynamic_weight   = 0;
    dw.weight_data      = make_weight(weight);
    if (with_bias) dw.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    if (dw.create_pipeline(opt) != 0) {
        fprintf(stderr, "  ConvolutionDepthWise_arm::create_pipeline failed\n");
        g_failed++; return false;
    }
    int ret = dw.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  ConvolutionDepthWise_arm::forward failed %d\n", ret); g_failed++; return false; }

    TestMat ref = ref_depthwise_conv2d(in, weight, bias, kh, kw, stride_h, stride_w, pad_top, pad_left, dil_h, dil_w);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data.data(), ref.total(), 1e-3f);
    return g_failed == before;
}

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

// Convolution1D (base) — ncnn layout: w=sequence_len, h=in_channels, c=1
// The ncnn Convolution1D treats the input as [w=length, h=channels] (2D mat)
static bool run_conv1d(int in_c, int out_c, int in_w, int kw,
                        int stride_w, int pad_left, int dil_w = 1,
                        bool with_bias = false)
{
    int wsize = out_c * in_c * kw;
    std::vector<float> weight = make_weights(wsize, 0.5f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.05f; }

    // Convolution1D input layout: w=length, h=channels (2D mat, no c dim)
    std::vector<float> in_flat(in_c * in_w);
    for (int i = 0; i < (int)in_flat.size(); ++i) in_flat[i] = (i + 1) * 0.1f;
    ncnn::Mat bottom = make_mat_2d(in_w, in_c, in_flat);
    ncnn::Mat top;

    ncnn::Convolution1D conv1d;
    conv1d.num_output       = out_c;
    conv1d.kernel_w         = kw;
    conv1d.dilation_w       = dil_w;
    conv1d.stride_w         = stride_w;
    conv1d.pad_left         = pad_left; conv1d.pad_right = pad_left;
    conv1d.pad_value        = 0.f;
    conv1d.bias_term        = with_bias ? 1 : 0;
    conv1d.weight_data_size = wsize;
    conv1d.activation_type  = 0;
    conv1d.dynamic_weight   = 0;
    conv1d.weight_data      = make_weight(weight);
    if (with_bias) conv1d.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    int ret = conv1d.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Convolution1D::forward failed %d\n", ret); g_failed++; return false; }

    // Reference uses TestMat (h=in_c, w=in_w)
    TestMat in_tm(in_w, in_c, 1, in_flat);
    TestMat ref = ref_conv1d(in_tm, weight, bias, out_c, kw, stride_w, pad_left, dil_w);

    // Output: h=out_c, w=out_len (2D mat)
    std::vector<float> got;
    if (top.dims == 2) {
        int out_len = top.w;
        got.resize(out_c * out_len);
        for (int oc = 0; oc < out_c; ++oc)
            memcpy(got.data() + oc * out_len, top.row(oc), out_len * sizeof(float));
    } else {
        read_mat(top, got);
    }

    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data.data(), (int)got.size(), 1e-3f);
    return g_failed == before;
}

// Convolution1D_arm
static bool run_conv1d_arm(int in_c, int out_c, int in_w, int kw,
                             int stride_w, int pad_left, int dil_w = 1,
                             bool with_bias = false)
{
    int wsize = out_c * in_c * kw;
    std::vector<float> weight = make_weights(wsize, 0.5f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.05f; }

    std::vector<float> in_flat(in_c * in_w);
    for (int i = 0; i < (int)in_flat.size(); ++i) in_flat[i] = (i + 1) * 0.1f;
    ncnn::Mat bottom = make_mat_2d(in_w, in_c, in_flat);
    ncnn::Mat top;

    ncnn::Convolution1D_arm conv1d;
    conv1d.num_output       = out_c;
    conv1d.kernel_w         = kw;
    conv1d.dilation_w       = dil_w;
    conv1d.stride_w         = stride_w;
    conv1d.pad_left         = pad_left; conv1d.pad_right = pad_left;
    conv1d.pad_value        = 0.f;
    conv1d.bias_term        = with_bias ? 1 : 0;
    conv1d.weight_data_size = wsize;
    conv1d.activation_type  = 0;
    conv1d.dynamic_weight   = 0;
    conv1d.weight_data      = make_weight(weight);
    if (with_bias) conv1d.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    if (conv1d.create_pipeline(opt) != 0) {
        fprintf(stderr, "  Convolution1D_arm::create_pipeline failed\n");
        g_failed++; return false;
    }
    int ret = conv1d.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Convolution1D_arm::forward failed %d\n", ret); g_failed++; return false; }

    TestMat in_tm(in_w, in_c, 1, in_flat);
    TestMat ref = ref_conv1d(in_tm, weight, bias, out_c, kw, stride_w, pad_left, dil_w);

    std::vector<float> got;
    if (top.dims == 2) {
        int out_len = top.w;
        got.resize(out_c * out_len);
        for (int oc = 0; oc < out_c; ++oc)
            memcpy(got.data() + oc * out_len, top.row(oc), out_len * sizeof(float));
    } else {
        read_mat(top, got);
    }

    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data.data(), (int)got.size(), 1e-3f);
    return g_failed == before;
}

// ═══════════════════════════════════════════════════════════════════
// ── Test cases ────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════

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

// ── ConvolutionDepthWise (base) ───────────────────────────────────

void test_dw_base_3x3() {
    ASSERT_TRUE(run_depthwise_conv2d(2, 6, 6, 3, 3, 1, 1, 0, 0));
    ASSERT_TRUE(run_depthwise_conv2d(4, 8, 8, 3, 3, 1, 1, 1, 1));
    ASSERT_TRUE(run_depthwise_conv2d(8, 12, 12, 3, 3, 2, 2, 0, 0));
}

void test_dw_base_5x5() {
    ASSERT_TRUE(run_depthwise_conv2d(4, 8, 8, 5, 5, 1, 1, 2, 2));
}

void test_dw_base_bias() {
    ASSERT_TRUE(run_depthwise_conv2d(4, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true));
}

// ── ConvolutionDepthWise_arm ──────────────────────────────────────

void test_dw_arm_3x3() {
    ASSERT_TRUE(run_depthwise_conv2d_arm(2, 6, 6, 3, 3, 1, 1, 0, 0));
    ASSERT_TRUE(run_depthwise_conv2d_arm(4, 8, 8, 3, 3, 1, 1, 1, 1));
    ASSERT_TRUE(run_depthwise_conv2d_arm(8, 12, 12, 3, 3, 2, 2, 0, 0));
}

void test_dw_arm_5x5() {
    ASSERT_TRUE(run_depthwise_conv2d_arm(4, 8, 8, 5, 5, 1, 1, 2, 2));
}

void test_dw_arm_bias() {
    ASSERT_TRUE(run_depthwise_conv2d_arm(4, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true));
}

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

// ── DeconvolutionDepthWise (base) ─────────────────────────────────

void test_dw_deconv_base_2x2_s2() {
    ASSERT_TRUE(run_depthwise_deconv2d(2, 3, 3, 2, 2, 2, 2));
    ASSERT_TRUE(run_depthwise_deconv2d(4, 4, 4, 2, 2, 2, 2));
}

void test_dw_deconv_base_3x3_s1() {
    ASSERT_TRUE(run_depthwise_deconv2d(2, 4, 4, 3, 3, 1, 1));
    ASSERT_TRUE(run_depthwise_deconv2d(4, 5, 5, 3, 3, 1, 1));
}

// ── DeconvolutionDepthWise_arm ────────────────────────────────────

void test_dw_deconv_arm_2x2_s2() {
    ASSERT_TRUE(run_depthwise_deconv2d_arm(2, 3, 3, 2, 2, 2, 2));
    ASSERT_TRUE(run_depthwise_deconv2d_arm(4, 4, 4, 2, 2, 2, 2));
}

void test_dw_deconv_arm_3x3_s1() {
    ASSERT_TRUE(run_depthwise_deconv2d_arm(2, 4, 4, 3, 3, 1, 1));
    ASSERT_TRUE(run_depthwise_deconv2d_arm(4, 5, 5, 3, 3, 1, 1));
}

// ── Convolution1D (base) ──────────────────────────────────────────

void test_conv1d_base_k3() {
    ASSERT_TRUE(run_conv1d(2, 4, 8, 3, 1, 0));
    ASSERT_TRUE(run_conv1d(4, 8, 16, 3, 1, 1));
    ASSERT_TRUE(run_conv1d(8, 4, 12, 3, 2, 0));
}

void test_conv1d_base_k1() {
    ASSERT_TRUE(run_conv1d(3, 4, 8, 1, 1, 0));
}

void test_conv1d_base_bias() {
    ASSERT_TRUE(run_conv1d(4, 8, 8, 3, 1, 1, 1, true));
}

// ── Convolution1D_arm ─────────────────────────────────────────────

void test_conv1d_arm_k3() {
    ASSERT_TRUE(run_conv1d_arm(2, 4, 8, 3, 1, 0));
    ASSERT_TRUE(run_conv1d_arm(4, 8, 16, 3, 1, 1));
    ASSERT_TRUE(run_conv1d_arm(8, 4, 12, 3, 2, 0));
}

void test_conv1d_arm_k1() {
    ASSERT_TRUE(run_conv1d_arm(3, 4, 8, 1, 1, 0));
}

void test_conv1d_arm_bias() {
    ASSERT_TRUE(run_conv1d_arm(4, 8, 8, 3, 1, 1, 1, true));
}

// ═══════════════════════════════════════════════════════════════════
// ── main ──────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════

int main()
{
    printf("=== test_mapped_conv ===\n");

    printf("\n-- Convolution (base) --\n");
    RUN_TEST(test_conv_base_1x1_s1);
    RUN_TEST(test_conv_base_3x3_s1);
    RUN_TEST(test_conv_base_3x3_s2);
    RUN_TEST(test_conv_base_5x5);
    RUN_TEST(test_conv_base_dilation);
    RUN_TEST(test_conv_base_bias);

    printf("\n-- Convolution_arm --\n");
    RUN_TEST(test_conv_arm_1x1_s1);
    RUN_TEST(test_conv_arm_3x3_s1);
    RUN_TEST(test_conv_arm_3x3_s2);
    RUN_TEST(test_conv_arm_5x5);
    RUN_TEST(test_conv_arm_bias);

    printf("\n-- ConvolutionDepthWise (base) --\n");
    RUN_TEST(test_dw_base_3x3);
    RUN_TEST(test_dw_base_5x5);
    RUN_TEST(test_dw_base_bias);

    printf("\n-- ConvolutionDepthWise_arm --\n");
    RUN_TEST(test_dw_arm_3x3);
    RUN_TEST(test_dw_arm_5x5);
    RUN_TEST(test_dw_arm_bias);

    printf("\n-- Deconvolution (base) --\n");
    RUN_TEST(test_deconv_base_2x2_s2);
    RUN_TEST(test_deconv_base_3x3_s1);
    RUN_TEST(test_deconv_base_bias);

    printf("\n-- Deconvolution_arm --\n");
    RUN_TEST(test_deconv_arm_2x2_s2);
    RUN_TEST(test_deconv_arm_3x3_s1);
    RUN_TEST(test_deconv_arm_bias);

    printf("\n-- DeconvolutionDepthWise (base) --\n");
    RUN_TEST(test_dw_deconv_base_2x2_s2);
    RUN_TEST(test_dw_deconv_base_3x3_s1);

    printf("\n-- DeconvolutionDepthWise_arm --\n");
    RUN_TEST(test_dw_deconv_arm_2x2_s2);
    RUN_TEST(test_dw_deconv_arm_3x3_s1);

    printf("\n-- Convolution1D (base) --\n");
    RUN_TEST(test_conv1d_base_k3);
    RUN_TEST(test_conv1d_base_k1);
    RUN_TEST(test_conv1d_base_bias);

    printf("\n-- Convolution1D_arm --\n");
    RUN_TEST(test_conv1d_arm_k3);
    RUN_TEST(test_conv1d_arm_k1);
    RUN_TEST(test_conv1d_arm_bias);

    print_summary("mapped_conv");
    return g_failed > 0 ? 1 : 0;
}
