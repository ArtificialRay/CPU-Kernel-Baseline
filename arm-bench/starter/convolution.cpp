#include "test_utils.h"
#include "ncnn_helpers.h"

// ── Base class headers ────────────────────────────────────────────────────────
#include "../ncnn/mapped/convolution/convolution.h"

// ── ARM class headers ─────────────────────────────────────────────────────────
#include "../ncnn/mapped/convolution/convolution_arm.h"


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
    std::vector<float> got; read_mat(top, got);
}
// CANDIDATE_INJECT_END

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

    std::vector<float> got; read_mat(top, got);
}

