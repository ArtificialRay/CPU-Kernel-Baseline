#include "test_utils.h"
#include "ncnn_helpers.h"

#include "../ncnn/mapped/convolutiondepthwise/convolutiondepthwise.h"
#include "../mapped/convolutiondepthwise/convolutiondepthwise_arm.h"

// CANDIDATE_INJECT_START
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

    std::vector<float> got; read_mat(top, got);
}
// CANDIDATE_INJECT_END

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

    std::vector<float> got; read_mat(top, got);
}