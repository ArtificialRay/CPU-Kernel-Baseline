#include "test_utils.h"
#include "ncnn_helpers.h"

#include "../ncnn/mapped/deconvolutiondepthwise/deconvolutiondepthwise.h"
#include "../ncnn/mapped/deconvolutiondepthwise/deconvolutiondepthwise_arm.h"

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

    std::vector<float> got; read_mat(top, got);
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

    std::vector<float> got; read_mat(top, got);
}
