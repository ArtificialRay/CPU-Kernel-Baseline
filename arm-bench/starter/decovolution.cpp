#include "test_utils.h"
#include "ncnn_helpers.h"

#include "../ncnn/mapped/deconvolution/deconvolution.h"
#include "../ncnn/mapped/deconvolution/deconvolution_arm.h


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

    std::vector<float> got; read_mat(top, got);
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

    std::vector<float> got; read_mat(top, got);
}