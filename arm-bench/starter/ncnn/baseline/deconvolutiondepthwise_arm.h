#include "ncnn_helpers.h"
#include "ref_conv.h"

// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DECONVOLUTIONDEPTHWISE_H
#define LAYER_DECONVOLUTIONDEPTHWISE_H

#include "layer.h"

namespace ncnn {

class DeconvolutionDepthWise : public Layer
{
public:
    DeconvolutionDepthWise();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    void cut_padding(const Mat& top_blob_bordered, Mat& top_blob, const Option& opt) const;

public:
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_left;
    int pad_right;
    int pad_top;
    int pad_bottom;
    int output_pad_right;
    int output_pad_bottom;
    int output_w;
    int output_h;
    int bias_term;

    int weight_data_size;
    int group;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    int dynamic_weight;

    // model
    Mat weight_data;
    Mat bias_data;
};

} // namespace ncnn

#endif // LAYER_DECONVOLUTIONDEPTHWISE_H

#ifndef LAYER_DECONVOLUTIONDEPTHWISE_ARM_H
#define LAYER_DECONVOLUTIONDEPTHWISE_ARM_H

namespace ncnn {

class DeconvolutionDepthWise_arm : public DeconvolutionDepthWise
{
public:
    DeconvolutionDepthWise_arm();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_ARM82
    int create_pipeline_fp16s(const Option& opt);
    int forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif
#if NCNN_BF16
    int forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif

public:
    std::vector<ncnn::Layer*> group_ops;

    Mat weight_data_tm;

    // fp16
    Mat bias_data_fp16;
};

} // namespace ncnn

#endif // LAYER_DECONVOLUTIONDEPTHWISE_ARM_H

// BASELINE_INJECT_START
[[maybe_unused]] static ncnn::Mat run_ref_depthwise_deconv2d(int c, int in_h, int in_w,
                                             int kh, int kw, int stride_h, int stride_w,
                                             bool with_bias = false)
{
    int wsize = c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(c); for (int i = 0; i < c; ++i) bias[i] = i * 0.1f; }

    ncnn::Mat in = make_mat_ramp(in_w, in_h, c);
    return ref_depthwise_deconv2d(in, weight, bias, kh, kw, stride_h, stride_w);
}

// Generic runner for DeconvolutionDepthWise_arm
[[maybe_unused]] static ncnn::Mat run_depthwise_deconv2d_arm(int c, int in_h, int in_w,
                                             int kh, int kw, int stride_h, int stride_w,
                                             bool with_bias = false)
{
    int wsize = c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(c); for (int i = 0; i < c; ++i) bias[i] = i * 0.1f; }

    ncnn::Mat bottom = make_mat_ramp(in_w, in_h, c);
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
        return ncnn::Mat();
    }
    int ret = ddw.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  DeconvolutionDepthWise_arm::forward failed %d\n", ret); return ncnn::Mat(); }
    return top;
}
// BASELINE_INJECT_END