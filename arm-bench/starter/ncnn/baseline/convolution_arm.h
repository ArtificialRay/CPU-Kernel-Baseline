#include "ncnn_helpers.h"
#include "ref_conv.h"

#ifndef LAYER_CONVOLUTION_H
#define LAYER_CONVOLUTION_H

#include "layer.h"

namespace ncnn {

class Convolution : public Layer
{
public:
    Convolution();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const;
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, int kernel_w, int kernel_h, const Option& opt) const;

#if NCNN_INT8
    int forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif

public:
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_left; // -233=SAME_UPPER -234=SAME_LOWER
    int pad_right;
    int pad_top;
    int pad_bottom;
    float pad_value;
    int bias_term;

    int weight_data_size;

    int int8_scale_term;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    int dynamic_weight;

    // model
    Mat weight_data;
    Mat bias_data;

#if NCNN_INT8
    Mat weight_data_int8_scales;
    Mat bottom_blob_int8_scales;
    Mat top_blob_int8_scales;
#endif
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION_H

#ifndef LAYER_CONVOLUTION_ARM_H
#define LAYER_CONVOLUTION_ARM_H


namespace ncnn {

class Convolution_arm : public Convolution
{
public:
    Convolution_arm();

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
    int create_pipeline_bf16s(const Option& opt);
    int forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif
#if NCNN_INT8
    int create_pipeline_int8_arm(const Option& opt);
    int forward_int8_arm(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif
    int forwardDilation_arm(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    Layer* activation;

    int nT;

    Mat weight_data_tm;
    Mat weight_3x3s2_data;

    Mat weight_sgemm_data;
    Mat weight_winograd23_data;
    Mat weight_winograd43_data;
    Mat weight_winograd63_data;

    // forwardDilation
    Layer* convolution_dilation1;

    // fp16
    Mat bias_data_fp16;

#if NCNN_INT8
    Mat scale_in_data;
#endif
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION_ARM_H


// BASELINE_INJECT_START
[[maybe_unused]] static ncnn::Mat run_ref_conv2d(int in_c, int out_c, int in_h, int in_w,
                                 int kh, int kw, int stride_h, int stride_w,
                                 int pad_top, int pad_left,
                                 int dil_h = 1, int dil_w = 1,
                                 bool with_bias = false)
{
    int wsize = out_c * in_c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.5f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.1f; }

    ncnn::Mat in = make_mat_ramp(in_w, in_h, in_c);
    return ref_conv2d(in, weight, bias, out_c, kh, kw, stride_h, stride_w, pad_top, pad_left, dil_h, dil_w);
}

// Generic runner for Convolution_arm
[[maybe_unused]] static ncnn::Mat run_conv2d_arm(int in_c, int out_c, int in_h, int in_w,
                                 int kh, int kw, int stride_h, int stride_w,
                                 int pad_top, int pad_left,
                                 int dil_h = 1, int dil_w = 1,
                                 bool with_bias = false)
{
    int wsize = out_c * in_c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.5f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.1f; }

    ncnn::Mat bottom = make_mat_ramp(in_w, in_h, in_c);
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
        return ncnn::Mat();
    }
    int ret = conv.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Convolution_arm::forward failed %d\n", ret); return ncnn::Mat(); }
    return top;
}
// BASELINE_INJECT_END