#include "ncnn_helpers.h"
#include "ref_conv.h"
#ifndef LAYER_CONVOLUTION1D_H
#define LAYER_CONVOLUTION1D_H

#include "layer.h"

namespace ncnn {

class Convolution1D : public Layer
{
public:
    Convolution1D();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const;
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, int kernel_w, const Option& opt) const;

public:
    // param
    int num_output;
    int kernel_w;
    int dilation_w;
    int stride_w;
    int pad_left; // -233=SAME_UPPER -234=SAME_LOWER
    int pad_right;
    float pad_value;
    int bias_term;

    int weight_data_size;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    int dynamic_weight;

    // model
    Mat weight_data;
    Mat bias_data;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION1D_H


#ifndef LAYER_CONVOLUTION1D_ARM_H
#define LAYER_CONVOLUTION1D_ARM_H

namespace ncnn {

class Convolution1D_arm : public Convolution1D
{
public:
    Convolution1D_arm();

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

public:
    Mat weight_data_tm;

    // fp16
    Mat bias_data_fp16;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION1D_ARM_H

// BASELINE_INJECT_START
// The ncnn Convolution1D treats the input as [w=length, h=channels] (2D mat)
[[maybe_unused]] static ncnn::Mat run_ref_conv1d(int in_c, int out_c, int in_w, int kw,
                                 int stride_w, int pad_left, int dil_w = 1,
                                 bool with_bias = false)
{
    int wsize = out_c * in_c * kw;
    std::vector<float> weight = make_weights(wsize, 0.5f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.05f; }

    ncnn::Mat in = make_mat_ramp_2d(in_w, in_c);
    return ref_conv1d(in, weight, bias, out_c, kw, stride_w, pad_left, dil_w);
}

// Generic runner for Convolution1D_arm
[[maybe_unused]] static ncnn::Mat run_conv1d_arm(int in_c, int out_c, int in_w, int kw,
                                 int stride_w, int pad_left, int dil_w = 1,
                                 bool with_bias = false)
{
    int wsize = out_c * in_c * kw;
    std::vector<float> weight = make_weights(wsize, 0.5f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.05f; }

    ncnn::Mat bottom = make_mat_ramp_2d(in_w, in_c);
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
        return ncnn::Mat();
    }
    int ret = conv1d.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Convolution1D_arm::forward failed %d\n", ret); return ncnn::Mat(); }
    return top;
}
// BASELINE_INJECT_END