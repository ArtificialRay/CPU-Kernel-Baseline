#include "ncnn_helpers.h"
#include "ref_conv.h"

#ifndef LAYER_DECONVOLUTION_H
#define LAYER_DECONVOLUTION_H

#include "layer.h"

namespace ncnn {

class Deconvolution : public Layer
{
public:
    Deconvolution();

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

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    Mat activation_params;

    int dynamic_weight;

    // model
    Mat weight_data;
    Mat bias_data;
};

} // namespace ncnn

#endif // LAYER_DECONVOLUTION_H


#ifndef LAYER_DECONVOLUTION_ARM_H
#define LAYER_DECONVOLUTION_ARM_H

namespace ncnn {

class Deconvolution_arm : public Deconvolution
{
public:
    Deconvolution_arm();

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
    Layer* activation;
    Layer* gemm;

    Mat weight_data_tm;

    // fp16
    Mat bias_data_fp16;
};

} // namespace ncnn

#endif // LAYER_DECONVOLUTION_ARM_H

// BASELINE_INJECT_START
[[maybe_unused]] static ncnn::Mat run_ref_deconv2d(int in_c, int out_c, int in_h, int in_w,
                                   int kh, int kw, int stride_h, int stride_w,
                                   bool with_bias = false)
{
    int wsize = in_c * out_c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.1f; }

    ncnn::Mat in = make_mat_ramp(in_w, in_h, in_c);
    return ref_deconv2d(in, weight, bias, in_c, out_c, kh, kw, stride_h, stride_w);
}

// Generic runner for Deconvolution_arm
[[maybe_unused]] static ncnn::Mat run_deconv2d_arm(int in_c, int out_c, int in_h, int in_w,
                                   int kh, int kw, int stride_h, int stride_w,
                                   bool with_bias = false)
{
    int wsize = in_c * out_c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.1f; }

    ncnn::Mat bottom = make_mat_ramp(in_w, in_h, in_c);
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
        return ncnn::Mat();
    }
    int ret = deconv.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Deconvolution_arm::forward failed %d\n", ret); return ncnn::Mat(); }
    return top;
}
// BASELINE_INJECT_END