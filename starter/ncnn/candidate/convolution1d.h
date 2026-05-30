#ifndef LAYER_CONVOLUTION1D_H
#define LAYER_CONVOLUTION1D_H

#include "layer.h"
#include "ncnn_helpers.h"
#include "ref_conv.h"

namespace ncnn {

class Convolution1D : public Layer
{
public:
    Convolution1D() { one_blob_only = true; support_inplace = false; }

    virtual int load_param(const ParamDict&) { return 0; }
    virtual int load_model(const ModelBin&) { return 0; }

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    virtual int forward(const std::vector<Mat>&, std::vector<Mat>&, const Option&) const { return -1; }

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

// Defined in candidates_src/ncnn/convolution1d.cpp — the part to optimize.
int convolution1d_kernel(const Mat& bottom_blob, Mat& top_blob,
                         const Mat& weight_data, const Mat& bias_data,
                         int kernel_w, int stride_w, int dilation_w,
                         int activation_type, const Mat& activation_params,
                         const Option& opt);

inline void Convolution1D::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const
{
    make_padding(bottom_blob, bottom_blob_bordered, kernel_w, opt);
}

inline void Convolution1D::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered,
                                        int _kernel_w, const Option& opt) const
{
    int w = bottom_blob.w;
    const int kernel_extent_w = dilation_w * (_kernel_w - 1) + 1;

    bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0)
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, 0, 0, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_left == -233 && pad_right == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, 0, 0, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_left == -234 && pad_right == -234)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, 0, 0, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}

inline int Convolution1D::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const int w = bottom_blob_bordered.w;
    const size_t elemsize = bottom_blob_bordered.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int outw = (w - kernel_extent_w) / stride_w + 1;

    top_blob.create(outw, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    return convolution1d_kernel(bottom_blob_bordered, top_blob,
                                weight_data, bias_data,
                                kernel_w, stride_w, dilation_w,
                                activation_type, activation_params, opt);
}

} // namespace ncnn

#endif // LAYER_CONVOLUTION1D_H

// CANDIDATE_INJECT_START
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

[[maybe_unused]] static ncnn::Mat run_conv1d(int in_c, int out_c, int in_w, int kw,
                             int stride_w, int pad_left, int dil_w = 1,
                             bool with_bias = false)
{
    int wsize = out_c * in_c * kw;
    std::vector<float> weight = make_weights(wsize, 0.5f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.05f; }

    // Convolution1D input layout: w=length, h=channels (2D mat, no c dim)
    ncnn::Mat bottom = make_mat_ramp_2d(in_w, in_c);
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
    if (ret != 0) { fprintf(stderr, "  Convolution1D::forward failed %d\n", ret); return ncnn::Mat(); }
    return top;
}
// CANDIDATE_INJECT_END
