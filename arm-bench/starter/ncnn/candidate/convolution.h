#ifndef LAYER_CONVOLUTION_H
#define LAYER_CONVOLUTION_H

#include "layer.h"
#include "ncnn_helpers.h"
#include "ref_conv.h"

namespace ncnn {

class Convolution : public Layer
{
public:
    Convolution() { one_blob_only = true; support_inplace = false; }

    virtual int load_param(const ParamDict&) { return 0; }
    virtual int load_model(const ModelBin&) { return 0; }

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    virtual int forward(const std::vector<Mat>&, std::vector<Mat>&, const Option&) const { return -1; }

protected:
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const;
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, int kernel_w, int kernel_h, const Option& opt) const;

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
};

// Defined in candidates_src/ncnn/convolution.cpp — the part to optimize.
int convolution_kernel(const Mat& bottom_blob, Mat& top_blob,
                       const Mat& weight_data, const Mat& bias_data,
                       int kernel_w, int kernel_h,
                       int stride_w, int stride_h,
                       int dilation_w, int dilation_h,
                       int activation_type, const Mat& activation_params,
                       const Option& opt);

inline void Convolution::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const
{
    make_padding(bottom_blob, bottom_blob_bordered, kernel_w, kernel_h, opt);
}

inline void Convolution::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered,
                                      int _kernel_w, int _kernel_h, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    const int kernel_extent_w = dilation_w * (_kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (_kernel_h - 1) + 1;

    bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        // tensorflow padding=SAME or onnx padding=SAME_UPPER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        // onnx padding=SAME_LOWER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}

inline int Convolution::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const int w = bottom_blob_bordered.w;
    const int h = bottom_blob_bordered.h;
    const size_t elemsize = bottom_blob_bordered.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    return convolution_kernel(bottom_blob_bordered, top_blob,
                              weight_data, bias_data,
                              kernel_w, kernel_h,
                              stride_w, stride_h,
                              dilation_w, dilation_h,
                              activation_type, activation_params, opt);
}

} // namespace ncnn

#endif // LAYER_CONVOLUTION_H

// CANDIDATE_INJECT_START
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

[[maybe_unused]] static ncnn::Mat run_conv2d(int in_c, int out_c, int in_h, int in_w,
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
    if (ret != 0) { fprintf(stderr, "  Convolution::forward failed %d\n", ret); return ncnn::Mat(); }
    return top;
}
// CANDIDATE_INJECT_END
