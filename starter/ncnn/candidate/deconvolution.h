#ifndef LAYER_DECONVOLUTION_H
#define LAYER_DECONVOLUTION_H

#include "layer.h"
#include "ncnn_helpers.h"
#include "ref_conv.h"

namespace ncnn {

class Deconvolution : public Layer
{
public:
    Deconvolution() { one_blob_only = true; support_inplace = false; }

    virtual int load_param(const ParamDict&) { return 0; }
    virtual int load_model(const ModelBin&) { return 0; }

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    virtual int forward(const std::vector<Mat>&, std::vector<Mat>&, const Option&) const { return -1; }

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

// Defined in candidates_src/ncnn/deconvolution.cpp — the part to optimize.
int deconvolution_kernel(const Mat& bottom_blob, Mat& top_blob,
                         const Mat& weight_data, const Mat& bias_data,
                         int kernel_w, int kernel_h,
                         int stride_w, int stride_h,
                         int dilation_w, int dilation_h,
                         int activation_type, const Mat& activation_params,
                         const Option& opt);

inline void Deconvolution::cut_padding(const Mat& top_blob_bordered, Mat& top_blob, const Option& opt) const
{
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        copy_cut_border(top_blob_bordered, top_blob, pad_top, pad_bottom, pad_left, pad_right, opt);
    }
    else if (output_w > 0 && output_h > 0)
    {
        int wcut = top_blob_bordered.w - output_w;
        int hcut = top_blob_bordered.h - output_h;

        if (pad_left == -233 || pad_right == -233 || pad_top == -233 || pad_bottom == -233)
        {
            copy_cut_border(top_blob_bordered, top_blob, hcut / 2, hcut - hcut / 2, wcut / 2, wcut - wcut / 2, opt);
        }
        else if (pad_left == -234 || pad_right == -234 || pad_top == -234 || pad_bottom == -234)
        {
            copy_cut_border(top_blob_bordered, top_blob, hcut - hcut / 2, hcut / 2, wcut - wcut / 2, wcut / 2, opt);
        }
    }
    else
    {
        top_blob = top_blob_bordered;
    }
}

inline int Deconvolution::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w + output_pad_right;
    int outh = (h - 1) * stride_h + kernel_extent_h + output_pad_bottom;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    int ret = deconvolution_kernel(bottom_blob, top_blob_bordered,
                                   weight_data, bias_data,
                                   kernel_w, kernel_h,
                                   stride_w, stride_h,
                                   dilation_w, dilation_h,
                                   activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn

#endif // LAYER_DECONVOLUTION_H

// CANDIDATE_INJECT_START
// Weight layout for deconvolution in ncnn: [in_c, out_c, kh, kw]
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

// ── Setup/forward split ──────────────────────────────────────────────
// Lets perf binaries pay the layer construction + weight/bias/bottom build
// cost once per shape (in setup_deconv2d) and time only forward_deconv2d.
// Symmetric with the baseline-side split so candidate vs baseline ms are
// apples-to-apples (both timed on forward() alone). Tests keep using the
// one-shot run_deconv2d wrapper, so EXPECT_MATCH(...) is unaffected.
struct DeconvCtx {
    std::unique_ptr<ncnn::Deconvolution> layer;   // heap-stable so move-by-value of ctx is safe
    ncnn::Mat bottom;
    ncnn::Option opt;
};

[[maybe_unused]] static DeconvCtx setup_deconv2d(int in_c, int out_c, int in_h, int in_w,
                                 int kh, int kw, int stride_h, int stride_w,
                                 bool with_bias = false)
{
    int wsize = in_c * out_c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.1f; }

    DeconvCtx ctx;
    ctx.layer.reset(new ncnn::Deconvolution());
    auto& d = *ctx.layer;
    d.num_output         = out_c;
    d.kernel_w           = kw;    d.kernel_h  = kh;
    d.dilation_w         = 1;     d.dilation_h = 1;
    d.stride_w           = stride_w; d.stride_h = stride_h;
    d.pad_left           = 0; d.pad_right  = 0;
    d.pad_top            = 0; d.pad_bottom = 0;
    d.output_pad_right   = 0; d.output_pad_bottom = 0;
    d.output_w           = 0; d.output_h = 0;
    d.bias_term          = with_bias ? 1 : 0;
    d.weight_data_size   = wsize;
    d.activation_type    = 0;
    d.dynamic_weight     = 0;
    d.weight_data        = make_weight(weight);   // memcpy into ncnn::Mat — `weight` can die after
    if (with_bias) d.bias_data = make_weight(bias);

    ctx.opt = make_opt();
    ctx.bottom = make_mat_ramp(in_w, in_h, in_c);
    return ctx;
}

// Hot path — this is what perf binaries time.
[[maybe_unused]] static ncnn::Mat forward_deconv2d(const DeconvCtx& ctx)
{
    if (!ctx.layer) return ncnn::Mat();
    ncnn::Mat top;
    int ret = ctx.layer->forward(ctx.bottom, top, ctx.opt);
    if (ret != 0) { fprintf(stderr, "  Deconvolution::forward failed %d\n", ret); return ncnn::Mat(); }
    return top;
}

// One-shot wrapper — keeps EXPECT_MATCH(run_deconv2d, run_ref_deconv2d, ...) working.
[[maybe_unused]] static ncnn::Mat run_deconv2d(int in_c, int out_c, int in_h, int in_w,
                               int kh, int kw, int stride_h, int stride_w,
                               bool with_bias = false)
{
    auto ctx = setup_deconv2d(in_c, out_c, in_h, in_w, kh, kw, stride_h, stride_w, with_bias);
    return forward_deconv2d(ctx);
}
// CANDIDATE_INJECT_END
