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

// ── Setup/forward split ──────────────────────────────────────────────
// Deconvolution_arm::create_pipeline does a full transpose+populate of
// weight_data_tm and then (for the four NEON-specialized (k, s) variants —
// 3×3 s1/s2, 4×4 s1/s2) discards it via `weight_data_tm = weight_data;`
// (deconvolution_arm.cpp:178-200). That work scales with weight size and is
// paid on every run_deconv2d_arm() call in the original API. Splitting lets
// perf binaries pay it once per shape (setup_deconv2d_arm) and time only
// forward(); tests keep using run_deconv2d_arm() as a one-shot wrapper.
struct DeconvArmCtx {
    std::unique_ptr<ncnn::Deconvolution_arm> layer;  // owns activation/gemm pointers — heap-stable
    ncnn::Mat bottom;
    ncnn::Option opt;
};

[[maybe_unused]] static DeconvArmCtx setup_deconv2d_arm(int in_c, int out_c, int in_h, int in_w,
                                       int kh, int kw, int stride_h, int stride_w,
                                       bool with_bias = false)
{
    int wsize = in_c * out_c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.1f; }

    DeconvArmCtx ctx;
    ctx.layer.reset(new ncnn::Deconvolution_arm());
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
    d.weight_data        = make_weight(weight);
    if (with_bias) d.bias_data = make_weight(bias);

    ctx.opt = make_opt();
    if (d.create_pipeline(ctx.opt) != 0) {
        fprintf(stderr, "  Deconvolution_arm::create_pipeline failed\n");
        ctx.layer.reset();   // empty ctx → forward returns empty Mat
        return ctx;
    }
    ctx.bottom = make_mat_ramp(in_w, in_h, in_c);
    return ctx;
}

// Hot path — this is what perf binaries time.
[[maybe_unused]] static ncnn::Mat forward_deconv2d_arm(const DeconvArmCtx& ctx)
{
    if (!ctx.layer) return ncnn::Mat();
    ncnn::Mat top;
    int ret = ctx.layer->forward(ctx.bottom, top, ctx.opt);
    if (ret != 0) { fprintf(stderr, "  Deconvolution_arm::forward failed %d\n", ret); return ncnn::Mat(); }
    return top;
}

// One-shot wrapper — keeps EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, ...) working.
[[maybe_unused]] static ncnn::Mat run_deconv2d_arm(int in_c, int out_c, int in_h, int in_w,
                                   int kh, int kw, int stride_h, int stride_w,
                                   bool with_bias = false)
{
    auto ctx = setup_deconv2d_arm(in_c, out_c, in_h, in_w, kh, kw, stride_h, stride_w, with_bias);
    return forward_deconv2d_arm(ctx);
}
// BASELINE_INJECT_END