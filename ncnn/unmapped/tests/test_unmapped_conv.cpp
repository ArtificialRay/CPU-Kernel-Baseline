// test_unmapped_conv.cpp
// Tests for unmapped conv kernel implementations (no ARM variants):
//   Convolution3D, ConvolutionDepthWise1D, ConvolutionDepthWise3D,
//   Deconvolution1D, Deconvolution3D,
//   DeconvolutionDepthWise1D, DeconvolutionDepthWise3D,
//   DeformableConv2D.

#include "test_utils.h"
#include "ncnn_helpers.h"

#include "../unmapped/convolution3d/convolution3d.h"
#include "../unmapped/convolutiondepthwise1d/convolutiondepthwise1d.h"
#include "../unmapped/convolutiondepthwise3d/convolutiondepthwise3d.h"
#include "../unmapped/deconvolution1d/deconvolution1d.h"
#include "../unmapped/deconvolution3d/deconvolution3d.h"
#include "../unmapped/deconvolutiondepthwise1d/deconvolutiondepthwise1d.h"
#include "../unmapped/deconvolutiondepthwise3d/deconvolutiondepthwise3d.h"
#include "../unmapped/deformableconv2d/deformableconv2d.h"

// ═══════════════════════════════════════════════════════════════════
// ── Reference implementations ────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════

// Flat index helpers for compact notation
// 3D mat (no depth): flat[c * h * w + y * w + x]
// 4D mat:            flat[c * d * h * w + z * h * w + y * w + x]

inline float at3(const std::vector<float>& f, int c, int y, int x, int H, int W) {
    return f[c * H * W + y * W + x];
}
inline float& at3(std::vector<float>& f, int c, int y, int x, int H, int W) {
    return f[c * H * W + y * W + x];
}
inline float at4(const std::vector<float>& f, int c, int z, int y, int x,
                  int D, int H, int W) {
    return f[c * D * H * W + z * H * W + y * W + x];
}
inline float& at4(std::vector<float>& f, int c, int z, int y, int x,
                   int D, int H, int W) {
    return f[c * D * H * W + z * H * W + y * W + x];
}

// 3-D convolution reference
// weight: [out_c, in_c, kd, kh, kw]
// input : [in_c, d, h, w]  (no padding)
// output: [out_c, outd, outh, outw]
static std::vector<float> ref_conv3d(
        const std::vector<float>& in,
        int in_c, int in_d, int in_h, int in_w,
        const std::vector<float>& weight,
        const std::vector<float>& bias,
        int out_c, int kd, int kh, int kw,
        int sd, int sh, int sw)
{
    int outd = (in_d - kd) / sd + 1;
    int outh = (in_h - kh) / sh + 1;
    int outw = (in_w - kw) / sw + 1;
    std::vector<float> out(out_c * outd * outh * outw, 0.f);

    for (int oc = 0; oc < out_c; ++oc)
    for (int oz = 0; oz < outd; ++oz)
    for (int oy = 0; oy < outh; ++oy)
    for (int ox = 0; ox < outw; ++ox) {
        float sum = bias.empty() ? 0.f : bias[oc];
        for (int ic = 0; ic < in_c; ++ic)
        for (int kdi = 0; kdi < kd; ++kdi)
        for (int khi = 0; khi < kh; ++khi)
        for (int kwi = 0; kwi < kw; ++kwi) {
            int iz = oz * sd + kdi;
            int iy = oy * sh + khi;
            int ix = ox * sw + kwi;
            float px = at4(in, ic, iz, iy, ix, in_d, in_h, in_w);
            int widx = ((oc * in_c + ic) * kd + kdi) * kh * kw + khi * kw + kwi;
            sum += px * weight[widx];
        }
        at4(out, oc, oz, oy, ox, outd, outh, outw) = sum;
    }
    return out;
}

// Depthwise 3-D convolution reference
// weight: [c, kd, kh, kw]
// input : [c, d, h, w]
static std::vector<float> ref_dw_conv3d(
        const std::vector<float>& in,
        int c, int in_d, int in_h, int in_w,
        const std::vector<float>& weight,
        const std::vector<float>& bias,
        int kd, int kh, int kw,
        int sd, int sh, int sw)
{
    int outd = (in_d - kd) / sd + 1;
    int outh = (in_h - kh) / sh + 1;
    int outw = (in_w - kw) / sw + 1;
    std::vector<float> out(c * outd * outh * outw, 0.f);

    for (int ch = 0; ch < c; ++ch)
    for (int oz = 0; oz < outd; ++oz)
    for (int oy = 0; oy < outh; ++oy)
    for (int ox = 0; ox < outw; ++ox) {
        float sum = bias.empty() ? 0.f : bias[ch];
        for (int kdi = 0; kdi < kd; ++kdi)
        for (int khi = 0; khi < kh; ++khi)
        for (int kwi = 0; kwi < kw; ++kwi) {
            float px = at4(in, ch, oz * sd + kdi, oy * sh + khi, ox * sw + kwi, in_d, in_h, in_w);
            int widx = (ch * kd + kdi) * kh * kw + khi * kw + kwi;
            sum += px * weight[widx];
        }
        at4(out, ch, oz, oy, ox, outd, outh, outw) = sum;
    }
    return out;
}

// 1-D depthwise convolution reference
// Input layout: [h=channels, w=length] (2D mat, no channel dim in ncnn)
// weight: [group, kw]
static std::vector<float> ref_dw_conv1d(
        const std::vector<float>& in,
        int channels, int in_w,
        const std::vector<float>& weight,
        const std::vector<float>& bias,
        int kw, int sw, int pad_l, int dil_w)
{
    int out_w = (in_w + 2 * pad_l - dil_w * (kw - 1) - 1) / sw + 1;
    std::vector<float> out(channels * out_w, 0.f);
    for (int ch = 0; ch < channels; ++ch)
    for (int ow = 0; ow < out_w; ++ow) {
        float sum = bias.empty() ? 0.f : bias[ch];
        for (int ki = 0; ki < kw; ++ki) {
            int iw = ow * sw - pad_l + ki * dil_w;
            float px = (iw >= 0 && iw < in_w) ? in[ch * in_w + iw] : 0.f;
            sum += px * weight[ch * kw + ki];
        }
        out[ch * out_w + ow] = sum;
    }
    return out;
}

// 1-D deconvolution reference
// Input: [h=in_c, w=in_len], weight: [out_c, in_c, kw] → flattened as [in_c * out_c * kw]
// Wait, looking at deconvolution1d.cpp:
//   weight_ptr[kernel_w * h * p + q * kernel_w + k]
//   → layout: [out_c, in_c, kw]
static std::vector<float> ref_deconv1d(
        const std::vector<float>& in,
        int in_c, int in_len,
        const std::vector<float>& weight,
        const std::vector<float>& bias,
        int out_c, int kw, int sw, int dil_w = 1)
{
    int ke_w = dil_w * (kw - 1) + 1;
    int out_len = (in_len - 1) * sw + ke_w;
    std::vector<float> out(out_c * out_len, 0.f);

    if (!bias.empty())
        for (int oc = 0; oc < out_c; ++oc)
            for (int i = 0; i < out_len; ++i)
                out[oc * out_len + i] = bias[oc];

    for (int p = 0; p < out_c; ++p)
    for (int j = 0; j < in_len; ++j) {
        for (int q = 0; q < in_c; ++q) {
            float val = in[q * in_len + j];
            for (int k = 0; k < kw; ++k) {
                int ox = j * sw + k * dil_w;
                float wt = weight[kw * in_c * p + q * kw + k];
                out[p * out_len + ox] += val * wt;
            }
        }
    }
    return out;
}

// Depthwise 1-D deconvolution reference (group = channels)
// weight: [channels, kw]
static std::vector<float> ref_dw_deconv1d(
        const std::vector<float>& in,
        int channels, int in_len,
        const std::vector<float>& weight,
        const std::vector<float>& bias,
        int kw, int sw, int dil_w = 1)
{
    int ke_w = dil_w * (kw - 1) + 1;
    int out_len = (in_len - 1) * sw + ke_w;
    std::vector<float> out(channels * out_len, 0.f);

    if (!bias.empty())
        for (int ch = 0; ch < channels; ++ch)
            for (int i = 0; i < out_len; ++i)
                out[ch * out_len + i] = bias[ch];

    for (int ch = 0; ch < channels; ++ch)
    for (int j = 0; j < in_len; ++j) {
        float val = in[ch * in_len + j];
        for (int k = 0; k < kw; ++k) {
            int ox = j * sw + k * dil_w;
            out[ch * out_len + ox] += val * weight[ch * kw + k];
        }
    }
    return out;
}

// 3-D deconvolution reference
// weight: [out_c, in_c, kd, kh, kw] (ncnn layout: per-outch block of in_c*kd*kh*kw)
static std::vector<float> ref_deconv3d(
        const std::vector<float>& in,
        int in_c, int in_d, int in_h, int in_w,
        const std::vector<float>& weight,
        const std::vector<float>& bias,
        int out_c, int kd, int kh, int kw,
        int sd, int sh, int sw)
{
    int out_d = (in_d - 1) * sd + kd;
    int out_h = (in_h - 1) * sh + kh;
    int out_w = (in_w - 1) * sw + kw;
    std::vector<float> out(out_c * out_d * out_h * out_w, 0.f);

    if (!bias.empty())
        for (int oc = 0; oc < out_c; ++oc)
            for (int i = 0; i < out_d * out_h * out_w; ++i)
                out[oc * out_d * out_h * out_w + i] = bias[oc];

    for (int oc = 0; oc < out_c; ++oc)
    for (int iz = 0; iz < in_d; ++iz)
    for (int iy = 0; iy < in_h; ++iy)
    for (int ix = 0; ix < in_w; ++ix) {
        for (int ic = 0; ic < in_c; ++ic) {
            float val = at4(in, ic, iz, iy, ix, in_d, in_h, in_w);
            for (int kdi = 0; kdi < kd; ++kdi)
            for (int khi = 0; khi < kh; ++khi)
            for (int kwi = 0; kwi < kw; ++kwi) {
                int oz = iz * sd + kdi;
                int oy = iy * sh + khi;
                int ox = ix * sw + kwi;
                // weight layout: [out_c, in_c, kd, kh, kw] matching ncnn deconv3d forward
                int widx = ((oc * in_c + ic) * kd + kdi) * kh * kw + khi * kw + kwi;
                at4(out, oc, oz, oy, ox, out_d, out_h, out_w) += val * weight[widx];
            }
        }
    }
    return out;
}

// Depthwise 3-D deconvolution reference
// weight: [channels, kd, kh, kw]
static std::vector<float> ref_dw_deconv3d(
        const std::vector<float>& in,
        int channels, int in_d, int in_h, int in_w,
        const std::vector<float>& weight,
        const std::vector<float>& bias,
        int kd, int kh, int kw,
        int sd, int sh, int sw)
{
    int out_d = (in_d - 1) * sd + kd;
    int out_h = (in_h - 1) * sh + kh;
    int out_w = (in_w - 1) * sw + kw;
    std::vector<float> out(channels * out_d * out_h * out_w, 0.f);

    if (!bias.empty())
        for (int ch = 0; ch < channels; ++ch)
            for (int i = 0; i < out_d * out_h * out_w; ++i)
                out[ch * out_d * out_h * out_w + i] = bias[ch];

    for (int ch = 0; ch < channels; ++ch)
    for (int iz = 0; iz < in_d; ++iz)
    for (int iy = 0; iy < in_h; ++iy)
    for (int ix = 0; ix < in_w; ++ix) {
        float val = at4(in, ch, iz, iy, ix, in_d, in_h, in_w);
        for (int kdi = 0; kdi < kd; ++kdi)
        for (int khi = 0; khi < kh; ++khi)
        for (int kwi = 0; kwi < kw; ++kwi) {
            int oz = iz * sd + kdi;
            int oy = iy * sh + khi;
            int ox = ix * sw + kwi;
            int widx = (ch * kd + kdi) * kh * kw + khi * kw + kwi;
            at4(out, ch, oz, oy, ox, out_d, out_h, out_w) += val * weight[widx];
        }
    }
    return out;
}

// Deformable conv 2D reference (no mask)
// weight: [out_c, in_c, kh, kw]
// offset: [2 * kh * kw, out_h, out_w]  (delta_h, delta_w for each kernel pos)
static std::vector<float> ref_deformable_conv2d(
        const std::vector<float>& in,
        int in_c, int in_h, int in_w,
        const std::vector<float>& weight,
        const std::vector<float>& bias,
        const std::vector<float>& offset,   // [kh*kw*2, out_h, out_w]
        int out_c, int kh, int kw,
        int sh, int sw, int dil_h, int dil_w,
        int pad_top, int pad_left)
{
    int out_h = (in_h + pad_top * 2 - dil_h * (kh - 1) - 1) / sh + 1;
    int out_w = (in_w + pad_left * 2 - dil_w * (kw - 1) - 1) / sw + 1;
    std::vector<float> out(out_c * out_h * out_w, 0.f);

    for (int h_col = 0; h_col < out_h; ++h_col)
    for (int w_col = 0; w_col < out_w; ++w_col) {
        int h_in_base = h_col * sh - pad_top;
        int w_in_base = w_col * sw - pad_left;
        for (int oc = 0; oc < out_c; ++oc) {
            float sum = bias.empty() ? 0.f : bias[oc];
            for (int ki = 0; ki < kh; ++ki)
            for (int kj = 0; kj < kw; ++kj) {
                int kern_idx = ki * kw + kj;
                // offset layout matches deformableconv2d.cpp:
                // offset.channel((i * kernel_w + j) * 2)     = delta_h
                // offset.channel((i * kernel_w + j) * 2 + 1) = delta_w
                float dh = offset[(kern_idx * 2 + 0) * out_h * out_w + h_col * out_w + w_col];
                float dw = offset[(kern_idx * 2 + 1) * out_h * out_w + h_col * out_w + w_col];
                float h_im = h_in_base + ki * dil_h + dh;
                float w_im = w_in_base + kj * dil_w + dw;
                // Bilinear interpolation
                bool cond = h_im > -1 && w_im > -1 && h_im < in_h && w_im < in_w;
                for (int ic = 0; ic < in_c; ++ic) {
                    float val = 0.f;
                    if (cond) {
                        int h_low  = (int)floorf(h_im), h_high = h_low + 1;
                        int w_low  = (int)floorf(w_im), w_high = w_low + 1;
                        float lh = h_im - h_low, lw = w_im - w_low;
                        float hh = 1 - lh, hw = 1 - lw;
                        float v1 = (h_low  >= 0 && w_low  >= 0) ? in[ic * in_h * in_w + h_low  * in_w + w_low ] : 0.f;
                        float v2 = (h_low  >= 0 && w_high <= in_w-1) ? in[ic * in_h * in_w + h_low  * in_w + w_high] : 0.f;
                        float v3 = (h_high <= in_h-1 && w_low  >= 0) ? in[ic * in_h * in_w + h_high * in_w + w_low ] : 0.f;
                        float v4 = (h_high <= in_h-1 && w_high <= in_w-1) ? in[ic * in_h * in_w + h_high * in_w + w_high] : 0.f;
                        val = hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4;
                    }
                    int widx = ((oc * in_c + ic) * kh + ki) * kw + kj;
                    sum += val * weight[widx];
                }
            }
            out[oc * out_h * out_w + h_col * out_w + w_col] = sum;
        }
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════
// ── Kernel runner helpers ─────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════

static bool run_conv3d(int in_c, int out_c,
                        int in_d, int in_h, int in_w,
                        int kd, int kh, int kw,
                        int sd = 1, int sh = 1, int sw = 1,
                        bool with_bias = false)
{
    int wsize = out_c * in_c * kd * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.05f; }

    std::vector<float> in_flat(in_c * in_d * in_h * in_w);
    for (int i = 0; i < (int)in_flat.size(); ++i) in_flat[i] = (i + 1) * 0.1f;

    ncnn::Mat bottom = make_mat_4d(in_w, in_h, in_d, in_c, in_flat);
    ncnn::Mat top;

    ncnn::Convolution3D conv3d;
    conv3d.num_output       = out_c;
    conv3d.kernel_w         = kw;    conv3d.kernel_h  = kh;    conv3d.kernel_d = kd;
    conv3d.dilation_w       = 1;     conv3d.dilation_h = 1;    conv3d.dilation_d = 1;
    conv3d.stride_w         = sw;    conv3d.stride_h  = sh;    conv3d.stride_d = sd;
    conv3d.pad_left         = 0; conv3d.pad_right  = 0;
    conv3d.pad_top          = 0; conv3d.pad_bottom = 0;
    conv3d.pad_front        = 0; conv3d.pad_behind = 0;
    conv3d.pad_value        = 0.f;
    conv3d.bias_term        = with_bias ? 1 : 0;
    conv3d.weight_data_size = wsize;
    conv3d.activation_type  = 0;
    conv3d.weight_data      = make_weight(weight);
    if (with_bias) conv3d.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    int ret = conv3d.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Convolution3D::forward failed %d\n", ret); g_failed++; return false; }

    std::vector<float> ref = ref_conv3d(in_flat, in_c, in_d, in_h, in_w, weight, bias, out_c, kd, kh, kw, sd, sh, sw);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_EQ((int)got.size(), (int)ref.size());
    ASSERT_VEC_NEAR(got, ref.data(), (int)ref.size(), 1e-3f);
    return g_failed == before;
}

static bool run_dw_conv3d(int c, int in_d, int in_h, int in_w,
                            int kd, int kh, int kw,
                            int sd = 1, int sh = 1, int sw = 1,
                            bool with_bias = false)
{
    int wsize = c * kd * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(c); for (int i = 0; i < c; ++i) bias[i] = i * 0.05f; }

    std::vector<float> in_flat(c * in_d * in_h * in_w);
    for (int i = 0; i < (int)in_flat.size(); ++i) in_flat[i] = (i + 1) * 0.1f;

    ncnn::Mat bottom = make_mat_4d(in_w, in_h, in_d, c, in_flat);
    ncnn::Mat top;

    ncnn::ConvolutionDepthWise3D dw3d;
    dw3d.num_output       = c;
    dw3d.kernel_w         = kw;    dw3d.kernel_h  = kh;    dw3d.kernel_d = kd;
    dw3d.dilation_w       = 1;     dw3d.dilation_h = 1;    dw3d.dilation_d = 1;
    dw3d.stride_w         = sw;    dw3d.stride_h  = sh;    dw3d.stride_d = sd;
    dw3d.pad_left         = 0; dw3d.pad_right  = 0;
    dw3d.pad_top          = 0; dw3d.pad_bottom = 0;
    dw3d.pad_front        = 0; dw3d.pad_behind = 0;
    dw3d.pad_value        = 0.f;
    dw3d.bias_term        = with_bias ? 1 : 0;
    dw3d.weight_data_size = wsize;
    dw3d.group            = c;
    dw3d.activation_type  = 0;
    dw3d.weight_data      = make_weight(weight);
    if (with_bias) dw3d.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    int ret = dw3d.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  ConvolutionDepthWise3D::forward failed %d\n", ret); g_failed++; return false; }

    std::vector<float> ref = ref_dw_conv3d(in_flat, c, in_d, in_h, in_w, weight, bias, kd, kh, kw, sd, sh, sw);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_EQ((int)got.size(), (int)ref.size());
    ASSERT_VEC_NEAR(got, ref.data(), (int)ref.size(), 1e-3f);
    return g_failed == before;
}

// ConvolutionDepthWise1D: input as 2D mat [h=channels, w=length]
static bool run_dw_conv1d(int channels, int in_len, int kw,
                            int sw = 1, int pad_l = 0, int dil_w = 1,
                            bool with_bias = false)
{
    int wsize = channels * kw;
    std::vector<float> weight = make_weights(wsize, 0.5f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(channels); for (int i = 0; i < channels; ++i) bias[i] = i * 0.05f; }

    std::vector<float> in_flat(channels * in_len);
    for (int i = 0; i < (int)in_flat.size(); ++i) in_flat[i] = (i + 1) * 0.1f;

    ncnn::Mat bottom = make_mat_2d(in_len, channels, in_flat);
    ncnn::Mat top;

    ncnn::ConvolutionDepthWise1D dw1d;
    dw1d.num_output       = channels;
    dw1d.kernel_w         = kw;
    dw1d.dilation_w       = dil_w;
    dw1d.stride_w         = sw;
    dw1d.pad_left         = pad_l; dw1d.pad_right = pad_l;
    dw1d.pad_value        = 0.f;
    dw1d.bias_term        = with_bias ? 1 : 0;
    dw1d.weight_data_size = wsize;
    dw1d.group            = channels;
    dw1d.activation_type  = 0;
    dw1d.dynamic_weight   = 0;
    dw1d.weight_data      = make_weight(weight);
    if (with_bias) dw1d.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    int ret = dw1d.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  ConvolutionDepthWise1D::forward failed %d\n", ret); g_failed++; return false; }

    std::vector<float> ref = ref_dw_conv1d(in_flat, channels, in_len, weight, bias, kw, sw, pad_l, dil_w);

    // Read top (2D mat: h=channels, w=out_len)
    int out_len = (in_len + 2 * pad_l - dil_w * (kw - 1) - 1) / sw + 1;
    std::vector<float> got(channels * out_len);
    for (int ch = 0; ch < channels; ++ch)
        memcpy(got.data() + ch * out_len, top.row(ch), out_len * sizeof(float));

    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data(), (int)ref.size(), 1e-3f);
    return g_failed == before;
}

// Deconvolution1D: input as 2D mat [h=in_c, w=in_len]
static bool run_deconv1d(int in_c, int out_c, int in_len, int kw,
                          int sw = 1, bool with_bias = false)
{
    int wsize = in_c * out_c * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.05f; }

    std::vector<float> in_flat(in_c * in_len);
    for (int i = 0; i < (int)in_flat.size(); ++i) in_flat[i] = (i + 1) * 0.1f;

    ncnn::Mat bottom = make_mat_2d(in_len, in_c, in_flat);
    ncnn::Mat top;

    ncnn::Deconvolution1D deconv1d;
    deconv1d.num_output         = out_c;
    deconv1d.kernel_w           = kw;
    deconv1d.dilation_w         = 1;
    deconv1d.stride_w           = sw;
    deconv1d.pad_left           = 0; deconv1d.pad_right = 0;
    deconv1d.output_pad_right   = 0;
    deconv1d.output_w           = 0;
    deconv1d.bias_term          = with_bias ? 1 : 0;
    deconv1d.weight_data_size   = wsize;
    deconv1d.activation_type    = 0;
    deconv1d.dynamic_weight     = 0;
    deconv1d.weight_data        = make_weight(weight);
    if (with_bias) deconv1d.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    int ret = deconv1d.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Deconvolution1D::forward failed %d\n", ret); g_failed++; return false; }

    std::vector<float> ref = ref_deconv1d(in_flat, in_c, in_len, weight, bias, out_c, kw, sw);

    int out_len = (in_len - 1) * sw + kw;
    std::vector<float> got(out_c * out_len);
    for (int oc = 0; oc < out_c; ++oc)
        memcpy(got.data() + oc * out_len, top.row(oc), out_len * sizeof(float));

    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data(), (int)ref.size(), 1e-3f);
    return g_failed == before;
}

// DeconvolutionDepthWise1D: depthwise transposed 1D conv
// Input: [h=channels, w=in_len], weight: [channels, kw]
static bool run_dw_deconv1d(int channels, int in_len, int kw,
                              int sw = 1, bool with_bias = false)
{
    int wsize = channels * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(channels); for (int i = 0; i < channels; ++i) bias[i] = i * 0.05f; }

    std::vector<float> in_flat(channels * in_len);
    for (int i = 0; i < (int)in_flat.size(); ++i) in_flat[i] = (i + 1) * 0.1f;

    ncnn::Mat bottom = make_mat_2d(in_len, channels, in_flat);
    ncnn::Mat top;

    ncnn::DeconvolutionDepthWise1D ddw1d;
    ddw1d.num_output         = channels;
    ddw1d.kernel_w           = kw;
    ddw1d.dilation_w         = 1;
    ddw1d.stride_w           = sw;
    ddw1d.pad_left           = 0; ddw1d.pad_right = 0;
    ddw1d.output_pad_right   = 0;
    ddw1d.output_w           = 0;
    ddw1d.bias_term          = with_bias ? 1 : 0;
    ddw1d.weight_data_size   = wsize;
    ddw1d.group              = channels;
    ddw1d.activation_type    = 0;
    ddw1d.dynamic_weight     = 0;
    ddw1d.weight_data        = make_weight(weight);
    if (with_bias) ddw1d.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    int ret = ddw1d.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  DeconvolutionDepthWise1D::forward failed %d\n", ret); g_failed++; return false; }

    std::vector<float> ref = ref_dw_deconv1d(in_flat, channels, in_len, weight, bias, kw, sw);

    int out_len = (in_len - 1) * sw + kw;
    std::vector<float> got(channels * out_len);
    for (int ch = 0; ch < channels; ++ch)
        memcpy(got.data() + ch * out_len, top.row(ch), out_len * sizeof(float));

    int before = g_failed;
    ASSERT_VEC_NEAR(got, ref.data(), (int)ref.size(), 1e-3f);
    return g_failed == before;
}

// Deconvolution3D
static bool run_deconv3d(int in_c, int out_c,
                          int in_d, int in_h, int in_w,
                          int kd, int kh, int kw,
                          int sd = 1, int sh = 1, int sw = 1,
                          bool with_bias = false)
{
    int wsize = in_c * out_c * kd * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.05f; }

    std::vector<float> in_flat(in_c * in_d * in_h * in_w);
    for (int i = 0; i < (int)in_flat.size(); ++i) in_flat[i] = (i + 1) * 0.1f;

    ncnn::Mat bottom = make_mat_4d(in_w, in_h, in_d, in_c, in_flat);
    ncnn::Mat top;

    ncnn::Deconvolution3D deconv3d;
    deconv3d.num_output         = out_c;
    deconv3d.kernel_w           = kw;    deconv3d.kernel_h  = kh;    deconv3d.kernel_d = kd;
    deconv3d.dilation_w         = 1;     deconv3d.dilation_h = 1;    deconv3d.dilation_d = 1;
    deconv3d.stride_w           = sw;    deconv3d.stride_h  = sh;    deconv3d.stride_d = sd;
    deconv3d.pad_left           = 0; deconv3d.pad_right  = 0;
    deconv3d.pad_top            = 0; deconv3d.pad_bottom = 0;
    deconv3d.pad_front          = 0; deconv3d.pad_behind = 0;
    deconv3d.output_pad_right   = 0; deconv3d.output_pad_bottom = 0; deconv3d.output_pad_behind = 0;
    deconv3d.output_w           = 0; deconv3d.output_h = 0; deconv3d.output_d = 0;
    deconv3d.bias_term          = with_bias ? 1 : 0;
    deconv3d.weight_data_size   = wsize;
    deconv3d.activation_type    = 0;
    deconv3d.weight_data        = make_weight(weight);
    if (with_bias) deconv3d.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    int ret = deconv3d.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  Deconvolution3D::forward failed %d\n", ret); g_failed++; return false; }

    std::vector<float> ref = ref_deconv3d(in_flat, in_c, in_d, in_h, in_w, weight, bias, out_c, kd, kh, kw, sd, sh, sw);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_EQ((int)got.size(), (int)ref.size());
    ASSERT_VEC_NEAR(got, ref.data(), (int)ref.size(), 1e-3f);
    return g_failed == before;
}

// DeconvolutionDepthWise3D
static bool run_dw_deconv3d(int channels,
                              int in_d, int in_h, int in_w,
                              int kd, int kh, int kw,
                              int sd = 1, int sh = 1, int sw = 1,
                              bool with_bias = false)
{
    int wsize = channels * kd * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(channels); for (int i = 0; i < channels; ++i) bias[i] = i * 0.05f; }

    std::vector<float> in_flat(channels * in_d * in_h * in_w);
    for (int i = 0; i < (int)in_flat.size(); ++i) in_flat[i] = (i + 1) * 0.1f;

    ncnn::Mat bottom = make_mat_4d(in_w, in_h, in_d, channels, in_flat);
    ncnn::Mat top;

    ncnn::DeconvolutionDepthWise3D ddw3d;
    ddw3d.num_output         = channels;
    ddw3d.kernel_w           = kw;    ddw3d.kernel_h  = kh;    ddw3d.kernel_d = kd;
    ddw3d.dilation_w         = 1;     ddw3d.dilation_h = 1;    ddw3d.dilation_d = 1;
    ddw3d.stride_w           = sw;    ddw3d.stride_h  = sh;    ddw3d.stride_d = sd;
    ddw3d.pad_left           = 0; ddw3d.pad_right  = 0;
    ddw3d.pad_top            = 0; ddw3d.pad_bottom = 0;
    ddw3d.pad_front          = 0; ddw3d.pad_behind = 0;
    ddw3d.output_pad_right   = 0; ddw3d.output_pad_bottom = 0; ddw3d.output_pad_behind = 0;
    ddw3d.output_w           = 0; ddw3d.output_h = 0; ddw3d.output_d = 0;
    ddw3d.bias_term          = with_bias ? 1 : 0;
    ddw3d.weight_data_size   = wsize;
    ddw3d.group              = channels;
    ddw3d.activation_type    = 0;
    ddw3d.weight_data        = make_weight(weight);
    if (with_bias) ddw3d.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    int ret = ddw3d.forward(bottom, top, opt);
    if (ret != 0) { fprintf(stderr, "  DeconvolutionDepthWise3D::forward failed %d\n", ret); g_failed++; return false; }

    std::vector<float> ref = ref_dw_deconv3d(in_flat, channels, in_d, in_h, in_w, weight, bias, kd, kh, kw, sd, sh, sw);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_EQ((int)got.size(), (int)ref.size());
    ASSERT_VEC_NEAR(got, ref.data(), (int)ref.size(), 1e-3f);
    return g_failed == before;
}

// DeformableConv2D (no mask)
static bool run_deformable_conv2d(int in_c, int out_c,
                                   int in_h, int in_w,
                                   int kh, int kw,
                                   int sh = 1, int sw = 1,
                                   int pad_top = 0, int pad_left = 0,
                                   bool with_bias = false)
{
    int out_h = (in_h + pad_top * 2 - kh) / sh + 1;
    int out_w = (in_w + pad_left * 2 - kw) / sw + 1;

    int wsize = out_c * in_c * kh * kw;
    std::vector<float> weight = make_weights(wsize, 0.3f);
    std::vector<float> bias;
    if (with_bias) { bias.resize(out_c); for (int i = 0; i < out_c; ++i) bias[i] = i * 0.05f; }

    std::vector<float> in_flat(in_c * in_h * in_w);
    for (int i = 0; i < (int)in_flat.size(); ++i) in_flat[i] = (i + 1) * 0.1f;

    // offset: [2 * kh * kw, out_h, out_w], set to 0 (no deformation)
    int offset_size = 2 * kh * kw * out_h * out_w;
    std::vector<float> offset_flat(offset_size, 0.f);

    ncnn::Mat bottom = make_mat(in_w, in_h, in_c, in_flat);
    // offset mat: channel = 2*kh*kw, h=out_h, w=out_w
    ncnn::Mat offset_mat = make_mat(out_w, out_h, 2 * kh * kw, offset_flat);
    ncnn::Mat top;

    ncnn::DeformableConv2D deform;
    deform.num_output       = out_c;
    deform.kernel_w         = kw;    deform.kernel_h  = kh;
    deform.dilation_w       = 1;     deform.dilation_h = 1;
    deform.stride_w         = sw;    deform.stride_h  = sh;
    deform.pad_left         = pad_left; deform.pad_right  = pad_left;
    deform.pad_top          = pad_top;  deform.pad_bottom = pad_top;
    deform.bias_term        = with_bias ? 1 : 0;
    deform.weight_data_size = wsize;
    deform.activation_type  = 0;
    deform.weight_data      = make_weight(weight);
    if (with_bias) deform.bias_data = make_weight(bias);

    ncnn::Option opt = make_opt();
    std::vector<ncnn::Mat> bottoms = {bottom, offset_mat};
    std::vector<ncnn::Mat> tops(1);
    int ret = deform.forward(bottoms, tops, opt);
    if (ret != 0) { fprintf(stderr, "  DeformableConv2D::forward failed %d\n", ret); g_failed++; return false; }

    top = tops[0];
    std::vector<float> ref = ref_deformable_conv2d(
        in_flat, in_c, in_h, in_w, weight, bias, offset_flat,
        out_c, kh, kw, sh, sw, 1, 1, pad_top, pad_left);
    std::vector<float> got; read_mat(top, got);
    int before = g_failed;
    ASSERT_EQ((int)got.size(), (int)ref.size());
    ASSERT_VEC_NEAR(got, ref.data(), (int)ref.size(), 1e-3f);
    return g_failed == before;
}

// ═══════════════════════════════════════════════════════════════════
// ── Test cases ────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════

void test_conv3d_1x1x1()  { ASSERT_TRUE(run_conv3d(2, 4, 4, 4, 4, 1, 1, 1)); }
void test_conv3d_1x3x3()  { ASSERT_TRUE(run_conv3d(2, 4, 4, 5, 5, 1, 3, 3)); }
void test_conv3d_3x3x3()  { ASSERT_TRUE(run_conv3d(2, 4, 4, 5, 5, 3, 3, 3)); }
void test_conv3d_stride2() { ASSERT_TRUE(run_conv3d(2, 4, 6, 6, 6, 2, 2, 2, 2, 2, 2)); }
void test_conv3d_bias()   { ASSERT_TRUE(run_conv3d(2, 4, 4, 4, 4, 2, 2, 2, 1, 1, 1, true)); }
void test_conv3d_sizes()  {
    ASSERT_TRUE(run_conv3d(1, 1, 3, 3, 3, 1, 1, 1));
    ASSERT_TRUE(run_conv3d(3, 2, 5, 5, 5, 2, 2, 2));
    ASSERT_TRUE(run_conv3d(4, 8, 6, 6, 6, 3, 3, 3));
}

void test_dw_conv3d_basic() { ASSERT_TRUE(run_dw_conv3d(4, 4, 5, 5, 2, 2, 2)); }
void test_dw_conv3d_1x1x1() { ASSERT_TRUE(run_dw_conv3d(4, 5, 5, 5, 1, 1, 1)); }
void test_dw_conv3d_bias()  { ASSERT_TRUE(run_dw_conv3d(4, 4, 5, 5, 2, 2, 2, 1, 1, 1, true)); }
void test_dw_conv3d_sizes() {
    ASSERT_TRUE(run_dw_conv3d(2, 4, 4, 4, 2, 2, 2));
    ASSERT_TRUE(run_dw_conv3d(8, 6, 6, 6, 3, 3, 3));
}

void test_dw_conv1d_basic() { ASSERT_TRUE(run_dw_conv1d(4, 8, 3)); }
void test_dw_conv1d_pad()   { ASSERT_TRUE(run_dw_conv1d(4, 8, 3, 1, 1)); }
void test_dw_conv1d_stride() { ASSERT_TRUE(run_dw_conv1d(4, 8, 3, 2, 0)); }
void test_dw_conv1d_bias()  { ASSERT_TRUE(run_dw_conv1d(4, 8, 3, 1, 1, 1, true)); }
void test_dw_conv1d_sizes() {
    ASSERT_TRUE(run_dw_conv1d(1, 10, 3));
    ASSERT_TRUE(run_dw_conv1d(8, 16, 5, 1, 2));
}

void test_deconv1d_basic()  { ASSERT_TRUE(run_deconv1d(2, 4, 5, 3)); }
void test_deconv1d_stride() { ASSERT_TRUE(run_deconv1d(2, 4, 4, 2, 2)); }
void test_deconv1d_bias()   { ASSERT_TRUE(run_deconv1d(2, 4, 5, 3, 1, true)); }
void test_deconv1d_sizes()  {
    ASSERT_TRUE(run_deconv1d(1, 1, 4, 2));
    ASSERT_TRUE(run_deconv1d(4, 4, 6, 3, 2));
}

void test_dw_deconv1d_basic()  { ASSERT_TRUE(run_dw_deconv1d(4, 6, 3)); }
void test_dw_deconv1d_stride() { ASSERT_TRUE(run_dw_deconv1d(4, 4, 2, 2)); }
void test_dw_deconv1d_bias()   { ASSERT_TRUE(run_dw_deconv1d(4, 6, 3, 1, true)); }

void test_deconv3d_1x1x1() { ASSERT_TRUE(run_deconv3d(2, 4, 3, 3, 3, 1, 1, 1)); }
void test_deconv3d_2x2x2() { ASSERT_TRUE(run_deconv3d(2, 4, 3, 3, 3, 2, 2, 2)); }
void test_deconv3d_stride() { ASSERT_TRUE(run_deconv3d(2, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2)); }
void test_deconv3d_bias()   { ASSERT_TRUE(run_deconv3d(2, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, true)); }
void test_deconv3d_sizes()  {
    ASSERT_TRUE(run_deconv3d(1, 1, 2, 2, 2, 1, 1, 1));
    ASSERT_TRUE(run_deconv3d(3, 2, 3, 4, 4, 2, 2, 2));
}

void test_dw_deconv3d_basic() { ASSERT_TRUE(run_dw_deconv3d(4, 3, 3, 3, 2, 2, 2)); }
void test_dw_deconv3d_stride() { ASSERT_TRUE(run_dw_deconv3d(4, 3, 3, 3, 2, 2, 2, 2, 2, 2)); }
void test_dw_deconv3d_bias()  { ASSERT_TRUE(run_dw_deconv3d(4, 3, 3, 3, 2, 2, 2, 1, 1, 1, true)); }

void test_deformable_3x3()  { ASSERT_TRUE(run_deformable_conv2d(2, 4, 6, 6, 3, 3)); }
void test_deformable_1x1()  { ASSERT_TRUE(run_deformable_conv2d(2, 4, 4, 4, 1, 1)); }
void test_deformable_pad()  { ASSERT_TRUE(run_deformable_conv2d(2, 4, 5, 5, 3, 3, 1, 1, 1, 1)); }
void test_deformable_bias() { ASSERT_TRUE(run_deformable_conv2d(2, 4, 5, 5, 3, 3, 1, 1, 0, 0, true)); }
void test_deformable_sizes() {
    ASSERT_TRUE(run_deformable_conv2d(1, 1, 4, 4, 3, 3));
    ASSERT_TRUE(run_deformable_conv2d(4, 8, 8, 8, 3, 3, 1, 1, 1, 1));
}

// ═══════════════════════════════════════════════════════════════════
int main()
{
    printf("=== test_unmapped_conv ===\n");

    printf("\n-- Convolution3D --\n");
    RUN_TEST(test_conv3d_1x1x1);
    RUN_TEST(test_conv3d_1x3x3);
    RUN_TEST(test_conv3d_3x3x3);
    RUN_TEST(test_conv3d_stride2);
    RUN_TEST(test_conv3d_bias);
    RUN_TEST(test_conv3d_sizes);

    printf("\n-- ConvolutionDepthWise3D --\n");
    RUN_TEST(test_dw_conv3d_basic);
    RUN_TEST(test_dw_conv3d_1x1x1);
    RUN_TEST(test_dw_conv3d_bias);
    RUN_TEST(test_dw_conv3d_sizes);

    printf("\n-- ConvolutionDepthWise1D --\n");
    RUN_TEST(test_dw_conv1d_basic);
    RUN_TEST(test_dw_conv1d_pad);
    RUN_TEST(test_dw_conv1d_stride);
    RUN_TEST(test_dw_conv1d_bias);
    RUN_TEST(test_dw_conv1d_sizes);

    printf("\n-- Deconvolution1D --\n");
    RUN_TEST(test_deconv1d_basic);
    RUN_TEST(test_deconv1d_stride);
    RUN_TEST(test_deconv1d_bias);
    RUN_TEST(test_deconv1d_sizes);

    printf("\n-- DeconvolutionDepthWise1D --\n");
    RUN_TEST(test_dw_deconv1d_basic);
    RUN_TEST(test_dw_deconv1d_stride);
    RUN_TEST(test_dw_deconv1d_bias);

    printf("\n-- Deconvolution3D --\n");
    RUN_TEST(test_deconv3d_1x1x1);
    RUN_TEST(test_deconv3d_2x2x2);
    RUN_TEST(test_deconv3d_stride);
    RUN_TEST(test_deconv3d_bias);
    RUN_TEST(test_deconv3d_sizes);

    printf("\n-- DeconvolutionDepthWise3D --\n");
    RUN_TEST(test_dw_deconv3d_basic);
    RUN_TEST(test_dw_deconv3d_stride);
    RUN_TEST(test_dw_deconv3d_bias);

    printf("\n-- DeformableConv2D --\n");
    RUN_TEST(test_deformable_3x3);
    RUN_TEST(test_deformable_1x1);
    RUN_TEST(test_deformable_pad);
    RUN_TEST(test_deformable_bias);
    RUN_TEST(test_deformable_sizes);

    print_summary("unmapped_conv");
    return g_failed > 0 ? 1 : 0;
}
