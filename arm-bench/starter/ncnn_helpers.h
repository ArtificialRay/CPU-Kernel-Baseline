// ncnn_helpers.h — shared helpers for test files that link real ncnn kernels.
#pragma once
#include "../../ncnn/framework/mat.h"
#include "../../ncnn/framework/option.h"
#include <cstring>
#include <vector>

// ── Mat construction ─────────────────────────────────────────────────────────

// Create a 1-D ncnn::Mat from a flat float vector.
static inline ncnn::Mat make_mat_1d(const std::vector<float>& v)
{
    ncnn::Mat m;
    m.create((int)v.size(), 4u, (ncnn::Allocator*)0);
    memcpy((float*)m, v.data(), v.size() * sizeof(float));
    return m;
}

// Create a 2-D ncnn::Mat [h × w] from a flat row-major vector.
static inline ncnn::Mat make_mat_2d(int w_, int h_, const std::vector<float>& flat)
{
    ncnn::Mat m;
    m.create(w_, h_, 4u, (ncnn::Allocator*)0);
    for (int hh = 0; hh < h_; ++hh)
        memcpy(m.row(hh), flat.data() + hh * w_, w_ * sizeof(float));
    return m;
}

// Create a 3-D ncnn::Mat [c × h × w] from a flat c-major vector.
static inline ncnn::Mat make_mat(int w_, int h_, int c_, const std::vector<float>& flat)
{
    ncnn::Mat m;
    m.create(w_, h_, c_, 4u, (ncnn::Allocator*)0);
    for (int cc = 0; cc < c_; ++cc)
        for (int hh = 0; hh < h_; ++hh) {
            float* dst = m.channel(cc).row(hh);
            const float* src = flat.data() + cc * h_ * w_ + hh * w_;
            memcpy(dst, src, w_ * sizeof(float));
        }
    return m;
}

// Create a 4-D ncnn::Mat [c × d × h × w] from a flat c-major vector.
// ncnn 4D layout: channel c, depth d, height h, width w
static inline ncnn::Mat make_mat_4d(int w_, int h_, int d_, int c_, const std::vector<float>& flat)
{
    ncnn::Mat m;
    m.create(w_, h_, d_, c_, 4u, (ncnn::Allocator*)0);
    for (int cc = 0; cc < c_; ++cc)
        for (int dd = 0; dd < d_; ++dd)
            for (int hh = 0; hh < h_; ++hh) {
                float* dst = m.channel(cc).depth(dd).row(hh);
                const float* src = flat.data() + cc * d_ * h_ * w_ + dd * h_ * w_ + hh * w_;
                memcpy(dst, src, w_ * sizeof(float));
            }
    return m;
}

// Create a flat 1-D weight Mat
static inline ncnn::Mat make_weight(const std::vector<float>& v)
{
    ncnn::Mat m;
    m.create((int)v.size(), 4u, (ncnn::Allocator*)0);
    memcpy((float*)m, v.data(), v.size() * sizeof(float));
    return m;
}

// Create a flat 1-D int8 weight Mat
static inline ncnn::Mat make_weight_int8(const std::vector<int8_t>& v)
{
    ncnn::Mat m;
    m.create((int)v.size(), (size_t)1u, (ncnn::Allocator*)0);
    memcpy((int8_t*)m, v.data(), v.size());
    return m;
}

// ── Mat readback ─────────────────────────────────────────────────────────────

// Read any ncnn::Mat into a flat float vector (c-major for 3D, row-major for 2D).
static inline void read_mat(const ncnn::Mat& m, std::vector<float>& flat)
{
    if (m.dims == 1) {
        flat.resize(m.w);
        memcpy(flat.data(), (const float*)m, m.w * sizeof(float));
    } else if (m.dims == 2) {
        flat.resize(m.h * m.w);
        for (int hh = 0; hh < m.h; ++hh)
            memcpy(flat.data() + hh * m.w, m.row(hh), m.w * sizeof(float));
    } else if (m.dims == 3) {
        flat.resize(m.c * m.h * m.w);
        for (int cc = 0; cc < m.c; ++cc)
            for (int hh = 0; hh < m.h; ++hh)
                memcpy(flat.data() + cc * m.h * m.w + hh * m.w,
                       m.channel(cc).row(hh), m.w * sizeof(float));
    } else { // dims == 4
        flat.resize(m.c * m.d * m.h * m.w);
        for (int cc = 0; cc < m.c; ++cc)
            for (int dd = 0; dd < m.d; ++dd)
                for (int hh = 0; hh < m.h; ++hh)
                    memcpy(flat.data() + cc * m.d * m.h * m.w + dd * m.h * m.w + hh * m.w,
                           m.channel(cc).depth(dd).row(hh), m.w * sizeof(float));
    }
}

// Read an int8 ncnn::Mat (elemsize==1) into a flat int8 vector (c-major, row-major).
// Correctly handles cstep padding between channels.
static inline void read_mat_int8(const ncnn::Mat& m, std::vector<int8_t>& flat)
{
    flat.resize(m.c * m.h * m.w);
    for (int cc = 0; cc < m.c; ++cc)
        for (int hh = 0; hh < m.h; ++hh)
            memcpy(flat.data() + cc * m.h * m.w + hh * m.w,
                   m.channel(cc).row(hh), m.w);
}

// ── Option helper ────────────────────────────────────────────────────────────

static inline ncnn::Option make_opt()
{
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    opt.use_fp16_storage = false;
    opt.use_bf16_storage = false;
    opt.use_sgemm_convolution = false;   // avoid stub Gemm layer in arm kernels
    opt.use_winograd_convolution = false; // avoid stub Winograd in arm kernels
    return opt;
}
