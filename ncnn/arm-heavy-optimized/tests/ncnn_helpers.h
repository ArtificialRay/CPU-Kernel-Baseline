// ncnn_helpers.h — shared helpers for test files that link real ncnn kernels.
// Include this after test_utils.h in any test compiled with add_ncnn_test().
#pragma once
#include "../../framework/mat.h"
#include "../../framework/option.h"
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

// Create a flat 1-D weight Mat (alias for make_mat_1d, explicit intent).
static inline ncnn::Mat make_weight(const std::vector<float>& v)
{
    ncnn::Mat m;
    m.create((int)v.size(), 4u, (ncnn::Allocator*)0);
    memcpy((float*)m, v.data(), v.size() * sizeof(float));
    return m;
}

// ── Mat readback ─────────────────────────────────────────────────────────────

// Read any ncnn::Mat into a flat float vector (c-major for 3-D, row-major for 2-D).
static inline void read_mat(const ncnn::Mat& m, std::vector<float>& flat)
{
    if (m.dims == 1) {
        flat.resize(m.w);
        memcpy(flat.data(), (const float*)m, m.w * sizeof(float));
    } else if (m.dims == 2) {
        flat.resize(m.h * m.w);
        for (int hh = 0; hh < m.h; ++hh)
            memcpy(flat.data() + hh * m.w, m.row(hh), m.w * sizeof(float));
    } else {
        flat.resize(m.c * m.h * m.w);
        for (int cc = 0; cc < m.c; ++cc)
            for (int hh = 0; hh < m.h; ++hh)
                memcpy(flat.data() + cc * m.h * m.w + hh * m.w,
                       m.channel(cc).row(hh), m.w * sizeof(float));
    }
}

// Read an int8 ncnn::Mat (elemsize==1) into a flat int8 vector.
static inline void read_mat_int8(const ncnn::Mat& m, std::vector<int8_t>& flat)
{
    flat.resize(m.w * m.h * m.c);
    memcpy(flat.data(), (const int8_t*)m, flat.size());
}

// ── Option helper ────────────────────────────────────────────────────────────

static inline ncnn::Option make_opt()
{
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    opt.use_fp16_storage = false;
    opt.use_bf16_storage = false;
    return opt;
}
