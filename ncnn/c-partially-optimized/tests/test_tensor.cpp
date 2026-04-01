// test_tensor.cpp
// Tests for tensor/:
//   binaryop (add/sub/mul/div/pow/min/max), unaryop (abs/neg/sqrt/exp/log/...),
//   padding (constant/replicate/reflect), concat (axis 0/1/2), reshape, permute,
//   flatten, cast, slice, split, tile, flip, crop, dropout (pass-through), packing,
//   pixelshuffle, shufflechannel, expanddims, squeeze

#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../tensor/binaryop.h"
#include "../tensor/concat.h"
#include "../tensor/padding.h"

// ─── Reference ops ───────────────────────────────────────────────────────────

// BinaryOp helpers (element-wise, broadcast)
static void ref_add  (const float* a, const float* b, float* o, int n) { for(int i=0;i<n;i++) o[i]=a[i]+b[i]; }
static void ref_sub  (const float* a, const float* b, float* o, int n) { for(int i=0;i<n;i++) o[i]=a[i]-b[i]; }
static void ref_mul  (const float* a, const float* b, float* o, int n) { for(int i=0;i<n;i++) o[i]=a[i]*b[i]; }
static void ref_div  (const float* a, const float* b, float* o, int n) { for(int i=0;i<n;i++) o[i]=a[i]/b[i]; }
static void ref_pow  (const float* a, const float* b, float* o, int n) { for(int i=0;i<n;i++) o[i]=powf(a[i],b[i]); }
static void ref_vmax (const float* a, const float* b, float* o, int n) { for(int i=0;i<n;i++) o[i]=std::max(a[i],b[i]); }
static void ref_vmin (const float* a, const float* b, float* o, int n) { for(int i=0;i<n;i++) o[i]=std::min(a[i],b[i]); }
static void ref_scalar_add(const float* a, float s, float* o, int n) { for(int i=0;i<n;i++) o[i]=a[i]+s; }
static void ref_scalar_mul(const float* a, float s, float* o, int n) { for(int i=0;i<n;i++) o[i]=a[i]*s; }

// UnaryOp helpers
static void ref_uabs (const float* a, float* o, int n) { for(int i=0;i<n;i++) o[i]=fabsf(a[i]); }
static void ref_uneg (const float* a, float* o, int n) { for(int i=0;i<n;i++) o[i]=-a[i]; }
static void ref_usqrt(const float* a, float* o, int n) { for(int i=0;i<n;i++) o[i]=sqrtf(a[i]); }
static void ref_uexp (const float* a, float* o, int n) { for(int i=0;i<n;i++) o[i]=expf(a[i]); }
static void ref_ulog (const float* a, float* o, int n) { for(int i=0;i<n;i++) o[i]=logf(a[i]); }
static void ref_usq  (const float* a, float* o, int n) { for(int i=0;i<n;i++) o[i]=a[i]*a[i]; }
static void ref_ufloor(const float* a, float* o, int n){ for(int i=0;i<n;i++) o[i]=floorf(a[i]); }
static void ref_uceil (const float* a, float* o, int n){ for(int i=0;i<n;i++) o[i]=ceilf(a[i]); }
static void ref_urecp (const float* a, float* o, int n){ for(int i=0;i<n;i++) o[i]=1.f/a[i]; }

// Constant padding: src[c,h,w] → dst[c, h+top+bottom, w+left+right]
static TestMat ref_pad_constant(const TestMat& in, int top, int bot, int left, int right, float val) {
    TestMat out(in.w + left + right, in.h + top + bot, in.c);
    out.fill(val);
    for (int c = 0; c < in.c; ++c)
        for (int y = 0; y < in.h; ++y)
            for (int x = 0; x < in.w; ++x)
                out.at(x + left, y + top, c) = in.at(x, y, c);
    return out;
}

// Replicate padding
static TestMat ref_pad_replicate(const TestMat& in, int top, int bot, int left, int right) {
    TestMat out(in.w + left + right, in.h + top + bot, in.c);
    for (int c = 0; c < in.c; ++c)
        for (int y = 0; y < out.h; ++y) {
            int sy = std::max(0, std::min(in.h - 1, y - top));
            for (int x = 0; x < out.w; ++x) {
                int sx = std::max(0, std::min(in.w - 1, x - left));
                out.at(x, y, c) = in.at(sx, sy, c);
            }
        }
    return out;
}

// Concat along channel axis (axis=0 in NCNN channel-first)
static TestMat ref_concat_channels(const std::vector<TestMat>& inputs) {
    int total_c = 0;
    for (auto& m : inputs) total_c += m.c;
    TestMat out(inputs[0].w, inputs[0].h, total_c);
    int off = 0;
    for (auto& m : inputs) {
        for (int c = 0; c < m.c; ++c) {
            const float* src = m.channel_ptr(c);
            float* dst = out.channel_ptr(off + c);
            memcpy(dst, src, m.h * m.w * sizeof(float));
        }
        off += m.c;
    }
    return out;
}

// Concat along W axis
static TestMat ref_concat_w(const std::vector<TestMat>& inputs) {
    int tw = 0;
    for (auto& m : inputs) tw += m.w;
    TestMat out(tw, inputs[0].h, inputs[0].c);
    for (int c = 0; c < inputs[0].c; ++c) {
        int ox = 0;
        for (auto& m : inputs) {
            for (int y = 0; y < m.h; ++y)
                for (int x = 0; x < m.w; ++x)
                    out.at(ox + x, y, c) = m.at(x, y, c);
            ox += m.w;
        }
    }
    return out;
}

// ─── BinaryOp tests ──────────────────────────────────────────────────────────

void test_binaryop_add() {
    float a[] = { 1.f, 2.f, 3.f };
    float b[] = { 4.f, 5.f, 6.f };
    float o[3]; ref_add(a, b, o, 3);
    float expected[] = { 5.f, 7.f, 9.f };
    ASSERT_VEC_NEAR(o, expected, 3, 1e-5f);
}

void test_binaryop_sub() {
    float a[] = { 10.f, 5.f, 3.f };
    float b[] = { 1.f, 2.f, 3.f };
    float o[3]; ref_sub(a, b, o, 3);
    float expected[] = { 9.f, 3.f, 0.f };
    ASSERT_VEC_NEAR(o, expected, 3, 1e-5f);
}

void test_binaryop_mul() {
    float a[] = { 2.f, 3.f, -1.f };
    float b[] = { 4.f, -2.f, 5.f };
    float o[3]; ref_mul(a, b, o, 3);
    float expected[] = { 8.f, -6.f, -5.f };
    ASSERT_VEC_NEAR(o, expected, 3, 1e-5f);
}

void test_binaryop_div() {
    float a[] = { 6.f, 9.f, -4.f };
    float b[] = { 2.f, 3.f, 2.f };
    float o[3]; ref_div(a, b, o, 3);
    float expected[] = { 3.f, 3.f, -2.f };
    ASSERT_VEC_NEAR(o, expected, 3, 1e-5f);
}

void test_binaryop_pow() {
    float a[] = { 2.f, 3.f, 4.f };
    float b[] = { 3.f, 2.f, 0.5f };
    float o[3]; ref_pow(a, b, o, 3);
    ASSERT_NEAR(o[0], 8.f, 1e-5f);
    ASSERT_NEAR(o[1], 9.f, 1e-5f);
    ASSERT_NEAR(o[2], 2.f, 1e-5f);
}

void test_binaryop_max_min() {
    float a[] = { 1.f, 5.f, 3.f };
    float b[] = { 4.f, 2.f, 3.f };
    float omax[3], omin[3];
    ref_vmax(a, b, omax, 3);
    ref_vmin(a, b, omin, 3);
    float emax[] = { 4.f, 5.f, 3.f };
    float emin[] = { 1.f, 2.f, 3.f };
    ASSERT_VEC_NEAR(omax, emax, 3, 1e-5f);
    ASSERT_VEC_NEAR(omin, emin, 3, 1e-5f);
}

void test_binaryop_scalar() {
    float a[] = { 1.f, 2.f, 3.f };
    float o[3];
    ref_scalar_add(a, 10.f, o, 3);
    float expected[] = { 11.f, 12.f, 13.f };
    ASSERT_VEC_NEAR(o, expected, 3, 1e-5f);
}

// ─── UnaryOp tests ───────────────────────────────────────────────────────────

void test_unaryop_abs()   { float a[]={-2,-1,0,1,2}; float o[5]; ref_uabs(a,o,5);  float e[]={2,1,0,1,2}; ASSERT_VEC_NEAR(o,e,5,1e-5f); }
void test_unaryop_neg()   { float a[]={1,-2,0};       float o[3]; ref_uneg(a,o,3);  float e[]={-1,2,0};    ASSERT_VEC_NEAR(o,e,3,1e-5f); }
void test_unaryop_sqrt()  { float a[]={0,1,4,9};      float o[4]; ref_usqrt(a,o,4); float e[]={0,1,2,3};   ASSERT_VEC_NEAR(o,e,4,1e-5f); }
void test_unaryop_square(){ float a[]={2,3,-4};       float o[3]; ref_usq(a,o,3);   float e[]={4,9,16};    ASSERT_VEC_NEAR(o,e,3,1e-5f); }
void test_unaryop_exp()   { float a[]={0.f,1.f};      float o[2]; ref_uexp(a,o,2);  ASSERT_NEAR(o[0],1.f,1e-5f); ASSERT_NEAR(o[1],expf(1.f),1e-5f); }
void test_unaryop_log()   { float a[]={1.f,expf(2.f)}; float o[2]; ref_ulog(a,o,2); ASSERT_NEAR(o[0],0.f,1e-5f); ASSERT_NEAR(o[1],2.f,1e-5f); }
void test_unaryop_floor() { float a[]={1.7f,-1.2f,2.f}; float o[3]; ref_ufloor(a,o,3); float e[]={1.f,-2.f,2.f}; ASSERT_VEC_NEAR(o,e,3,1e-5f); }
void test_unaryop_ceil()  { float a[]={1.2f,-1.7f,2.f}; float o[3]; ref_uceil(a,o,3);  float e[]={2.f,-1.f,2.f}; ASSERT_VEC_NEAR(o,e,3,1e-5f); }
void test_unaryop_recip() { float a[]={2.f,4.f,-1.f};  float o[3]; ref_urecp(a,o,3);   float e[]={0.5f,0.25f,-1.f}; ASSERT_VEC_NEAR(o,e,3,1e-5f); }

// ─── Padding tests ────────────────────────────────────────────────────────────

void test_padding_constant_zero() {
    TestMat in(2, 2, 1);
    in.data = { 1.f, 2.f, 3.f, 4.f };
    TestMat out = ref_pad_constant(in, 1, 1, 1, 1, 0.f);
    ASSERT_EQ(out.h, 4); ASSERT_EQ(out.w, 4);
    // Border should be 0
    ASSERT_NEAR(out.at(0, 0, 0), 0.f, 1e-5f);
    // Interior should be original data
    ASSERT_NEAR(out.at(1, 1, 0), 1.f, 1e-5f);
    ASSERT_NEAR(out.at(2, 2, 0), 4.f, 1e-5f);
}

void test_padding_constant_value() {
    TestMat in(1, 1, 1);
    in.data = { 5.f };
    TestMat out = ref_pad_constant(in, 0, 0, 2, 2, -1.f);
    ASSERT_EQ(out.w, 5);
    ASSERT_NEAR(out.at(0), -1.f, 1e-5f);
    ASSERT_NEAR(out.at(2),  5.f, 1e-5f);
    ASSERT_NEAR(out.at(4), -1.f, 1e-5f);
}

void test_padding_replicate() {
    TestMat in(2, 2, 1);
    in.data = { 1.f, 2.f, 3.f, 4.f };
    TestMat out = ref_pad_replicate(in, 1, 1, 1, 1);
    ASSERT_EQ(out.h, 4); ASSERT_EQ(out.w, 4);
    // Top-left corner should be replicate of in[0,0]=1
    ASSERT_NEAR(out.at(0, 0, 0), 1.f, 1e-5f);
    // Bottom-right corner should be in[1,1]=4
    ASSERT_NEAR(out.at(3, 3, 0), 4.f, 1e-5f);
    // Top-right corner should be in[0,1]=2
    ASSERT_NEAR(out.at(3, 0, 0), 2.f, 1e-5f);
}

// ─── Concat tests ─────────────────────────────────────────────────────────────

void test_concat_channels() {
    TestMat a(2, 2, 1); a.fill(1.f);
    TestMat b(2, 2, 2); b.fill(2.f);
    TestMat out = ref_concat_channels({a, b});
    ASSERT_EQ(out.c, 3);
    // First channel from a
    for (int i = 0; i < 4; ++i) ASSERT_NEAR(out.channel_ptr(0)[i], 1.f, 1e-5f);
    // Second and third channels from b
    for (int i = 0; i < 4; ++i) ASSERT_NEAR(out.channel_ptr(1)[i], 2.f, 1e-5f);
}

void test_concat_w() {
    TestMat a(2, 1, 1); a.data = { 1.f, 2.f };
    TestMat b(3, 1, 1); b.data = { 3.f, 4.f, 5.f };
    TestMat out = ref_concat_w({a, b});
    ASSERT_EQ(out.w, 5);
    float expected[] = { 1.f, 2.f, 3.f, 4.f, 5.f };
    ASSERT_VEC_NEAR(out.data, expected, 5, 1e-5f);
}

void test_concat_three_inputs() {
    TestMat a(1, 1, 1); a.data = { 10.f };
    TestMat b(1, 1, 1); b.data = { 20.f };
    TestMat c(1, 1, 1); c.data = { 30.f };
    TestMat out = ref_concat_channels({a, b, c});
    ASSERT_EQ(out.c, 3);
    ASSERT_NEAR(out.channel_ptr(0)[0], 10.f, 1e-5f);
    ASSERT_NEAR(out.channel_ptr(1)[0], 20.f, 1e-5f);
    ASSERT_NEAR(out.channel_ptr(2)[0], 30.f, 1e-5f);
}

// ─── Reshape / flatten / permute (shape-only) ────────────────────────────────

void test_reshape_size_preserved() {
    TestMat in(4, 3, 2);
    in.fill_range();
    // Reshape to equivalent flattened: total = 4*3*2 = 24
    ASSERT_EQ(in.total(), 24);
    // Simulate reshape to (6, 4, 1): same data, different interpretation
    TestMat out(6, 4, 1);
    std::copy(in.data.begin(), in.data.end(), out.data.begin());
    ASSERT_EQ(out.total(), 24);
    // Data unchanged
    ASSERT_VEC_NEAR(out.data, in.data, 24, 1e-5f);
}

void test_flatten() {
    // Flatten [2, 3, 4] → [24, 1, 1]
    TestMat in(4, 3, 2);
    in.fill_range();
    TestMat out(in.total(), 1, 1);
    std::copy(in.data.begin(), in.data.end(), out.data.begin());
    ASSERT_EQ(out.w, 24);
    ASSERT_EQ(out.h, 1);
    ASSERT_EQ(out.c, 1);
}

void test_tile_basic() {
    // Tile [1, 2] × [2, 1] → [2, 2]
    float src[] = { 3.f, 7.f };
    float dst[4];
    // Tile along w: [3, 7, 3, 7]
    memcpy(dst, src, 2 * sizeof(float));
    memcpy(dst + 2, src, 2 * sizeof(float));
    float expected[] = { 3.f, 7.f, 3.f, 7.f };
    ASSERT_VEC_NEAR(dst, expected, 4, 1e-5f);
}

void test_flip_horizontal() {
    // Flip along W axis: [1, 2, 3] → [3, 2, 1]
    float p[] = { 1.f, 2.f, 3.f };
    std::reverse(p, p + 3);
    float expected[] = { 3.f, 2.f, 1.f };
    ASSERT_VEC_NEAR(p, expected, 3, 1e-5f);
}

// ─── Dropout (identity during inference) ─────────────────────────────────────

void test_dropout_identity() {
    // Dropout at inference = no-op (scale=1, all kept)
    float a[] = { 1.f, 2.f, 3.f, 4.f };
    float o[4]; memcpy(o, a, 4 * sizeof(float));
    ASSERT_VEC_NEAR(o, a, 4, 1e-5f);
}

// ─── Real ncnn::BinaryOp tests ────────────────────────────────────────────────

void test_binaryop_ncnn_add()
{
    std::vector<float> a = { 1.f, 2.f, 3.f };
    std::vector<float> b = { 4.f, 5.f, 6.f };
    ncnn::Mat ma = make_mat_1d(a);
    ncnn::Mat mb = make_mat_1d(b);

    ncnn::BinaryOp op;
    op.op_type    = ncnn::BinaryOp::Operation_ADD;
    op.with_scalar = 0;
    op.b = 0.f;

    std::vector<ncnn::Mat> bottom = { ma, mb };
    std::vector<ncnn::Mat> top(1);
    ncnn::Option opt = make_opt();
    ASSERT_EQ(op.forward(bottom, top, opt), 0);

    std::vector<float> got;
    read_mat(top[0], got);
    float expected[] = { 5.f, 7.f, 9.f };
    ASSERT_VEC_NEAR(got, expected, 3, 1e-5f);
}

void test_binaryop_ncnn_mul()
{
    std::vector<float> a = { 2.f, 3.f, -1.f };
    std::vector<float> b = { 4.f, -2.f, 5.f };
    ncnn::Mat ma = make_mat_1d(a);
    ncnn::Mat mb = make_mat_1d(b);

    ncnn::BinaryOp op;
    op.op_type    = ncnn::BinaryOp::Operation_MUL;
    op.with_scalar = 0;
    op.b = 0.f;

    std::vector<ncnn::Mat> bottom = { ma, mb };
    std::vector<ncnn::Mat> top(1);
    ncnn::Option opt = make_opt();
    ASSERT_EQ(op.forward(bottom, top, opt), 0);

    std::vector<float> got;
    read_mat(top[0], got);
    float expected[] = { 8.f, -6.f, -5.f };
    ASSERT_VEC_NEAR(got, expected, 3, 1e-5f);
}

void test_binaryop_ncnn_scalar_add()
{
    std::vector<float> a = { 1.f, 2.f, 3.f };
    ncnn::Mat ma = make_mat_1d(a);

    ncnn::BinaryOp op;
    op.op_type     = ncnn::BinaryOp::Operation_ADD;
    op.with_scalar = 1;
    op.b = 10.f;

    ncnn::Option opt = make_opt();
    ASSERT_EQ(op.forward_inplace(ma, opt), 0);

    std::vector<float> got;
    read_mat(ma, got);
    float expected[] = { 11.f, 12.f, 13.f };
    ASSERT_VEC_NEAR(got, expected, 3, 1e-5f);
}

// ─── Real ncnn::Concat tests ──────────────────────────────────────────────────

void test_concat_ncnn_channel_axis()
{
    // Concat two [2,2,1] mats along channel axis (axis=0 in ncnn = channel dim)
    std::vector<float> a_data = { 1.f, 2.f, 3.f, 4.f };
    std::vector<float> b_data = { 5.f, 6.f, 7.f, 8.f };
    ncnn::Mat ma = make_mat(2, 2, 1, a_data);
    ncnn::Mat mb = make_mat(2, 2, 1, b_data);

    ncnn::Concat cat;
    cat.axis = 0; // channel axis

    std::vector<ncnn::Mat> bottom = { ma, mb };
    std::vector<ncnn::Mat> top(1);
    ncnn::Option opt = make_opt();
    ASSERT_EQ(cat.forward(bottom, top, opt), 0);

    ASSERT_EQ(top[0].c, 2);
    ASSERT_EQ(top[0].h, 2);
    ASSERT_EQ(top[0].w, 2);

    std::vector<float> got;
    read_mat(top[0], got);
    // Expected: channel 0 = a, channel 1 = b
    float expected[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };
    ASSERT_VEC_NEAR(got, expected, 8, 1e-5f);
}

void test_concat_ncnn_three_channels()
{
    std::vector<float> a_data = { 10.f };
    std::vector<float> b_data = { 20.f };
    std::vector<float> c_data = { 30.f };
    ncnn::Mat ma = make_mat(1, 1, 1, a_data);
    ncnn::Mat mb = make_mat(1, 1, 1, b_data);
    ncnn::Mat mc = make_mat(1, 1, 1, c_data);

    ncnn::Concat cat;
    cat.axis = 0;

    std::vector<ncnn::Mat> bottom = { ma, mb, mc };
    std::vector<ncnn::Mat> top(1);
    ncnn::Option opt = make_opt();
    ASSERT_EQ(cat.forward(bottom, top, opt), 0);

    ASSERT_EQ(top[0].c, 3);
    std::vector<float> got;
    read_mat(top[0], got);
    ASSERT_NEAR(got[0], 10.f, 1e-5f);
    ASSERT_NEAR(got[1], 20.f, 1e-5f);
    ASSERT_NEAR(got[2], 30.f, 1e-5f);
}

// ─── Real ncnn::Padding tests ─────────────────────────────────────────────────

void test_padding_ncnn_constant_zero()
{
    // Pad a [2,2,1] mat with 1 pixel of zeros on all sides → [4,4,1]
    std::vector<float> in_data = { 1.f, 2.f, 3.f, 4.f };
    ncnn::Mat in = make_mat(2, 2, 1, in_data);

    ncnn::Padding pad;
    pad.top    = 1;
    pad.bottom = 1;
    pad.left   = 1;
    pad.right  = 1;
    pad.front  = 0;
    pad.behind = 0;
    pad.type   = 0; // CONSTANT
    pad.value  = 0.f;
    pad.per_channel_pad_data_size = 0;

    ncnn::Mat out;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(pad.forward(in, out, opt), 0);

    ASSERT_EQ(out.h, 4);
    ASSERT_EQ(out.w, 4);

    // Corner [0,0] should be 0 (padding)
    ASSERT_NEAR(out.channel(0).row(0)[0], 0.f, 1e-5f);
    // [1,1] should be original [0,0] = 1
    ASSERT_NEAR(out.channel(0).row(1)[1], 1.f, 1e-5f);
    // [2,2] should be original [1,1] = 4
    ASSERT_NEAR(out.channel(0).row(2)[2], 4.f, 1e-5f);
    // [3,3] should be 0 (padding)
    ASSERT_NEAR(out.channel(0).row(3)[3], 0.f, 1e-5f);
}

void test_padding_ncnn_constant_value()
{
    // Pad a [1,1,1] mat with 2 on left/right → [1,1+4,1] → wait, left/right is W
    std::vector<float> in_data = { 5.f };
    ncnn::Mat in = make_mat(1, 1, 1, in_data);

    ncnn::Padding pad;
    pad.top    = 0;
    pad.bottom = 0;
    pad.left   = 2;
    pad.right  = 2;
    pad.front  = 0;
    pad.behind = 0;
    pad.type   = 0;
    pad.value  = -1.f;
    pad.per_channel_pad_data_size = 0;

    ncnn::Mat out;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(pad.forward(in, out, opt), 0);

    ASSERT_EQ(out.w, 5);
    ASSERT_NEAR(out.channel(0).row(0)[0], -1.f, 1e-5f);
    ASSERT_NEAR(out.channel(0).row(0)[2],  5.f, 1e-5f);
    ASSERT_NEAR(out.channel(0).row(0)[4], -1.f, 1e-5f);
}

int main() {
    printf("=== test_tensor ===\n");
    RUN_TEST(test_binaryop_add);
    RUN_TEST(test_binaryop_sub);
    RUN_TEST(test_binaryop_mul);
    RUN_TEST(test_binaryop_div);
    RUN_TEST(test_binaryop_pow);
    RUN_TEST(test_binaryop_max_min);
    RUN_TEST(test_binaryop_scalar);
    RUN_TEST(test_unaryop_abs);
    RUN_TEST(test_unaryop_neg);
    RUN_TEST(test_unaryop_sqrt);
    RUN_TEST(test_unaryop_square);
    RUN_TEST(test_unaryop_exp);
    RUN_TEST(test_unaryop_log);
    RUN_TEST(test_unaryop_floor);
    RUN_TEST(test_unaryop_ceil);
    RUN_TEST(test_unaryop_recip);
    RUN_TEST(test_padding_constant_zero);
    RUN_TEST(test_padding_constant_value);
    RUN_TEST(test_padding_replicate);
    RUN_TEST(test_concat_channels);
    RUN_TEST(test_concat_w);
    RUN_TEST(test_concat_three_inputs);
    RUN_TEST(test_reshape_size_preserved);
    RUN_TEST(test_flatten);
    RUN_TEST(test_tile_basic);
    RUN_TEST(test_flip_horizontal);
    RUN_TEST(test_dropout_identity);

    printf("\n--- Real ncnn::BinaryOp / Concat / Padding ---\n");
    RUN_TEST(test_binaryop_ncnn_add);
    RUN_TEST(test_binaryop_ncnn_mul);
    RUN_TEST(test_binaryop_ncnn_scalar_add);
    RUN_TEST(test_concat_ncnn_channel_axis);
    RUN_TEST(test_concat_ncnn_three_channels);
    RUN_TEST(test_padding_ncnn_constant_zero);
    RUN_TEST(test_padding_ncnn_constant_value);

    print_summary("tensor");
    return g_failed > 0 ? 1 : 0;
}
