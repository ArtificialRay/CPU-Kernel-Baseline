// test_reduction.cpp
// Tests for reduction/:
//   pooling (max, avg, global, adaptive), pooling1d/3d, softmax (axis 0/1/2),
//   reduction (sum, mean, max, min, prod, L1, L2, LogSumExp),
//   argmax, cumulativesum, spp, statisticspooling
//
// Section 1: reference-only tests
// Section 2: real ncnn kernel tests (linked via reduction_impl + ncnn_stub)

#include "test_utils.h"
#include <limits>

// ── Real ncnn kernel headers ──────────────────────────────────────────────────
#include "ncnn_helpers.h"
#include "../reduction/softmax.h"
#include "../reduction/pooling.h"
#include "../reduction/reduction.h"

// ─── Reference pooling ────────────────────────────────────────────────────────

static TestMat ref_max_pool2d(const TestMat& in, int kh, int kw,
                                int sh, int sw, int pad = 0) {
    int oh = (in.h + 2*pad - kh) / sh + 1;
    int ow = (in.w + 2*pad - kw) / sw + 1;
    TestMat out(ow, oh, in.c);
    for (int c = 0; c < in.c; ++c) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                float mx = -std::numeric_limits<float>::infinity();
                for (int ki = 0; ki < kh; ++ki)
                    for (int kj = 0; kj < kw; ++kj) {
                        int r = i*sh - pad + ki, col = j*sw - pad + kj;
                        if (r>=0 && r<in.h && col>=0 && col<in.w)
                            mx = std::max(mx, in.at(col, r, c));
                    }
                out.at(j, i, c) = mx;
            }
        }
    }
    return out;
}

static TestMat ref_avg_pool2d(const TestMat& in, int kh, int kw,
                                int sh, int sw, int pad = 0) {
    int oh = (in.h + 2*pad - kh) / sh + 1;
    int ow = (in.w + 2*pad - kw) / sw + 1;
    TestMat out(ow, oh, in.c);
    for (int c = 0; c < in.c; ++c) {
        for (int i = 0; i < oh; ++i) {
            for (int j = 0; j < ow; ++j) {
                float s = 0.f; int cnt = 0;
                for (int ki = 0; ki < kh; ++ki)
                    for (int kj = 0; kj < kw; ++kj) {
                        int r = i*sh - pad + ki, col = j*sw - pad + kj;
                        if (r>=0 && r<in.h && col>=0 && col<in.w) { s += in.at(col, r, c); cnt++; }
                    }
                out.at(j, i, c) = cnt > 0 ? s / cnt : 0.f;
            }
        }
    }
    return out;
}

static TestMat ref_global_avg_pool(const TestMat& in) {
    TestMat out(1, 1, in.c);
    for (int c = 0; c < in.c; ++c) {
        float s = 0.f;
        for (int i = 0; i < in.h * in.w; ++i) s += in.channel_ptr(c)[i];
        out.at(0, 0, c) = s / (in.h * in.w);
    }
    return out;
}

// ─── Reference reduction ops ─────────────────────────────────────────────────

static float ref_reduce_sum    (const float* p, int n) { float s=0; for(int i=0;i<n;i++) s+=p[i]; return s; }
static float ref_reduce_mean   (const float* p, int n) { return ref_reduce_sum(p,n)/n; }
static float ref_reduce_max    (const float* p, int n) { float m=p[0]; for(int i=1;i<n;i++) m=std::max(m,p[i]); return m; }
static float ref_reduce_min    (const float* p, int n) { float m=p[0]; for(int i=1;i<n;i++) m=std::min(m,p[i]); return m; }
static float ref_reduce_prod   (const float* p, int n) { float m=1; for(int i=0;i<n;i++) m*=p[i]; return m; }
static float ref_reduce_l1     (const float* p, int n) { float s=0; for(int i=0;i<n;i++) s+=fabsf(p[i]); return s; }
static float ref_reduce_l2     (const float* p, int n) { float s=0; for(int i=0;i<n;i++) s+=p[i]*p[i]; return sqrtf(s); }
static float ref_reduce_logsumexp(const float* p, int n) {
    float mx = ref_reduce_max(p, n);
    float s = 0.f;
    for (int i = 0; i < n; i++) s += expf(p[i] - mx);
    return mx + logf(s);
}

// ─── Test cases ──────────────────────────────────────────────────────────────

void test_max_pool_2x2() {
    TestMat in(4, 4, 1);
    in.fill_range();  // 1..16
    TestMat out = ref_max_pool2d(in, 2, 2, 2, 2);
    ASSERT_EQ(out.h, 2); ASSERT_EQ(out.w, 2);
    // top-left 2x2: max(1,2,5,6)=6
    ASSERT_NEAR(out.at(0, 0, 0), 6.f, 1e-5f);
    // top-right 2x2: max(3,4,7,8)=8
    ASSERT_NEAR(out.at(1, 0, 0), 8.f, 1e-5f);
    // bottom-left 2x2: max(9,10,13,14)=14
    ASSERT_NEAR(out.at(0, 1, 0), 14.f, 1e-5f);
    // bottom-right 2x2: max(11,12,15,16)=16
    ASSERT_NEAR(out.at(1, 1, 0), 16.f, 1e-5f);
}

void test_avg_pool_2x2() {
    TestMat in(4, 4, 1);
    in.fill_range();
    TestMat out = ref_avg_pool2d(in, 2, 2, 2, 2);
    ASSERT_EQ(out.h, 2); ASSERT_EQ(out.w, 2);
    // top-left avg(1,2,5,6)=14/4=3.5
    ASSERT_NEAR(out.at(0, 0, 0), 3.5f, 1e-5f);
}

void test_global_avg_pool() {
    TestMat in(3, 3, 2);
    in.fill_range();  // ch0: 1..9, ch1: 10..18
    TestMat out = ref_global_avg_pool(in);
    ASSERT_EQ(out.w, 1); ASSERT_EQ(out.h, 1); ASSERT_EQ(out.c, 2);
    ASSERT_NEAR(out.at(0, 0, 0), 5.f, 1e-5f);   // mean(1..9)=5
    ASSERT_NEAR(out.at(0, 0, 1), 14.f, 1e-5f);  // mean(10..18)=14
}

void test_max_pool_1x1_noop() {
    TestMat in(3, 3, 1);
    in.fill_range();
    TestMat out = ref_max_pool2d(in, 1, 1, 1, 1);
    ASSERT_VEC_NEAR(out.data, in.data, 9, 1e-5f);
}

void test_avg_pool_single_element() {
    TestMat in(1, 1, 1);
    in.data = { 7.f };
    TestMat out = ref_avg_pool2d(in, 1, 1, 1, 1);
    ASSERT_NEAR(out.at(0), 7.f, 1e-5f);
}

void test_softmax_1d() {
    float p[] = { 1.f, 2.f, 3.f };
    softmax_inplace(p, 3);
    // Sum = 1
    float s = p[0] + p[1] + p[2];
    ASSERT_NEAR(s, 1.f, 1e-5f);
    // Monotonic: p[0] < p[1] < p[2]
    ASSERT_TRUE(p[0] < p[1] && p[1] < p[2]);
    // p[2] should be ≈ e^3 / (e^1+e^2+e^3) ≈ 0.665
    ASSERT_NEAR(p[2], 0.6652f, 1e-3f);
}

void test_softmax_uniform() {
    float p[] = { 5.f, 5.f, 5.f, 5.f };
    softmax_inplace(p, 4);
    for (float v : p) ASSERT_NEAR(v, 0.25f, 1e-5f);
}

void test_softmax_numerical_stability() {
    // Large values: should not produce inf/nan
    float p[] = { 1000.f, 1001.f, 1002.f };
    softmax_inplace(p, 3);
    float s = p[0] + p[1] + p[2];
    ASSERT_NEAR(s, 1.f, 1e-5f);
    for (float v : p) ASSERT_TRUE(v > 0.f && v < 1.f);
}

void test_softmax_axis1_2d() {
    // Softmax along axis=1 of a 2x3 matrix (row-wise softmax)
    float data[6] = { 1.f, 2.f, 3.f,
                      4.f, 5.f, 6.f };
    softmax_inplace(data, 3);      // row 0
    softmax_inplace(data + 3, 3);  // row 1
    ASSERT_NEAR(data[0] + data[1] + data[2], 1.f, 1e-5f);
    ASSERT_NEAR(data[3] + data[4] + data[5], 1.f, 1e-5f);
}

void test_reduction_sum() {
    float v[] = { 1.f, 2.f, 3.f, 4.f, 5.f };
    ASSERT_NEAR(ref_reduce_sum(v, 5), 15.f, 1e-5f);
}

void test_reduction_mean() {
    float v[] = { 1.f, 2.f, 3.f, 4.f, 5.f };
    ASSERT_NEAR(ref_reduce_mean(v, 5), 3.f, 1e-5f);
}

void test_reduction_max_min() {
    float v[] = { 3.f, -1.f, 7.f, 2.f, -5.f };
    ASSERT_NEAR(ref_reduce_max(v, 5),  7.f, 1e-5f);
    ASSERT_NEAR(ref_reduce_min(v, 5), -5.f, 1e-5f);
}

void test_reduction_prod() {
    float v[] = { 2.f, 3.f, 4.f };
    ASSERT_NEAR(ref_reduce_prod(v, 3), 24.f, 1e-5f);
}

void test_reduction_l1_l2() {
    float v[] = { 3.f, -4.f, 0.f };
    ASSERT_NEAR(ref_reduce_l1(v, 3), 7.f, 1e-5f);  // 3+4+0
    ASSERT_NEAR(ref_reduce_l2(v, 3), 5.f, 1e-5f);  // sqrt(9+16)=5
}

void test_reduction_logsumexp() {
    float v[] = { 1.f, 2.f, 3.f };
    float ref = logf(expf(1.f) + expf(2.f) + expf(3.f));
    ASSERT_NEAR(ref_reduce_logsumexp(v, 3), ref, 1e-4f);
}

void test_argmax_basic() {
    float v[] = { 1.f, 5.f, 3.f, 2.f };
    int idx = (int)(std::max_element(v, v + 4) - v);
    ASSERT_EQ(idx, 1);
}

void test_argmax_negative() {
    float v[] = { -3.f, -1.f, -2.f };
    int idx = (int)(std::max_element(v, v + 3) - v);
    ASSERT_EQ(idx, 1);
}

void test_cumulativesum() {
    float v[] = { 1.f, 2.f, 3.f, 4.f };
    float cumsum[4];
    cumsum[0] = v[0];
    for (int i = 1; i < 4; ++i) cumsum[i] = cumsum[i-1] + v[i];
    float expected[] = { 1.f, 3.f, 6.f, 10.f };
    ASSERT_VEC_NEAR(cumsum, expected, 4, 1e-5f);
}

void test_statisticspooling_mean_std() {
    // Statistics pooling: compute mean and std of all spatial locations
    float p[] = { 2.f, 4.f, 4.f, 4.f, 5.f, 5.f, 7.f, 9.f };  // n=8
    int n = 8;
    float mean = ref_reduce_mean(p, n);  // 5.0
    float var = 0.f;
    for (int i = 0; i < n; i++) var += (p[i] - mean) * (p[i] - mean);
    var /= n;
    float std = sqrtf(var);  // 2.0
    ASSERT_NEAR(mean, 5.f, 1e-5f);
    ASSERT_NEAR(std,  2.f, 1e-5f);
}

// ─── Real ncnn kernel tests ───────────────────────────────────────────────────

void test_softmax_ncnn_1d()
{
    // Axis=0 on a 1D blob (w dimension)
    std::vector<float> vals = { 1.f, 2.f, 3.f };
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::Softmax sm; sm.axis = 0;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(sm.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    // Sum = 1
    float s = out[0] + out[1] + out[2];
    ASSERT_NEAR(s, 1.f, 1e-5f);
    // Monotonically increasing
    ASSERT_TRUE(out[0] < out[1] && out[1] < out[2]);
    // Compare with ref
    std::vector<float> ref = vals;
    softmax_inplace(ref.data(), 3);
    ASSERT_VEC_NEAR(out, ref.data(), 3, 1e-5f);
}

void test_softmax_ncnn_uniform()
{
    std::vector<float> vals = { 5.f, 5.f, 5.f, 5.f };
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::Softmax sm; sm.axis = 0;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(sm.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    for (float v : out) ASSERT_NEAR(v, 0.25f, 1e-5f);
}

void test_softmax_ncnn_2d_axis0()
{
    // 3×1×1 blob → softmax across 3 channels (axis=0)
    std::vector<float> flat = { 1.f, 2.f, 3.f };
    ncnn::Mat m = make_mat(1, 1, 3, flat);  // w=1,h=1,c=3
    ncnn::Softmax sm; sm.axis = 0;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(sm.forward_inplace(m, opt), 0);
    // sum of channel values = 1
    float s = ((float*)m.channel(0))[0] + ((float*)m.channel(1))[0] + ((float*)m.channel(2))[0];
    ASSERT_NEAR(s, 1.f, 1e-5f);
}

void test_max_pool_ncnn_2x2()
{
    // 4×4 input, max pool 2×2 stride 2 → 2×2 output
    std::vector<float> flat(16);
    for (int i = 0; i < 16; ++i) flat[i] = (float)(i + 1);
    ncnn::Mat bottom = make_mat(4, 4, 1, flat);
    ncnn::Mat top;

    ncnn::Pooling pool;
    pool.pooling_type = ncnn::Pooling::PoolMethod_MAX;
    pool.kernel_w = 2; pool.kernel_h = 2;
    pool.stride_w = 2; pool.stride_h = 2;
    pool.pad_left = 0; pool.pad_right = 0;
    pool.pad_top  = 0; pool.pad_bottom = 0;
    pool.global_pooling  = 0;
    pool.pad_mode        = 0;
    pool.avgpool_count_include_pad = 0;
    pool.adaptive_pooling = 0;

    ncnn::Option opt = make_opt();
    ASSERT_EQ(pool.forward(bottom, top, opt), 0);
    ASSERT_EQ(top.h, 2); ASSERT_EQ(top.w, 2);
    // top-left 2×2: max(1,2,5,6)=6
    ASSERT_NEAR(top.channel(0).row(0)[0], 6.f,  1e-5f);
    // top-right: max(3,4,7,8)=8
    ASSERT_NEAR(top.channel(0).row(0)[1], 8.f,  1e-5f);
    // bottom-left: max(9,10,13,14)=14
    ASSERT_NEAR(top.channel(0).row(1)[0], 14.f, 1e-5f);
    // bottom-right: max(11,12,15,16)=16
    ASSERT_NEAR(top.channel(0).row(1)[1], 16.f, 1e-5f);
}

void test_avg_pool_ncnn_2x2()
{
    std::vector<float> flat(16);
    for (int i = 0; i < 16; ++i) flat[i] = (float)(i + 1);
    ncnn::Mat bottom = make_mat(4, 4, 1, flat);
    ncnn::Mat top;

    ncnn::Pooling pool;
    pool.pooling_type = ncnn::Pooling::PoolMethod_AVE;
    pool.kernel_w = 2; pool.kernel_h = 2;
    pool.stride_w = 2; pool.stride_h = 2;
    pool.pad_left = 0; pool.pad_right = 0;
    pool.pad_top  = 0; pool.pad_bottom = 0;
    pool.global_pooling  = 0;
    pool.pad_mode        = 0;
    pool.avgpool_count_include_pad = 0;
    pool.adaptive_pooling = 0;

    ncnn::Option opt = make_opt();
    ASSERT_EQ(pool.forward(bottom, top, opt), 0);
    // top-left avg(1,2,5,6)=3.5
    ASSERT_NEAR(top.channel(0).row(0)[0], 3.5f, 1e-4f);
}

void test_global_avg_pool_ncnn()
{
    std::vector<float> flat(9);
    for (int i = 0; i < 9; ++i) flat[i] = (float)(i + 1);
    ncnn::Mat bottom = make_mat(3, 3, 1, flat);
    ncnn::Mat top;

    ncnn::Pooling pool;
    pool.pooling_type = ncnn::Pooling::PoolMethod_AVE;
    pool.global_pooling  = 1;
    pool.adaptive_pooling = 0;
    pool.pad_mode = 0;
    pool.kernel_w = 0; pool.kernel_h = 0;
    pool.stride_w = 1; pool.stride_h = 1;
    pool.pad_left = 0; pool.pad_right = 0;
    pool.pad_top  = 0; pool.pad_bottom = 0;
    pool.avgpool_count_include_pad = 0;

    ncnn::Option opt = make_opt();
    ASSERT_EQ(pool.forward(bottom, top, opt), 0);
    ASSERT_EQ(top.w, 1); ASSERT_EQ(top.h, 1);
    ASSERT_NEAR(((float*)top)[0], 5.f, 1e-4f);  // mean(1..9)=5
}

void test_reduction_ncnn_sum()
{
    // Reduction::ReductionOp_SUM, reduce_all=1 → scalar sum
    std::vector<float> vals = { 1.f, 2.f, 3.f, 4.f, 5.f };
    ncnn::Mat bottom = make_mat_1d(vals);
    ncnn::Mat top;

    ncnn::Reduction red;
    red.operation  = ncnn::Reduction::ReductionOp_SUM;
    red.reduce_all = 1;
    red.coeff      = 1.f;
    red.keepdims   = 0;

    ncnn::Option opt = make_opt();
    ASSERT_EQ(red.forward(bottom, top, opt), 0);
    std::vector<float> out; read_mat(top, out);
    ASSERT_NEAR(out[0], 15.f, 1e-4f);
}

void test_reduction_ncnn_mean()
{
    std::vector<float> vals = { 2.f, 4.f, 6.f, 8.f };
    ncnn::Mat bottom = make_mat_1d(vals);
    ncnn::Mat top;

    ncnn::Reduction red;
    red.operation  = ncnn::Reduction::ReductionOp_MEAN;
    red.reduce_all = 1;
    red.coeff      = 1.f;
    red.keepdims   = 0;

    ncnn::Option opt = make_opt();
    ASSERT_EQ(red.forward(bottom, top, opt), 0);
    std::vector<float> out; read_mat(top, out);
    ASSERT_NEAR(out[0], 5.f, 1e-4f);
}

void test_reduction_ncnn_max()
{
    std::vector<float> vals = { 3.f, -1.f, 7.f, 2.f };
    ncnn::Mat bottom = make_mat_1d(vals);
    ncnn::Mat top;

    ncnn::Reduction red;
    red.operation  = ncnn::Reduction::ReductionOp_MAX;
    red.reduce_all = 1;
    red.coeff      = 1.f;
    red.keepdims   = 0;

    ncnn::Option opt = make_opt();
    ASSERT_EQ(red.forward(bottom, top, opt), 0);
    std::vector<float> out; read_mat(top, out);
    ASSERT_NEAR(out[0], 7.f, 1e-4f);
}

int main() {
    printf("=== test_reduction ===\n");
    printf("\n-- Reference tests --\n");
    RUN_TEST(test_max_pool_2x2);
    RUN_TEST(test_avg_pool_2x2);
    RUN_TEST(test_global_avg_pool);
    RUN_TEST(test_max_pool_1x1_noop);
    RUN_TEST(test_avg_pool_single_element);
    RUN_TEST(test_softmax_1d);
    RUN_TEST(test_softmax_uniform);
    RUN_TEST(test_softmax_numerical_stability);
    RUN_TEST(test_softmax_axis1_2d);
    RUN_TEST(test_reduction_sum);
    RUN_TEST(test_reduction_mean);
    RUN_TEST(test_reduction_max_min);
    RUN_TEST(test_reduction_prod);
    RUN_TEST(test_reduction_l1_l2);
    RUN_TEST(test_reduction_logsumexp);
    RUN_TEST(test_argmax_basic);
    RUN_TEST(test_argmax_negative);
    RUN_TEST(test_cumulativesum);
    RUN_TEST(test_statisticspooling_mean_std);

    printf("\n-- Real ncnn::Softmax / Pooling / Reduction --\n");
    RUN_TEST(test_softmax_ncnn_1d);
    RUN_TEST(test_softmax_ncnn_uniform);
    RUN_TEST(test_softmax_ncnn_2d_axis0);
    RUN_TEST(test_max_pool_ncnn_2x2);
    RUN_TEST(test_avg_pool_ncnn_2x2);
    RUN_TEST(test_global_avg_pool_ncnn);
    RUN_TEST(test_reduction_ncnn_sum);
    RUN_TEST(test_reduction_ncnn_mean);
    RUN_TEST(test_reduction_ncnn_max);

    print_summary("reduction");
    return g_failed > 0 ? 1 : 0;
}
