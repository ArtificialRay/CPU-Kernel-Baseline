// test_reduction.cpp — ARM reduction kernel tests
// Tests Pooling_arm, Softmax_arm

#include "test_utils.h"
#include <limits>

#include "ncnn_helpers.h"
#include "../reduction/softmax_arm.h"
#include "../reduction/pooling_arm.h"

// ─── Reference implementations ───────────────────────────────────────────────

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

// ─── Reference test cases ────────────────────────────────────────────────────

void test_max_pool_2x2() {
    TestMat in(4, 4, 1);
    in.fill_range();
    TestMat out = ref_max_pool2d(in, 2, 2, 2, 2);
    ASSERT_EQ(out.h, 2); ASSERT_EQ(out.w, 2);
    ASSERT_NEAR(out.at(0, 0, 0), 6.f,  1e-5f);
    ASSERT_NEAR(out.at(1, 0, 0), 8.f,  1e-5f);
    ASSERT_NEAR(out.at(0, 1, 0), 14.f, 1e-5f);
    ASSERT_NEAR(out.at(1, 1, 0), 16.f, 1e-5f);
}

void test_avg_pool_2x2() {
    TestMat in(4, 4, 1);
    in.fill_range();
    TestMat out = ref_avg_pool2d(in, 2, 2, 2, 2);
    ASSERT_NEAR(out.at(0, 0, 0), 3.5f, 1e-5f);
}

void test_softmax_1d() {
    float p[] = { 1.f, 2.f, 3.f };
    softmax_inplace(p, 3);
    float s = p[0] + p[1] + p[2];
    ASSERT_NEAR(s, 1.f, 1e-5f);
    ASSERT_TRUE(p[0] < p[1] && p[1] < p[2]);
    ASSERT_NEAR(p[2], 0.6652f, 1e-3f);
}

void test_softmax_uniform() {
    float p[] = { 5.f, 5.f, 5.f, 5.f };
    softmax_inplace(p, 4);
    for (float v : p) ASSERT_NEAR(v, 0.25f, 1e-5f);
}

void test_softmax_numerical_stability() {
    float p[] = { 1000.f, 1001.f, 1002.f };
    softmax_inplace(p, 3);
    float s = p[0] + p[1] + p[2];
    ASSERT_NEAR(s, 1.f, 1e-5f);
    for (float v : p) ASSERT_TRUE(v > 0.f && v < 1.f);
}

// ─── Real ARM kernel tests ────────────────────────────────────────────────────

void test_softmax_arm_1d()
{
    std::vector<float> vals = { 1.f, 2.f, 3.f };
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::Softmax_arm sm; sm.axis = 0;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(sm.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    float s = out[0] + out[1] + out[2];
    ASSERT_NEAR(s, 1.f, 1e-4f);
    ASSERT_TRUE(out[0] < out[1] && out[1] < out[2]);
    std::vector<float> ref = vals;
    softmax_inplace(ref.data(), 3);
    ASSERT_VEC_NEAR(out, ref.data(), 3, 1e-3f);
}

void test_softmax_arm_uniform()
{
    std::vector<float> vals = { 5.f, 5.f, 5.f, 5.f };
    ncnn::Mat m = make_mat_1d(vals);
    ncnn::Softmax_arm sm; sm.axis = 0;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(sm.forward_inplace(m, opt), 0);
    std::vector<float> out; read_mat(m, out);
    for (float v : out) ASSERT_NEAR(v, 0.25f, 1e-4f);
}

void test_softmax_arm_2d_axis0()
{
    std::vector<float> flat = { 1.f, 2.f, 3.f };
    ncnn::Mat m = make_mat(1, 1, 3, flat);
    ncnn::Softmax_arm sm; sm.axis = 0;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(sm.forward_inplace(m, opt), 0);
    float s = ((float*)m.channel(0))[0] + ((float*)m.channel(1))[0] + ((float*)m.channel(2))[0];
    ASSERT_NEAR(s, 1.f, 1e-4f);
}

void test_max_pool_arm_2x2()
{
    std::vector<float> flat(16);
    for (int i = 0; i < 16; ++i) flat[i] = (float)(i + 1);
    ncnn::Mat bottom = make_mat(4, 4, 1, flat);
    ncnn::Mat top;

    ncnn::Pooling_arm pool;
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
    pool.create_pipeline(opt);
    ASSERT_EQ(pool.forward(bottom, top, opt), 0);
    ASSERT_EQ(top.h, 2); ASSERT_EQ(top.w, 2);
    ASSERT_NEAR(top.channel(0).row(0)[0], 6.f,  1e-4f);
    ASSERT_NEAR(top.channel(0).row(0)[1], 8.f,  1e-4f);
    ASSERT_NEAR(top.channel(0).row(1)[0], 14.f, 1e-4f);
    ASSERT_NEAR(top.channel(0).row(1)[1], 16.f, 1e-4f);
}

void test_avg_pool_arm_2x2()
{
    std::vector<float> flat(16);
    for (int i = 0; i < 16; ++i) flat[i] = (float)(i + 1);
    ncnn::Mat bottom = make_mat(4, 4, 1, flat);
    ncnn::Mat top;

    ncnn::Pooling_arm pool;
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
    pool.create_pipeline(opt);
    ASSERT_EQ(pool.forward(bottom, top, opt), 0);
    ASSERT_NEAR(top.channel(0).row(0)[0], 3.5f, 1e-3f);
}

void test_global_avg_pool_arm()
{
    std::vector<float> flat(9);
    for (int i = 0; i < 9; ++i) flat[i] = (float)(i + 1);
    ncnn::Mat bottom = make_mat(3, 3, 1, flat);
    ncnn::Mat top;

    ncnn::Pooling_arm pool;
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
    pool.create_pipeline(opt);
    ASSERT_EQ(pool.forward(bottom, top, opt), 0);
    ASSERT_EQ(top.w, 1); ASSERT_EQ(top.h, 1);
    ASSERT_NEAR(((float*)top)[0], 5.f, 1e-3f);
}

int main() {
    printf("=== test_reduction (ARM) ===\n");
    printf("\n-- Reference tests --\n");
    RUN_TEST(test_max_pool_2x2);
    RUN_TEST(test_avg_pool_2x2);
    RUN_TEST(test_softmax_1d);
    RUN_TEST(test_softmax_uniform);
    RUN_TEST(test_softmax_numerical_stability);

    printf("\n-- Real ARM Softmax / Pooling --\n");
    RUN_TEST(test_softmax_arm_1d);
    RUN_TEST(test_softmax_arm_uniform);
    RUN_TEST(test_softmax_arm_2d_axis0);
    RUN_TEST(test_max_pool_arm_2x2);
    RUN_TEST(test_avg_pool_arm_2x2);
    RUN_TEST(test_global_avg_pool_arm);

    print_summary("reduction_arm");
    return g_failed > 0 ? 1 : 0;
}
