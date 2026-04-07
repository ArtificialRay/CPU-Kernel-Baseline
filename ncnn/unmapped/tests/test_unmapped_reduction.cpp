// test_unmapped_reduction.cpp — reference vs ncnn for Reduction
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../unmapped/reduction/reduction.h"
#include <vector>

static bool run_reduction_sum(int n)
{
    std::vector<float> data(n);
    for (int i = 0; i < n; i++) data[i] = (float)(i + 1) * 0.1f;

    ncnn::Mat in = make_mat_1d(data);
    ncnn::Mat out;

    ncnn::Reduction red;
    red.operation = ncnn::Reduction::ReductionOp_SUM;
    red.reduce_all = 1;
    red.coeff = 1.f;
    red.keepdims = 0;

    ncnn::Option opt = make_opt();
    if (red.forward(in, out, opt) != 0) { g_failed++; return false; }

    float ref_sum = 0.f;
    for (auto v : data) ref_sum += v;

    std::vector<float> got;
    read_mat(out, got);
    int before = g_failed;
    ASSERT_NEAR(got[0], ref_sum, 1e-3f);
    return g_failed == before;
}

static bool run_reduction_mean(int n)
{
    std::vector<float> data(n);
    for (int i = 0; i < n; i++) data[i] = (float)(i + 1) * 0.1f;

    ncnn::Mat in = make_mat_1d(data);
    ncnn::Mat out;

    ncnn::Reduction red;
    red.operation = ncnn::Reduction::ReductionOp_MEAN;
    red.reduce_all = 1;
    red.coeff = 1.f;
    red.keepdims = 0;

    ncnn::Option opt = make_opt();
    if (red.forward(in, out, opt) != 0) { g_failed++; return false; }

    float ref_sum = 0.f;
    for (auto v : data) ref_sum += v;

    std::vector<float> got;
    read_mat(out, got);
    int before = g_failed;
    ASSERT_NEAR(got[0], ref_sum / n, 1e-4f);
    return g_failed == before;
}

static bool run_reduction_3d(int c, int h, int w)
{
    int n = c * h * w;
    std::vector<float> data(n);
    for (int i = 0; i < n; i++) data[i] = (float)(i + 1) * 0.1f;

    ncnn::Mat in = make_mat(w, h, c, data);
    ncnn::Mat out;

    ncnn::Reduction red;
    red.operation = ncnn::Reduction::ReductionOp_SUM;
    red.reduce_all = 1;
    red.coeff = 1.f;
    red.keepdims = 0;

    ncnn::Option opt = make_opt();
    if (red.forward(in, out, opt) != 0) { g_failed++; return false; }

    float ref_sum = 0.f;
    for (auto v : data) ref_sum += v;

    std::vector<float> got;
    read_mat(out, got);
    int before = g_failed;
    ASSERT_NEAR(got[0], ref_sum, 1e-3f);
    return g_failed == before;
}

static void test_reduction()
{
    run_reduction_sum(16);
    run_reduction_sum(32);
    run_reduction_mean(16);
    run_reduction_mean(32);
    run_reduction_3d(2, 3, 4);
}

int main()
{
    RUN_TEST(test_reduction);
    print_summary("test_unmapped_reduction");
    return g_failed ? 1 : 0;
}
