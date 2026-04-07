// test_mapped_gelu.cpp — base vs ARM comparison for GELU
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/gelu/gelu.h"
#include "../mapped/gelu/gelu_arm.h"

static bool run_gelu(int c, int h, int w, int fast)
{
    std::vector<float> data = make_weights(c * h * w);

    ncnn::Mat m1 = make_mat(w, h, c, data);
    ncnn::Mat m2 = make_mat(w, h, c, data);

    ncnn::GELU base; base.fast_gelu = fast;
    ncnn::GELU_arm arm; arm.fast_gelu = fast;
    ncnn::Option opt = make_opt();

    arm.create_pipeline(opt);

    if (base.forward_inplace(m1, opt) != 0 || arm.forward_inplace(m2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(m1, o1); read_mat(m2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-3f);
    return g_failed == before;
}

static void test_gelu()
{
    run_gelu(1, 1, 16, 0);
    run_gelu(3, 4, 8, 0);
    run_gelu(1, 1, 16, 1);
    run_gelu(3, 4, 8, 1);
}

int main()
{
    RUN_TEST(test_gelu);
    print_summary("test_mapped_gelu");
    return g_failed ? 1 : 0;
}
