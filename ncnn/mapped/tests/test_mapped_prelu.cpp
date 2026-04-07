// test_mapped_prelu.cpp — base vs ARM comparison for PReLU
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/prelu/prelu.h"
#include "../mapped/prelu/prelu_arm.h"

static bool run_prelu(int c, int h, int w)
{
    std::vector<float> data = make_weights(c * h * w);
    for (auto& v : data) v -= 0.3f;

    std::vector<float> slopes(c);
    for (int i = 0; i < c; i++) slopes[i] = 0.1f + 0.05f * i;

    ncnn::Mat m1 = make_mat(w, h, c, data);
    ncnn::Mat m2 = make_mat(w, h, c, data);

    ncnn::PReLU base;
    base.num_slope = c;
    base.slope_data = make_weight(slopes);

    ncnn::PReLU_arm arm;
    arm.num_slope = c;
    arm.slope_data = make_weight(slopes);

    ncnn::Option opt = make_opt();

    if (base.forward_inplace(m1, opt) != 0 || arm.forward_inplace(m2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(m1, o1); read_mat(m2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-5f);
    return g_failed == before;
}

static void test_prelu()
{
    run_prelu(1, 1, 16);
    run_prelu(3, 4, 8);
    run_prelu(4, 5, 7);
}

int main()
{
    RUN_TEST(test_prelu);
    print_summary("test_mapped_prelu");
    return g_failed ? 1 : 0;
}
