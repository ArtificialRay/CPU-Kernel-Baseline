// test_mapped_swish.cpp — base vs ARM comparison for Swish
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/swish/swish.h"
#include "../mapped/swish/swish_arm.h"

static bool run_swish(int c, int h, int w)
{
    std::vector<float> data = make_weights(c * h * w);

    ncnn::Mat m1 = make_mat(w, h, c, data);
    ncnn::Mat m2 = make_mat(w, h, c, data);

    ncnn::Swish base;
    ncnn::Swish_arm arm;
    ncnn::Option opt = make_opt();

    if (base.forward_inplace(m1, opt) != 0 || arm.forward_inplace(m2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(m1, o1); read_mat(m2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-4f);
    return g_failed == before;
}

static void test_swish()
{
    run_swish(1, 1, 16);
    run_swish(3, 4, 8);
    run_swish(4, 5, 7);
}

int main()
{
    RUN_TEST(test_swish);
    print_summary("test_mapped_swish");
    return g_failed ? 1 : 0;
}
