// test_mapped_mish.cpp — base vs ARM comparison for Mish
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/mish/mish.h"
#include "../mapped/mish/mish_arm.h"

static bool run_mish(int c, int h, int w)
{
    std::vector<float> data = make_weights(c * h * w);

    ncnn::Mat m1 = make_mat(w, h, c, data);
    ncnn::Mat m2 = make_mat(w, h, c, data);

    ncnn::Mish base;
    ncnn::Mish_arm arm;
    ncnn::Option opt = make_opt();

    if (base.forward_inplace(m1, opt) != 0 || arm.forward_inplace(m2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(m1, o1); read_mat(m2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-4f);
    return g_failed == before;
}

static void test_mish()
{
    run_mish(1, 1, 16);
    run_mish(3, 4, 8);
    run_mish(4, 5, 7);
}

int main()
{
    RUN_TEST(test_mish);
    print_summary("test_mapped_mish");
    return g_failed ? 1 : 0;
}
