// test_mapped_flatten.cpp — base vs ARM comparison for Flatten
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/flatten/flatten.h"
#include "../mapped/flatten/flatten_arm.h"

static bool run_flatten(int c, int h, int w)
{
    std::vector<float> data = make_weights(c * h * w);

    ncnn::Mat in1 = make_mat(w, h, c, data);
    ncnn::Mat in2 = make_mat(w, h, c, data);
    ncnn::Mat out1, out2;

    ncnn::Flatten base;
    ncnn::Flatten_arm arm;
    ncnn::Option opt = make_opt();

    if (base.forward(in1, out1, opt) != 0 || arm.forward(in2, out2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(out1, o1); read_mat(out2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-5f);
    return g_failed == before;
}

static void test_flatten()
{
    run_flatten(1, 1, 16);
    run_flatten(3, 4, 8);
    run_flatten(4, 5, 7);
}

int main()
{
    RUN_TEST(test_flatten);
    print_summary("test_mapped_flatten");
    return g_failed ? 1 : 0;
}
