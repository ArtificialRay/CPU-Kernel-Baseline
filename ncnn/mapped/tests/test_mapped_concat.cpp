// test_mapped_concat.cpp — base vs ARM comparison for Concat
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/concat/concat.h"
#include "../mapped/concat/concat_arm.h"

static bool run_concat(int c1, int c2, int h, int w)
{
    std::vector<float> dataA(c1 * h * w), dataB(c2 * h * w);
    for (int i = 0; i < c1 * h * w; i++) dataA[i] = (float)(i + 1) * 0.1f;
    for (int i = 0; i < c2 * h * w; i++) dataB[i] = (float)(i + 2) * 0.1f;

    ncnn::Mat a1 = make_mat(w, h, c1, dataA), b1 = make_mat(w, h, c2, dataB);
    ncnn::Mat a2 = make_mat(w, h, c1, dataA), b2 = make_mat(w, h, c2, dataB);

    ncnn::Concat base; base.axis = 0;
    ncnn::Concat_arm arm; arm.axis = 0;
    ncnn::Option opt = make_opt();

    std::vector<ncnn::Mat> in1 = {a1, b1}, in2 = {a2, b2};
    std::vector<ncnn::Mat> out1(1), out2(1);

    if (base.forward(in1, out1, opt) != 0 || arm.forward(in2, out2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(out1[0], o1); read_mat(out2[0], o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-5f);
    return g_failed == before;
}

static void test_concat()
{
    run_concat(2, 3, 4, 8);
    run_concat(1, 2, 5, 7);
    run_concat(3, 3, 3, 6);
}

int main()
{
    RUN_TEST(test_concat);
    print_summary("test_mapped_concat");
    return g_failed ? 1 : 0;
}
