// test_mapped_eltwise.cpp — base vs ARM comparison for Eltwise
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/eltwise/eltwise.h"
#include "../mapped/eltwise/eltwise_arm.h"

static bool run_eltwise(int c, int h, int w, int op_type)
{
    std::vector<float> dataA(c * h * w), dataB(c * h * w);
    for (int i = 0; i < c * h * w; i++) {
        dataA[i] = (float)(i + 1) * 0.1f;
        dataB[i] = (float)(i + 2) * 0.1f;
    }

    ncnn::Mat a1 = make_mat(w, h, c, dataA), b1 = make_mat(w, h, c, dataB);
    ncnn::Mat a2 = make_mat(w, h, c, dataA), b2 = make_mat(w, h, c, dataB);

    ncnn::Eltwise base; base.op_type = op_type;
    ncnn::Eltwise_arm arm; arm.op_type = op_type;
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

static void test_eltwise()
{
    run_eltwise(3, 4, 8, ncnn::Eltwise::Operation_SUM);
    run_eltwise(3, 4, 8, ncnn::Eltwise::Operation_PROD);
    run_eltwise(4, 5, 7, ncnn::Eltwise::Operation_SUM);
}

int main()
{
    RUN_TEST(test_eltwise);
    print_summary("test_mapped_eltwise");
    return g_failed ? 1 : 0;
}
