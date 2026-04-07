// test_mapped_unaryop.cpp — base vs ARM comparison for UnaryOp
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/unaryop/unaryop.h"
#include "../mapped/unaryop/unaryop_arm.h"
#include <cmath>

static bool run_unaryop(int c, int h, int w, int op_type, bool positive = false)
{
    std::vector<float> data = make_weights(c * h * w);
    if (positive)
        for (auto& v : data) v = fabsf(v) + 0.01f;
    else
        for (auto& v : data) v -= 0.3f;

    ncnn::Mat m1 = make_mat(w, h, c, data);
    ncnn::Mat m2 = make_mat(w, h, c, data);

    ncnn::UnaryOp base; base.op_type = op_type;
    ncnn::UnaryOp_arm arm; arm.op_type = op_type;
    ncnn::Option opt = make_opt();

    if (base.forward_inplace(m1, opt) != 0 || arm.forward_inplace(m2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(m1, o1); read_mat(m2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-4f);
    return g_failed == before;
}

static void test_unaryop()
{
    run_unaryop(3, 4, 8, ncnn::UnaryOp::Operation_ABS);
    run_unaryop(3, 4, 8, ncnn::UnaryOp::Operation_NEG);
    run_unaryop(3, 4, 8, ncnn::UnaryOp::Operation_SQUARE);
    run_unaryop(3, 4, 8, ncnn::UnaryOp::Operation_SQRT, true);
    run_unaryop(3, 4, 8, ncnn::UnaryOp::Operation_EXP);
    run_unaryop(3, 4, 8, ncnn::UnaryOp::Operation_LOG, true);
}

int main()
{
    RUN_TEST(test_unaryop);
    print_summary("test_mapped_unaryop");
    return g_failed ? 1 : 0;
}
