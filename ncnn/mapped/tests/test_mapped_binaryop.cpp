// test_mapped_binaryop.cpp — base vs ARM comparison for BinaryOp
#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../mapped/binaryop/binaryop.h"
#include "../mapped/binaryop/binaryop_arm.h"

static bool run_binaryop_scalar(int c, int h, int w, int op_type, float scalar_b)
{
    std::vector<float> data = make_weights(c * h * w);
    for (auto& v : data) v -= 0.3f;

    ncnn::Mat m1 = make_mat(w, h, c, data);
    ncnn::Mat m2 = make_mat(w, h, c, data);

    ncnn::BinaryOp base;
    base.op_type = op_type; base.with_scalar = 1; base.b = scalar_b;
    base.one_blob_only = true; base.support_inplace = true;

    ncnn::BinaryOp_arm arm;
    arm.op_type = op_type; arm.with_scalar = 1; arm.b = scalar_b;
    arm.one_blob_only = true; arm.support_inplace = true;

    ncnn::Option opt = make_opt();

    if (base.forward_inplace(m1, opt) != 0 || arm.forward_inplace(m2, opt) != 0) { g_failed++; return false; }

    std::vector<float> o1, o2;
    read_mat(m1, o1); read_mat(m2, o2);
    int before = g_failed;
    ASSERT_VEC_NEAR(o1, o2.data(), (int)o1.size(), 1e-5f);
    return g_failed == before;
}

static bool run_binaryop_tensor(int c, int h, int w, int op_type)
{
    std::vector<float> dataA = make_weights(c * h * w);
    std::vector<float> dataB = make_weights(c * h * w);
    for (auto& v : dataA) v -= 0.3f;
    for (auto& v : dataB) v -= 0.2f;

    ncnn::Mat a1 = make_mat(w, h, c, dataA), b1 = make_mat(w, h, c, dataB);
    ncnn::Mat a2 = make_mat(w, h, c, dataA), b2 = make_mat(w, h, c, dataB);

    ncnn::BinaryOp base; base.op_type = op_type; base.with_scalar = 0;
    ncnn::BinaryOp_arm arm; arm.op_type = op_type; arm.with_scalar = 0;
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

static void test_binaryop()
{
    run_binaryop_scalar(3, 4, 8, ncnn::BinaryOp::Operation_ADD, 2.5f);
    run_binaryop_scalar(3, 4, 8, ncnn::BinaryOp::Operation_MUL, 0.5f);
    run_binaryop_tensor(3, 4, 8, ncnn::BinaryOp::Operation_ADD);
    run_binaryop_tensor(3, 4, 8, ncnn::BinaryOp::Operation_MUL);
}

int main()
{
    RUN_TEST(test_binaryop);
    print_summary("test_mapped_binaryop");
    return g_failed ? 1 : 0;
}
