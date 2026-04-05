// test_tensor.cpp — ARM tensor kernel tests
// Tests BinaryOp_arm, UnaryOp_arm, Flatten_arm, Concat_arm

#include "test_utils.h"
#include "ncnn_helpers.h"

#include "../tensor/binaryop_arm.h"
#include "../tensor/unaryop_arm.h"
#include "../tensor/flatten_arm.h"
#include "../tensor/concat_arm.h"

// ─── Reference ops ───────────────────────────────────────────────────────────

static void ref_add  (const float* a, const float* b, float* o, int n) { for(int i=0;i<n;i++) o[i]=a[i]+b[i]; }
static void ref_sub  (const float* a, const float* b, float* o, int n) { for(int i=0;i<n;i++) o[i]=a[i]-b[i]; }
static void ref_mul  (const float* a, const float* b, float* o, int n) { for(int i=0;i<n;i++) o[i]=a[i]*b[i]; }
static void ref_uabs (const float* a, float* o, int n) { for(int i=0;i<n;i++) o[i]=fabsf(a[i]); }
static void ref_uneg (const float* a, float* o, int n) { for(int i=0;i<n;i++) o[i]=-a[i]; }
static void ref_usqrt(const float* a, float* o, int n) { for(int i=0;i<n;i++) o[i]=sqrtf(a[i]); }

// ─── Reference test cases ────────────────────────────────────────────────────

void test_binaryop_add() {
    float a[] = { 1.f, 2.f, 3.f };
    float b[] = { 4.f, 5.f, 6.f };
    float o[3]; ref_add(a, b, o, 3);
    float expected[] = { 5.f, 7.f, 9.f };
    ASSERT_VEC_NEAR(o, expected, 3, 1e-5f);
}

void test_binaryop_sub() {
    float a[] = { 10.f, 5.f, 3.f };
    float b[] = { 1.f, 2.f, 3.f };
    float o[3]; ref_sub(a, b, o, 3);
    float expected[] = { 9.f, 3.f, 0.f };
    ASSERT_VEC_NEAR(o, expected, 3, 1e-5f);
}

void test_binaryop_mul() {
    float a[] = { 2.f, 3.f, -1.f };
    float b[] = { 4.f, -2.f, 5.f };
    float o[3]; ref_mul(a, b, o, 3);
    float expected[] = { 8.f, -6.f, -5.f };
    ASSERT_VEC_NEAR(o, expected, 3, 1e-5f);
}

void test_unaryop_abs() {
    float a[]={-2,-1,0,1,2}; float o[5]; ref_uabs(a,o,5);
    float e[]={2,1,0,1,2}; ASSERT_VEC_NEAR(o,e,5,1e-5f);
}

void test_unaryop_neg() {
    float a[]={1,-2,0}; float o[3]; ref_uneg(a,o,3);
    float e[]={-1,2,0}; ASSERT_VEC_NEAR(o,e,3,1e-5f);
}

void test_unaryop_sqrt() {
    float a[]={0,1,4,9}; float o[4]; ref_usqrt(a,o,4);
    float e[]={0,1,2,3}; ASSERT_VEC_NEAR(o,e,4,1e-5f);
}

void test_concat_channels_ref() {
    // Channel concat reference
    float a[] = { 1.f, 2.f, 3.f, 4.f };
    float b[] = { 5.f, 6.f, 7.f, 8.f };
    float expected[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };
    // Just test that values match expected layout
    float concat[8];
    memcpy(concat, a, 4 * sizeof(float));
    memcpy(concat + 4, b, 4 * sizeof(float));
    ASSERT_VEC_NEAR(concat, expected, 8, 1e-5f);
}

void test_flatten_ref() {
    TestMat in(4, 3, 2);
    in.fill_range();
    ASSERT_EQ(in.total(), 24);
}

// ─── Real ARM kernel tests ────────────────────────────────────────────────────

void test_binaryop_arm_add()
{
    std::vector<float> a = { 1.f, 2.f, 3.f };
    std::vector<float> b = { 4.f, 5.f, 6.f };
    ncnn::Mat ma = make_mat_1d(a);
    ncnn::Mat mb = make_mat_1d(b);

    ncnn::BinaryOp_arm op;
    op.op_type    = ncnn::BinaryOp::Operation_ADD;
    op.with_scalar = 0;
    op.b = 0.f;

    std::vector<ncnn::Mat> bottom = { ma, mb };
    std::vector<ncnn::Mat> top(1);
    ncnn::Option opt = make_opt();
    ASSERT_EQ(op.forward(bottom, top, opt), 0);

    std::vector<float> got; read_mat(top[0], got);
    float expected[] = { 5.f, 7.f, 9.f };
    ASSERT_VEC_NEAR(got, expected, 3, 1e-4f);
}

void test_binaryop_arm_mul()
{
    std::vector<float> a = { 2.f, 3.f, -1.f };
    std::vector<float> b = { 4.f, -2.f, 5.f };
    ncnn::Mat ma = make_mat_1d(a);
    ncnn::Mat mb = make_mat_1d(b);

    ncnn::BinaryOp_arm op;
    op.op_type    = ncnn::BinaryOp::Operation_MUL;
    op.with_scalar = 0;
    op.b = 0.f;

    std::vector<ncnn::Mat> bottom = { ma, mb };
    std::vector<ncnn::Mat> top(1);
    ncnn::Option opt = make_opt();
    ASSERT_EQ(op.forward(bottom, top, opt), 0);

    std::vector<float> got; read_mat(top[0], got);
    float expected[] = { 8.f, -6.f, -5.f };
    ASSERT_VEC_NEAR(got, expected, 3, 1e-4f);
}

void test_binaryop_arm_scalar_add()
{
    std::vector<float> a = { 1.f, 2.f, 3.f };
    ncnn::Mat ma = make_mat_1d(a);

    ncnn::BinaryOp_arm op;
    op.op_type     = ncnn::BinaryOp::Operation_ADD;
    op.with_scalar = 1;
    op.b = 10.f;

    ncnn::Option opt = make_opt();
    ASSERT_EQ(op.forward_inplace(ma, opt), 0);

    std::vector<float> got; read_mat(ma, got);
    float expected[] = { 11.f, 12.f, 13.f };
    ASSERT_VEC_NEAR(got, expected, 3, 1e-4f);
}

void test_concat_arm_channel_axis()
{
    std::vector<float> a_data = { 1.f, 2.f, 3.f, 4.f };
    std::vector<float> b_data = { 5.f, 6.f, 7.f, 8.f };
    ncnn::Mat ma = make_mat(2, 2, 1, a_data);
    ncnn::Mat mb = make_mat(2, 2, 1, b_data);

    ncnn::Concat_arm cat;
    cat.axis = 0;

    std::vector<ncnn::Mat> bottom = { ma, mb };
    std::vector<ncnn::Mat> top(1);
    ncnn::Option opt = make_opt();
    ASSERT_EQ(cat.forward(bottom, top, opt), 0);

    ASSERT_EQ(top[0].c, 2);
    ASSERT_EQ(top[0].h, 2);
    ASSERT_EQ(top[0].w, 2);

    std::vector<float> got; read_mat(top[0], got);
    float expected[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };
    ASSERT_VEC_NEAR(got, expected, 8, 1e-4f);
}

void test_concat_arm_three_channels()
{
    std::vector<float> a_data = { 10.f };
    std::vector<float> b_data = { 20.f };
    std::vector<float> c_data = { 30.f };
    ncnn::Mat ma = make_mat(1, 1, 1, a_data);
    ncnn::Mat mb = make_mat(1, 1, 1, b_data);
    ncnn::Mat mc = make_mat(1, 1, 1, c_data);

    ncnn::Concat_arm cat;
    cat.axis = 0;

    std::vector<ncnn::Mat> bottom = { ma, mb, mc };
    std::vector<ncnn::Mat> top(1);
    ncnn::Option opt = make_opt();
    ASSERT_EQ(cat.forward(bottom, top, opt), 0);

    ASSERT_EQ(top[0].c, 3);
    std::vector<float> got; read_mat(top[0], got);
    ASSERT_NEAR(got[0], 10.f, 1e-4f);
    ASSERT_NEAR(got[1], 20.f, 1e-4f);
    ASSERT_NEAR(got[2], 30.f, 1e-4f);
}

void test_flatten_arm()
{
    // Flatten a [c=2, h=2, w=3] mat → [1D, 12 elements]
    std::vector<float> flat_in(12);
    for (int i = 0; i < 12; ++i) flat_in[i] = (float)(i + 1);
    ncnn::Mat in = make_mat(3, 2, 2, flat_in);
    ncnn::Mat out;

    ncnn::Flatten_arm fl;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(fl.forward(in, out, opt), 0);

    ASSERT_EQ(out.dims, 1);
    ASSERT_EQ(out.w, 12);
}

void test_unaryop_arm_abs()
{
    std::vector<float> in_data = { -3.f, -1.f, 0.f, 1.f, 2.f };
    ncnn::Mat m = make_mat_1d(in_data);

    ncnn::UnaryOp_arm op;
    op.op_type = ncnn::UnaryOp::Operation_ABS;

    ncnn::Option opt = make_opt();
    ASSERT_EQ(op.forward_inplace(m, opt), 0);

    std::vector<float> got; read_mat(m, got);
    float expected[] = { 3.f, 1.f, 0.f, 1.f, 2.f };
    ASSERT_VEC_NEAR(got, expected, 5, 1e-4f);
}

void test_unaryop_arm_neg()
{
    std::vector<float> in_data = { 1.f, -2.f, 0.f };
    ncnn::Mat m = make_mat_1d(in_data);

    ncnn::UnaryOp_arm op;
    op.op_type = ncnn::UnaryOp::Operation_NEG;

    ncnn::Option opt = make_opt();
    ASSERT_EQ(op.forward_inplace(m, opt), 0);

    std::vector<float> got; read_mat(m, got);
    float expected[] = { -1.f, 2.f, 0.f };
    ASSERT_VEC_NEAR(got, expected, 3, 1e-4f);
}

int main() {
    printf("=== test_tensor (ARM) ===\n");
    printf("\n-- Reference tests --\n");
    RUN_TEST(test_binaryop_add);
    RUN_TEST(test_binaryop_sub);
    RUN_TEST(test_binaryop_mul);
    RUN_TEST(test_unaryop_abs);
    RUN_TEST(test_unaryop_neg);
    RUN_TEST(test_unaryop_sqrt);
    RUN_TEST(test_concat_channels_ref);
    RUN_TEST(test_flatten_ref);

    printf("\n--- Real ARM BinaryOp / UnaryOp / Flatten / Concat ---\n");
    RUN_TEST(test_binaryop_arm_add);
    RUN_TEST(test_binaryop_arm_mul);
    RUN_TEST(test_binaryop_arm_scalar_add);
    RUN_TEST(test_concat_arm_channel_axis);
    RUN_TEST(test_concat_arm_three_channels);
    RUN_TEST(test_flatten_arm);
    RUN_TEST(test_unaryop_arm_abs);
    RUN_TEST(test_unaryop_arm_neg);

    print_summary("tensor_arm");
    return g_failed > 0 ? 1 : 0;
}
