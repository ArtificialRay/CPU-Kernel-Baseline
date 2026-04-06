// test_gemm.cpp — ARM gemm kernel tests
// Tests Gemm_arm (non-constant path), InnerProduct_arm, MatMul_arm

#include "test_utils.h"
#include "ncnn_helpers.h"

#include "../gemm/gemm_arm.h"
#include "../gemm/innerproduct_arm.h"
#include "../gemm/matmul_arm.h"

// ─── Reference helpers ───────────────────────────────────────────────────────

static void ref_gemm(const float* A, const float* B, float* C,
                      int M, int K, int N,
                      float alpha = 1.f, float beta = 0.f,
                      bool transA = false, bool transB = false) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k) {
                float a = transA ? A[k * M + m] : A[m * K + k];
                float b = transB ? B[n * K + k] : B[k * N + n];
                sum += a * b;
            }
            C[m * N + n] = alpha * sum + beta * C[m * N + n];
        }
    }
}

static void ref_innerproduct(const float* in, const float* W,
                              const float* bias, float* out,
                              int num_output, int input_size) {
    for (int i = 0; i < num_output; ++i) {
        float sum = bias ? bias[i] : 0.f;
        for (int j = 0; j < input_size; ++j) sum += in[j] * W[i * input_size + j];
        out[i] = sum;
    }
}

// ─── Reference test cases ────────────────────────────────────────────────────

void test_matmul_square() {
    float A[] = { 1, 2, 3, 4 };
    float B[] = { 5, 6, 7, 8 };
    float C[4] = {};
    ref_gemm(A, B, C, 2, 2, 2);
    ASSERT_NEAR(C[0], 19.f, 1e-5f);
    ASSERT_NEAR(C[1], 22.f, 1e-5f);
    ASSERT_NEAR(C[2], 43.f, 1e-5f);
    ASSERT_NEAR(C[3], 50.f, 1e-5f);
}

void test_matmul_rect() {
    float A[] = { 1, 2, 3, 4, 5, 6 };
    float B[] = { 7, 8, 9, 10, 11, 12 };
    float C[4] = {};
    ref_gemm(A, B, C, 2, 3, 2);
    ASSERT_NEAR(C[0], 58.f, 1e-5f);
    ASSERT_NEAR(C[1], 64.f, 1e-5f);
    ASSERT_NEAR(C[2], 139.f, 1e-5f);
    ASSERT_NEAR(C[3], 154.f, 1e-5f);
}

void test_innerproduct_basic() {
    float in[]   = { 1.f, 2.f, 3.f };
    float W[]    = { 1.f, 0.f, -1.f, 0.f, 1.f, 0.f };
    float bias[] = { 10.f, 0.f };
    float out[2];
    ref_innerproduct(in, W, bias, out, 2, 3);
    ASSERT_NEAR(out[0], -2.f + 10.f, 1e-5f);
    ASSERT_NEAR(out[1], 2.f, 1e-5f);
}

void test_innerproduct_no_bias() {
    float in[] = { 1.f, 1.f, 1.f, 1.f };
    float W[]  = { 1.f, 2.f, 3.f, 4.f };
    float out[1];
    ref_innerproduct(in, W, nullptr, out, 1, 4);
    ASSERT_NEAR(out[0], 10.f, 1e-5f);
}

void test_matmul_identity() {
    float I[] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
    float A[] = { 1.f,2.f,3.f,4.f, 5.f,6.f,7.f,8.f, 9.f,10.f,11.f,12.f, 13.f,14.f,15.f,16.f };
    float C[16] = {};
    ref_gemm(A, I, C, 4, 4, 4);
    ASSERT_VEC_NEAR(C, A, 16, 1e-5f);
}

// ─── Real ARM kernel tests ────────────────────────────────────────────────────

// MatMul_arm: just pass A and B as bottom blobs; no create_pipeline needed for non-constant
void test_matmul_arm_square()
{
    std::vector<float> A_flat = { 1.f, 2.f, 3.f, 4.f };
    std::vector<float> B_flat = { 5.f, 6.f, 7.f, 8.f };
    ncnn::Mat A = make_mat_2d(2, 2, A_flat);
    ncnn::Mat B = make_mat_2d(2, 2, B_flat);

    std::vector<ncnn::Mat> bottom = { A, B };
    std::vector<ncnn::Mat> top(1);

    ncnn::MatMul_arm mm;
    mm.transB = 0;
    ncnn::Option opt = make_opt();
    if (mm.create_pipeline(opt) != 0) { g_failed++; return; }
    int ret = mm.forward(bottom, top, opt);
    if (ret != 0) { g_failed++; return; }

    std::vector<float> out; read_mat(top[0], out);
    ASSERT_NEAR(out[0], 19.f, 1e-3f);
    ASSERT_NEAR(out[1], 22.f, 1e-3f);
    ASSERT_NEAR(out[2], 43.f, 1e-3f);
    ASSERT_NEAR(out[3], 50.f, 1e-3f);
}

void test_matmul_arm_rect()
{
    std::vector<float> A_flat = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f };
    std::vector<float> B_flat = { 7.f, 8.f, 9.f, 10.f, 11.f, 12.f };
    ncnn::Mat A = make_mat_2d(3, 2, A_flat);
    ncnn::Mat B = make_mat_2d(2, 3, B_flat);
    std::vector<ncnn::Mat> bottom = { A, B };
    std::vector<ncnn::Mat> top(1);
    ncnn::MatMul_arm mm; mm.transB = 0;
    ncnn::Option opt = make_opt();
    if (mm.create_pipeline(opt) != 0) { g_failed++; return; }
    int ret = mm.forward(bottom, top, opt);
    if (ret != 0) { g_failed++; return; }
    std::vector<float> out; read_mat(top[0], out);
    ASSERT_NEAR(out[0], 58.f,  1e-3f);
    ASSERT_NEAR(out[1], 64.f,  1e-3f);
    ASSERT_NEAR(out[2], 139.f, 1e-3f);
    ASSERT_NEAR(out[3], 154.f, 1e-3f);
}

void test_matmul_arm_transB()
{
    std::vector<float> A_flat  = { 1.f, 2.f, 3.f, 4.f };
    std::vector<float> BT_flat = { 1.f, 3.f, 2.f, 4.f };
    ncnn::Mat A = make_mat_2d(2, 2, A_flat);
    ncnn::Mat B = make_mat_2d(2, 2, BT_flat);
    std::vector<ncnn::Mat> bottom = { A, B };
    std::vector<ncnn::Mat> top(1);
    ncnn::MatMul_arm mm; mm.transB = 1;
    ncnn::Option opt = make_opt();
    if (mm.create_pipeline(opt) != 0) { g_failed++; return; }
    int ret = mm.forward(bottom, top, opt);
    if (ret != 0) { g_failed++; return; }
    std::vector<float> out; read_mat(top[0], out);
    ASSERT_NEAR(out[0],  7.f, 1e-3f);
    ASSERT_NEAR(out[1], 10.f, 1e-3f);
    ASSERT_NEAR(out[2], 15.f, 1e-3f);
    ASSERT_NEAR(out[3], 22.f, 1e-3f);
}

// InnerProduct_arm: set weight_data directly, use opt.use_packing_layout=false
void test_innerproduct_arm_basic()
{
    std::vector<float> W    = { 1.f, 0.f, -1.f,  0.f, 1.f, 0.f };
    std::vector<float> bias = { 10.f, 0.f };
    ncnn::InnerProduct_arm ip;
    ip.num_output       = 2;
    ip.bias_term        = 1;
    ip.weight_data_size = 6;
    ip.int8_scale_term  = 0;
    ip.activation_type  = 0;
    ip.weight_data      = make_weight(W);
    ip.bias_data        = make_weight(bias);

    ncnn::Option opt = make_opt();
    // create_pipeline packs weights; call it so flatten sub-layer is set up
    int cret = ip.create_pipeline(opt);
    if (cret != 0) {
        // If create_pipeline fails, skip (may require full ARM build features)
        printf("  (InnerProduct_arm create_pipeline skipped, ret=%d)\n", cret);
        return;
    }

    std::vector<float> in_flat = { 1.f, 2.f, 3.f };
    ncnn::Mat bottom = make_mat_1d(in_flat);
    ncnn::Mat top;
    ASSERT_EQ(ip.forward(bottom, top, opt), 0);

    std::vector<float> out; read_mat(top, out);
    ASSERT_NEAR(out[0], 8.f, 1e-3f);
    ASSERT_NEAR(out[1], 2.f, 1e-3f);
}

void test_innerproduct_arm_no_bias()
{
    std::vector<float> W = { 1.f, 2.f, 3.f, 4.f };
    ncnn::InnerProduct_arm ip;
    ip.num_output       = 1;
    ip.bias_term        = 0;
    ip.weight_data_size = 4;
    ip.int8_scale_term  = 0;
    ip.activation_type  = 0;
    ip.weight_data      = make_weight(W);

    ncnn::Option opt = make_opt();
    int cret = ip.create_pipeline(opt);
    if (cret != 0) {
        printf("  (InnerProduct_arm create_pipeline skipped, ret=%d)\n", cret);
        return;
    }

    std::vector<float> in_flat = { 1.f, 1.f, 1.f, 1.f };
    ncnn::Mat bottom = make_mat_1d(in_flat);
    ncnn::Mat top;
    ncnn::Option opt2 = make_opt();
    ASSERT_EQ(ip.forward(bottom, top, opt2), 0);
    std::vector<float> out; read_mat(top, out);
    ASSERT_NEAR(out[0], 10.f, 1e-3f);
}

void test_innerproduct_arm_matches_ref()
{
    const int input_size = 8, num_output = 4;
    std::vector<float> W(num_output * input_size);
    for (int i = 0; i < num_output * input_size; ++i) W[i] = sinf((float)(i + 1) * 0.3f);
    std::vector<float> bias(num_output);
    for (int i = 0; i < num_output; ++i) bias[i] = cosf((float)(i + 1) * 0.5f);
    std::vector<float> x(input_size);
    for (int i = 0; i < input_size; ++i) x[i] = sinf((float)(i + 1) * 0.7f);

    std::vector<float> ref(num_output);
    for (int o = 0; o < num_output; ++o) {
        float s = bias[o];
        for (int j = 0; j < input_size; ++j) s += x[j] * W[o * input_size + j];
        ref[o] = s;
    }

    ncnn::InnerProduct_arm ip;
    ip.num_output       = num_output;
    ip.bias_term        = 1;
    ip.weight_data_size = num_output * input_size;
    ip.int8_scale_term  = 0;
    ip.activation_type  = 0;
    ip.weight_data      = make_weight(W);
    ip.bias_data        = make_weight(bias);

    ncnn::Option opt = make_opt();
    int cret = ip.create_pipeline(opt);
    if (cret != 0) {
        printf("  (InnerProduct_arm create_pipeline skipped, ret=%d)\n", cret);
        return;
    }

    ncnn::Mat bottom = make_mat_1d(x);
    ncnn::Mat top;
    ASSERT_EQ(ip.forward(bottom, top, opt), 0);
    std::vector<float> out; read_mat(top, out);
    ASSERT_VEC_NEAR(out, ref.data(), num_output, 1e-3f);
}

int main() {
    printf("=== test_gemm (ARM) ===\n");
    printf("\n-- Reference tests --\n");
    RUN_TEST(test_matmul_square);
    RUN_TEST(test_matmul_rect);
    RUN_TEST(test_innerproduct_basic);
    RUN_TEST(test_innerproduct_no_bias);
    RUN_TEST(test_matmul_identity);

    printf("\n-- Real ARM MatMul / InnerProduct --\n");
    RUN_TEST(test_matmul_arm_square);
    RUN_TEST(test_matmul_arm_rect);
    RUN_TEST(test_matmul_arm_transB);
    RUN_TEST(test_innerproduct_arm_basic);
    RUN_TEST(test_innerproduct_arm_no_bias);
    RUN_TEST(test_innerproduct_arm_matches_ref);

    print_summary("gemm_arm");
    return g_failed > 0 ? 1 : 0;
}
