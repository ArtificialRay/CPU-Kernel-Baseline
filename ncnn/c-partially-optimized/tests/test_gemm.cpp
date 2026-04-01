// test_gemm.cpp
// Tests for gemm/:
//   matmul, gemm (alpha/beta/transA/transB/broadcast-C), innerproduct, einsum
//
// Section 1: reference-only tests
// Section 2: real ncnn kernel tests (linked via gemm_impl + ncnn_stub)

#include "test_utils.h"

// ── Real ncnn kernel headers ──────────────────────────────────────────────────
#include "ncnn_helpers.h"
#include "../gemm/matmul.h"
#include "../gemm/innerproduct.h"
#include "../gemm/gemm.h"

// ─── Reference helpers ───────────────────────────────────────────────────────

// General matrix multiply: C = alpha * A * B + beta * C
// A[M,K]  B[K,N]  C[M,N]  (row-major)
static void ref_gemm(const float* A, const float* B, float* C,
                      int M, int K, int N,
                      float alpha = 1.f, float beta = 0.f,
                      bool transA = false, bool transB = false) {
    // Handle transposes by adjusting indexing
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

// Inner product (fully connected): out[i] = sum_j(in[j] * W[i,j]) + bias[i]
static void ref_innerproduct(const float* in, const float* W,
                              const float* bias, float* out,
                              int num_output, int input_size) {
    for (int i = 0; i < num_output; ++i) {
        float sum = bias ? bias[i] : 0.f;
        for (int j = 0; j < input_size; ++j) sum += in[j] * W[i * input_size + j];
        out[i] = sum;
    }
}

// ─── Test cases ──────────────────────────────────────────────────────────────

void test_matmul_square() {
    // 2x2 * 2x2
    float A[] = { 1, 2, 3, 4 };
    float B[] = { 5, 6, 7, 8 };
    float C[4] = {};
    ref_gemm(A, B, C, 2, 2, 2);
    // [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
    // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
    ASSERT_NEAR(C[0], 19.f, 1e-5f);
    ASSERT_NEAR(C[1], 22.f, 1e-5f);
    ASSERT_NEAR(C[2], 43.f, 1e-5f);
    ASSERT_NEAR(C[3], 50.f, 1e-5f);
}

void test_matmul_rect() {
    // 2x3 * 3x2 = 2x2
    float A[] = { 1, 2, 3,
                  4, 5, 6 };
    float B[] = { 7,  8,
                  9,  10,
                  11, 12 };
    float C[4] = {};
    ref_gemm(A, B, C, 2, 3, 2);
    // Row0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
    ASSERT_NEAR(C[0], 58.f, 1e-5f);
    ASSERT_NEAR(C[1], 64.f, 1e-5f);
    // Row1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
    ASSERT_NEAR(C[2], 139.f, 1e-5f);
    ASSERT_NEAR(C[3], 154.f, 1e-5f);
}

void test_matmul_vec_dot() {
    // [1,4] row vector dot [1,4] col → scalar
    float A[] = { 1.f, 2.f, 3.f, 4.f };  // 1x4
    float B[] = { 1.f, 2.f, 3.f, 4.f };  // 4x1
    float C[1] = {};
    ref_gemm(A, B, C, 1, 4, 1);
    // 1+4+9+16 = 30
    ASSERT_NEAR(C[0], 30.f, 1e-5f);
}

void test_gemm_alpha_beta() {
    float A[] = { 1, 2, 3, 4 };
    float B[] = { 1, 0, 0, 1 };  // identity
    float C[] = { 10.f, 10.f, 10.f, 10.f };
    // C = 2 * A*I + 0.5 * C_in
    ref_gemm(A, B, C, 2, 2, 2, 2.f, 0.5f);
    // A*I = A; 2*A + 0.5*[10,10,10,10] = [2+5, 4+5, 6+5, 8+5] = [7,9,11,13]
    ASSERT_NEAR(C[0], 7.f,  1e-5f);
    ASSERT_NEAR(C[1], 9.f,  1e-5f);
    ASSERT_NEAR(C[2], 11.f, 1e-5f);
    ASSERT_NEAR(C[3], 13.f, 1e-5f);
}

void test_gemm_transA() {
    // Compute A^T * B where A=[2,3], B=[2,3] → result [3,3]
    // A stored row-major as [K=2, M=3] = {1,2,3, 4,5,6}
    // When transA=true: element A^T[m,k] = A_stored[k*M+m]
    float A[] = { 1, 2, 3,
                  4, 5, 6 }; // A stored as [K=2, M=3]
    float B[] = { 1, 2, 3,
                  4, 5, 6 }; // [K=2, N=3]
    float C[9] = {};
    // transA=true: M=3, K=2, N=3; A_stored is [K=2, M=3]
    ref_gemm(A, B, C, 3, 2, 3, 1.f, 0.f, true, false);
    // A = [[1,2,3],[4,5,6]]  A^T = [[1,4],[2,5],[3,6]]
    // A^T * B = [[1*1+4*4, 1*2+4*5, 1*3+4*6],...]
    //         = [[17,22,27], [22,29,36], [27,36,45]]
    ASSERT_NEAR(C[0], 17.f, 1e-5f);
    ASSERT_NEAR(C[4], 29.f, 1e-5f);
    ASSERT_NEAR(C[8], 45.f, 1e-5f);
}

void test_gemm_transB() {
    float A[] = { 1, 2, 3, 4 };   // 2x2
    float B_T[] = { 1, 3,          // B stored as B^T → B is 2x2, B^T is 2x2
                    2, 4 };
    float C[4] = {};
    // transB: A[2,2]*B^T→B[2,2] i.e. A[2,2]*(B_T stored as [2,2] rows)
    ref_gemm(A, B_T, C, 2, 2, 2, 1.f, 0.f, false, true);
    // B_T rows = B columns → B = [[1,2],[3,4]]
    // A * B = [1*1+2*3, 1*2+2*4] = [7, 10]; [3*1+4*3, 3*2+4*4] = [15, 22]
    ASSERT_NEAR(C[0], 7.f,  1e-5f);
    ASSERT_NEAR(C[1], 10.f, 1e-5f);
    ASSERT_NEAR(C[2], 15.f, 1e-5f);
    ASSERT_NEAR(C[3], 22.f, 1e-5f);
}

void test_innerproduct_basic() {
    float in[]   = { 1.f, 2.f, 3.f };
    float W[]    = { 1.f, 0.f, -1.f,   // out[0] = 1-3 = -2
                     0.f, 1.f,  0.f };  // out[1] = 2
    float bias[] = { 10.f, 0.f };
    float out[2];
    ref_innerproduct(in, W, bias, out, 2, 3);
    ASSERT_NEAR(out[0], -2.f + 10.f, 1e-5f);  // 8
    ASSERT_NEAR(out[1], 2.f,          1e-5f);
}

void test_innerproduct_no_bias() {
    float in[] = { 1.f, 1.f, 1.f, 1.f };
    float W[]  = { 1.f, 2.f, 3.f, 4.f };
    float out[1];
    ref_innerproduct(in, W, nullptr, out, 1, 4);
    ASSERT_NEAR(out[0], 10.f, 1e-5f);  // 1+2+3+4
}

void test_innerproduct_batch() {
    // Batched FC: treat each row of input as a separate sample
    float in[] = { 1.f, 2.f,
                   3.f, 4.f }; // 2 samples of size 2
    float W[]  = { 1.f, 1.f }; // 1 output
    float C[2] = {};
    // sample0: 1+2=3, sample1: 3+4=7
    ref_gemm(in, W, C, 2, 2, 1);
    ASSERT_NEAR(C[0], 3.f, 1e-5f);
    ASSERT_NEAR(C[1], 7.f, 1e-5f);
}

void test_matmul_identity() {
    float I[] = { 1, 0, 0, 0,
                  0, 1, 0, 0,
                  0, 0, 1, 0,
                  0, 0, 0, 1 };
    float A[] = { 1.f, 2.f, 3.f, 4.f,
                  5.f, 6.f, 7.f, 8.f,
                  9.f, 10.f, 11.f, 12.f,
                  13.f, 14.f, 15.f, 16.f };
    float C[16] = {};
    ref_gemm(A, I, C, 4, 4, 4);
    ASSERT_VEC_NEAR(C, A, 16, 1e-5f);
}

void test_gemm_broadcast_C_scalar() {
    // C = A*B + beta * scalar_C  (broadcast scalar add)
    float A[] = { 1, 0, 0, 1 };  // 2x2 identity
    float B[] = { 3, 4, 5, 6 };  // 2x2
    float C[] = { 2, 2, 2, 2 };  // C initialized to 2s
    ref_gemm(A, B, C, 2, 2, 2, 1.f, 1.f);
    // A*B = B (since A=I), + C = [3+2, 4+2, 5+2, 6+2]
    float expected[] = { 5, 6, 7, 8 };
    ASSERT_VEC_NEAR(C, expected, 4, 1e-5f);
}

void test_matmul_1d_dot() {
    // Dot product: [n] · [n] → scalar, modelled as [1,n]*[n,1]
    float a[] = { 1.f, 2.f, 3.f };
    float b[] = { 4.f, 5.f, 6.f };
    float c[1] = {};
    ref_gemm(a, b, c, 1, 3, 1);
    // 4+10+18 = 32
    ASSERT_NEAR(c[0], 32.f, 1e-5f);
}

void test_innerproduct_multi_output() {
    float in[] = { 1.f, 2.f };
    // W[4,2]: each row is one output weight
    float W[] = { 1.f, 0.f,
                  0.f, 1.f,
                  1.f, 1.f,
                 -1.f, 1.f };
    float out[4];
    ref_innerproduct(in, W, nullptr, out, 4, 2);
    ASSERT_NEAR(out[0], 1.f, 1e-5f);   // [1*1+2*0]
    ASSERT_NEAR(out[1], 2.f, 1e-5f);   // [1*0+2*1]
    ASSERT_NEAR(out[2], 3.f, 1e-5f);   // [1+2]
    ASSERT_NEAR(out[3], 1.f, 1e-5f);   // [-1+2]
}

// ─── Real ncnn kernel tests ───────────────────────────────────────────────────

void test_matmul_ncnn_square()
{
    // [2,2] × [2,2] — both stored as 2D Mats (h=rows, w=cols)
    // ncnn MatMul: A[M,K] stored as Mat(w=K, h=M), B[K,N] as Mat(w=N, h=K)
    std::vector<float> A_flat = { 1.f, 2.f,   3.f, 4.f };
    std::vector<float> B_flat = { 5.f, 6.f,   7.f, 8.f };
    ncnn::Mat A = make_mat_2d(2, 2, A_flat);  // w=2, h=2
    ncnn::Mat B = make_mat_2d(2, 2, B_flat);

    std::vector<ncnn::Mat> bottom = { A, B };
    std::vector<ncnn::Mat> top(1);

    ncnn::MatMul mm; mm.transB = 0;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(mm.forward(bottom, top, opt), 0);

    std::vector<float> out; read_mat(top[0], out);
    // [1 2]*[5 6] = [1*5+2*7=19, 1*6+2*8=22 ; 3*5+4*7=43, 3*6+4*8=50]
    //        [7 8]
    ASSERT_NEAR(out[0], 19.f, 1e-4f);
    ASSERT_NEAR(out[1], 22.f, 1e-4f);
    ASSERT_NEAR(out[2], 43.f, 1e-4f);
    ASSERT_NEAR(out[3], 50.f, 1e-4f);
}

void test_matmul_ncnn_rect()
{
    // [2,3] × [3,2] = [2,2]
    std::vector<float> A_flat = { 1.f,2.f,3.f,  4.f,5.f,6.f };  // w=3,h=2
    std::vector<float> B_flat = { 7.f,8.f,  9.f,10.f,  11.f,12.f };  // w=2,h=3
    ncnn::Mat A = make_mat_2d(3, 2, A_flat);
    ncnn::Mat B = make_mat_2d(2, 3, B_flat);
    std::vector<ncnn::Mat> bottom = { A, B };
    std::vector<ncnn::Mat> top(1);
    ncnn::MatMul mm; mm.transB = 0;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(mm.forward(bottom, top, opt), 0);
    std::vector<float> out; read_mat(top[0], out);
    ASSERT_NEAR(out[0], 58.f,  1e-4f);
    ASSERT_NEAR(out[1], 64.f,  1e-4f);
    ASSERT_NEAR(out[2], 139.f, 1e-4f);
    ASSERT_NEAR(out[3], 154.f, 1e-4f);
}

void test_matmul_ncnn_transB()
{
    // A[2,2] * B^T[2,2] where B is stored transposed → same as A*B
    // transB=1: B rows are treated as B^T columns
    std::vector<float> A_flat = { 1.f,2.f, 3.f,4.f };
    // B^T stored row-major: B^T[0]=[1,3], B^T[1]=[2,4] → B=[[1,2],[3,4]]
    std::vector<float> BT_flat = { 1.f,3.f, 2.f,4.f };
    ncnn::Mat A = make_mat_2d(2, 2, A_flat);
    ncnn::Mat B = make_mat_2d(2, 2, BT_flat);
    std::vector<ncnn::Mat> bottom = { A, B };
    std::vector<ncnn::Mat> top(1);
    ncnn::MatMul mm; mm.transB = 1;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(mm.forward(bottom, top, opt), 0);
    std::vector<float> out; read_mat(top[0], out);
    // A * B = [[1*1+2*3, 1*2+2*4], [3*1+4*3, 3*2+4*4]] = [[7,10],[15,22]]
    ASSERT_NEAR(out[0],  7.f, 1e-4f);
    ASSERT_NEAR(out[1], 10.f, 1e-4f);
    ASSERT_NEAR(out[2], 15.f, 1e-4f);
    ASSERT_NEAR(out[3], 22.f, 1e-4f);
}

void test_innerproduct_ncnn_basic()
{
    // input=[1,2,3], W=[2,3], bias=[10,0]
    // out[0] = 1*1 + 2*0 + 3*(-1) + 10 = 8
    // out[1] = 1*0 + 2*1 + 3*0     + 0 = 2
    std::vector<float> W = { 1.f, 0.f, -1.f,
                              0.f, 1.f,  0.f };
    std::vector<float> bias = { 10.f, 0.f };
    ncnn::InnerProduct ip;
    ip.num_output       = 2;
    ip.bias_term        = 1;
    ip.weight_data_size = 6;
    ip.int8_scale_term  = 0;
    ip.activation_type  = 0;
    ip.weight_data      = make_weight(W);
    ip.bias_data        = make_weight(bias);

    std::vector<float> in_flat = { 1.f, 2.f, 3.f };
    ncnn::Mat bottom = make_mat_1d(in_flat);
    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(ip.forward(bottom, top, opt), 0);

    std::vector<float> out; read_mat(top, out);
    ASSERT_NEAR(out[0], 8.f, 1e-4f);
    ASSERT_NEAR(out[1], 2.f, 1e-4f);
}

void test_innerproduct_ncnn_no_bias()
{
    std::vector<float> W = { 1.f, 2.f, 3.f, 4.f };
    ncnn::InnerProduct ip;
    ip.num_output       = 1;
    ip.bias_term        = 0;
    ip.weight_data_size = 4;
    ip.int8_scale_term  = 0;
    ip.activation_type  = 0;
    ip.weight_data      = make_weight(W);

    std::vector<float> in_flat = { 1.f, 1.f, 1.f, 1.f };
    ncnn::Mat bottom = make_mat_1d(in_flat);
    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(ip.forward(bottom, top, opt), 0);
    std::vector<float> out; read_mat(top, out);
    ASSERT_NEAR(out[0], 10.f, 1e-4f);  // 1+2+3+4
}

void test_innerproduct_ncnn_relu_activation()
{
    // activation_type=1 (ReLU): negative outputs → 0
    std::vector<float> W = { 1.f, -2.f };  // out = x0 - 2*x1
    ncnn::InnerProduct ip;
    ip.num_output       = 1;
    ip.bias_term        = 0;
    ip.weight_data_size = 2;
    ip.int8_scale_term  = 0;
    ip.activation_type  = 1;  // ReLU
    ip.weight_data      = make_weight(W);

    // 1*1 + (-2)*3 = -5 → after relu → 0
    std::vector<float> in_flat = { 1.f, 3.f };
    ncnn::Mat bottom = make_mat_1d(in_flat);
    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(ip.forward(bottom, top, opt), 0);
    std::vector<float> out; read_mat(top, out);
    ASSERT_NEAR(out[0], 0.f, 1e-6f);
}

void test_innerproduct_ncnn_matches_ref()
{
    // Verify ncnn InnerProduct matches plain matmul reference
    const int input_size = 8, num_output = 4;
    std::vector<float> W(num_output * input_size);
    for (int i = 0; i < num_output * input_size; ++i) W[i] = sinf((float)(i + 1) * 0.3f);
    std::vector<float> bias(num_output);
    for (int i = 0; i < num_output; ++i) bias[i] = cosf((float)(i + 1) * 0.5f);
    std::vector<float> x(input_size);
    for (int i = 0; i < input_size; ++i) x[i] = sinf((float)(i + 1) * 0.7f);

    // Reference
    std::vector<float> ref(num_output);
    for (int o = 0; o < num_output; ++o) {
        float s = bias[o];
        for (int j = 0; j < input_size; ++j) s += x[j] * W[o * input_size + j];
        ref[o] = s;
    }

    ncnn::InnerProduct ip;
    ip.num_output       = num_output;
    ip.bias_term        = 1;
    ip.weight_data_size = num_output * input_size;
    ip.int8_scale_term  = 0;
    ip.activation_type  = 0;
    ip.weight_data      = make_weight(W);
    ip.bias_data        = make_weight(bias);

    ncnn::Mat bottom = make_mat_1d(x);
    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(ip.forward(bottom, top, opt), 0);
    std::vector<float> out; read_mat(top, out);
    ASSERT_VEC_NEAR(out, ref.data(), num_output, 1e-4f);
}

int main() {
    printf("=== test_gemm ===\n");
    printf("\n-- Reference tests --\n");
    RUN_TEST(test_matmul_square);
    RUN_TEST(test_matmul_rect);
    RUN_TEST(test_matmul_vec_dot);
    RUN_TEST(test_gemm_alpha_beta);
    RUN_TEST(test_gemm_transA);
    RUN_TEST(test_gemm_transB);
    RUN_TEST(test_innerproduct_basic);
    RUN_TEST(test_innerproduct_no_bias);
    RUN_TEST(test_innerproduct_batch);
    RUN_TEST(test_matmul_identity);
    RUN_TEST(test_gemm_broadcast_C_scalar);
    RUN_TEST(test_matmul_1d_dot);
    RUN_TEST(test_innerproduct_multi_output);

    printf("\n-- Real ncnn::MatMul / InnerProduct --\n");
    RUN_TEST(test_matmul_ncnn_square);
    RUN_TEST(test_matmul_ncnn_rect);
    RUN_TEST(test_matmul_ncnn_transB);
    RUN_TEST(test_innerproduct_ncnn_basic);
    RUN_TEST(test_innerproduct_ncnn_no_bias);
    RUN_TEST(test_innerproduct_ncnn_relu_activation);
    RUN_TEST(test_innerproduct_ncnn_matches_ref);

    print_summary("gemm");
    return g_failed > 0 ? 1 : 0;
}
