// test_quant.cpp — ARM quant kernel tests
// Tests Quantize_arm, Dequantize_arm

#include "test_utils.h"
#include "ncnn_helpers.h"

#include "../quant/quantize_arm.h"
#include "../quant/dequantize_arm.h"

// ─── Reference quantization ──────────────────────────────────────────────────

static int8_t ref_quantize(float x, float scale) {
    return float_to_int8(x * scale);
}

static float ref_dequantize(int8_t x, float scale) {
    return (float)x * scale;
}

static float ref_dequantize_with_bias(int8_t x, float scale, float bias) {
    return (float)x * scale + bias;
}

static int8_t ref_requantize(int8_t x, float scale_in, float scale_out,
                               float bias_val = 0.f, int activation = 0,
                               float act_p0 = 0.f, float act_p1 = 0.f) {
    float v = (float)x * scale_in + bias_val;
    switch (activation) {
        case 0: break;
        case 1: v = v > 0.f ? v : 0.f; break;
        case 2: v = v > 0.f ? v : act_p0 * v; break;
        case 3: v = std::max(act_p0, std::min(act_p1, v)); break;
        default: break;
    }
    return float_to_int8(v * scale_out);
}

// ─── Reference test cases ────────────────────────────────────────────────────

void test_quantize_basic() {
    ASSERT_EQ((int)ref_quantize(0.f,  1.f), 0);
    ASSERT_EQ((int)ref_quantize(1.f,  1.f), 1);
    ASSERT_EQ((int)ref_quantize(-1.f, 1.f), -1);
    ASSERT_EQ((int)ref_quantize(0.4f, 1.f), 0);
    ASSERT_EQ((int)ref_quantize(0.6f, 1.f), 1);
}

void test_quantize_scale() {
    ASSERT_EQ((int)ref_quantize(1.f, 2.f), 2);
    ASSERT_EQ((int)ref_quantize(0.5f, 4.f), 2);
    ASSERT_EQ((int)ref_quantize(0.3f, 10.f), 3);
}

void test_quantize_clamp() {
    ASSERT_EQ((int)ref_quantize(200.f, 1.f), 127);
    ASSERT_EQ((int)ref_quantize(-200.f, 1.f), -128);
}

void test_dequantize_basic() {
    ASSERT_NEAR(ref_dequantize(0, 1.f),   0.f, 1e-5f);
    ASSERT_NEAR(ref_dequantize(1, 1.f),   1.f, 1e-5f);
    ASSERT_NEAR(ref_dequantize(-1, 1.f), -1.f, 1e-5f);
}

void test_dequantize_scale() {
    ASSERT_NEAR(ref_dequantize(2,  0.5f), 1.f, 1e-5f);
    ASSERT_NEAR(ref_dequantize(10, 0.1f), 1.f, 1e-5f);
}

void test_dequantize_with_bias() {
    ASSERT_NEAR(ref_dequantize_with_bias(1, 1.f, 5.f), 6.f, 1e-5f);
    ASSERT_NEAR(ref_dequantize_with_bias(0, 2.f, -3.f), -3.f, 1e-5f);
}

void test_requantize_no_activation() {
    ASSERT_EQ((int)ref_requantize(10, 0.5f, 2.f), 10);
    ASSERT_EQ((int)ref_requantize(-4, 0.5f, 2.f), -4);
}

void test_requantize_relu() {
    ASSERT_EQ((int)ref_requantize(-5, 1.f, 1.f, 0.f, 1), 0);
    ASSERT_EQ((int)ref_requantize( 5, 1.f, 1.f, 0.f, 1), 5);
}

void test_quantize_range() {
    int8_t q_max = ref_quantize( 127.f, 1.f);
    int8_t q_min = ref_quantize(-128.f, 1.f);
    ASSERT_EQ((int)q_max,  127);
    ASSERT_EQ((int)q_min, -128);
}

// ─── Real ARM kernel tests ────────────────────────────────────────────────────

void test_quantize_arm_basic()
{
    std::vector<float> in_data = { 1.0f, -1.0f, 2.5f, -2.5f };
    ncnn::Mat in = make_mat_1d(in_data);

    ncnn::Quantize_arm q;
    q.scale_data_size = 1;
    std::vector<float> scale_vec = { 2.0f };
    q.scale_data = make_mat_1d(scale_vec);

    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(q.forward(in, top, opt), 0);

    std::vector<int8_t> got;
    read_mat_int8(top, got);
    int expected[] = { 2, -2, 5, -5 };
    for (int i = 0; i < 4; i++)
        ASSERT_EQ((int)got[i], expected[i]);
}

void test_quantize_arm_clamp()
{
    std::vector<float> in_data = { 200.f, -200.f };
    ncnn::Mat in = make_mat_1d(in_data);

    ncnn::Quantize_arm q;
    q.scale_data_size = 1;
    std::vector<float> scale_vec = { 1.0f };
    q.scale_data = make_mat_1d(scale_vec);

    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(q.forward(in, top, opt), 0);

    std::vector<int8_t> got;
    read_mat_int8(top, got);
    ASSERT_EQ((int)got[0],  127);
    ASSERT_TRUE((int)got[1] <= -127);  // ncnn clamps to -127 or -128
}

void test_dequantize_arm_basic()
{
    // Dequantize takes int32 inputs reinterpreted as float
    int32_t int_vals[] = { 10, 20, -5 };
    float float_bits[3];
    memcpy(float_bits, int_vals, sizeof(int_vals));
    std::vector<float> as_float(float_bits, float_bits + 3);
    ncnn::Mat in = make_mat_1d(as_float);

    ncnn::Dequantize_arm dq;
    dq.scale_data_size = 1;
    dq.bias_data_size  = 0;
    std::vector<float> scale_vec = { 0.1f };
    dq.scale_data = make_mat_1d(scale_vec);

    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(dq.forward(in, top, opt), 0);

    std::vector<float> got; read_mat(top, got);
    ASSERT_NEAR(got[0],  1.0f, 1e-4f);
    ASSERT_NEAR(got[1],  2.0f, 1e-4f);
    ASSERT_NEAR(got[2], -0.5f, 1e-4f);
}

void test_dequantize_arm_with_bias()
{
    int32_t int_vals[] = { 0, 4 };
    float float_bits[2];
    memcpy(float_bits, int_vals, sizeof(int_vals));
    std::vector<float> as_float(float_bits, float_bits + 2);
    ncnn::Mat in = make_mat_1d(as_float);

    ncnn::Dequantize_arm dq;
    dq.scale_data_size = 1;
    dq.bias_data_size  = 1;
    std::vector<float> scale_vec = { 1.0f };
    std::vector<float> bias_vec  = { 5.0f };
    dq.scale_data = make_mat_1d(scale_vec);
    dq.bias_data  = make_mat_1d(bias_vec);

    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(dq.forward(in, top, opt), 0);

    std::vector<float> got; read_mat(top, got);
    ASSERT_NEAR(got[0], 5.0f, 1e-4f);
    ASSERT_NEAR(got[1], 9.0f, 1e-4f);
}

int main() {
    printf("=== test_quant (ARM) ===\n");
    RUN_TEST(test_quantize_basic);
    RUN_TEST(test_quantize_scale);
    RUN_TEST(test_quantize_clamp);
    RUN_TEST(test_dequantize_basic);
    RUN_TEST(test_dequantize_scale);
    RUN_TEST(test_dequantize_with_bias);
    RUN_TEST(test_requantize_no_activation);
    RUN_TEST(test_requantize_relu);
    RUN_TEST(test_quantize_range);

    printf("\n--- Real ARM Quantize / Dequantize ---\n");
    RUN_TEST(test_quantize_arm_basic);
    RUN_TEST(test_quantize_arm_clamp);
    RUN_TEST(test_dequantize_arm_basic);
    RUN_TEST(test_dequantize_arm_with_bias);

    print_summary("quant_arm");
    return g_failed > 0 ? 1 : 0;
}
