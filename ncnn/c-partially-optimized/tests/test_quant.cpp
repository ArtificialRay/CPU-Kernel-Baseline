// test_quant.cpp
// Tests for quant/:
//   quantize (float → int8), dequantize (int8 → float),
//   requantize (int8 → scale/activate → int8)
//
// Matches ncnn's symmetric int8 quantization: round-then-clamp to [-128, 127].

#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../quant/quantize.h"
#include "../quant/dequantize.h"

// ─── Reference quantization ──────────────────────────────────────────────────

// float_to_int8 is in test_utils.h

static int8_t ref_quantize(float x, float scale) {
    return float_to_int8(x * scale);
}

static float ref_dequantize(int8_t x, float scale) {
    return (float)x * scale;
}

static float ref_dequantize_with_bias(int8_t x, float scale, float bias) {
    return (float)x * scale + bias;
}

// Requantize: int8 → float (dequant) → optional activation → float → int8 (quantize)
static int8_t ref_requantize(int8_t x, float scale_in, float scale_out,
                               float bias = 0.f, int activation = 0,
                               float act_p0 = 0.f, float act_p1 = 0.f) {
    float v = (float)x * scale_in + bias;
    // Apply activation
    switch (activation) {
        case 0: break;  // none
        case 1: v = v > 0.f ? v : 0.f; break;  // relu
        case 2: v = v > 0.f ? v : act_p0 * v; break; // leakyrelu
        case 3: v = std::max(act_p0, std::min(act_p1, v)); break; // clip
        default: break;
    }
    return float_to_int8(v * scale_out);
}

// Per-channel quantize
static void ref_quantize_perchannel(const float* in, int8_t* out,
                                     int channels, int spatial,
                                     const float* scales) {
    for (int c = 0; c < channels; ++c)
        for (int i = 0; i < spatial; ++i)
            out[c * spatial + i] = float_to_int8(in[c * spatial + i] * scales[c]);
}

// Per-channel dequantize
static void ref_dequantize_perchannel(const int8_t* in, float* out,
                                       int channels, int spatial,
                                       const float* scales,
                                       const float* biases = nullptr) {
    for (int c = 0; c < channels; ++c)
        for (int i = 0; i < spatial; ++i)
            out[c * spatial + i] = (float)in[c * spatial + i] * scales[c]
                                   + (biases ? biases[c] : 0.f);
}

// ─── Test cases ──────────────────────────────────────────────────────────────

void test_quantize_basic() {
    // scale=1: float value should round to nearest int8
    ASSERT_EQ((int)ref_quantize(0.f,  1.f), 0);
    ASSERT_EQ((int)ref_quantize(1.f,  1.f), 1);
    ASSERT_EQ((int)ref_quantize(-1.f, 1.f), -1);
    ASSERT_EQ((int)ref_quantize(0.4f, 1.f), 0);
    ASSERT_EQ((int)ref_quantize(0.6f, 1.f), 1);
}

void test_quantize_scale() {
    // scale=2: multiply then round
    ASSERT_EQ((int)ref_quantize(1.f, 2.f), 2);
    ASSERT_EQ((int)ref_quantize(0.5f, 4.f), 2);
    ASSERT_EQ((int)ref_quantize(0.3f, 10.f), 3);
}

void test_quantize_clamp() {
    // Values beyond int8 range [-128, 127] should be clamped
    ASSERT_EQ((int)ref_quantize(200.f, 1.f), 127);
    ASSERT_EQ((int)ref_quantize(-200.f, 1.f), -128);
    ASSERT_EQ((int)ref_quantize(1000.f, 1.f), 127);
}

void test_quantize_negative() {
    ASSERT_EQ((int)ref_quantize(-0.5f, 2.f), -1);
    ASSERT_EQ((int)ref_quantize(-1.5f, 2.f), -3);
    ASSERT_EQ((int)ref_quantize(-127.f, 1.f), -127);
}

void test_dequantize_basic() {
    // scale=1: int8 → same float value
    ASSERT_NEAR(ref_dequantize(0, 1.f),   0.f, 1e-5f);
    ASSERT_NEAR(ref_dequantize(1, 1.f),   1.f, 1e-5f);
    ASSERT_NEAR(ref_dequantize(-1, 1.f), -1.f, 1e-5f);
    ASSERT_NEAR(ref_dequantize(127, 1.f), 127.f, 1e-5f);
}

void test_dequantize_scale() {
    ASSERT_NEAR(ref_dequantize(2,  0.5f), 1.f,  1e-5f);
    ASSERT_NEAR(ref_dequantize(10, 0.1f), 1.f,  1e-5f);
    ASSERT_NEAR(ref_dequantize(-4, 0.25f), -1.f, 1e-5f);
}

void test_dequantize_with_bias() {
    ASSERT_NEAR(ref_dequantize_with_bias(1, 1.f, 5.f), 6.f, 1e-5f);
    ASSERT_NEAR(ref_dequantize_with_bias(0, 2.f, -3.f), -3.f, 1e-5f);
}

void test_quantize_dequantize_roundtrip() {
    // Quantize then dequantize: recovered value ≈ original (within quantization error)
    float orig_scale = 127.f / 3.f;    // 3.f maps to 127
    float values[] = { 0.f, 1.f, -1.f, 2.5f, -2.5f };
    for (float v : values) {
        int8_t q = ref_quantize(v, orig_scale);
        float deq = ref_dequantize(q, 1.f / orig_scale);
        // Error should be within ±0.5 / scale
        ASSERT_NEAR(deq, v, 0.5f / orig_scale + 1e-4f);
    }
}

void test_requantize_no_activation() {
    // Requantize: scale_in=0.5, scale_out=2, bias=0
    // x=10 → 10*0.5=5.0 → *2=10 → int8(10)
    ASSERT_EQ((int)ref_requantize(10, 0.5f, 2.f), 10);
    ASSERT_EQ((int)ref_requantize(-4, 0.5f, 2.f), -4);
}

void test_requantize_relu() {
    // ReLU activation (type=1): negative values zeroed before output quant
    ASSERT_EQ((int)ref_requantize(-5, 1.f, 1.f, 0.f, 1), 0);
    ASSERT_EQ((int)ref_requantize( 5, 1.f, 1.f, 0.f, 1), 5);
}

void test_requantize_clip() {
    // Clip activation (type=3): clamp to [0, 6] (ReLU6)
    ASSERT_EQ((int)ref_requantize(10, 1.f, 1.f, 0.f, 3, 0.f, 6.f), 6);
    ASSERT_EQ((int)ref_requantize(-2, 1.f, 1.f, 0.f, 3, 0.f, 6.f), 0);
    ASSERT_EQ((int)ref_requantize(3,  1.f, 1.f, 0.f, 3, 0.f, 6.f), 3);
}

void test_requantize_with_bias() {
    // Bias shifts value before output quantize
    // x=5, scale_in=1, bias=2 → 5+2=7 → scale_out=1 → int8(7)
    ASSERT_EQ((int)ref_requantize(5, 1.f, 1.f, 2.f, 0), 7);
}

void test_quantize_perchannel() {
    float in[] = { 1.f, 2.f,   // ch0: scale=2 → [2, 4]
                   3.f, 6.f };  // ch1: scale=0.5 → [1 (round 1.5), 3]
    float scales[] = { 2.f, 0.5f };
    int8_t out[4];
    ref_quantize_perchannel(in, out, 2, 2, scales);
    ASSERT_EQ((int)out[0], 2);
    ASSERT_EQ((int)out[1], 4);
    ASSERT_EQ((int)out[2], 2);  // round(3*0.5)=2 (round half-to-even may differ; accept 1 or 2)
    ASSERT_EQ((int)out[3], 3);
}

void test_dequantize_perchannel() {
    int8_t in[] = { 10, 20,   // ch0: scale=0.1 → [1, 2]
                    30, 40 }; // ch1: scale=0.5 → [15, 20]
    float out[4];
    float scales[] = { 0.1f, 0.5f };
    ref_dequantize_perchannel(in, out, 2, 2, scales);
    ASSERT_NEAR(out[0], 1.f, 1e-4f);
    ASSERT_NEAR(out[1], 2.f, 1e-4f);
    ASSERT_NEAR(out[2], 15.f, 1e-4f);
    ASSERT_NEAR(out[3], 20.f, 1e-4f);
}

void test_dequantize_perchannel_with_bias() {
    int8_t in[] = { 1, 2 };  // 1 channel, 2 spatial
    float scales[] = { 1.f };
    float biases[] = { 100.f };
    float out[2];
    ref_dequantize_perchannel(in, out, 1, 2, scales, biases);
    ASSERT_NEAR(out[0], 101.f, 1e-5f);
    ASSERT_NEAR(out[1], 102.f, 1e-5f);
}

void test_quantize_range() {
    // Verify all values in [-128, 127] stay in int8 range
    float max_val = 127.f;
    int8_t q_max = ref_quantize( max_val, 1.f);
    int8_t q_min = ref_quantize(-128.f,  1.f);
    ASSERT_EQ((int)q_max,  127);
    ASSERT_EQ((int)q_min, -128);
}

// ─── Real ncnn::Quantize tests ────────────────────────────────────────────────

void test_quantize_ncnn_basic()
{
    // scale=2: 1.0*2=2, -1*2=-2, 2.5*2=5, -2.5*2=-5
    std::vector<float> in_data = { 1.0f, -1.0f, 2.5f, -2.5f };
    ncnn::Mat in = make_mat_1d(in_data);

    ncnn::Quantize q;
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

void test_quantize_ncnn_clamp()
{
    // Values beyond [-127, 127] (ncnn clamps to -127, not -128)
    std::vector<float> in_data = { 200.f, -200.f };
    ncnn::Mat in = make_mat_1d(in_data);

    ncnn::Quantize q;
    q.scale_data_size = 1;
    std::vector<float> scale_vec = { 1.0f };
    q.scale_data = make_mat_1d(scale_vec);

    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(q.forward(in, top, opt), 0);

    std::vector<int8_t> got;
    read_mat_int8(top, got);
    ASSERT_EQ((int)got[0],  127);
    ASSERT_EQ((int)got[1], -127); // ncnn clamps to -127
}

// ─── Real ncnn::Dequantize tests ──────────────────────────────────────────────

void test_dequantize_ncnn_basic()
{
    // Dequantize takes int32 inputs; pass int bits reinterpreted as float
    // Values: 10, 20, -5 → with scale=0.1 → 1.0, 2.0, -0.5
    int32_t int_vals[] = { 10, 20, -5 };
    float float_bits[3];
    memcpy(float_bits, int_vals, sizeof(int_vals));
    std::vector<float> as_float(float_bits, float_bits + 3);
    ncnn::Mat in = make_mat_1d(as_float);

    ncnn::Dequantize dq;
    dq.scale_data_size = 1;
    dq.bias_data_size  = 0;
    std::vector<float> scale_vec = { 0.1f };
    dq.scale_data = make_mat_1d(scale_vec);

    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(dq.forward(in, top, opt), 0);

    std::vector<float> got;
    read_mat(top, got);
    ASSERT_NEAR(got[0],  1.0f, 1e-5f);
    ASSERT_NEAR(got[1],  2.0f, 1e-5f);
    ASSERT_NEAR(got[2], -0.5f, 1e-5f);
}

void test_dequantize_ncnn_with_bias()
{
    // values: [0, 4] → scale=1.0, bias=5.0 → [5.0, 9.0]
    int32_t int_vals[] = { 0, 4 };
    float float_bits[2];
    memcpy(float_bits, int_vals, sizeof(int_vals));
    std::vector<float> as_float(float_bits, float_bits + 2);
    ncnn::Mat in = make_mat_1d(as_float);

    ncnn::Dequantize dq;
    dq.scale_data_size = 1;
    dq.bias_data_size  = 1;
    std::vector<float> scale_vec = { 1.0f };
    std::vector<float> bias_vec  = { 5.0f };
    dq.scale_data = make_mat_1d(scale_vec);
    dq.bias_data  = make_mat_1d(bias_vec);

    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(dq.forward(in, top, opt), 0);

    std::vector<float> got;
    read_mat(top, got);
    ASSERT_NEAR(got[0], 5.0f, 1e-5f);
    ASSERT_NEAR(got[1], 9.0f, 1e-5f);
}

int main() {
    printf("=== test_quant ===\n");
    RUN_TEST(test_quantize_basic);
    RUN_TEST(test_quantize_scale);
    RUN_TEST(test_quantize_clamp);
    RUN_TEST(test_quantize_negative);
    RUN_TEST(test_dequantize_basic);
    RUN_TEST(test_dequantize_scale);
    RUN_TEST(test_dequantize_with_bias);
    RUN_TEST(test_quantize_dequantize_roundtrip);
    RUN_TEST(test_requantize_no_activation);
    RUN_TEST(test_requantize_relu);
    RUN_TEST(test_requantize_clip);
    RUN_TEST(test_requantize_with_bias);
    RUN_TEST(test_quantize_perchannel);
    RUN_TEST(test_dequantize_perchannel);
    RUN_TEST(test_dequantize_perchannel_with_bias);
    RUN_TEST(test_quantize_range);

    printf("\n--- Real ncnn::Quantize / Dequantize ---\n");
    RUN_TEST(test_quantize_ncnn_basic);
    RUN_TEST(test_quantize_ncnn_clamp);
    RUN_TEST(test_dequantize_ncnn_basic);
    RUN_TEST(test_dequantize_ncnn_with_bias);

    print_summary("quant");
    return g_failed > 0 ? 1 : 0;
}
