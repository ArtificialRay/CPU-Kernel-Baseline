// test_recurrent.cpp — ARM recurrent kernel tests
// Tests GRU_arm, LSTM_arm, RNN_arm

#include "test_utils.h"
#include "ncnn_helpers.h"

#include "../recurrent/lstm_arm.h"
#include "../recurrent/gru_arm.h"
#include "../recurrent/rnn_arm.h"

// ─── Reference implementations ───────────────────────────────────────────────

static void ref_lstm_step(const float* x, int input_size,
                            const float* h, const float* c,
                            float* h_out, float* c_out,
                            const float* Wx,
                            const float* Wh,
                            const float* bias,
                            int H) {
    std::vector<float> gates(4 * H);
    for (int gi = 0; gi < 4 * H; ++gi) {
        float v = bias ? bias[gi] : 0.f;
        for (int j = 0; j < input_size; ++j) v += Wx[gi * input_size + j] * x[j];
        for (int j = 0; j < H;          ++j) v += Wh[gi * H + j] * h[j];
        gates[gi] = v;
    }
    for (int hi = 0; hi < H; ++hi) {
        float ig = sigmoid_f(gates[0*H + hi]);
        float fg = sigmoid_f(gates[1*H + hi]);
        float og = sigmoid_f(gates[2*H + hi]);
        float gg = tanh_f   (gates[3*H + hi]);
        c_out[hi] = fg * c[hi] + ig * gg;
        h_out[hi] = og * tanh_f(c_out[hi]);
    }
}

static void ref_gru_step(const float* x, int input_size,
                           const float* h, float* h_out,
                           const float* Wx,
                           const float* Wh,
                           const float* bias,
                           int H) {
    std::vector<float> wx(3 * H, 0.f);
    std::vector<float> wh(3 * H, 0.f);
    for (int gi = 0; gi < 3 * H; ++gi) {
        float vx = bias ? bias[gi] : 0.f;
        float vh = 0.f;
        for (int j = 0; j < input_size; ++j) vx += Wx[gi * input_size + j] * x[j];
        for (int j = 0; j < H;          ++j) vh += Wh[gi * H + j] * h[j];
        wx[gi] = vx;
        wh[gi] = vh;
    }
    for (int hi = 0; hi < H; ++hi) {
        float r = sigmoid_f(wx[0*H + hi] + wh[0*H + hi]);
        float u = sigmoid_f(wx[1*H + hi] + wh[1*H + hi]);
        float n = tanh_f   (wx[2*H + hi] + r * wh[2*H + hi]);
        h_out[hi] = (1.f - u) * n + u * h[hi];
    }
}

static void ref_rnn_step(const float* x, int input_size,
                           const float* h, float* h_out,
                           const float* Wx,
                           const float* Wh,
                           const float* bias,
                           int H) {
    for (int hi = 0; hi < H; ++hi) {
        float v = bias ? bias[hi] : 0.f;
        for (int j = 0; j < input_size; ++j) v += Wx[hi * input_size + j] * x[j];
        for (int j = 0; j < H;          ++j) v += Wh[hi * H + j] * h[j];
        h_out[hi] = tanh_f(v);
    }
}

// ncnn-layout reference for LSTM
static void ref_lstm_ncnn(const float* x_seq, int input_size, int T,
                           const float* weight_xc, const float* weight_hc,
                           const float* bias_c,
                           int H, float* output)
{
    std::vector<float> h(H, 0.f), c(H, 0.f), h_new(H), c_new(H);
    for (int t = 0; t < T; t++) {
        const float* x = x_seq + t * input_size;
        for (int q = 0; q < H; q++) {
            float gI = bias_c[0*H + q];
            float gF = bias_c[1*H + q];
            float gO = bias_c[2*H + q];
            float gG = bias_c[3*H + q];
            for (int i = 0; i < input_size; i++) {
                float xi = x[i];
                gI += weight_xc[(0*H + q) * input_size + i] * xi;
                gF += weight_xc[(1*H + q) * input_size + i] * xi;
                gO += weight_xc[(2*H + q) * input_size + i] * xi;
                gG += weight_xc[(3*H + q) * input_size + i] * xi;
            }
            for (int i = 0; i < H; i++) {
                float hi = h[i];
                gI += weight_hc[(0*H + q) * H + i] * hi;
                gF += weight_hc[(1*H + q) * H + i] * hi;
                gO += weight_hc[(2*H + q) * H + i] * hi;
                gG += weight_hc[(3*H + q) * H + i] * hi;
            }
            float I = 1.f / (1.f + expf(-gI));
            float F = 1.f / (1.f + expf(-gF));
            float O = 1.f / (1.f + expf(-gO));
            float G = tanhf(gG);
            c_new[q] = F * c[q] + I * G;
            h_new[q] = O * tanhf(c_new[q]);
        }
        memcpy(output + t * H, h_new.data(), H * sizeof(float));
        h = h_new; c = c_new;
    }
}

static void ref_gru_ncnn(const float* x_seq, int input_size, int T,
                          const float* weight_xc, const float* weight_hc,
                          const float* bias_c,
                          int H, float* output)
{
    std::vector<float> h(H, 0.f), h_new(H);
    for (int t = 0; t < T; t++) {
        const float* x = x_seq + t * input_size;
        const float* bias_R  = bias_c + 0*H;
        const float* bias_U  = bias_c + 1*H;
        const float* bias_WN = bias_c + 2*H;
        const float* bias_BN = bias_c + 3*H;
        for (int q = 0; q < H; q++) {
            float R = bias_R[q], U = bias_U[q];
            for (int i = 0; i < input_size; i++) {
                R += weight_xc[(0*H + q) * input_size + i] * x[i];
                U += weight_xc[(1*H + q) * input_size + i] * x[i];
            }
            for (int i = 0; i < H; i++) {
                R += weight_hc[(0*H + q) * H + i] * h[i];
                U += weight_hc[(1*H + q) * H + i] * h[i];
            }
            R = 1.f / (1.f + expf(-R));
            U = 1.f / (1.f + expf(-U));
            float N = bias_BN[q];
            for (int i = 0; i < H; i++) N += weight_hc[(2*H + q) * H + i] * h[i];
            N = bias_WN[q] + R * N;
            for (int i = 0; i < input_size; i++) N += weight_xc[(2*H + q) * input_size + i] * x[i];
            N = tanhf(N);
            h_new[q] = (1.f - U) * N + U * h[q];
        }
        memcpy(output + t * H, h_new.data(), H * sizeof(float));
        h = h_new;
    }
}

// ─── Reference test cases ────────────────────────────────────────────────────

void test_lstm_zero_input() {
    int H = 2, I = 3;
    std::vector<float> Wx(4*H*I, 0.f), Wh(4*H*H, 0.f), bias(4*H, 0.f);
    std::vector<float> x(I, 0.f), h(H, 0.f), c(H, 0.f);
    std::vector<float> h_out(H), c_out(H);
    ref_lstm_step(x.data(), I, h.data(), c.data(), h_out.data(), c_out.data(),
                  Wx.data(), Wh.data(), bias.data(), H);
    for (int i = 0; i < H; ++i) {
        ASSERT_NEAR(h_out[i], 0.f, 1e-5f);
        ASSERT_NEAR(c_out[i], 0.f, 1e-5f);
    }
}

void test_gru_zero_weights() {
    int H = 2, I = 2;
    std::vector<float> Wx(3*H*I, 0.f), Wh(3*H*H, 0.f), bias(3*H, 0.f);
    float x[] = { 0.f, 0.f };
    float h_prev[] = { 1.f, -1.f };
    std::vector<float> h_out(H);
    ref_gru_step(x, I, h_prev, h_out.data(), Wx.data(), Wh.data(), bias.data(), H);
    ASSERT_NEAR(h_out[0],  0.5f, 1e-4f);
    ASSERT_NEAR(h_out[1], -0.5f, 1e-4f);
}

void test_rnn_step_basic() {
    int H = 2, I = 2;
    float Wx[] = { 1.f, 0.f, 0.f, 1.f };
    float Wh[] = { 0.f, 0.f, 0.f, 0.f };
    float x[] = { 1.f, 2.f };
    float h[] = { 0.f, 0.f };
    std::vector<float> h_out(H);
    ref_rnn_step(x, I, h, h_out.data(), Wx, Wh, nullptr, H);
    ASSERT_NEAR(h_out[0], tanh_f(1.f), 1e-5f);
    ASSERT_NEAR(h_out[1], tanh_f(2.f), 1e-5f);
}

// ─── Real ARM kernel tests ────────────────────────────────────────────────────

void test_lstm_arm_zero_weights()
{
    int I = 3, H = 2, T = 1;
    std::vector<float> wx(H*4 * I, 0.f);
    std::vector<float> wh(H*4 * H, 0.f);
    std::vector<float> bias(4 * H, 0.f);
    std::vector<float> x(T * I, 1.f);

    ncnn::LSTM_arm lstm;
    lstm.num_output      = H;
    lstm.hidden_size     = H;
    lstm.direction       = 0;
    lstm.int8_scale_term = 0;
    lstm.weight_xc_data  = make_mat(I, H*4, 1, wx);
    lstm.weight_hc_data  = make_mat(H, H*4, 1, wh);
    lstm.bias_c_data     = make_mat(H, 4,   1, bias);

    ncnn::Mat bottom = make_mat_2d(I, T, x);
    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(lstm.forward(bottom, top, opt), 0);

    std::vector<float> got; read_mat(top, got);
    for (int i = 0; i < (int)got.size(); i++)
        ASSERT_NEAR(got[i], 0.f, 1e-4f);
}

void test_lstm_arm_matches_ref()
{
    int I = 3, H = 2, T = 4;
    std::vector<float> wx(H*4 * I);
    std::vector<float> wh(H*4 * H);
    std::vector<float> bias(4 * H);
    std::vector<float> x(T * I);
    for (int i = 0; i < (int)wx.size(); i++)   wx[i]   = 0.1f * sinf((float)i * 0.7f);
    for (int i = 0; i < (int)wh.size(); i++)   wh[i]   = 0.1f * cosf((float)i * 0.5f);
    for (int i = 0; i < (int)bias.size(); i++) bias[i] = 0.05f * sinf((float)i * 1.3f);
    for (int i = 0; i < (int)x.size(); i++)    x[i]    = 0.5f * sinf((float)i * 0.9f);

    std::vector<float> ref(T * H);
    ref_lstm_ncnn(x.data(), I, T, wx.data(), wh.data(), bias.data(), H, ref.data());

    ncnn::LSTM_arm lstm;
    lstm.num_output      = H;
    lstm.hidden_size     = H;
    lstm.direction       = 0;
    lstm.int8_scale_term = 0;
    lstm.weight_xc_data  = make_mat(I, H*4, 1, wx);
    lstm.weight_hc_data  = make_mat(H, H*4, 1, wh);
    lstm.bias_c_data     = make_mat(H, 4,   1, bias);

    ncnn::Mat bottom = make_mat_2d(I, T, x);
    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(lstm.forward(bottom, top, opt), 0);

    std::vector<float> got; read_mat(top, got);
    ASSERT_EQ((int)got.size(), T * H);
    for (int i = 0; i < T * H; i++)
        ASSERT_NEAR(got[i], ref[i], 1e-3f);
}

void test_gru_arm_zero_weights()
{
    int I = 2, H = 2, T = 1;
    std::vector<float> wx(H*3 * I, 0.f);
    std::vector<float> wh(H*3 * H, 0.f);
    std::vector<float> bias(4 * H, 0.f);
    std::vector<float> x(T * I, 1.f);

    ncnn::GRU_arm gru;
    gru.num_output      = H;
    gru.direction       = 0;
    gru.int8_scale_term = 0;
    gru.weight_xc_data  = make_mat(I, H*3, 1, wx);
    gru.weight_hc_data  = make_mat(H, H*3, 1, wh);
    gru.bias_c_data     = make_mat(H, 4,   1, bias);

    ncnn::Mat bottom = make_mat_2d(I, T, x);
    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(gru.forward(bottom, top, opt), 0);

    std::vector<float> got; read_mat(top, got);
    for (int i = 0; i < (int)got.size(); i++)
        ASSERT_NEAR(got[i], 0.f, 1e-4f);
}

void test_gru_arm_matches_ref()
{
    int I = 3, H = 2, T = 4;
    std::vector<float> wx(H*3 * I);
    std::vector<float> wh(H*3 * H);
    std::vector<float> bias(4 * H);
    std::vector<float> x(T * I);
    for (int i = 0; i < (int)wx.size(); i++)   wx[i]   = 0.1f * sinf((float)i * 0.6f);
    for (int i = 0; i < (int)wh.size(); i++)   wh[i]   = 0.1f * cosf((float)i * 0.4f);
    for (int i = 0; i < (int)bias.size(); i++) bias[i] = 0.05f * cosf((float)i * 1.1f);
    for (int i = 0; i < (int)x.size(); i++)    x[i]    = 0.5f * cosf((float)i * 0.8f);

    std::vector<float> ref(T * H);
    ref_gru_ncnn(x.data(), I, T, wx.data(), wh.data(), bias.data(), H, ref.data());

    ncnn::GRU_arm gru;
    gru.num_output      = H;
    gru.direction       = 0;
    gru.int8_scale_term = 0;
    gru.weight_xc_data  = make_mat(I, H*3, 1, wx);
    gru.weight_hc_data  = make_mat(H, H*3, 1, wh);
    gru.bias_c_data     = make_mat(H, 4,   1, bias);

    ncnn::Mat bottom = make_mat_2d(I, T, x);
    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(gru.forward(bottom, top, opt), 0);

    std::vector<float> got; read_mat(top, got);
    ASSERT_EQ((int)got.size(), T * H);
    for (int i = 0; i < T * H; i++)
        ASSERT_NEAR(got[i], ref[i], 1e-3f);
}

void test_rnn_arm_zero_weights()
{
    int I = 2, H = 2, T = 1;
    std::vector<float> wx(H * I, 0.f);
    std::vector<float> wh(H * H, 0.f);
    std::vector<float> bias(H, 0.f);
    std::vector<float> x(T * I, 1.f);

    ncnn::RNN_arm rnn;
    rnn.num_output      = H;
    rnn.direction       = 0;
    rnn.weight_xc_data  = make_mat(I, H, 1, wx);
    rnn.weight_hc_data  = make_mat(H, H, 1, wh);
    rnn.bias_c_data     = make_mat(H, 1, 1, bias);

    ncnn::Mat bottom = make_mat_2d(I, T, x);
    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(rnn.forward(bottom, top, opt), 0);

    // Output should be tanh(0) = 0 for all units since weights are zero
    std::vector<float> got; read_mat(top, got);
    for (int i = 0; i < (int)got.size(); i++)
        ASSERT_NEAR(got[i], 0.f, 1e-4f);
}

int main() {
    printf("=== test_recurrent (ARM) ===\n");
    printf("\n-- Reference tests --\n");
    RUN_TEST(test_lstm_zero_input);
    RUN_TEST(test_gru_zero_weights);
    RUN_TEST(test_rnn_step_basic);

    printf("\n--- Real ARM LSTM / GRU / RNN ---\n");
    RUN_TEST(test_lstm_arm_zero_weights);
    RUN_TEST(test_lstm_arm_matches_ref);
    RUN_TEST(test_gru_arm_zero_weights);
    RUN_TEST(test_gru_arm_matches_ref);
    RUN_TEST(test_rnn_arm_zero_weights);

    print_summary("recurrent_arm");
    return g_failed > 0 ? 1 : 0;
}
