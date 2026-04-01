// test_recurrent.cpp
// Tests for recurrent/:
//   lstm, gru, rnn
//
// Reference implementations of one time-step for each cell type.
// Tests cover gate values, hidden state updates, direction settings,
// and hidden/cell-state passing.

#include "test_utils.h"
#include "ncnn_helpers.h"
#include "../recurrent/lstm.h"
#include "../recurrent/gru.h"

// ─── Reference single time-step cells ────────────────────────────────────────

// sigmoid and tanh from test_utils.h (sigmoid_f, tanh_f)

// LSTM single step:
//   i = sigmoid(Wx_i * x + Wh_i * h + b_i)
//   f = sigmoid(Wx_f * x + Wh_f * h + b_f)
//   o = sigmoid(Wx_o * x + Wh_o * h + b_o)
//   g = tanh   (Wx_g * x + Wh_g * h + b_g)
//   c' = f*c + i*g
//   h' = o * tanh(c')
//
// Weights packed as [4*H, input_size] and [4*H, hidden_size] for the 4 gates (i,f,o,g).
static void ref_lstm_step(const float* x, int input_size,
                            const float* h, const float* c,
                            float* h_out, float* c_out,
                            const float* Wx,   // [4*H, input_size]
                            const float* Wh,   // [4*H, H]
                            const float* bias, // [4*H]
                            int H) {
    std::vector<float> gates(4 * H);
    for (int gi = 0; gi < 4 * H; ++gi) {
        float v = bias ? bias[gi] : 0.f;
        for (int j = 0; j < input_size; ++j) v += Wx[gi * input_size + j] * x[j];
        for (int j = 0; j < H;          ++j) v += Wh[gi * H + j] * h[j];
        gates[gi] = v;
    }
    // gates layout: [H gates_i | H gates_f | H gates_o | H gates_g]
    for (int hi = 0; hi < H; ++hi) {
        float ig = sigmoid_f(gates[0*H + hi]);
        float fg = sigmoid_f(gates[1*H + hi]);
        float og = sigmoid_f(gates[2*H + hi]);
        float gg = tanh_f   (gates[3*H + hi]);
        c_out[hi] = fg * c[hi] + ig * gg;
        h_out[hi] = og * tanh_f(c_out[hi]);
    }
}

// GRU single step:
//   r = sigmoid(Wx_r * x + Wh_r * h + b_r)
//   u = sigmoid(Wx_u * x + Wh_u * h + b_u)
//   n = tanh   (Wx_n * x + r * (Wh_n * h) + b_n)
//   h'= (1-u)*n + u*h
//
// Weights packed as [3*H, input_size] for gates (r, u, n)
static void ref_gru_step(const float* x, int input_size,
                           const float* h, float* h_out,
                           const float* Wx,   // [3*H, input_size]
                           const float* Wh,   // [3*H, H]
                           const float* bias, // [3*H]
                           int H) {
    // Input contributions
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

// Simple RNN step: h' = tanh(Wx*x + Wh*h + b)
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

// ─── Test cases ──────────────────────────────────────────────────────────────

void test_lstm_zero_input() {
    // All-zero input, weights, biases and init state → gates all sigmoid(0)/tanh(0)
    // i=f=o=0.5, g=0, c'=f*0+i*0=0, h'=o*tanh(0)=0
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

void test_lstm_forget_gate_closed() {
    // Force forget gate to -inf → f≈0 → cell state is wiped
    int H = 1, I = 1;
    // Wx layout: [i, f, o, g] × I, set f bias very negative
    std::vector<float> Wx(4*H*I, 0.f), Wh(4*H*H, 0.f);
    std::vector<float> bias = { 0.f, -100.f, 0.f, 0.f };  // f gate bias = -100
    float x[] = { 1.f };
    float h[] = { 0.f };
    float c[] = { 10.f };  // large previous cell state
    float h_out[1], c_out[1];
    ref_lstm_step(x, I, h, c, h_out, c_out, Wx.data(), Wh.data(), bias.data(), H);
    // f ≈ 0 → c_out ≈ 0 + i*g (i=sigmoid(0)=0.5, g=tanh(0)=0)
    ASSERT_NEAR(c_out[0], 0.f, 0.01f);
}

void test_lstm_input_gate_closed() {
    // Force input gate to -inf → i≈0, no new info written
    int H = 1, I = 1;
    std::vector<float> Wx(4*H*I, 0.f), Wh(4*H*H, 0.f);
    std::vector<float> bias = { -100.f, 100.f, 0.f, 0.f }; // i gate=-inf, f gate=+inf
    float x[] = { 1.f };
    float h[] = { 0.f };
    float c[] = { 5.f };
    float h_out[1], c_out[1];
    ref_lstm_step(x, I, h, c, h_out, c_out, Wx.data(), Wh.data(), bias.data(), H);
    // f≈1 → c_out ≈ c_prev = 5
    ASSERT_NEAR(c_out[0], 5.f, 0.01f);
}

void test_lstm_multi_step() {
    // Unroll 3 time steps with identity weights
    int H = 2, I = 2;
    // Use all-zero weights; bias: all gates set to 0 except input=1
    std::vector<float> Wx(4*H*I, 0.f), Wh(4*H*H, 0.f);
    std::vector<float> bias(4*H, 0.f);
    std::vector<float> h(H, 0.f), c(H, 0.f);
    float inputs[][2] = { {0.5f, 0.5f}, {0.5f, 0.5f}, {0.5f, 0.5f} };
    for (auto& x_step : inputs) {
        std::vector<float> h_out(H), c_out(H);
        ref_lstm_step(x_step, I, h.data(), c.data(), h_out.data(), c_out.data(),
                      Wx.data(), Wh.data(), bias.data(), H);
        h = h_out; c = c_out;
    }
    // State is bounded (tanh/sigmoid keep values in range)
    for (int i = 0; i < H; ++i) {
        ASSERT_TRUE(h[i] >= -1.f && h[i] <= 1.f);
        ASSERT_TRUE(fabsf(c[i]) < 10.f);
    }
}

void test_gru_zero_weights() {
    // All zero weights & bias: r=0.5, u=0.5, n=tanh(0)=0 → h' = (1-0.5)*0 + 0.5*h_prev
    int H = 2, I = 2;
    std::vector<float> Wx(3*H*I, 0.f), Wh(3*H*H, 0.f), bias(3*H, 0.f);
    float x[] = { 0.f, 0.f };
    float h_prev[] = { 1.f, -1.f };
    std::vector<float> h_out(H);
    ref_gru_step(x, I, h_prev, h_out.data(), Wx.data(), Wh.data(), bias.data(), H);
    // u=sigmoid(0)=0.5, n=tanh(0)=0 → h'=0.5*h_prev
    ASSERT_NEAR(h_out[0],  0.5f, 1e-4f);
    ASSERT_NEAR(h_out[1], -0.5f, 1e-4f);
}

void test_gru_reset_gate_open() {
    // Reset gate = 1 → n = tanh(Wx*x + 1*Wh*h) = full hidden contribution
    int H = 1, I = 1;
    std::vector<float> Wx(3*H*I, 0.f), Wh(3*H*H, 0.f);
    std::vector<float> bias = { 100.f,   // r gate very large → r≈1
                                  -100.f,  // u gate very negative → u≈0
                                  0.f };
    float x[] = { 0.f };
    float h_prev[] = { 0.5f };
    std::vector<float> h_out(H);
    ref_gru_step(x, I, h_prev, h_out.data(), Wx.data(), Wh.data(), bias.data(), H);
    // u≈0 → h' ≈ (1-0)*n = n = tanh(r * Wh*h + Wx*x) = tanh(1*0 + 0) = 0
    ASSERT_NEAR(h_out[0], 0.f, 0.01f);
}

void test_gru_update_gate_closed() {
    // u≈0 → output = new candidate (ignores previous hidden)
    int H = 1, I = 1;
    std::vector<float> Wx(3*H*I, 0.f), Wh(3*H*H, 0.f);
    std::vector<float> bias = { 0.f,    // r=0.5
                                  -100.f, // u≈0
                                  100.f }; // n gates → tanh(100) ≈ 1
    float x[] = { 0.f };
    float h_prev[] = { -999.f };  // h_prev should be ignored when u≈0
    std::vector<float> h_out(H);
    ref_gru_step(x, I, h_prev, h_out.data(), Wx.data(), Wh.data(), bias.data(), H);
    // h' = (1-0)*tanh(100) + 0*h_prev ≈ 1
    ASSERT_NEAR(h_out[0], 1.f, 0.01f);
}

void test_rnn_step_basic() {
    int H = 2, I = 2;
    // Simple weights: identity-like
    float Wx[] = { 1.f, 0.f,
                   0.f, 1.f }; // [2, 2]
    float Wh[] = { 0.f, 0.f,
                   0.f, 0.f }; // [2, 2] zero
    float x[] = { 1.f, 2.f };
    float h[] = { 0.f, 0.f };
    std::vector<float> h_out(H);
    ref_rnn_step(x, I, h, h_out.data(), Wx, Wh, nullptr, H);
    // h' = tanh([1, 2])
    ASSERT_NEAR(h_out[0], tanh_f(1.f), 1e-5f);
    ASSERT_NEAR(h_out[1], tanh_f(2.f), 1e-5f);
}

void test_rnn_step_with_hidden() {
    int H = 1, I = 1;
    float Wx[] = { 1.f };  // W_x = 1
    float Wh[] = { 1.f };  // W_h = 1
    float bias[] = { 0.f };
    float x[]  = { 0.5f };
    float h[]  = { 0.5f };
    float h_out[1];
    ref_rnn_step(x, I, h, h_out, Wx, Wh, bias, H);
    // v = 1*0.5 + 1*0.5 = 1.0 → h' = tanh(1)
    ASSERT_NEAR(h_out[0], tanh_f(1.f), 1e-5f);
}

void test_rnn_bounds() {
    // RNN output always in (-1, 1) due to tanh
    int H = 4, I = 4;
    std::vector<float> Wx(H*I), Wh(H*H);
    // Large weights to test tanh saturation
    for (auto& v : Wx) v = 100.f;
    for (auto& v : Wh) v = 100.f;
    float x[] = { 1.f, 1.f, 1.f, 1.f };
    float h[] = { 1.f, 1.f, 1.f, 1.f };
    std::vector<float> h_out(H);
    ref_rnn_step(x, I, h, h_out.data(), Wx.data(), Wh.data(), nullptr, H);
    for (int i = 0; i < H; ++i) {
        // tanhf saturates to exactly ±1 in floating point for large inputs
        ASSERT_TRUE(h_out[i] >= -1.f && h_out[i] <= 1.f);
    }
}

// ─── Reference matching ncnn weight layout ───────────────────────────────────

// LSTM: weight_xc [h=H*4, w=input_size], weight_hc [h=H*4, w=H], bias_c [h=4, w=H]
// Output: output[T * H] row-major
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
            // input contribution: weight_xc row = (H*gate + q) of width input_size
            for (int i = 0; i < input_size; i++) {
                float xi = x[i];
                gI += weight_xc[(0*H + q) * input_size + i] * xi;
                gF += weight_xc[(1*H + q) * input_size + i] * xi;
                gO += weight_xc[(2*H + q) * input_size + i] * xi;
                gG += weight_xc[(3*H + q) * input_size + i] * xi;
            }
            // hidden contribution: weight_hc row = (H*gate + q) of width H
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
        h = h_new;
        c = c_new;
    }
}

// GRU: weight_xc [h=H*3, w=input_size], weight_hc [h=H*3, w=H], bias_c [h=4, w=H]
// bias rows: R, U, WN, BN (ncnn splits new gate bias into two parts)
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
            // new gate: ncnn splits bias into WN and BN parts
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

// ─── Real ncnn LSTM/GRU tests ─────────────────────────────────────────────────

void test_lstm_ncnn_zero_weights()
{
    // zero weights → all gates = 0, outputs = 0
    int I = 3, H = 2, T = 1;
    std::vector<float> wx(H*4 * I, 0.f);
    std::vector<float> wh(H*4 * H, 0.f);
    std::vector<float> bias(4 * H, 0.f);
    std::vector<float> x(T * I, 1.f); // non-zero input, but weights are zero

    // ncnn weight mats: weight_xc=[w=I, h=H*4, c=1], weight_hc=[w=H, h=H*4, c=1], bias=[w=H, h=4, c=1]
    ncnn::LSTM lstm;
    lstm.num_output = H;
    lstm.hidden_size = H;
    lstm.direction = 0;
    lstm.int8_scale_term = 0;
    lstm.weight_xc_data = make_mat(I, H*4, 1, wx);
    lstm.weight_hc_data = make_mat(H, H*4, 1, wh);
    lstm.bias_c_data    = make_mat(H, 4,   1, bias);

    ncnn::Mat bottom = make_mat_2d(I, T, x);
    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(lstm.forward(bottom, top, opt), 0);

    std::vector<float> got;
    read_mat(top, got);
    for (int i = 0; i < (int)got.size(); i++)
        ASSERT_NEAR(got[i], 0.f, 1e-5f);
}

void test_lstm_ncnn_matches_ref()
{
    int I = 3, H = 2, T = 4;
    // Deterministic small weights
    std::vector<float> wx(H*4 * I);
    std::vector<float> wh(H*4 * H);
    std::vector<float> bias(4 * H);
    std::vector<float> x(T * I);
    for (int i = 0; i < (int)wx.size(); i++)   wx[i]   = 0.1f * sinf((float)i * 0.7f);
    for (int i = 0; i < (int)wh.size(); i++)   wh[i]   = 0.1f * cosf((float)i * 0.5f);
    for (int i = 0; i < (int)bias.size(); i++) bias[i] = 0.05f * sinf((float)i * 1.3f);
    for (int i = 0; i < (int)x.size(); i++)    x[i]    = 0.5f * sinf((float)i * 0.9f);

    // Reference
    std::vector<float> ref(T * H);
    ref_lstm_ncnn(x.data(), I, T, wx.data(), wh.data(), bias.data(), H, ref.data());

    ncnn::LSTM lstm;
    lstm.num_output = H;
    lstm.hidden_size = H;
    lstm.direction = 0;
    lstm.int8_scale_term = 0;
    lstm.weight_xc_data = make_mat(I, H*4, 1, wx);
    lstm.weight_hc_data = make_mat(H, H*4, 1, wh);
    lstm.bias_c_data    = make_mat(H, 4,   1, bias);

    ncnn::Mat bottom = make_mat_2d(I, T, x);
    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(lstm.forward(bottom, top, opt), 0);

    std::vector<float> got;
    read_mat(top, got);
    ASSERT_EQ((int)got.size(), T * H);
    for (int i = 0; i < T * H; i++)
        ASSERT_NEAR(got[i], ref[i], 1e-4f);
}

void test_gru_ncnn_zero_weights()
{
    // zero weights + zero biases, zero hidden → h' = (1-0.5)*tanh(0) + 0.5*0 = 0
    int I = 2, H = 2, T = 1;
    std::vector<float> wx(H*3 * I, 0.f);
    std::vector<float> wh(H*3 * H, 0.f);
    std::vector<float> bias(4 * H, 0.f);
    std::vector<float> x(T * I, 1.f);

    ncnn::GRU gru;
    gru.num_output = H;
    gru.direction = 0;
    gru.int8_scale_term = 0;
    gru.weight_xc_data = make_mat(I, H*3, 1, wx);
    gru.weight_hc_data = make_mat(H, H*3, 1, wh);
    gru.bias_c_data    = make_mat(H, 4,   1, bias);

    ncnn::Mat bottom = make_mat_2d(I, T, x);
    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(gru.forward(bottom, top, opt), 0);

    std::vector<float> got;
    read_mat(top, got);
    for (int i = 0; i < (int)got.size(); i++)
        ASSERT_NEAR(got[i], 0.f, 1e-5f);
}

void test_gru_ncnn_matches_ref()
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

    ncnn::GRU gru;
    gru.num_output = H;
    gru.direction = 0;
    gru.int8_scale_term = 0;
    gru.weight_xc_data = make_mat(I, H*3, 1, wx);
    gru.weight_hc_data = make_mat(H, H*3, 1, wh);
    gru.bias_c_data    = make_mat(H, 4,   1, bias);

    ncnn::Mat bottom = make_mat_2d(I, T, x);
    ncnn::Mat top;
    ncnn::Option opt = make_opt();
    ASSERT_EQ(gru.forward(bottom, top, opt), 0);

    std::vector<float> got;
    read_mat(top, got);
    ASSERT_EQ((int)got.size(), T * H);
    for (int i = 0; i < T * H; i++)
        ASSERT_NEAR(got[i], ref[i], 1e-4f);
}

int main() {
    printf("=== test_recurrent ===\n");
    RUN_TEST(test_lstm_zero_input);
    RUN_TEST(test_lstm_forget_gate_closed);
    RUN_TEST(test_lstm_input_gate_closed);
    RUN_TEST(test_lstm_multi_step);
    RUN_TEST(test_gru_zero_weights);
    RUN_TEST(test_gru_reset_gate_open);
    RUN_TEST(test_gru_update_gate_closed);
    RUN_TEST(test_rnn_step_basic);
    RUN_TEST(test_rnn_step_with_hidden);
    RUN_TEST(test_rnn_bounds);

    printf("\n--- Real ncnn::LSTM / GRU ---\n");
    RUN_TEST(test_lstm_ncnn_zero_weights);
    RUN_TEST(test_lstm_ncnn_matches_ref);
    RUN_TEST(test_gru_ncnn_zero_weights);
    RUN_TEST(test_gru_ncnn_matches_ref);

    print_summary("recurrent");
    return g_failed > 0 ? 1 : 0;
}
