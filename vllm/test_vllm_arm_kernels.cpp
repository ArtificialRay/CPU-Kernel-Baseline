// test_vllm_arm_kernels.cpp
//
// Standalone test harness for vLLM's ARM CPU bottleneck kernels.
// No PyTorch / LibTorch / pybind11 dependency.
//
// Build (host x86, reference mode):
//   g++ -std=c++17 -O2 -DVLLM_STANDALONE \
//       -I./include test_vllm_arm_kernels.cpp -lm -o test_vllm_arm
//   ./test_vllm_arm
//
// Build (AArch64, with NEON):
//   aarch64-linux-gnu-g++ -std=c++17 -O3 \
//       -march=armv8.2-a+dotprod+fp16 \
//       -DVLLM_STANDALONE \
//       -I./include test_vllm_arm_kernels.cpp -lm -lpthread -o test_vllm_arm_neon
//
// Build (AArch64, with BF16 — Graviton3/Neoverse V1/Apple M series):
//   aarch64-linux-gnu-g++ -std=c++17 -O3 \
//       -march=armv8.2-a+dotprod+fp16+bf16 \
//       -DVLLM_STANDALONE -DARM_BF16_SUPPORT \
//       -I./include test_vllm_arm_kernels.cpp -lm -lpthread -o test_vllm_arm_bf16

#define VLLM_STANDALONE
#include "vllm_arm_kernels.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace vllm_test;

// ── Micro test framework ────────────────────────────────────────────────────
static int g_pass = 0, g_fail = 0;

#define EXPECT_CLOSE(name, val, tol) do {                              \
    float _v = (val);                                                   \
    if (_v <= (tol)) {                                                  \
        std::printf("  [PASS] %-45s (max_diff=%.2e)\n", name, _v);     \
        ++g_pass;                                                       \
    } else {                                                            \
        std::printf("  [FAIL] %-45s (max_diff=%.2e > tol=%.2e)\n",     \
                    name, _v, (float)(tol));                            \
        ++g_fail;                                                       \
    }                                                                   \
} while(0)

#define EXPECT_TRUE(name, cond) do {                                    \
    if (cond) { std::printf("  [PASS] %s\n", name); ++g_pass; }        \
    else      { std::printf("  [FAIL] %s\n", name); ++g_fail; }        \
} while(0)

static float tensor_max_diff(const at::Tensor& a, const at::Tensor& b) {
    assert(a.numel() == b.numel());
    const float* pa = a.data_ptr<float>();
    const float* pb = b.data_ptr<float>();
    float d = 0.f;
    for (int64_t i = 0; i < a.numel(); ++i)
        d = std::max(d, std::abs(pa[i] - pb[i]));
    return d;
}

// ── Reference implementations (golden) ─────────────────────────────────────

// RMS norm reference
static void ref_rms_norm(float* out, const float* inp,
                          const float* weight, int64_t T, int64_t H, float eps) {
    for (int64_t t = 0; t < T; ++t) {
        const float* x = inp + t*H;
        float ss = 0.f;
        for (int64_t i=0;i<H;++i) ss += x[i]*x[i];
        float inv = 1.f / std::sqrt(ss/H + eps);
        for (int64_t i=0;i<H;++i) out[t*H+i] = x[i]*inv*weight[i];
    }
}

// SiLU-and-mul reference
static void ref_silu_and_mul(float* out, const float* inp, int64_t T, int64_t D) {
    for (int64_t t=0;t<T;++t)
        for (int64_t i=0;i<D;++i) {
            float g = inp[t*2*D+i];
            float m = inp[t*2*D+D+i];
            out[t*D+i] = (g/(1.f+std::exp(-g)))*m;
        }
}

// GELU-and-mul reference
static void ref_gelu_and_mul(float* out, const float* inp, int64_t T, int64_t D) {
    for (int64_t t=0;t<T;++t)
        for (int64_t i=0;i<D;++i) {
            float g = inp[t*2*D+i];
            float m = inp[t*2*D+D+i];
            out[t*D+i] = 0.5f*g*(1.f+std::erf(g*0.70710678f))*m;
        }
}

// Naive paged attention reference (single-head, arbitrary ctx_len)
static void ref_paged_attention(
        float*         out,
        const float*   query,       // [H]
        const float*   key_cache,   // [total_slots, H]
        const float*   val_cache,   // [total_slots, H]
        const int*     slot_map,    // ctx_len physical slot indices
        int            ctx_len,
        int            head_size,
        float          scale) {
    std::vector<float> logits(ctx_len);
    for (int i=0;i<ctx_len;++i) {
        float dot=0;
        for (int d=0;d<head_size;++d)
            dot += query[d] * key_cache[slot_map[i]*head_size+d];
        logits[i] = dot*scale;
    }
    // Softmax
    float mx = *std::max_element(logits.begin(), logits.end());
    float sm = 0; for (auto& l:logits) { l=std::exp(l-mx); sm+=l; }
    for (auto& l:logits) l/=sm;
    // Weighted sum
    std::fill(out, out+head_size, 0.f);
    for (int i=0;i<ctx_len;++i)
        for (int d=0;d<head_size;++d)
            out[d] += logits[i]*val_cache[slot_map[i]*head_size+d];
}

// ── Test: RMS Norm ──────────────────────────────────────────────────────────
static void test_rms_norm() {
    std::puts("\n[RMSNorm]");
    const int64_t T=16, H=256;
    const float EPS=1e-5f;

    auto inp    = rand_tensor({T, H});
    auto weight = rand_tensor({H}, c10::ScalarType::Float, 0.5f, 1.5f);
    auto out    = make_tensor({T, H});
    auto out_ref= make_tensor({T, H});

    // Reference
    ref_rms_norm(out_ref.data_ptr<float>(), inp.data_ptr<float>(),
                 weight.data_ptr<float>(), T, H, EPS);

    // Kernel
    rms_norm(out, inp, weight, EPS);
    EXPECT_CLOSE("rms_norm fp32 [T=16,H=256]", tensor_max_diff(out, out_ref), 1e-5f);

    // BF16
    auto inp_bf = rand_tensor({T, H}, c10::ScalarType::BFloat16);
    auto wt_bf  = rand_tensor({H},   c10::ScalarType::BFloat16, 0.5f, 1.5f);
    auto out_bf = make_tensor({T, H}, c10::ScalarType::BFloat16);
    rms_norm(out_bf, inp_bf, wt_bf, EPS);
    EXPECT_TRUE("rms_norm bf16 runs without crash", true);

    // FP16
    auto inp_f16= rand_tensor({T, H}, c10::ScalarType::Half);
    auto wt_f16 = rand_tensor({H},   c10::ScalarType::Half, 0.5f, 1.5f);
    auto out_f16= make_tensor({T, H}, c10::ScalarType::Half);
    rms_norm(out_f16, inp_f16, wt_f16, EPS);
    EXPECT_TRUE("rms_norm fp16 runs without crash", true);

    // Larger batch
    const int64_t T2=128, H2=4096;
    auto inp2  = rand_tensor({T2,H2});
    auto wt2   = rand_tensor({H2}, c10::ScalarType::Float, 0.5f, 1.5f);
    auto out2  = make_tensor({T2,H2});
    auto ref2  = make_tensor({T2,H2});
    ref_rms_norm(ref2.data_ptr<float>(), inp2.data_ptr<float>(),
                 wt2.data_ptr<float>(), T2, H2, EPS);
    rms_norm(out2, inp2, wt2, EPS);
    EXPECT_CLOSE("rms_norm fp32 [T=128,H=4096]", tensor_max_diff(out2,ref2), 1e-4f);
}

// ── Test: fused_add_rms_norm ────────────────────────────────────────────────
static void test_fused_add_rms_norm() {
    std::puts("\n[fused_add_rms_norm]");
    const int64_t T=8, H=512;
    const float EPS=1e-5f;

    auto inp      = rand_tensor({T,H});
    auto residual = rand_tensor({T,H});
    auto weight   = rand_tensor({H}, c10::ScalarType::Float, 0.5f, 1.5f);

    // Build reference: residual_ref = residual + inp; out_ref = rms_norm(residual_ref)
    auto res_ref = make_tensor({T,H});
    auto out_ref = make_tensor({T,H});
    {
        const float* r=residual.data_ptr<float>();
        const float* x=inp.data_ptr<float>();
        float* rr=res_ref.data_ptr<float>();
        for (int64_t i=0;i<T*H;++i) rr[i]=r[i]+x[i];
    }
    ref_rms_norm(out_ref.data_ptr<float>(), res_ref.data_ptr<float>(),
                 weight.data_ptr<float>(), T, H, EPS);

    fused_add_rms_norm(inp, residual, weight, EPS);

    // After call: inp should contain the rms-normed output
    EXPECT_CLOSE("fused_add_rms_norm output fp32", tensor_max_diff(inp, out_ref), 1e-4f);
    // residual should contain the updated residual
    EXPECT_CLOSE("fused_add_rms_norm residual fp32", tensor_max_diff(residual, res_ref), 1e-5f);
}

// ── Test: silu_and_mul ──────────────────────────────────────────────────────
static void test_silu_and_mul() {
    std::puts("\n[silu_and_mul]");
    const int64_t T=8, D=512;

    auto inp = rand_tensor({T, 2*D});
    auto out = make_tensor({T, D});
    auto ref = make_tensor({T, D});

    ref_silu_and_mul(ref.data_ptr<float>(), inp.data_ptr<float>(), T, D);
    silu_and_mul(out, inp);
    EXPECT_CLOSE("silu_and_mul fp32 [T=8,D=512]", tensor_max_diff(out,ref), 1e-5f);

    // BF16
    auto inp_bf = rand_tensor({T,2*D}, c10::ScalarType::BFloat16);
    auto out_bf = make_tensor({T,D},   c10::ScalarType::BFloat16);
    silu_and_mul(out_bf, inp_bf);
    EXPECT_TRUE("silu_and_mul bf16 runs", true);

    // Large
    const int64_t T2=64, D2=8192;
    auto inp2=rand_tensor({T2,2*D2});
    auto out2=make_tensor({T2,D2});
    auto ref2=make_tensor({T2,D2});
    ref_silu_and_mul(ref2.data_ptr<float>(), inp2.data_ptr<float>(), T2, D2);
    silu_and_mul(out2, inp2);
    EXPECT_CLOSE("silu_and_mul fp32 [T=64,D=8192]", tensor_max_diff(out2,ref2), 1e-4f);
}

// ── Test: gelu_and_mul ──────────────────────────────────────────────────────
static void test_gelu_and_mul() {
    std::puts("\n[gelu_and_mul / gelu_tanh_and_mul]");
    const int64_t T=4, D=256;
    auto inp=rand_tensor({T,2*D});
    auto out=make_tensor({T,D});
    auto ref=make_tensor({T,D});

    ref_gelu_and_mul(ref.data_ptr<float>(), inp.data_ptr<float>(), T, D);
    gelu_and_mul(out, inp);
    EXPECT_CLOSE("gelu_and_mul fp32", tensor_max_diff(out,ref), 1e-4f);

    auto out2=make_tensor({T,D});
    gelu_tanh_and_mul(out2, inp);
    EXPECT_TRUE("gelu_tanh_and_mul runs without crash", true);
}

// ── Test: paged_attention_v1 ────────────────────────────────────────────────
static void test_paged_attention_v1() {
    std::puts("\n[paged_attention_v1]");

    const int num_seqs     = 2;
    const int num_heads    = 4;
    const int num_kv_heads = 2;
    const int head_size    = 64;
    const int block_size   = 16;
    const int ctx_len      = 48;   // 3 full blocks
    const int max_blocks   = 4;
    const int num_blocks   = 12;   // physical blocks in pool
    const int x            = 8;    // float32 interleave
    const float scale      = 1.f / std::sqrt((float)head_size);

    // ── Allocate tensors ────────────────────────────────────────────────────
    auto query = rand_tensor({num_seqs, num_heads, head_size});

    // key_cache:   [num_blocks, num_kv_heads, head_size/x, block_size, x]
    auto key_cache = rand_tensor(
        {num_blocks, num_kv_heads, head_size/x, block_size, x});
    // value_cache: [num_blocks, num_kv_heads, head_size, block_size]
    auto val_cache = rand_tensor(
        {num_blocks, num_kv_heads, head_size, block_size});

    // block_tables: [num_seqs, max_blocks]
    auto btables = at::Tensor::empty({num_seqs, max_blocks}, c10::ScalarType::Int);
    // seq 0 → physical blocks 0,1,2;  seq 1 → blocks 3,4,5
    {
        int32_t* p = btables.data_ptr<int32_t>();
        p[0]=0; p[1]=1; p[2]=2; p[3]=0;   // seq 0
        p[4]=3; p[5]=4; p[6]=5; p[7]=0;   // seq 1
    }

    // seq_lens: both sequences have ctx_len tokens
    auto seq_lens = at::Tensor::empty({num_seqs}, c10::ScalarType::Int);
    seq_lens.data_ptr<int32_t>()[0] = ctx_len;
    seq_lens.data_ptr<int32_t>()[1] = ctx_len;

    auto out = make_tensor({num_seqs, num_heads, head_size});

    paged_attention_v1(out, query, key_cache, val_cache,
                       num_kv_heads, scale, btables, seq_lens,
                       block_size, ctx_len, std::nullopt, "auto", 1.f);

    // Smoke test: output should be finite and non-zero
    const float* op = out.data_ptr<float>();
    bool finite = true, nonzero = false;
    for (int64_t i=0; i<out.numel(); ++i) {
        if (!std::isfinite(op[i])) { finite=false; break; }
        if (std::abs(op[i]) > 1e-6f) nonzero=true;
    }
    EXPECT_TRUE("paged_attention_v1 output is finite", finite);
    EXPECT_TRUE("paged_attention_v1 output is non-zero", nonzero);

    // ── Verify against single-head reference (seq 0, head 0) ───────────────
    // Build flat KV for seq 0, kv-head 0:
    int kv_h0 = 0;  // head 0 maps to kv-head 0  (4heads/2kv_heads = 2)
    // Convert key from interleaved layout to plain [ctx_len, head_size]
    std::vector<float> ref_keys  (ctx_len * head_size);
    std::vector<float> ref_vals  (ctx_len * head_size);
    std::vector<int>   slot_map  (ctx_len);

    // Physical blocks for seq 0: 0,1,2
    int32_t phys[3] = {0,1,2};
    for (int ti=0; ti<ctx_len; ++ti) {
        int bl = ti/block_size, sl = ti%block_size;
        slot_map[ti] = ti;  // we'll lay out flat
        int32_t pb = phys[bl];

        // Load key from interleaved layout
        const float* kbase = key_cache.data_ptr<float>()
            + pb * num_kv_heads * (head_size/x) * block_size * x
            + kv_h0 * (head_size/x) * block_size * x;
        for (int d=0;d<head_size;++d) {
            int g=d/x, w=d%x;
            ref_keys[ti*head_size+d] = kbase[g*block_size*x + sl*x + w];
        }

        // Load value from [num_kv_heads, head_size, block_size]
        const float* vbase = val_cache.data_ptr<float>()
            + pb * num_kv_heads * head_size * block_size
            + kv_h0 * head_size * block_size;
        for (int d=0;d<head_size;++d)
            ref_vals[ti*head_size+d] = vbase[d*block_size + sl];
    }

    std::vector<float> ref_out(head_size);
    ref_paged_attention(ref_out.data(),
                        query.data_ptr<float>(),  // seq0, head0
                        ref_keys.data(), ref_vals.data(),
                        slot_map.data(), ctx_len, head_size, scale);

    const float* ker_out = out.data_ptr<float>();  // seq0, head0
    float diff = 0.f;
    for (int d=0;d<head_size;++d)
        diff = std::max(diff, std::abs(ker_out[d] - ref_out[d]));
    EXPECT_CLOSE("paged_attention_v1 vs reference (seq0,h0)", diff, 1e-4f);
}

// ── Test: paged_attention_v2 ────────────────────────────────────────────────
static void test_paged_attention_v2() {
    std::puts("\n[paged_attention_v2]");

    // Use a long context (> PARTITION_SIZE=512) to exercise the two-pass path
    const int num_seqs     = 1;
    const int num_heads    = 2;
    const int num_kv_heads = 2;
    const int head_size    = 64;
    const int block_size   = 16;
    const int ctx_len      = 640;  // 40 blocks, spans 2 partitions
    const int max_blocks   = 50;
    const int num_blocks   = 50;
    const int max_parts    = 2;
    const int x            = 8;
    const float scale      = 1.f / std::sqrt((float)head_size);

    auto query     = rand_tensor({num_seqs, num_heads, head_size});
    auto key_cache = rand_tensor({num_blocks, num_kv_heads, head_size/x, block_size, x});
    auto val_cache = rand_tensor({num_blocks, num_kv_heads, head_size, block_size});

    auto btables   = at::Tensor::empty({num_seqs, max_blocks}, c10::ScalarType::Int);
    {
        int32_t* p = btables.data_ptr<int32_t>();
        for (int i=0;i<max_blocks;++i) p[i] = i;  // consecutive physical blocks
    }
    auto seq_lens  = at::Tensor::empty({num_seqs}, c10::ScalarType::Int);
    seq_lens.data_ptr<int32_t>()[0] = ctx_len;

    auto out        = make_tensor({num_seqs, num_heads, head_size});
    auto exp_sums   = make_tensor({num_seqs, num_heads, max_parts});
    auto max_logits = make_tensor({num_seqs, num_heads, max_parts});
    auto tmp_out    = make_tensor({num_seqs, num_heads, max_parts, head_size});

    paged_attention_v2(out, exp_sums, max_logits, tmp_out,
                       query, key_cache, val_cache,
                       num_kv_heads, scale, btables, seq_lens,
                       block_size, ctx_len, std::nullopt, "auto", 1.f);

    // Check output is finite + non-trivial
    const float* op = out.data_ptr<float>();
    bool finite=true, nonzero=false;
    for (int64_t i=0;i<out.numel();++i) {
        if (!std::isfinite(op[i])) { finite=false; break; }
        if (std::abs(op[i])>1e-6f) nonzero=true;
    }
    EXPECT_TRUE("paged_attention_v2 output is finite",   finite);
    EXPECT_TRUE("paged_attention_v2 output is non-zero", nonzero);

    // v1 vs v2 consistency: same inputs, same expected output
    auto out_v1 = make_tensor({num_seqs, num_heads, head_size});
    paged_attention_v1(out_v1, query, key_cache, val_cache,
                       num_kv_heads, scale, btables, seq_lens,
                       block_size, ctx_len, std::nullopt, "auto", 1.f);

    float diff = tensor_max_diff(out, out_v1);
    EXPECT_CLOSE("paged_attention_v2 matches v1 (ctx=640)", diff, 1e-3f);
}

// ── Micro-benchmark ─────────────────────────────────────────────────────────
static void bench_rms_norm() {
    std::puts("\n[Benchmark: rms_norm]");
    const int64_t T=256, H=4096;
    auto inp=rand_tensor({T,H}), wt=rand_tensor({H}), out=make_tensor({T,H});
    const int ITERS = 500;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i=0;i<ITERS;++i) rms_norm(out,inp,wt,1e-5);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    std::printf("  rms_norm T=%ld H=%ld : %.2f ms/iter  (%.1f GB/s est.)\n",
                (long)T,(long)H, ms/ITERS,
                2.0*T*H*4 / (ms/ITERS*1e-3) / 1e9);
}

static void bench_silu_and_mul() {
    std::puts("\n[Benchmark: silu_and_mul]");
    const int64_t T=256, D=8192;
    auto inp=rand_tensor({T,2*D}), out=make_tensor({T,D});
    const int ITERS = 200;
    auto t0=std::chrono::high_resolution_clock::now();
    for (int i=0;i<ITERS;++i) silu_and_mul(out,inp);
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    std::printf("  silu_and_mul T=%ld D=%ld : %.2f ms/iter\n",
                (long)T,(long)D, ms/ITERS);
}

static void bench_paged_attention_v1() {
    std::puts("\n[Benchmark: paged_attention_v1]");
    const int num_seqs=8, num_heads=32, num_kv_heads=8;
    const int head_size=128, block_size=16, ctx_len=512;
    const int max_blocks=(ctx_len+block_size-1)/block_size;
    const int num_blocks=num_seqs*max_blocks+16;
    const int x=8;
    float scale=1.f/std::sqrt((float)head_size);

    auto query=rand_tensor({num_seqs,num_heads,head_size});
    auto kc=rand_tensor({num_blocks,num_kv_heads,head_size/x,block_size,x});
    auto vc=rand_tensor({num_blocks,num_kv_heads,head_size,block_size});
    auto bt=at::Tensor::empty({num_seqs,max_blocks},c10::ScalarType::Int);
    {
        int32_t* p=bt.data_ptr<int32_t>();
        for(int s=0;s<num_seqs;++s)
            for(int b=0;b<max_blocks;++b)
                p[s*max_blocks+b]=s*max_blocks+b;
    }
    auto sl=at::Tensor::empty({num_seqs},c10::ScalarType::Int);
    for(int i=0;i<num_seqs;++i) sl.data_ptr<int32_t>()[i]=ctx_len;
    auto out=make_tensor({num_seqs,num_heads,head_size});

    const int ITERS=20;
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<ITERS;++i)
        paged_attention_v1(out,query,kc,vc,num_kv_heads,scale,bt,sl,
                           block_size,ctx_len,std::nullopt,"auto",1.f);
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    std::printf("  paged_attn_v1 seqs=%d heads=%d ctx=%d head_sz=%d : %.1f ms/iter\n",
                num_seqs,num_heads,ctx_len,head_size,ms/ITERS);
}

// ── main ────────────────────────────────────────────────────────────────────
int main() {
    std::puts("=============================================================");
    std::puts(" vLLM ARM CPU Kernel Standalone Tests");
#if defined(VLLM_NEON)
    std::puts(" ISA: NEON (AArch64)");
#  ifdef ARM_BF16_SUPPORT
    std::puts(" ISA: +BF16 (BFMMLA)");
#  endif
#else
    std::puts(" ISA: Scalar fallback (host x86)");
#endif
    std::puts("=============================================================");

    test_rms_norm();
    test_fused_add_rms_norm();
    test_silu_and_mul();
    test_gelu_and_mul();
    test_paged_attention_v1();
    test_paged_attention_v2();

    bench_rms_norm();
    bench_silu_and_mul();
    bench_paged_attention_v1();

    std::puts("\n=============================================================");
    std::printf(" Results: %d passed, %d failed\n", g_pass, g_fail);
    std::puts("=============================================================");
    return g_fail > 0 ? 1 : 0;
}
