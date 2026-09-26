// bench_repack.cpp -- single-threaded per-shape timing of one quantized GEMM shape:
//   (a) ggml mul_mat with the weight in the CPU_REPACK extra buffer type (what stock llama.cpp runs:
//       interleaved 8x8/8x4 i8mm/dotprod gemm+gemv, repack.cpp),
//   (b) ggml mul_mat with the weight in a plain CPU buffer (= llama.cpp --no-repack / build-norepack),
//   (c) an agent kernel .so via dlopen, called exactly like scripts/e2e/override does
//       (f32 activations -> bf16 via ggml_cpu_fp32_to_bf16, raw ggml block rows, row-major [M,N] out),
// on identical weights and activations. Target: llama.cpp v0.4.1 (b29c606). One JSON line per M on stdout.
//
// ggml mapping: src0 = W [K,N] (ne00=K, ne01=N), src1 = X f32 [K,M], dst = [N,M] == row-major C[M][N].
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

typedef int (*entry_fn)(const uint16_t * A_bf16, float * out, const uint8_t * B, int M);
typedef int (*llamacpp_fn)(const void * A, const void * B, float * C, int64_t M, int64_t N, int64_t K);

struct Args {
    std::string type = "q4_K";
    int64_t N = 0, K = 0;
    std::vector<int> Ms = {1, 2, 4, 8, 16, 32, 128, 512};
    std::string so, symbol = "armbench_entry_gemm", abi = "entry", shape;
    std::string weights, act;       // optional input files
    std::string act_dtype = "auto"; // for raw act files: f32 | bf16
    int reps = 20, warmup = 3, min_reps = 3;
    double max_sec = 4.0;           // time budget per M (all variants together, timed phase)
    int ref_rows = 64;              // weight rows sampled for the fp64 dequant reference
    int flush_mb = 0;               // >0: stream through this many MB between timed calls (cold-cache mode)
    bool no_norepack = false, no_repack = false;
    uint64_t seed = 1234;
    int gen_threads = 0;            // weight generation/quantization threads (untimed); 0 = hw concurrency
};

static void usage() {
    fprintf(stderr,
        "usage: bench_repack --type q4_K|q5_K|q6_K|q8_0 --N n --K k [--M 1,2,4,...]\n"
        "   [--so kernel.so --symbol armbench_entry_gemm --abi entry|entry_rows|llamacpp] [--shape name]\n"
        "   [--reps 20] [--warmup 3] [--min-reps 3] [--max-sec 4] [--flush-mb 0] [--ref-rows 64] [--seed 1234]\n"
        "   [--weights raw_block_rows.bin] [--act file.npy|file.bin --act-dtype auto|f32|bf16]\n"
        "   [--no-norepack] [--no-repack] [--gen-threads 0]\n");
    exit(2);
}

static std::vector<int> parse_ints(const std::string & s) {
    std::vector<int> v; size_t p = 0;
    while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
        if (q > p) v.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1; }
    return v;
}

static Args parse(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto nxt = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (k == "--type") a.type = nxt();
        else if (k == "--N") a.N = atoll(nxt().c_str());
        else if (k == "--K") a.K = atoll(nxt().c_str());
        else if (k == "--M") a.Ms = parse_ints(nxt());
        else if (k == "--so") a.so = nxt();
        else if (k == "--symbol") a.symbol = nxt();
        else if (k == "--abi") a.abi = nxt();
        else if (k == "--shape") a.shape = nxt();
        else if (k == "--weights") a.weights = nxt();
        else if (k == "--act") a.act = nxt();
        else if (k == "--act-dtype") a.act_dtype = nxt();
        else if (k == "--reps") a.reps = atoi(nxt().c_str());
        else if (k == "--warmup") a.warmup = atoi(nxt().c_str());
        else if (k == "--min-reps") a.min_reps = atoi(nxt().c_str());
        else if (k == "--max-sec") a.max_sec = atof(nxt().c_str());
        else if (k == "--ref-rows") a.ref_rows = atoi(nxt().c_str());
        else if (k == "--flush-mb") a.flush_mb = atoi(nxt().c_str());
        else if (k == "--seed") a.seed = strtoull(nxt().c_str(), nullptr, 10);
        else if (k == "--gen-threads") a.gen_threads = atoi(nxt().c_str());
        else if (k == "--no-norepack") a.no_norepack = true;
        else if (k == "--no-repack") a.no_repack = true;
        else usage();
    }
    if (a.N <= 0 || a.K <= 0 || a.Ms.empty()) usage();
    if (a.shape.empty()) a.shape = "gemm_ggml_" + a.type + "_n" + std::to_string(a.N) + "_k" + std::to_string(a.K);
    return a;
}

static ggml_type type_from_name(const std::string & s) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && s == n) return (ggml_type) t;
    }
    fprintf(stderr, "unknown ggml type %s\n", s.c_str()); exit(2);
}

// ggml logs "repack tensor <name> with q4_K_8x8" (GGML_LOG_DEBUG) when it picks an interleaved kernel;
// capture the kernel id for the JSON and still forward everything to stderr.
static std::string g_repack_kernel;
static void log_cb(enum ggml_log_level, const char * text, void *) {
    const char * p = strstr(text, "repack tensor");
    if (p && (p = strstr(p, " with "))) { g_repack_kernel = p + 6; while (!g_repack_kernel.empty() && isspace((unsigned char) g_repack_kernel.back())) g_repack_kernel.pop_back(); }
    fputs(text, stderr);
}

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static inline float bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }

// ---- activations: .npy (<f4 or <u2/bf16 as 2-byte) or raw (--act-dtype) -> f32 [rows][K]
static std::vector<float> load_act(const Args & a, int64_t K, int64_t & rows) {
    std::ifstream f(a.act, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", a.act.c_str()); exit(1); }
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t off = 0; std::string dt = a.act_dtype;
    if (buf.size() >= 10 && memcmp(buf.data(), "\x93NUMPY", 6) == 0) {
        int major = (unsigned char) buf[6];
        size_t hl = major == 1 ? (size_t)((unsigned char) buf[8] | ((unsigned char) buf[9] << 8))
                               : (size_t)((unsigned char) buf[8] | ((unsigned char) buf[9] << 8) | ((unsigned char) buf[10] << 16) | ((unsigned char) buf[11] << 24));
        size_t hs = major == 1 ? 10 : 12;
        std::string hdr(buf.data() + hs, hl);
        off = hs + hl;
        if (hdr.find("'fortran_order': True") != std::string::npos) { fprintf(stderr, "npy: fortran order unsupported\n"); exit(1); }
        if (dt == "auto") {
            if (hdr.find("<f4") != std::string::npos) dt = "f32";
            else if (hdr.find("<u2") != std::string::npos || hdr.find("<i2") != std::string::npos ||
                     hdr.find("bfloat16") != std::string::npos || hdr.find("V2") != std::string::npos) dt = "bf16";
            else { fprintf(stderr, "npy: unsupported dtype in header %s\n", hdr.c_str()); exit(1); }
        }
    }
    if (dt == "auto") dt = "f32";
    const size_t es = dt == "bf16" ? 2 : 4;
    const size_t n = (buf.size() - off) / es;
    if (n < (size_t) K || n % K) { fprintf(stderr, "act file: %zu elements not a multiple of K=%lld\n", n, (long long) K); exit(1); }
    rows = n / K;
    std::vector<float> out(n);
    if (es == 4) memcpy(out.data(), buf.data() + off, n * 4);
    else for (size_t i = 0; i < n; i++) { uint16_t h; memcpy(&h, buf.data() + off + 2 * i, 2); out[i] = bf16_to_f32(h); }
    return out;
}

// ---- weights: Gaussian f32 quantized with ggml's own quantizer, row chunks in parallel (untimed)
static void gen_weights(const Args & a, ggml_type t, uint8_t * dst) {
    const size_t rs = ggml_row_size(t, a.K);
    if (!a.weights.empty()) {
        FILE * f = fopen(a.weights.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", a.weights.c_str()); exit(1); }
        size_t got = fread(dst, 1, rs * a.N, f);
        fclose(f);
        if (got != rs * (size_t) a.N) { fprintf(stderr, "weights file: got %zu bytes, need %zu (N*row_size)\n", got, rs * a.N); exit(1); }
        return;
    }
    ggml_quantize_init(t);
    int nt = a.gen_threads > 0 ? a.gen_threads : (int) std::max(1u, std::thread::hardware_concurrency());
    const int64_t CH = 256;
    const int64_t nch = (a.N + CH - 1) / CH;
    std::vector<std::thread> th;
    for (int w = 0; w < nt; w++) th.emplace_back([&, w]() {
        std::vector<float> src(CH * a.K);
        for (int64_t c = w; c < nch; c += nt) {
            const int64_t r0 = c * CH, nr = std::min(CH, a.N - r0);
            std::mt19937_64 rng(a.seed * 1000003ULL + c);
            std::normal_distribution<float> nd(0.0f, 0.02f);
            for (int64_t i = 0; i < nr * a.K; i++) src[i] = nd(rng);
            ggml_quantize_chunk(t, src.data(), dst + r0 * rs, 0, nr, a.K, nullptr);
        }
    });
    for (auto & x : th) x.join();
}

struct Stat { double min_us = NAN, med_us = NAN; int n = 0; };
static Stat summarize(std::vector<double> v) {
    Stat s; if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n = (int) v.size(); s.min_us = v[0] * 1e6;
    s.med_us = (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2])) * 1e6;
    return s;
}

struct Cmp { double sqnr_db = NAN, max_abs = NAN; };
static Cmp compare(const float * ref, const float * x, size_t n) {
    double s = 0, e = 0, m = 0;
    for (size_t i = 0; i < n; i++) { double d = (double) ref[i] - x[i]; s += (double) ref[i] * ref[i]; e += d * d; m = std::max(m, std::fabs(d)); }
    Cmp c; c.max_abs = m; c.sqnr_db = e == 0 ? INFINITY : 10.0 * std::log10(s / e); return c;
}
// SQNR of x vs a dense fp64 reference on a subset of output columns
static Cmp compare_cols(const std::vector<double> & ref, const float * x, int64_t M, int64_t N, const std::vector<int64_t> & cols) {
    double s = 0, e = 0, m = 0;
    for (int64_t mi = 0; mi < M; mi++) for (size_t j = 0; j < cols.size(); j++) {
        double r = ref[mi * cols.size() + j], d = r - x[mi * N + cols[j]];
        s += r * r; e += d * d; m = std::max(m, std::fabs(d));
    }
    Cmp c; c.max_abs = m; c.sqnr_db = e == 0 ? INFINITY : 10.0 * std::log10(s / e); return c;
}

static std::string jnum(double v) {
    if (std::isnan(v)) return "null";
    if (std::isinf(v)) return v > 0 ? "1e999" : "-1e999";
    char b[64]; snprintf(b, sizeof b, "%.6g", v); return b;
}
static std::string jstr(const std::string & s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o + "\"";
}

int main(int argc, char ** argv) {
    Args a = parse(argc, argv);
    const ggml_type t = type_from_name(a.type);
    if (a.K % ggml_blck_size(t)) { fprintf(stderr, "K=%lld not a multiple of block size %lld\n", (long long) a.K, (long long) ggml_blck_size(t)); return 2; }
    const size_t rs = ggml_row_size(t, a.K);
    const int Mmax = *std::max_element(a.Ms.begin(), a.Ms.end());

    // the e2e override hook (if compiled into this libggml-cpu) must not intercept the plain-buffer baseline
    if (getenv("ARMBENCH_OVERRIDES")) { fprintf(stderr, "[bench] unsetting ARMBENCH_OVERRIDES\n"); unsetenv("ARMBENCH_OVERRIDES"); }
    ggml_log_set(log_cb, nullptr);
    ggml_cpu_init();
    char cpu[256];
    snprintf(cpu, sizeof cpu, "neon=%d dotprod=%d i8mm=%d sve=%d sve_bytes=%d sme=%d",
             ggml_cpu_has_neon(), ggml_cpu_has_dotprod(), ggml_cpu_has_matmul_int8(), ggml_cpu_has_sve(),
             ggml_cpu_has_sve() ? ggml_cpu_get_sve_cnt() : 0, ggml_cpu_has_sme());
    fprintf(stderr, "[bench] %s  type=%s N=%lld K=%lld row_size=%zu weights=%.1f MB  cpu: %s\n", a.shape.c_str(), a.type.c_str(),
            (long long) a.N, (long long) a.K, rs, rs * a.N / 1e6, cpu);

    // ---- CPU backend, 1 thread
    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) { fprintf(stderr, "ggml_backend_cpu_init failed\n"); return 1; }
    ggml_backend_cpu_set_n_threads(be, 1);

    // ---- find the CPU_REPACK extra buffer type through the public registry (what llama.cpp does)
    ggml_backend_buffer_type_t repack_buft = nullptr;
    {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        auto get_extra = reg ? (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts") : nullptr;
        if (get_extra) {
            for (ggml_backend_buffer_type_t * p = get_extra(dev); p && *p; ++p) {
                fprintf(stderr, "[bench] extra buffer type: %s\n", ggml_backend_buft_name(*p));
                if (strcmp(ggml_backend_buft_name(*p), "CPU_REPACK") == 0) repack_buft = *p;
            }
        }
        if (!repack_buft) fprintf(stderr, "[bench] WARNING: no CPU_REPACK buffer type (built with GGML_CPU_REPACK=OFF?)\n");
    }

    // ---- plain weight tensor (also the raw-block source for the agent kernel and for the repack upload)
    ggml_init_params ipw = { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_wp = ggml_init(ipw);
    ggml_tensor * Wp = ggml_new_tensor_2d(ctx_wp, t, a.K, a.N);
    ggml_backend_buffer_t buf_wp = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wp, ggml_backend_cpu_buffer_type());
    if (!buf_wp) { fprintf(stderr, "alloc plain weights failed\n"); return 1; }
    double tg = now_s();
    gen_weights(a, t, (uint8_t *) Wp->data);   // plain CPU buffer is host memory: write in place
    fprintf(stderr, "[bench] weights ready in %.1fs (%s)\n", now_s() - tg, a.weights.empty() ? "gaussian, ggml_quantize_chunk" : a.weights.c_str());

    // ---- repacked weight tensor
    ggml_context * ctx_wr = nullptr; ggml_tensor * Wr = nullptr; ggml_backend_buffer_t buf_wr = nullptr;
    bool repack_used = false; std::string repack_note = "no CPU_REPACK buft";
    if (repack_buft && !a.no_repack) {
        ctx_wr = ggml_init(ipw);
        Wr = ggml_new_tensor_2d(ctx_wr, t, a.K, a.N);
        buf_wr = ggml_backend_alloc_ctx_tensors_from_buft(ctx_wr, repack_buft);   // runs repack init_tensor -> sets Wr->extra
        if (!buf_wr) { fprintf(stderr, "alloc repack weights failed\n"); return 1; }
        if (Wr->extra == nullptr) {
            // ggml_repack_get_optimal_repack_type() returned nullptr for this type/N/CPU: set_tensor would
            // assert, and at compute time the op would not be claimed. Report instead of timing the plain path.
            repack_note = "repack buft has no kernel for this type/N on this CPU (tensor->extra == NULL)";
        } else {
            double tr = now_s();
            ggml_backend_tensor_set(Wr, Wp->data, 0, ggml_nbytes(Wr));   // repack happens here
            repack_used = ggml_backend_buffer_get_type(Wr->buffer) == repack_buft && Wr->extra != nullptr;
            repack_note = "ok";
            fprintf(stderr, "[bench] repacked into %s in %.1fs\n", ggml_backend_buft_name(ggml_backend_buffer_get_type(Wr->buffer)), now_s() - tr);
        }
    } else if (a.no_repack) repack_note = "--no-repack";

    // ---- agent kernel
    void * so_h = nullptr; entry_fn fe = nullptr; llamacpp_fn fl = nullptr;
    if (!a.so.empty()) {
        so_h = dlopen(a.so.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!so_h) { fprintf(stderr, "dlopen %s: %s\n", a.so.c_str(), dlerror()); return 1; }
        void * sym = dlsym(so_h, a.symbol.c_str());
        if (!sym) { fprintf(stderr, "dlsym %s: %s\n", a.symbol.c_str(), dlerror()); return 1; }
        if (a.abi == "entry" || a.abi == "entry_rows") fe = (entry_fn) sym;   // rows ABI's full-N entry == entry
        else if (a.abi == "llamacpp") fl = (llamacpp_fn) sym;
        else { fprintf(stderr, "unknown abi %s\n", a.abi.c_str()); return 2; }
    }

    // ---- activation pool [Mmax][K]
    std::vector<float> act;
    int64_t act_rows = 0;
    if (!a.act.empty()) act = load_act(a, a.K, act_rows);
    else {
        act_rows = Mmax; act.resize((size_t) Mmax * a.K);
        std::mt19937_64 rng(a.seed ^ 0x9e3779b97f4a7c15ULL); std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto & v : act) v = nd(rng);
    }

    // ---- fp64 reference columns (dequantized weights), evenly spaced
    std::vector<int64_t> ref_cols;
    std::vector<std::vector<float>> ref_w;
    if (a.ref_rows > 0) {
        int nr = (int) std::min<int64_t>(a.ref_rows, a.N);
        auto to_f = ggml_get_type_traits(t)->to_float;
        for (int j = 0; j < nr; j++) {
            int64_t c = (int64_t) ((double) j * a.N / nr);
            ref_cols.push_back(c);
            std::vector<float> w(a.K);
            to_f((const uint8_t *) Wp->data + c * rs, w.data(), a.K);
            ref_w.push_back(std::move(w));
        }
    }

    std::vector<uint8_t> flush;
    if (a.flush_mb > 0) flush.assign((size_t) a.flush_mb << 20, 1);
    volatile uint64_t flush_sink = 0;
    auto do_flush = [&]() {
        if (flush.empty()) return;
        uint64_t s = 0;
        for (size_t i = 0; i < flush.size(); i += 64) { flush[i]++; s += flush[i]; }
        flush_sink = flush_sink + s;
    };

    for (int M : a.Ms) {
        // graphs for this M
        ggml_init_params ipg = { 16 * ggml_tensor_overhead() + 3 * ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx_g = ggml_init(ipg);
        ggml_tensor * X = ggml_new_tensor_2d(ctx_g, GGML_TYPE_F32, a.K, M);
        ggml_tensor * Yr = Wr && repack_used ? ggml_mul_mat(ctx_g, Wr, X) : nullptr;
        ggml_tensor * Yp = a.no_norepack ? nullptr : ggml_mul_mat(ctx_g, Wp, X);
        ggml_cgraph * gr = nullptr, * gp = nullptr;
        if (Yr) { gr = ggml_new_graph(ctx_g); ggml_build_forward_expand(gr, Yr); }
        if (Yp) { gp = ggml_new_graph(ctx_g); ggml_build_forward_expand(gp, Yp); }
        ggml_backend_buffer_t buf_g = ggml_backend_alloc_ctx_tensors(ctx_g, be);
        if (!buf_g) { fprintf(stderr, "alloc graph tensors failed (M=%d)\n", M); return 1; }

        std::vector<float> xh((size_t) M * a.K);
        for (int m = 0; m < M; m++) memcpy(&xh[(size_t) m * a.K], &act[(size_t) (m % act_rows) * a.K], a.K * sizeof(float));
        ggml_backend_tensor_set(X, xh.data(), 0, xh.size() * sizeof(float));

        std::vector<uint16_t> A_bf16((size_t) M * a.K);
        std::vector<float> Ya((size_t) M * a.N, 0.0f);
        int agent_rc = 0;
        auto run_agent = [&]() {
            // same work the override hook does per call: f32 -> bf16 (RNE) then the kernel
            ggml_cpu_fp32_to_bf16((const float *) X->data, (ggml_bf16_t *) A_bf16.data(), (int64_t) M * a.K);
            int rc = fe ? fe(A_bf16.data(), Ya.data(), (const uint8_t *) Wp->data, M)
                        : fl(A_bf16.data(), Wp->data, Ya.data(), M, a.N, a.K);
            if (rc) agent_rc = rc;
        };
        auto run_convert = [&]() { ggml_cpu_fp32_to_bf16((const float *) X->data, (ggml_bf16_t *) A_bf16.data(), (int64_t) M * a.K); };
        auto run_r = [&]() { if (ggml_backend_graph_compute(be, gr) != GGML_STATUS_SUCCESS) { fprintf(stderr, "repack compute failed\n"); exit(1); } };
        auto run_p = [&]() { if (ggml_backend_graph_compute(be, gp) != GGML_STATUS_SUCCESS) { fprintf(stderr, "plain compute failed\n"); exit(1); } };

        struct V { const char * name; std::function<void()> fn; std::vector<double> ts; };
        std::vector<V> vs;
        if (gr) vs.push_back({"repack", run_r, {}});
        if (gp) vs.push_back({"norepack", run_p, {}});
        if (fe || fl) { vs.push_back({"agent", run_agent, {}}); vs.push_back({"convert", run_convert, {}}); }

        for (auto & v : vs) for (int w = 0; w < a.warmup; w++) v.fn();   // also produces the outputs compared below
        if (agent_rc) { fprintf(stderr, "agent kernel returned %d (M=%d)\n", agent_rc, M); }

        // interleaved timed rounds (repack, norepack, agent, convert, repeat) so drift hits all equally
        const double t0 = now_s();
        for (int r = 0; r < a.reps; r++) {
            for (auto & v : vs) {
                do_flush();
                const double s = now_s(); v.fn(); v.ts.push_back(now_s() - s);
            }
            if (r + 1 >= a.min_reps && now_s() - t0 > a.max_sec) break;
        }

        Stat sr, sp, sa, sc;
        for (auto & v : vs) {
            Stat s = summarize(v.ts);
            if (!strcmp(v.name, "repack")) sr = s; else if (!strcmp(v.name, "norepack")) sp = s;
            else if (!strcmp(v.name, "agent")) sa = s; else sc = s;
        }

        // correctness
        const size_t n = (size_t) M * a.N;
        std::vector<float> yr, yp;
        if (Yr) { yr.resize(n); ggml_backend_tensor_get(Yr, yr.data(), 0, n * 4); }
        if (Yp) { yp.resize(n); ggml_backend_tensor_get(Yp, yp.data(), 0, n * 4); }
        const float * base = Yr ? yr.data() : (Yp ? yp.data() : nullptr);   // comparison reference: repack if available
        Cmp c_a, c_p;
        if (base && (fe || fl)) c_a = compare(base, Ya.data(), n);
        if (Yr && Yp) c_p = compare(yr.data(), yp.data(), n);
        Cmp rr, rp, ra;
        if (!ref_cols.empty()) {
            std::vector<double> ref((size_t) M * ref_cols.size());
            for (int m = 0; m < M; m++) for (size_t j = 0; j < ref_cols.size(); j++) {
                double s = 0; const float * x = &xh[(size_t) m * a.K]; const float * w = ref_w[j].data();
                for (int64_t k = 0; k < a.K; k++) s += (double) x[k] * w[k];
                ref[(size_t) m * ref_cols.size() + j] = s;
            }
            if (Yr) rr = compare_cols(ref, yr.data(), M, a.N, ref_cols);
            if (Yp) rp = compare_cols(ref, yp.data(), M, a.N, ref_cols);
            if (fe || fl) ra = compare_cols(ref, Ya.data(), M, a.N, ref_cols);
        }

        std::string o = "{";
        o += "\"shape\":" + jstr(a.shape) + ",\"type\":" + jstr(a.type) + ",\"N\":" + std::to_string(a.N) + ",\"K\":" + std::to_string(a.K);
        o += ",\"M\":" + std::to_string(M);
        o += ",\"repack_used\":" + std::string(repack_used ? "true" : "false") + ",\"repack_note\":" + jstr(repack_note)
           + ",\"repack_kernel\":" + (g_repack_kernel.empty() ? std::string("null") : jstr(g_repack_kernel));
        o += ",\"t_repack_min_us\":" + jnum(sr.min_us) + ",\"t_repack_med_us\":" + jnum(sr.med_us);
        o += ",\"t_norepack_min_us\":" + jnum(sp.min_us) + ",\"t_norepack_med_us\":" + jnum(sp.med_us);
        o += ",\"t_agent_min_us\":" + jnum(sa.min_us) + ",\"t_agent_med_us\":" + jnum(sa.med_us);
        o += ",\"t_bf16_convert_min_us\":" + jnum(sc.min_us);
        o += ",\"agent_vs_repack\":" + jnum(sr.min_us / sa.min_us);        // >1: agent faster than repacked ggml
        o += ",\"agent_vs_norepack\":" + jnum(sp.min_us / sa.min_us);      // >1: agent faster than plain ggml
        o += ",\"repack_vs_norepack\":" + jnum(sp.min_us / sr.min_us);     // >1: repack faster than plain
        o += ",\"sqnr_db\":" + jnum(c_a.sqnr_db) + ",\"max_abs_agent\":" + jnum(c_a.max_abs);
        o += ",\"sqnr_ref_is\":" + jstr(Yr ? "repack" : "norepack");
        o += ",\"sqnr_norepack_vs_repack_db\":" + jnum(c_p.sqnr_db);
        o += ",\"sqnr_fp64ref_repack_db\":" + jnum(rr.sqnr_db) + ",\"sqnr_fp64ref_norepack_db\":" + jnum(rp.sqnr_db) + ",\"sqnr_fp64ref_agent_db\":" + jnum(ra.sqnr_db);
        o += ",\"agent_rc\":" + std::to_string(agent_rc);
        o += ",\"reps\":" + std::to_string(sa.n ? sa.n : (sr.n ? sr.n : sp.n)) + ",\"flush_mb\":" + std::to_string(a.flush_mb);
        o += ",\"threads\":1,\"cpu\":" + jstr(cpu) + ",\"so\":" + jstr(a.so) + ",\"abi\":" + jstr(a.so.empty() ? "" : a.abi);
        o += "}";
        printf("%s\n", o.c_str()); fflush(stdout);
        fprintf(stderr, "[bench] M=%-4d repack %s us  norepack %s us  agent %s us  agent/repack %sx  sqnr %s dB\n", M,
                jnum(sr.min_us).c_str(), jnum(sp.min_us).c_str(), jnum(sa.min_us).c_str(), jnum(sr.min_us / sa.min_us).c_str(), jnum(c_a.sqnr_db).c_str());

        ggml_backend_buffer_free(buf_g);
        ggml_free(ctx_g);
    }

    if (buf_wr) ggml_backend_buffer_free(buf_wr);
    if (ctx_wr) ggml_free(ctx_wr);
    ggml_backend_buffer_free(buf_wp);
    ggml_free(ctx_wp);
    ggml_backend_free(be);
    if (so_h) dlclose(so_h);
    return 0;
}
