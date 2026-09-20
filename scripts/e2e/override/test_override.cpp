// test_override.cpp - end-to-end check of the armbench kernel-override hook.
//
//   test_override ref   <ref.bin>                 run the graph with the stock path, dump dst
//   test_override check <ref.bin> <test_kernel.so> run with ARMBENCH_OVERRIDES set, compare
//
// Graph: C1 = mul_mat(B1 q4_K [K=512,N=64], A1 f32 [K=512,M=3])   -> "llamacpp" ABI kernel
//        C2 = mul_mat(B2 q4_K [K=256,N=32], A2 f32 [K=256,M=3])   -> "entry"    ABI kernel
// A is drawn as k/127 with a +-1 in every 256-block so the stock path's Q8_K activation
// quantization is exact; the remaining stock-vs-override difference is bf16 rounding of A.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "armbench_override.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>

extern "C" void dequantize_row_q4_K(const void * x, float * y, int64_t k);

static const float ATOL = 1e-2f, RTOL = 1e-2f;

struct mm { int64_t K, N, M; std::vector<float> A, Bf, ref; std::vector<uint8_t> Bq; ggml_tensor *a, *b, *c; };

static uint32_t rng = 12345;
static float urand() { rng = rng * 1664525u + 1013904223u; return (rng >> 8) / 16777216.0f; }

static void fill(mm & x) {
    x.A.resize(x.K * x.M);
    for (int64_t m = 0; m < x.M; m++)
        for (int64_t k = 0; k < x.K; k++) {
            int v = (int) (urand() * 255.0f) - 127;                    // [-127,127]
            if (k % 256 == 0) v = (m % 2) ? 127 : -127;                 // Q8_K amax exact per block
            x.A[m * x.K + k] = v / 127.0f;
        }
    std::vector<float> Bsrc(x.K * x.N);
    for (auto & v : Bsrc) v = (urand() - 0.5f) * 0.2f;                 // +-0.1
    x.Bq.resize(ggml_row_size(GGML_TYPE_Q4_K, x.K) * x.N);
    size_t got = ggml_quantize_chunk(GGML_TYPE_Q4_K, Bsrc.data(), x.Bq.data(), 0, x.N, x.K, nullptr);
    if (got != x.Bq.size()) { fprintf(stderr, "quantize size mismatch\n"); exit(2); }
    x.Bf.resize(x.K * x.N);
    for (int64_t n = 0; n < x.N; n++) dequantize_row_q4_K(x.Bq.data() + n * ggml_row_size(GGML_TYPE_Q4_K, x.K), &x.Bf[n * x.K], x.K);
    x.ref.assign(x.M * x.N, 0.0f);
    for (int64_t m = 0; m < x.M; m++)
        for (int64_t n = 0; n < x.N; n++) {
            double acc = 0;
            for (int64_t k = 0; k < x.K; k++) acc += (double) x.A[m * x.K + k] * x.Bf[n * x.K + k];
            x.ref[m * x.N + n] = (float) acc;
        }
}

static bool compare(const char * what, const std::vector<float> & got, const std::vector<float> & want) {
    float max_abs = 0, max_rel = 0; size_t bad = 0;
    for (size_t i = 0; i < got.size(); i++) {
        float d = fabsf(got[i] - want[i]);
        float r = d / std::max(fabsf(want[i]), 1e-6f);
        max_abs = std::max(max_abs, d); max_rel = std::max(max_rel, r);
        if (d > ATOL + RTOL * fabsf(want[i])) bad++;
    }
    printf("  %-28s max_abs=%.3e max_rel=%.3e violations=%zu/%zu -> %s\n", what, max_abs, max_rel, bad, got.size(), bad ? "FAIL" : "ok");
    return bad == 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s ref|check <ref.bin> [test_kernel.so]\n", argv[0]); return 2; }
    const bool check = strcmp(argv[1], "check") == 0;
    const char * ref_path = argv[2];

    mm m1 { 512, 64, 3 }, m2 { 256, 32, 3 };
    fill(m1); fill(m2);

    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    for (mm * x : { &m1, &m2 }) {
        x->b = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, x->K, x->N);
        x->a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->K, x->M);
        x->c = ggml_mul_mat(ctx, x->b, x->a);   // [N, M]
    }
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 4);  // exercise the ith != 0 early-return path
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);   // plain CPU buffer: src0->extra == NULL
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 2; }
    for (mm * x : { &m1, &m2 }) {
        ggml_backend_tensor_set(x->b, x->Bq.data(), 0, x->Bq.size());
        ggml_backend_tensor_set(x->a, x->A.data(), 0, x->A.size() * sizeof(float));
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, m1.c);
    ggml_build_forward_expand(gf, m2.c);

    uint64_t hits_before = armbench_override_hit_count();
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "graph compute failed\n"); return 2; }
    uint64_t hits = armbench_override_hit_count() - hits_before;

    std::vector<float> out1(m1.M * m1.N), out2(m2.M * m2.N);
    ggml_backend_tensor_get(m1.c, out1.data(), 0, out1.size() * sizeof(float));
    ggml_backend_tensor_get(m2.c, out2.data(), 0, out2.size() * sizeof(float));

    bool ok = true;
    printf("[%s] kernels loaded=%d override hits=%llu\n", check ? "override" : "stock", armbench_override_num_kernels(), (unsigned long long) hits);
    ok &= compare("C1 vs fp32 reference", out1, m1.ref);
    ok &= compare("C2 vs fp32 reference", out2, m2.ref);

    if (!check) {
        if (hits != 0) { printf("  expected 0 override hits in stock run\n"); ok = false; }
        FILE * f = fopen(ref_path, "wb");
        fwrite(out1.data(), sizeof(float), out1.size(), f);
        fwrite(out2.data(), sizeof(float), out2.size(), f);
        fclose(f);
        printf("  wrote %s\n", ref_path);
    } else {
        std::vector<float> ref1(out1.size()), ref2(out2.size());
        FILE * f = fopen(ref_path, "rb");
        if (!f || fread(ref1.data(), sizeof(float), ref1.size(), f) != ref1.size() || fread(ref2.data(), sizeof(float), ref2.size(), f) != ref2.size()) {
            fprintf(stderr, "cannot read %s (run 'ref' first)\n", ref_path); return 2;
        }
        fclose(f);
        ok &= compare("C1 override vs stock", out1, ref1);
        ok &= compare("C2 override vs stock", out2, ref2);
        if (armbench_override_num_kernels() != 2) { printf("  expected 2 kernels loaded\n"); ok = false; }
        if (hits != 2) { printf("  expected 2 override hits, got %llu\n", (unsigned long long) hits); ok = false; }
        if (argc > 3) {   // read the test kernel's own call counters
            void * h = dlopen(argv[3], RTLD_NOW);
            int * c_ll = h ? (int *) dlsym(h, "test_kernel_calls_llamacpp") : nullptr;
            int * c_en = h ? (int *) dlsym(h, "test_kernel_calls_entry")    : nullptr;
            printf("  test kernel calls: llamacpp=%d entry=%d\n", c_ll ? *c_ll : -1, c_en ? *c_en : -1);
            if (!c_ll || !c_en || *c_ll != 1 || *c_en != 1) { printf("  expected exactly one call per ABI\n"); ok = false; }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
