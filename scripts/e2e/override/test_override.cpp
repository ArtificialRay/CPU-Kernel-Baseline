// test_override.cpp - end-to-end check of the armbench kernel-override hook.
//
//   test_override ref   <ref.bin>                              stock path, dump dst
//   test_override check <ref.bin> <test_kernel.so> <rows_kernel.so> [expect_threads]
//                                                              with ARMBENCH_OVERRIDES set, compare
//
// Graph: C1 = mul_mat(B1 q4_K [K=512,N=64],  A1 f32 [K=512,M=3])  -> "llamacpp"   ABI kernel
//        C2 = mul_mat(B2 q4_K [K=256,N=32],  A2 f32 [K=256,M=3])  -> "entry"      ABI kernel
//        C3 = mul_mat(B3 q4_K [K=256,N=512], A3 f32 [K=256,M=1])  -> "entry_rows" N-split (4 x 128 rows)
//        C4 = mul_mat(B4 q4_K [K=256,N=512], A4 f32 [K=256,M=6])  -> "entry_rows" M-split (2 rows/thread)
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

    mm m1 { 512, 64, 3 }, m2 { 256, 32, 3 }, m3 { 256, 512, 1 }, m4 { 256, 512, 6 };
    mm * all[] = { &m1, &m2, &m3, &m4 };
    for (mm * x : all) fill(*x);

    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    for (mm * x : all) {
        x->b = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, x->K, x->N);
        x->a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->K, x->M);
        x->c = ggml_mul_mat(ctx, x->b, x->a);   // [N, M]
    }
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 4);  // exercise the ith != 0 early-return path
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);   // plain CPU buffer: src0->extra == NULL
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 2; }
    for (mm * x : all) {
        ggml_backend_tensor_set(x->b, x->Bq.data(), 0, x->Bq.size());
        ggml_backend_tensor_set(x->a, x->A.data(), 0, x->A.size() * sizeof(float));
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (mm * x : all) ggml_build_forward_expand(gf, x->c);

    uint64_t hits_before = armbench_override_hit_count();
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "graph compute failed\n"); return 2; }
    uint64_t hits = armbench_override_hit_count() - hits_before;

    std::vector<std::vector<float>> out(4);
    for (int i = 0; i < 4; i++) {
        out[i].resize(all[i]->M * all[i]->N);
        ggml_backend_tensor_get(all[i]->c, out[i].data(), 0, out[i].size() * sizeof(float));
    }

    bool ok = true;
    printf("[%s] kernels loaded=%d override hits=%llu\n", check ? "override" : "stock", armbench_override_num_kernels(), (unsigned long long) hits);
    const char * names[] = { "C1 (llamacpp, M=3)", "C2 (entry, M=3)", "C3 (entry_rows, M=1)", "C4 (entry_rows, M=6)" };
    for (int i = 0; i < 4; i++) ok &= compare((std::string(names[i]) + " vs fp32 ref").c_str(), out[i], all[i]->ref);

    if (!check) {
        if (hits != 0) { printf("  expected 0 override hits in stock run\n"); ok = false; }
        FILE * f = fopen(ref_path, "wb");
        for (auto & o : out) fwrite(o.data(), sizeof(float), o.size(), f);
        fclose(f);
        printf("  wrote %s\n", ref_path);
    } else {
        FILE * f = fopen(ref_path, "rb");
        if (!f) { fprintf(stderr, "cannot read %s (run 'ref' first)\n", ref_path); return 2; }
        for (int i = 0; i < 4; i++) {
            std::vector<float> ref(out[i].size());
            if (fread(ref.data(), sizeof(float), ref.size(), f) != ref.size()) { fprintf(stderr, "short read %s\n", ref_path); return 2; }
            ok &= compare((std::string(names[i]) + " vs stock").c_str(), out[i], ref);
        }
        fclose(f);
        if (armbench_override_num_kernels() != 3) { printf("  expected 3 kernels loaded\n"); ok = false; }
        if (hits != 4) { printf("  expected 4 override hits, got %llu\n", (unsigned long long) hits); ok = false; }
        auto counter = [](void * h, const char * sym) { int * p = h ? (int *) dlsym(h, sym) : nullptr; return p ? *p : -1; };
        if (argc > 3) {   // "llamacpp"/"entry" test kernel: exactly one call per ABI, thread 0 only
            void * h = dlopen(argv[3], RTLD_NOW);
            int c_ll = counter(h, "test_kernel_calls_llamacpp"), c_en = counter(h, "test_kernel_calls_entry"), th = counter(h, "test_kernel_distinct_threads");
            printf("  single-thread kernel: calls llamacpp=%d entry=%d distinct_threads=%d\n", c_ll, c_en, th);
            if (c_ll != 1 || c_en != 1 || th != 1) { printf("  expected one call per ABI on one thread\n"); ok = false; }
        }
        if (argc > 4) {   // "entry_rows" test kernel: N-split (M=1) + M-split (M=6)
            int expect_threads = argc > 5 ? atoi(argv[5]) : 4;
            void * h = dlopen(argv[4], RTLD_NOW);
            int c_rows = counter(h, "test_kernel_calls_rows"), c_en = counter(h, "test_kernel_calls_entry"), th = counter(h, "test_kernel_distinct_threads");
            printf("  entry_rows kernel: calls rows=%d entry=%d distinct_threads=%d (expect threads=%d)\n", c_rows, c_en, th, expect_threads);
            if (expect_threads > 1) {
                // nth=4: M=1,N=512 -> chunk 128 -> 4 rows calls; M=6 -> 2 rows/thread -> 3 entry calls
                if (c_rows != 4 || c_en != 3 || th < 2) { printf("  expected rows=4 entry=3 distinct_threads>=2\n"); ok = false; }
            } else {
                if (c_rows != 0 || c_en != 2 || th != 1) { printf("  expected rows=0 entry=2 distinct_threads=1 (forced single)\n"); ok = false; }
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
