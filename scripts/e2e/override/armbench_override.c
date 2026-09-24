// armbench_override.c - see armbench_override.h for the contract.
//
// Compiled into ggml-cpu (added to ggml/src/ggml-cpu/CMakeLists.txt by the patch).
// No third-party deps: the manifest is parsed by a tiny hand-written JSON scanner
// that understands exactly the shape in manifest.example.json (a flat array of
// flat objects with string/number values; unknown keys and nested values are skipped).

#include "armbench_override.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"   // struct ggml_compute_params

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <inttypes.h>
#include <limits.h>

#if defined(_WIN32)
// dlopen is POSIX-only; the hook compiles but is permanently disabled on Windows.
#define ARMBENCH_HAVE_DLOPEN 0
#else
#define ARMBENCH_HAVE_DLOPEN 1
#include <dlfcn.h>
#include <pthread.h>
#endif

// ---------------------------------------------------------------------------
// kernel table
// ---------------------------------------------------------------------------

// Standalone kernel ABIs (must match the project's llama.cpp-target kernels).
//   A: row-major bf16 [M,K] (raw uint16 bits)     B: raw ggml quantized rows (src0->data)
//   C: row-major f32 [M,N]                         returns 0 on success
typedef int (*armbench_gemm_fn)(const void * A, const void * B, float * C, int64_t M, int64_t N, int64_t K);
// "entry" ABI: reference-scalar harness entry of agent-submitted kernels; N and K are
// baked in as constexpr on the kernel side (must equal the manifest's N/K), B_blocks is
// the raw ggml block rows (src0->data as-is), output is row-major f32 [M,N].
typedef int (*armbench_entry_gemm_fn)(const uint16_t * A_bf16, float * output, const uint8_t * B_blocks, int M);
// "entry_rows" ABI: the .so exports BOTH armbench_entry_gemm (N baked = full N) and this
// row-range variant: computes only weight rows [0, n_rows) relative to the given B pointer
// and writes output[m*n_rows + n] (output row stride == n_rows, NOT the full N).
typedef int (*armbench_entry_gemm_rows_fn)(const uint16_t * A_bf16, float * output, const uint8_t * B_blocks, int M, int n_rows);
//   x: [M,D] input, weight: [D], out: [M,D]        (unused: see rms_norm stub)
typedef int (*armbench_rms_norm_fn)(const void * x, const void * weight, float * out, int64_t M, int64_t D, float eps);

enum armbench_op {
    ARMBENCH_OP_MUL_MAT,
    ARMBENCH_OP_RMS_NORM,
    ARMBENCH_OP_COUNT,
};

enum armbench_abi {
    ARMBENCH_ABI_LLAMACPP,  // "llamacpp" (default): armbench_llamacpp_gemm(A, B, C, M, N, K)
    ARMBENCH_ABI_ENTRY,     // "entry":              armbench_entry_gemm(A_bf16, output, B_blocks, M)
    ARMBENCH_ABI_ENTRY_ROWS,// "entry_rows":         "entry" + armbench_entry_gemm_rows(..., M, n_rows); multithreaded dispatch
    ARMBENCH_ABI_COUNT,
};

static const char * const armbench_abi_names[ARMBENCH_ABI_COUNT] = {
    [ARMBENCH_ABI_LLAMACPP]   = "llamacpp",
    [ARMBENCH_ABI_ENTRY]      = "entry",
    [ARMBENCH_ABI_ENTRY_ROWS] = "entry_rows",
};

#define ARMBENCH_ROWS_SYMBOL_DEFAULT "armbench_entry_gemm_rows"
#define ARMBENCH_N_CHUNK 64   // decode N-split granularity (rows)

// op name table: adding an op = add an enum value + a row here + a hook function.
static const char * const armbench_op_names[ARMBENCH_OP_COUNT] = {
    [ARMBENCH_OP_MUL_MAT]  = "mul_mat",
    [ARMBENCH_OP_RMS_NORM] = "rms_norm",
};

struct armbench_kernel {
    enum armbench_op  op;
    enum armbench_abi abi;
    enum ggml_type   type;     // src0->type (mul_mat); unused for rms_norm
    int64_t          K;        // mul_mat: ne00 ; rms_norm: ne00 (D)
    int64_t          N;        // mul_mat: ne01
    int              threads;  // reserved, parsed but unused
    char *           so;
    char *           symbol;
    char *           symbol_rows;  // entry_rows only ("symbol_rows", default armbench_entry_gemm_rows)
    void *           fn;
    void *           fn_rows;      // entry_rows only
    uint64_t         hits;
    int              dumped;       // ARMBENCH_DUMP_DIR: calls written so far
    int              seen;         // ARMBENCH_DUMP_DIR: calls observed (for striding)
};

static struct armbench_kernel * g_kernels   = NULL;
static int                      g_nkernels  = 0;
static bool                     g_enabled   = false;  // fast path: false unless a manifest loaded
static bool                     g_log       = false;
static bool                     g_force_single = false;  // ARMBENCH_OVERRIDE_THREADS=1
static const char *             g_dump_dir     = NULL;   // ARMBENCH_DUMP_DIR
static int                      g_dump_calls   = 4;      // ARMBENCH_DUMP_CALLS
static bool                     g_dump_weights = false;  // ARMBENCH_DUMP_WEIGHTS
static int                      g_dump_stride  = 8;      // ARMBENCH_DUMP_STRIDE (layer spread)
static uint64_t                 g_total_hits = 0;

// ---------------------------------------------------------------------------
// minimal JSON scanner
// ---------------------------------------------------------------------------

struct js { const char * p; const char * end; };

static void js_ws(struct js * s) { while (s->p < s->end && isspace((unsigned char) *s->p)) s->p++; }

static bool js_expect(struct js * s, char c) {
    js_ws(s);
    if (s->p < s->end && *s->p == c) { s->p++; return true; }
    return false;
}

// parses "..." (handles \" \\ \/ \n \t; no \u), returns malloc'd string or NULL
static char * js_string(struct js * s) {
    js_ws(s);
    if (s->p >= s->end || *s->p != '"') return NULL;
    s->p++;
    size_t cap = 64, n = 0;
    char * out = malloc(cap);
    if (!out) return NULL;
    while (s->p < s->end && *s->p != '"') {
        char c = *s->p++;
        if (c == '\\' && s->p < s->end) {
            char e = *s->p++;
            switch (e) {
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                default:  c = e;    break;  // \" \\ \/
            }
        }
        if (n + 1 >= cap) { cap *= 2; char * t = realloc(out, cap); if (!t) { free(out); return NULL; } out = t; }
        out[n++] = c;
    }
    if (s->p >= s->end) { free(out); return NULL; }
    s->p++;  // closing quote
    out[n] = 0;
    return out;
}

// skips any JSON value (string, number, literal, nested object/array)
static bool js_skip_value(struct js * s) {
    js_ws(s);
    if (s->p >= s->end) return false;
    if (*s->p == '"') { char * t = js_string(s); if (!t) return false; free(t); return true; }
    if (*s->p == '{' || *s->p == '[') {
        int depth = 0;
        while (s->p < s->end) {
            char c = *s->p;
            if (c == '"') { char * t = js_string(s); if (!t) return false; free(t); continue; }
            if (c == '{' || c == '[') depth++;
            if (c == '}' || c == ']') { depth--; if (depth == 0) { s->p++; return true; } }
            s->p++;
        }
        return false;
    }
    while (s->p < s->end && !strchr(",}] \t\r\n", *s->p)) s->p++;  // number / true / false / null
    return true;
}

static bool js_number(struct js * s, int64_t * out) {
    js_ws(s);
    char * endp = NULL;
    long long v = strtoll(s->p, &endp, 10);
    if (endp == s->p) return false;
    s->p = endp;
    *out = (int64_t) v;
    return true;
}

// ---------------------------------------------------------------------------
// manifest loading
// ---------------------------------------------------------------------------

static char * read_file(const char * path, size_t * len) {
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char * buf = malloc((size_t) n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t) n, f);
    fclose(f);
    buf[got] = 0;
    *len = got;
    return buf;
}

static bool lookup_type(const char * name, enum ggml_type * out) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * tn = ggml_type_name((enum ggml_type) t);
        if (tn && strcmp(tn, name) == 0) { *out = (enum ggml_type) t; return true; }
    }
    return false;
}

static bool lookup_abi(const char * name, enum armbench_abi * out) {
    for (int i = 0; i < ARMBENCH_ABI_COUNT; i++) {
        if (strcmp(armbench_abi_names[i], name) == 0) { *out = (enum armbench_abi) i; return true; }
    }
    return false;
}

static bool lookup_op(const char * name, enum armbench_op * out) {
    for (int i = 0; i < ARMBENCH_OP_COUNT; i++) {
        if (strcmp(armbench_op_names[i], name) == 0) { *out = (enum armbench_op) i; return true; }
    }
    return false;
}

// parse one {...} kernel object; returns false on hard parse error
static bool parse_kernel(struct js * s, const char * manifest_path) {
    if (!js_expect(s, '{')) return false;
    char * op = NULL, * type = NULL, * so = NULL, * symbol = NULL, * abi = NULL, * symbol_rows = NULL;
    int64_t K = -1, N = -1, threads = 1;
    bool ok = true;
    js_ws(s);
    if (s->p < s->end && *s->p == '}') { s->p++; return true; }  // empty object: ignore
    for (;;) {
        char * key = js_string(s);
        if (!key || !js_expect(s, ':')) { free(key); ok = false; break; }
        if      (strcmp(key, "op")      == 0) { free(op);     op     = js_string(s); ok = op     != NULL; }
        else if (strcmp(key, "type")    == 0) { free(type);   type   = js_string(s); ok = type   != NULL; }
        else if (strcmp(key, "so")      == 0) { free(so);     so     = js_string(s); ok = so     != NULL; }
        else if (strcmp(key, "symbol")  == 0) { free(symbol); symbol = js_string(s); ok = symbol != NULL; }
        else if (strcmp(key, "abi")     == 0) { free(abi);    abi    = js_string(s); ok = abi    != NULL; }
        else if (strcmp(key, "symbol_rows") == 0) { free(symbol_rows); symbol_rows = js_string(s); ok = symbol_rows != NULL; }
        else if (strcmp(key, "K")       == 0) { ok = js_number(s, &K); }
        else if (strcmp(key, "N")       == 0) { ok = js_number(s, &N); }
        else if (strcmp(key, "threads") == 0) { ok = js_number(s, &threads); }
        else                                  { ok = js_skip_value(s); }
        free(key);
        if (!ok) break;
        js_ws(s);
        if (s->p < s->end && *s->p == ',') { s->p++; continue; }
        if (s->p < s->end && *s->p == '}') { s->p++; break; }
        ok = false; break;
    }

    if (ok) {
        struct armbench_kernel k;
        memset(&k, 0, sizeof(k));
        const char * err = NULL;
        k.abi = ARMBENCH_ABI_LLAMACPP;
        if (!op || !lookup_op(op, &k.op))            err = "missing/unknown \"op\"";
        else if (!so || !symbol)                     err = "missing \"so\" or \"symbol\"";
        else if (abi && !lookup_abi(abi, &k.abi))    err = "unknown \"abi\" (use \"llamacpp\" or \"entry\")";
        else if (k.op == ARMBENCH_OP_MUL_MAT) {
            if (!type || !lookup_type(type, &k.type)) err = "missing/unknown \"type\" (use a ggml type name, e.g. q4_K)";
            else if (K <= 0 || N <= 0)                err = "\"K\" and \"N\" must be > 0";
        } else if (k.op == ARMBENCH_OP_RMS_NORM) {
            if (K <= 0)                               err = "\"K\" (=D) must be > 0";
        }
        if (err) {
            fprintf(stderr, "[armbench-override] %s: skipping entry (%s)\n", manifest_path, err);
        } else {
            k.K = K; k.N = N; k.threads = (int) threads;
            k.so = so; k.symbol = symbol; so = symbol = NULL;
            if (k.abi == ARMBENCH_ABI_ENTRY_ROWS) {
                k.symbol_rows = symbol_rows ? symbol_rows : strdup(ARMBENCH_ROWS_SYMBOL_DEFAULT);
                symbol_rows = NULL;
            }
            struct armbench_kernel * t = realloc(g_kernels, (size_t) (g_nkernels + 1) * sizeof(*t));
            if (t) { g_kernels = t; g_kernels[g_nkernels++] = k; }
            else   { free(k.so); free(k.symbol); ok = false; }
        }
    }
    free(op); free(type); free(so); free(symbol); free(abi); free(symbol_rows);
    return ok;
}

static bool parse_manifest(const char * buf, size_t len, const char * manifest_path) {
    struct js s = { buf, buf + len };
    if (!js_expect(&s, '{')) return false;
    for (;;) {
        js_ws(&s);
        if (s.p < s.end && *s.p == '}') return true;
        char * key = js_string(&s);
        if (!key || !js_expect(&s, ':')) { free(key); return false; }
        bool is_kernels = strcmp(key, "kernels") == 0;
        free(key);
        if (is_kernels) {
            if (!js_expect(&s, '[')) return false;
            js_ws(&s);
            if (s.p < s.end && *s.p == ']') { s.p++; }
            else for (;;) {
                if (!parse_kernel(&s, manifest_path)) return false;
                js_ws(&s);
                if (s.p < s.end && *s.p == ',') { s.p++; continue; }
                if (s.p < s.end && *s.p == ']') { s.p++; break; }
                return false;
            }
        } else if (!js_skip_value(&s)) {
            return false;
        }
        js_ws(&s);
        if (s.p < s.end && *s.p == ',') { s.p++; continue; }
        if (s.p < s.end && *s.p == '}') return true;
        return false;
    }
}

static void armbench_override_init_impl(void) {
#if ARMBENCH_HAVE_DLOPEN
    const char * path = getenv("ARMBENCH_OVERRIDES");
    if (!path || !*path) return;
    const char * lg = getenv("ARMBENCH_OVERRIDE_LOG");
    g_log = lg && *lg && strcmp(lg, "0") != 0;
    const char * th = getenv("ARMBENCH_OVERRIDE_THREADS");
    g_force_single = th && strcmp(th, "1") == 0;
    // ARMBENCH_DUMP_DIR: capture real activations (and the real weight rows once) per
    // intercepted shape, so definition workloads can be built from what the model
    // actually computes instead of synthetic random tensors. Forces single-thread
    // dispatch so A is the whole ubatch, not one thread's row slice.
    g_dump_dir = getenv("ARMBENCH_DUMP_DIR");
    if (g_dump_dir && !*g_dump_dir) g_dump_dir = NULL;
    if (g_dump_dir) {
        const char * nc = getenv("ARMBENCH_DUMP_CALLS");
        if (nc && *nc) g_dump_calls = atoi(nc);
        const char * db = getenv("ARMBENCH_DUMP_WEIGHTS");
        g_dump_weights = db && strcmp(db, "0") != 0;
        const char * st = getenv("ARMBENCH_DUMP_STRIDE");
        if (st && *st) g_dump_stride = atoi(st);
        if (g_dump_stride < 1) g_dump_stride = 1;
        g_force_single = true;
        fprintf(stderr, "[armbench-override] dumping %d call(s) per shape to %s (weights=%d)\n",
                g_dump_calls, g_dump_dir, (int) g_dump_weights);
    }

    size_t len = 0;
    char * buf = read_file(path, &len);
    if (!buf) {
        fprintf(stderr, "[armbench-override] cannot read manifest %s; overrides disabled\n", path);
        return;
    }
    bool ok = parse_manifest(buf, len, path);
    free(buf);
    if (!ok) {
        fprintf(stderr, "[armbench-override] parse error in manifest %s; overrides disabled\n", path);
        g_nkernels = 0;
        return;
    }

    // resolve every entry; a failure disables all overrides (a partially-overridden
    // run would be a misleading measurement)
    for (int i = 0; i < g_nkernels; i++) {
        struct armbench_kernel * k = &g_kernels[i];
        void * h = dlopen(k->so, RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            fprintf(stderr, "[armbench-override] dlopen(%s) failed: %s; overrides disabled\n", k->so, dlerror());
            g_nkernels = 0;
            return;
        }
        k->fn = dlsym(h, k->symbol);
        if (!k->fn) {
            fprintf(stderr, "[armbench-override] dlsym(%s, %s) failed: %s; overrides disabled\n", k->so, k->symbol, dlerror());
            g_nkernels = 0;
            return;
        }
        if (k->abi == ARMBENCH_ABI_ENTRY_ROWS) {
            k->fn_rows = dlsym(h, k->symbol_rows);
            if (!k->fn_rows) {
                fprintf(stderr, "[armbench-override] dlsym(%s, %s) failed: %s; overrides disabled\n", k->so, k->symbol_rows, dlerror());
                g_nkernels = 0;
                return;
            }
        }
        if (k->op == ARMBENCH_OP_RMS_NORM) {
            fprintf(stderr, "[armbench-override] note: rms_norm override is a stub in this build; entry %s will never be hit\n", k->so);
        }
    }
    g_enabled = g_nkernels > 0;
    if (g_log) {
        fprintf(stderr, "[armbench-override] loaded %d kernel(s) from %s%s\n", g_nkernels, path,
                g_force_single ? " (ARMBENCH_OVERRIDE_THREADS=1: single-thread dispatch forced)" : "");
        for (int i = 0; i < g_nkernels; i++) {
            const struct armbench_kernel * k = &g_kernels[i];
            fprintf(stderr, "[armbench-override]   %s type=%s K=%" PRId64 " N=%" PRId64 " abi=%s threads=%d -> %s:%s\n",
                    armbench_op_names[k->op], k->op == ARMBENCH_OP_MUL_MAT ? ggml_type_name(k->type) : "-",
                    k->K, k->N, armbench_abi_names[k->abi], k->threads, k->so, k->symbol);
        }
    }
#endif
}

#if ARMBENCH_HAVE_DLOPEN
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
#endif
static volatile bool g_inited = false;

// Fast path: one static-bool check once initialised. Init is guarded by
// pthread_once because ggml_compute_forward runs on all worker threads at once.
static inline bool armbench_override_ready(void) {
    if (!g_inited) {
#if ARMBENCH_HAVE_DLOPEN
        pthread_once(&g_once, armbench_override_init_impl);
#endif
        g_inited = true;
    }
    return g_enabled;
}

int armbench_override_num_kernels(void) {
    return armbench_override_ready() ? g_nkernels : 0;
}

uint64_t armbench_override_hit_count(void) {
    return g_total_hits;
}

// ---------------------------------------------------------------------------
// mul_mat
// ---------------------------------------------------------------------------

// Thread-local so that independent graph computations (each driven by its own
// thread 0) never share the scratch buffer.
static _Thread_local void * tl_scratch = NULL;
static _Thread_local size_t tl_scratch_size = 0;

static void * scratch_get(size_t bytes) {
    if (bytes > tl_scratch_size) {
        free(tl_scratch);
        tl_scratch = malloc(bytes);
        tl_scratch_size = tl_scratch ? bytes : 0;
    }
    return tl_scratch;
}

static struct armbench_kernel * find_mul_mat(enum ggml_type type, int64_t K, int64_t N) {
    for (int i = 0; i < g_nkernels; i++) {
        struct armbench_kernel * k = &g_kernels[i];
        if (k->op == ARMBENCH_OP_MUL_MAT && k->type == type && k->K == K && k->N == N) return k;
    }
    return NULL;
}

// hit accounting + first-hit log; called from thread 0 only (once per op)
static void note_hit(struct armbench_kernel * k, const struct ggml_tensor * src0, int64_t K, int64_t N, int64_t M, int nth, const char * mode) {
    if (g_log && k->hits == 0) {
        fprintf(stderr, "[armbench-override] hit mul_mat type=%s K=%" PRId64 " N=%" PRId64 " (M=%" PRId64 ", nth=%d, abi=%s, dispatch=%s) -> %s:%s\n",
                ggml_type_name(src0->type), K, N, M, nth, armbench_abi_names[k->abi], mode, k->so, k->symbol);
    }
    k->hits++;
    g_total_hits++;
}

// Threads that already returned cannot be recalled, so there is no fallback to the
// stock path for this op: fail loudly rather than emit garbage.
static void kernel_failed(const struct armbench_kernel * k, int rc, const struct ggml_tensor * src0, int64_t K, int64_t N, int64_t M) {
    fprintf(stderr, "[armbench-override] kernel %s:%s returned %d for mul_mat type=%s K=%" PRId64 " N=%" PRId64 " M=%" PRId64 "\n",
            k->so, k->symbol, rc, ggml_type_name(src0->type), K, N, M);
    abort();
}


// ---------------------------------------------------------------------------
// Activation / weight capture (ARMBENCH_DUMP_DIR)
// ---------------------------------------------------------------------------
// Writes raw little-endian tensors next to a one-line .meta descriptor:
//   <type>_K<K>_N<N>.call<i>.a.bin   bf16 activations, row-major [M, K]
//   <type>_K<K>_N<N>.b.bin           the tensor's own ggml block rows [N, K_bytes]
// scripts/e2e/dump_to_workloads.py turns these into bench-trace workloads.
static void armbench_dump(struct armbench_kernel * k, const struct ggml_tensor * src0,
                          const uint16_t * A, int64_t M, int64_t N, int64_t K) {
    char path[2048], meta[2048];
    const char * tn = ggml_type_name(src0->type);
    snprintf(path, sizeof path, "%s/%s_K%lld_N%lld.call%d.a.bin",
             g_dump_dir, tn, (long long) K, (long long) N, k->dumped);
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[armbench-override] dump: cannot write %s\n", path); return; }
    fwrite(A, sizeof(uint16_t), (size_t) M * (size_t) K, f);
    fclose(f);
    snprintf(meta, sizeof meta, "%s/%s_K%lld_N%lld.call%d.a.meta",
             g_dump_dir, tn, (long long) K, (long long) N, k->dumped);
    f = fopen(meta, "w");
    if (f) {
        fprintf(f, "{\"type\":\"%s\",\"K\":%lld,\"N\":%lld,\"M\":%lld,\"dtype\":\"bfloat16\",\"call\":%d}\n",
                tn, (long long) K, (long long) N, (long long) M, k->dumped);
        fclose(f);
    }
    if (g_dump_weights && k->dumped == 0) {
        const size_t nb = (size_t) src0->nb[1] * (size_t) N;
        snprintf(path, sizeof path, "%s/%s_K%lld_N%lld.b.bin",
                 g_dump_dir, tn, (long long) K, (long long) N);
        f = fopen(path, "wb");
        if (f) { fwrite(src0->data, 1, nb, f); fclose(f); }
    }
    k->dumped++;
}

bool armbench_override_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (!armbench_override_ready()) return false;

    const struct ggml_tensor * src0 = dst->src[0];  // weights     [K,N]  ne00=K ne01=N
    const struct ggml_tensor * src1 = dst->src[1];  // activations [K,M]  ne10=K ne11=M
                                                    // dst         [N,M]  ne0=N  ne1=M

    // Eligibility must be a pure function of tensor metadata (all threads must agree).
    // src0->extra != NULL means an extra buffer type (repack / KleidiAI / AMX) owns the
    // weights and its data layout is not the plain ggml block layout; never claim those.
    // (In v0.4.1 those ops are intercepted by ggml_cpu_extra_compute_forward before
    // reaching this point anyway; this is a belt-and-braces guard.)
    if (src0->extra != NULL) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (!ggml_is_contiguous(src0)) return false;
    if (src1->nb[0] != sizeof(float)) return false;   // rows may be strided (nb[1]); elements must not be
    if (!ggml_is_contiguous(dst)) return false;

    const int64_t K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    struct armbench_kernel * k = find_mul_mat(src0->type, K, N);
    if (!k) return false;
    if (k->abi != ARMBENCH_ABI_LLAMACPP && (M > INT_MAX || N > INT_MAX)) return false;  // 4/5-arg ABIs take int

    // ---- multithreaded dispatch (abi "entry_rows" only) -------------------------------
    // Barrier-free: each thread computes a disjoint slice of dst from its own thread-local
    // bf16 copy of the activations it needs; ggml barriers after the op.
    if (k->abi == ARMBENCH_ABI_ENTRY_ROWS && !g_force_single && params->nth > 1) {
        const int ith = params->ith, nth = params->nth;
        int rc = 0;
        if (M < nth) {
            // Not enough activation rows to keep every thread busy, so split N instead of M.
            // M-splitting here is what made small micro-batches catastrophic: at M=2 with 16
            // threads only 2 threads got work and the build ran ~4.7x SLOWER than stock ggml,
            // which parallelises over N and uses all of them. Covers decode (M==1) and the
            // small-batch prefill / speculative-decode range in one branch.
            const int64_t chunk = ((N + nth - 1) / nth + ARMBENCH_N_CHUNK - 1) / ARMBENCH_N_CHUNK * ARMBENCH_N_CHUNK;
            const int64_t r0 = (int64_t) ith * chunk;
            if (r0 >= N) return true;
            const int64_t n_i = (r0 + chunk < N ? r0 + chunk : N) - r0;
            // The rows ABI writes a contiguous [M, n_i] block (row stride n_i), but dst has row
            // stride N -- so for M == 1 we can aim it straight at dst, and for M > 1 we need a
            // staging buffer and a per-row copy. One allocation, carved in two.
            const size_t a_bytes   = (size_t) M * (size_t) K * sizeof(uint16_t);
            const size_t out_bytes = (M > 1) ? (size_t) M * (size_t) n_i * sizeof(float) : 0;
            uint16_t * A = scratch_get(a_bytes + out_bytes);
            if (!A) goto oom;
            for (int64_t m = 0; m < M; m++) {
                const float * row = (const float *) ((const char *) src1->data + m * src1->nb[1]);
                ggml_cpu_fp32_to_bf16(row, (ggml_bf16_t *) (A + m * K), K);
            }
            float * out = (M > 1) ? (float *) ((char *) A + a_bytes) : (float *) dst->data + r0;
            if (ith == 0) note_hit(k, src0, K, N, M, params->nth, "N-split");
            rc = ((armbench_entry_gemm_rows_fn) k->fn_rows)(A, out,
                     (const uint8_t *) src0->data + r0 * src0->nb[1], (int) M, (int) n_i);
            if (M > 1) {
                for (int64_t m = 0; m < M; m++) {
                    memcpy((float *) dst->data + m * N + r0, out + m * n_i, (size_t) n_i * sizeof(float));
                }
            }
        } else {
            // prefill with M >= nth: split M across threads; full-N entry (row stride N is correct there)
            const int64_t mpt = (M + nth - 1) / nth;
            const int64_t m0  = (int64_t) ith * mpt;
            if (m0 >= M) return true;
            const int64_t M_i = (m0 + mpt < M ? m0 + mpt : M) - m0;
            uint16_t * A = scratch_get((size_t) M_i * (size_t) K * sizeof(uint16_t));
            if (!A) goto oom;
            for (int64_t m = 0; m < M_i; m++) {
                const float * row = (const float *) ((const char *) src1->data + (m0 + m) * src1->nb[1]);
                ggml_cpu_fp32_to_bf16(row, (ggml_bf16_t *) (A + m * K), K);
            }
            if (ith == 0) note_hit(k, src0, K, N, M, params->nth, "M-split");
            rc = ((armbench_entry_gemm_fn) k->fn)(A, (float *) dst->data + m0 * N, (const uint8_t *) src0->data, (int) M_i);
        }
        if (rc != 0) kernel_failed(k, rc, src0, K, N, M);
        return true;
    oom:
        fprintf(stderr, "[armbench-override] out of memory for bf16 scratch (M=%" PRId64 " K=%" PRId64 ")\n", M, K);
        abort();
    }

    // ---- single-threaded dispatch ("llamacpp", "entry", or forced) ---------------------
    // Thread 0 does all the work; the others return immediately and wait at the
    // per-node barrier in ggml_graph_compute_thread.
    if (params->ith != 0) return true;

    // src1 f32 [K,M] -> bf16 row-major [M,K] (round-to-nearest-even via ggml_cpu_fp32_to_bf16)
    uint16_t * A = scratch_get((size_t) M * (size_t) K * sizeof(uint16_t));
    if (!A) {
        fprintf(stderr, "[armbench-override] out of memory for %" PRId64 "x%" PRId64 " bf16 scratch\n", M, K);
        abort();
    }
    for (int64_t m = 0; m < M; m++) {
        const float * row = (const float *) ((const char *) src1->data + m * src1->nb[1]);
        ggml_cpu_fp32_to_bf16(row, (ggml_bf16_t *) (A + m * K), K);
    }
    note_hit(k, src0, K, N, M, params->nth, "single");
    // One dump every g_dump_stride calls: the same (type,K,N) shape recurs once per
    // layer, and early and late layers have very different activation statistics.
    if (g_dump_dir && k->dumped < g_dump_calls && (k->seen++ % g_dump_stride) == 0)
        armbench_dump(k, src0, A, M, N, K);

    int rc;
    switch (k->abi) {
        case ARMBENCH_ABI_ENTRY:
        case ARMBENCH_ABI_ENTRY_ROWS:
            rc = ((armbench_entry_gemm_fn) k->fn)(A, (float *) dst->data, (const uint8_t *) src0->data, (int) M);
            break;
        case ARMBENCH_ABI_LLAMACPP:
        default:
            rc = ((armbench_gemm_fn) k->fn)(A, src0->data, (float *) dst->data, M, N, K);
            break;
    }
    if (rc != 0) kernel_failed(k, rc, src0, K, N, M);
    return true;
}

// ---------------------------------------------------------------------------
// rms_norm - STUB
// ---------------------------------------------------------------------------
// TODO(armbench): ggml's GGML_OP_RMS_NORM has no weight; llama.cpp emits
// rms_norm followed by a separate GGML_OP_MUL with the norm weight, and the CPU
// backend fuses that pair in ggml_cpu_try_fuse_ops (ggml-cpu.c, RMS_NORM+MUL).
// The standalone ABI armbench_llamacpp_rms_norm(x, weight, out, M, D, eps)
// requires a non-NULL weight, so a correct override needs graph-level matching
// (claim the RMS_NORM node *and* its consuming MUL, writing the MUL's dst).
// Until then this returns false and is not wired into the patch.
bool armbench_override_rms_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    (void) params; (void) dst;
    return false;
}
