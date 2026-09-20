// test_kernel.c - trivial reference kernels used by test_override.sh to exercise
// the override hook. Dequantizes block_q4_K rows with ggml's dequantize_row_q4_K
// (resolved at runtime from the host process, which links libggml-base statically)
// and does a plain fp32 dot against the bf16 activations.
//
// Exposes both manifest ABIs:
//   "llamacpp": armbench_llamacpp_gemm(A, B, C, M, N, K)          (any N,K)
//   "entry":    armbench_entry_gemm(A_bf16, output, B_blocks, M)   (N,K baked in below)
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ENTRY_N 32
#define ENTRY_K 256

#define QK_K 256
#define Q4_K_ROW_BYTES(K) (((K) / QK_K) * 144)   // sizeof(block_q4_K) == 144

// call counters, read back by test_override via dlsym
int test_kernel_calls_llamacpp = 0;
int test_kernel_calls_entry    = 0;

typedef void (*dequantize_row_fn)(const void * x, float * y, int64_t k);

static dequantize_row_fn get_dequantize(void) {
    static dequantize_row_fn fn = NULL;
    if (!fn) fn = (dequantize_row_fn) dlsym(RTLD_DEFAULT, "dequantize_row_q4_K");
    return fn;
}

static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static int gemm(const uint16_t * A, const uint8_t * B, float * C, int64_t M, int64_t N, int64_t K) {
    dequantize_row_fn deq = get_dequantize();
    if (!deq) return -1;
    if (K % QK_K != 0) return -2;
    float * brow = (float *) malloc((size_t) K * sizeof(float));
    if (!brow) return -3;
    for (int64_t n = 0; n < N; n++) {
        deq(B + n * Q4_K_ROW_BYTES(K), brow, K);
        for (int64_t m = 0; m < M; m++) {
            const uint16_t * arow = A + m * K;
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) acc += bf16_to_f32(arow[k]) * brow[k];
            C[m * N + n] = acc;
        }
    }
    free(brow);
    return 0;
}

int armbench_llamacpp_gemm(const void * A, const void * B, float * C, int64_t M, int64_t N, int64_t K) {
    test_kernel_calls_llamacpp++;
    return gemm((const uint16_t *) A, (const uint8_t *) B, C, M, N, K);
}

int armbench_entry_gemm(const uint16_t * A_bf16, float * output, const uint8_t * B_blocks, int M) {
    test_kernel_calls_entry++;
    return gemm(A_bf16, B_blocks, output, M, ENTRY_N, ENTRY_K);
}
