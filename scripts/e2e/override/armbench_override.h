// armbench_override.h - runtime "kernel override" hook for the ggml CPU backend.
//
// Lets agent-optimized standalone kernels (shared objects with a C ABI) replace
// specific ggml ops at runtime, selected by an env var:
//
//   ARMBENCH_OVERRIDES=/path/manifest.json   (see manifest.example.json)
//   ARMBENCH_OVERRIDE_LOG=1                  (print one stderr line per matched key on first hit)
//
// If ARMBENCH_OVERRIDES is unset the hook costs one static-bool check per op.
//
// Kernel ABIs (manifest field "abi", per kernel):
//   "llamacpp" (default): int armbench_llamacpp_gemm(const void* A, const void* B, float* C,
//                                                     int64_t M, int64_t N, int64_t K)
//   "entry":              int armbench_entry_gemm(const uint16_t* A_bf16, float* output,
//                                                  const uint8_t* B_blocks, int M)
//                         (N and K baked into the kernel; must equal the manifest's N/K)
//   "entry_rows":         the .so exports the "entry" symbol AND
//                         int armbench_entry_gemm_rows(const uint16_t* A_bf16, float* output,
//                                                       const uint8_t* B_blocks, int M, int n_rows)
//                         computing weight rows [0,n_rows) relative to B_blocks with output row
//                         stride n_rows. Only this ABI is dispatched multithreaded (see below).
// In both: A = row-major bf16 [M,K] (raw bits), B = raw ggml quantized weight rows
// (src0->data as-is), C/output = row-major f32 [M,N], return 0 on success.
//
// Threading: ggml calls ggml_compute_forward on every worker thread with
// params->ith / params->nth; the graph loop barriers after every node, so the
// override is barrier-free. "llamacpp"/"entry" kernels run on thread 0 only (the
// others return true immediately). "entry_rows" kernels are dispatched on all
// threads: M==1 (decode) splits N in 64-row chunks via entry_gemm_rows; M>=2
// (prefill) splits M via entry_gemm. ARMBENCH_OVERRIDE_THREADS=1 forces the
// thread-0-only path for every ABI. The manifest field "threads" is reserved.
#pragma once

#include <stdbool.h>
#include <stdint.h>

struct ggml_compute_params;
struct ggml_tensor;

#ifdef __cplusplus
extern "C" {
#endif

// GGML_OP_MUL_MAT. Returns true if the op was handled (on every thread), false to
// fall through to ggml_compute_forward_mul_mat. The claim decision is a pure
// function of tensor metadata, so all threads agree.
bool armbench_override_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);

// GGML_OP_RMS_NORM. STUB: always returns false (see README, "rms_norm TODO").
bool armbench_override_rms_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);

// Introspection (for tests / coverage checks).
int      armbench_override_num_kernels(void);  // entries loaded from the manifest (0 if disabled)
uint64_t armbench_override_hit_count(void);    // total ops handled by override kernels

#ifdef __cplusplus
}
#endif
