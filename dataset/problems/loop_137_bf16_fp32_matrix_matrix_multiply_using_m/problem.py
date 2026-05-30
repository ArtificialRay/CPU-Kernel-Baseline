"""
loop_137: BF16-FP32 matrix-matrix multiply using MMLA

Purpose: Use of bf16 to fp32 MMLA instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_137",
    "num": "137",
    "name": "BF16-FP32 matrix-matrix multiply using MMLA",
    "description": "Multiply BF16 matrices accumulating into FP32 using tiled dot products",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_137_bf16_fp32_matrix_matrix_multiply_using_m",
    "tags": ['sve2', 'matmul', 'fp32', 'bf16'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_137_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  bfloat16_t *restrict a;
  bfloat16_t *restrict b;
  float32_t *restrict c;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_137(struct loop_137_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  float *restrict c = data->c;

  for (uint64_t x = 0; x < m; x++)
    for (uint64_t y = 0; y < n; y++) c[x * n + y] = 0.0f;

  // Loops ordered for contiguous memory access in inner loop
  for (uint64_t z = 0; z < k; z += 4)
    for (uint64_t x = 0; x < m; x++)
      for (uint64_t y = 0; y < n; y++) c[x * n + y] += bf16_dot4(x, y, z, data);
}
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: BF16-FP32 matrix-matrix multiply using MMLA
Purpose: Use of bf16 to fp32 MMLA instructions
Target: {isa_upper} on {isa_desc}

Struct definition:
```c
{struct_def}
```

Scalar implementation to optimize:
```c
{scalar_code}
```

Write an optimized {isa_upper} implementation. Output only the C function.
"""

# Input sizes for edge-case correctness testing at submit time.
# Empty list = skip (loop uses non-SIZE parameters or fixed dimensions).
EDGE_SIZES = []

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = []
