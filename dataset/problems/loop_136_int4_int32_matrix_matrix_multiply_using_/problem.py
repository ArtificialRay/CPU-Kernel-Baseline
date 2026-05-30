"""
loop_136: INT4-INT32 matrix-matrix multiply using MMLA

Purpose: Use of 4-bit dequantization (LUT) with i8 to i32 MMLA instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_136",
    "num": "136",
    "name": "INT4-INT32 matrix-matrix multiply using MMLA",
    "description": "Multiply 4-bit quantized matrices into INT32 using lookup-table dequantization",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_136_int4_int32_matrix_matrix_multiply_using_",
    "tags": ['sve2', 'matmul'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_136_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  int8_t *restrict a;
  int8_t *restrict b;
  // weight matrix rearranged
  int8_t *restrict b_r;
  int32_t *restrict c;
  int8_t lut[16];
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_136(struct loop_136_data *data) {
  uint64_t m = data->m;
  uint64_t n = data->n;
  uint64_t k = data->k;
  int32_t *restrict c = data->c;

  for (uint64_t m_i = 0; m_i < m; m_i++)
    for (uint64_t n_i = 0; n_i < n; n_i++) c[m_i * n + n_i] = 0;

  // Loops ordered for contiguous memory access in inner loop
  for (uint64_t m_i = 0; m_i < m; m_i++)
    for (uint64_t n_i = 0; n_i < n; n_i++)
      for (uint64_t k_i = 0; k_i < k; k_i += 8)
        c[m_i * n + n_i] += int32_dot8(m_i, n_i, k_i, data);
}
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: INT4-INT32 matrix-matrix multiply using MMLA
Purpose: Use of 4-bit dequantization (LUT) with i8 to i32 MMLA instructions
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
