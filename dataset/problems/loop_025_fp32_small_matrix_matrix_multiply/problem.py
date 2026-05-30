"""
loop_025: FP32 small matrix-matrix multiply

Purpose: Use of fp32 indexed MLA instruction

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_025",
    "num": "025",
    "name": "FP32 small matrix-matrix multiply",
    "description": "Multiply two small fixed-size FP32 matrices",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_025_fp32_small_matrix_matrix_multiply",
    "tags": ['sve2', 'matmul', 'fp32'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_025_data {
  float *restrict a;
  float *restrict b;
  float *restrict c;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void NOINLINE matrix_multiply_8x8(float *restrict a, float *restrict b,
                                         float *restrict c) {
  memset(c, 0, sizeof(float) * 8 * 8);

  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      for (int i = 0; i < 8; i++) {
        c[col + row * 8] += a[i + row * 8] * b[col + i * 8];
      }
    }
  }
}
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: FP32 small matrix-matrix multiply
Purpose: Use of fp32 indexed MLA instruction
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
