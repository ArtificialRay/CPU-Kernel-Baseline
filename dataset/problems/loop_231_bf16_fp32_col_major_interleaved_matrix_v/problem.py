"""
loop_231: BF16-FP32 col-major interleaved matrix-vector multiply

Purpose: Use of bf16 to fp32 DOT instruction

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_231",
    "num": "231",
    "name": "BF16-FP32 col-major interleaved matrix-vector multiply",
    "description": "Multiply a BF16 matrix by a BF16 vector accumulating into FP32 (column-major interleaved GEMV)",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_231_bf16_fp32_col_major_interleaved_matrix_v",
    "tags": ['sve2', 'matmul', 'dot-product', 'fp32', 'bf16'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_231_data {
  uint64_t m;
  uint64_t n;
  bfloat16_t *restrict a;
  bfloat16_t *restrict x;
  float32_t   *restrict b;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: BF16-FP32 col-major interleaved matrix-vector multiply
Purpose: Use of bf16 to fp32 DOT instruction
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
