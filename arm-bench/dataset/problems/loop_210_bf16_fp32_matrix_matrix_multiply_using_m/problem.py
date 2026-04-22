"""
loop_210: BF16-FP32 matrix-matrix multiply using MOPA / DOT

Purpose: Use of bf16 to fp32 MOPA (or DOT) instructions

ISA target: SME2 on Arm Neoverse V2 (AWS Graviton4, SME2 128-bit)
"""

METADATA = {
    "id": "loop_210",
    "num": "210",
    "name": "BF16-FP32 matrix-matrix multiply using MOPA / DOT",
    "description": "Multiply BF16 matrices accumulating into FP32 using tiled outer products or dot products",
    "isa_target": "sme2",
    "instance_type": "c8g.large",
    "dir_name": "loop_210_bf16_fp32_matrix_matrix_multiply_using_m",
    "tags": ['sme2', 'streaming', 'matmul', 'dot-product', 'fp32', 'bf16'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_210_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  bfloat16_t *restrict a;
  bfloat16_t *restrict b;
  float *restrict c;
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

USER_PROMPT_TEMPLATE = """Problem: BF16-FP32 matrix-matrix multiply using MOPA / DOT
Purpose: Use of bf16 to fp32 MOPA (or DOT) instructions
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
