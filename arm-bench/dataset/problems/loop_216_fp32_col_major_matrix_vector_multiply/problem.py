"""
loop_216: FP32 col-major matrix-vector multiply

Purpose: Use of fp32 MLA instruction

ISA target: SME2 on Arm Neoverse V2 (AWS Graviton4, SME2 128-bit)
"""

METADATA = {
    "id": "loop_216",
    "num": "216",
    "name": "FP32 col-major matrix-vector multiply",
    "description": "Multiply an FP32 matrix by an FP32 vector (column-major GEMV)",
    "isa_target": "sme2",
    "instance_type": "c8g.large",
    "dir_name": "loop_216_fp32_col_major_matrix_vector_multiply",
    "tags": ['sme2', 'streaming', 'matmul', 'fp32'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_216_data {
  uint64_t m;
  uint64_t n;
  float32_t *restrict a;
  float32_t *restrict x;
  float32_t *restrict b;
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

USER_PROMPT_TEMPLATE = """Problem: FP32 col-major matrix-vector multiply
Purpose: Use of fp32 MLA instruction
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
