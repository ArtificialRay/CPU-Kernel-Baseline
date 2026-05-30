"""
loop_208: BF16-BF16 matrix-matrix multiply using MOPA / DOT

Purpose: Use of bf16 to bf16 MOPA (or MLA) instructions

ISA target: SME2 on Arm Neoverse V2 (AWS Graviton4, SME2 128-bit)
"""

METADATA = {
    "id": "loop_208",
    "num": "208",
    "name": "BF16-BF16 matrix-matrix multiply using MOPA / DOT",
    "description": "Multiply BF16 matrices accumulating into BF16 using tiled outer products",
    "isa_target": "sme2",
    "instance_type": "c8g.large",
    "dir_name": "loop_208_bf16_bf16_matrix_matrix_multiply_using_m",
    "tags": ['sme2', 'streaming', 'matmul', 'dot-product', 'bf16'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_208_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  bfloat16_t *restrict a;
  bfloat16_t *restrict b;
  bfloat16_t *restrict c;
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

USER_PROMPT_TEMPLATE = """Problem: BF16-BF16 matrix-matrix multiply using MOPA / DOT
Purpose: Use of bf16 to bf16 MOPA (or MLA) instructions
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
