"""
loop_245: INT8-INT32 matrix-matrix multiply with rearrangement

Purpose: Use of MOPA, DOT & MMLA instructions with matrix rearrangements

ISA target: SME2 on Arm Neoverse V2 (AWS Graviton4, SME2 128-bit)
"""

METADATA = {
    "id": "loop_245",
    "num": "245",
    "name": "INT8-INT32 matrix-matrix multiply with rearrangement",
    "description": "Multiply INT8 matrices into INT32 using a mix of tiled outer products, dot products, and matrix rearrangement",
    "isa_target": "sme2",
    "instance_type": "c8g.large",
    "dir_name": "loop_245_int8_int32_matrix_matrix_multiply_with_r",
    "tags": ['sme2', 'streaming', 'matmul', 'dot-product', 'int8'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_245_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  int8_t *restrict a;
  int8_t *restrict a_mod;
  int8_t *restrict b;
  int8_t *restrict b_mod;
  int32_t *restrict c;
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

USER_PROMPT_TEMPLATE = """Problem: INT8-INT32 matrix-matrix multiply with rearrangement
Purpose: Use of MOPA, DOT & MMLA instructions with matrix rearrangements
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
