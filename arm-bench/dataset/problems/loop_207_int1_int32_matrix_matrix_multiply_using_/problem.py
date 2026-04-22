"""
loop_207: INT1-INT32 matrix-matrix multiply using MOPA / DOT

Purpose: Use of 1-bit MOPA instructions

ISA target: SME2 on Arm Neoverse V2 (AWS Graviton4, SME2 128-bit)
"""

METADATA = {
    "id": "loop_207",
    "num": "207",
    "name": "INT1-INT32 matrix-matrix multiply using MOPA / DOT",
    "description": "Multiply 1-bit binary matrices accumulating into INT32",
    "isa_target": "sme2",
    "instance_type": "c8g.large",
    "dir_name": "loop_207_int1_int32_matrix_matrix_multiply_using_",
    "tags": ['sme2', 'streaming', 'matmul', 'dot-product'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_207_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  uint32_t *restrict a;
  uint32_t *restrict b;
  uint32_t *restrict c;
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

USER_PROMPT_TEMPLATE = """Problem: INT1-INT32 matrix-matrix multiply using MOPA / DOT
Purpose: Use of 1-bit MOPA instructions
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
