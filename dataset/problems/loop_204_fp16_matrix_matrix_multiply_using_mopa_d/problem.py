"""
loop_204: FP16 matrix-matrix multiply using MOPA / DOT

Purpose: Use of fp16 to fp16 MOPA (or MLA) instructions

ISA target: SME2 on Arm Neoverse V2 (AWS Graviton4, SME2 128-bit)
"""

METADATA = {
    "id": "loop_204",
    "num": "204",
    "name": "FP16 matrix-matrix multiply using MOPA / DOT",
    "description": "Multiply two FP16 matrices accumulating into FP16 using tiled outer products",
    "isa_target": "sme2",
    "instance_type": "c8g.large",
    "dir_name": "loop_204_fp16_matrix_matrix_multiply_using_mopa_d",
    "tags": ['sme2', 'streaming', 'matmul', 'dot-product', 'fp16'],
}

# Data struct definition — parsed by scripts/gen_simd_loop_harness.py
STRUCT_DEF = r"""
struct loop_204_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  float16_t *restrict a;
  float16_t *restrict b;
  float16_t *restrict c;
};
"""

# Scalar reference implementation
SCALAR_CODE = r"""
#define LOOP_ATTR
#define OUTER_LOOP_ATTR
"""

# Prompt template
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: FP16 matrix-matrix multiply using MOPA / DOT
Purpose: Use of fp16 to fp16 MOPA (or MLA) instructions
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

# Input sizes for edge-case correctness testing.
# Empty list = skip (loop uses non-SIZE parameters or fixed dimensions).
EDGE_SIZES = []

# Input sizes for performance measurement.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = []
