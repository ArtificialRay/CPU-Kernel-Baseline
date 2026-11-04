"""
loop_202: FP32 matrix-matrix multiply using MOPA / DOT

Purpose: Use of fp32 MOPA (or MLA) instructions

ISA target: SME2 on Arm Neoverse V2 (AWS Graviton4, SME2 128-bit)
"""

METADATA = {
    "id": "loop_202",
    "num": "202",
    "name": "FP32 matrix-matrix multiply using MOPA / DOT",
    "description": "Multiply two FP32 matrices using tiled outer-product accumulation",
    "isa_target": "sme2",
    "instance_type": "c8g.large",
    "dir_name": "loop_202_fp32_matrix_matrix_multiply_using_mopa_d",
    "tags": ['sme2', 'streaming', 'matmul', 'dot-product', 'fp32'],
}

# Data struct definition — parsed by scripts/gen_simd_loop_harness.py
STRUCT_DEF = r"""
struct loop_202_data {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  float *restrict a;
  float *restrict b;
  float *restrict c;
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

USER_PROMPT_TEMPLATE = """Problem: FP32 matrix-matrix multiply using MOPA / DOT
Purpose: Use of fp32 MOPA (or MLA) instructions
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
