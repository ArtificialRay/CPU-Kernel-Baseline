"""
loop_212: 4-bit-FP32 col-major interleaved matrix-vector multiply

Purpose: Use of 4-bit dequantization (LUT) and DOT instructions

ISA target: SME2 on Arm Neoverse V2 (AWS Graviton4, SME2 128-bit)
"""

METADATA = {
    "id": "loop_212",
    "num": "212",
    "name": "4-bit-FP32 col-major interleaved matrix-vector multiply",
    "description": "Multiply 4-bit quantized matrices into FP32 using lookup dequantization and dot products",
    "isa_target": "sme2",
    "instance_type": "c8g.large",
    "dir_name": "loop_212_4_bit_fp32_col_major_interleaved_matrix_",
    "tags": ['sme2', 'streaming', 'matmul', 'dot-product', 'fp32'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_212_data {
  uint64_t m;
  uint64_t n;
  // Dequantized element is fp32
  // size = M * N / 2 bytes
  uint8_t *restrict a;
  int8_t *restrict x;
  float *restrict b;
  // size = M * N / 32 * 4 = M * N / 16 bytes
  float *restrict a_scales;
  // size = N / 32 * 4 = N / 16 bytes
  float *restrict x_scales;
  // 512b Look-up table for dequantization.
  uint8_t lut[64];
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

USER_PROMPT_TEMPLATE = """Problem: 4-bit-FP32 col-major interleaved matrix-vector multiply
Purpose: Use of 4-bit dequantization (LUT) and DOT instructions
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
