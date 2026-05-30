"""
loop_215: UINT8-UINT32 col-major interleaved matrix-vector multiply

Purpose: Use of u8 to u32 DOT instruction

ISA target: SME2 on Arm Neoverse V2 (AWS Graviton4, SME2 128-bit)
"""

METADATA = {
    "id": "loop_215",
    "num": "215",
    "name": "UINT8-UINT32 col-major interleaved matrix-vector multiply",
    "description": "Multiply uint8 matrices accumulating into uint32 using column-major tiled dot products",
    "isa_target": "sme2",
    "instance_type": "c8g.large",
    "dir_name": "loop_215_uint8_uint32_col_major_interleaved_matri",
    "tags": ['sme2', 'streaming', 'matmul', 'dot-product', 'int8', 'uint'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_215_data {
  uint64_t m;
  uint64_t n;
  uint8_t *restrict a;
  uint8_t *restrict b;
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

USER_PROMPT_TEMPLATE = """Problem: UINT8-UINT32 col-major interleaved matrix-vector multiply
Purpose: Use of u8 to u32 DOT instruction
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
