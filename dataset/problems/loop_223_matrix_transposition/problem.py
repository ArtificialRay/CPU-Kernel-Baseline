"""
loop_223: Matrix transposition

Purpose: Use of simd instructions (LD & ST ZA, LD & ZIP) in transposition

ISA target: SME2 on Arm Neoverse V2 (AWS Graviton4, SME2 128-bit)
"""

METADATA = {
    "id": "loop_223",
    "num": "223",
    "name": "Matrix transposition",
    "description": "Transpose a matrix in-place using interleaved load and store",
    "isa_target": "sme2",
    "instance_type": "c8g.large",
    "dir_name": "loop_223_matrix_transposition",
    "tags": ['sme2', 'streaming', 'matmul'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_223_data {
  uint64_t m;
  uint64_t n;
  uint32_t *restrict a;
  uint32_t *restrict at;
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

USER_PROMPT_TEMPLATE = """Problem: Matrix transposition
Purpose: Use of simd instructions (LD & ST ZA, LD & ZIP) in transposition
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
