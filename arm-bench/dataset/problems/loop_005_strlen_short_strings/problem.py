"""
loop_005: strlen short strings

Purpose: Use of FF and NF loads instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_005",
    "num": "005",
    "name": "strlen short strings",
    "description": "Compute the length of many short null-terminated strings",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_005_strlen_short_strings",
    "tags": ['sve2', 'string'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_005_data {
  uint8_t *p;
  uint8_t *lmt;
  uint32_t checksum;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""

"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: strlen short strings
Purpose: Use of FF and NF loads instructions
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
EDGE_SIZES = [1, 10, 100, 3999, 4001]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [8000, 15000]
