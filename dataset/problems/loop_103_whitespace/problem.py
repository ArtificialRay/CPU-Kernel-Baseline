"""
loop_103: whitespace

Purpose: Use of MATCH and NMATCH instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_103",
    "num": "103",
    "name": "whitespace",
    "description": "Find all whitespace character positions in a byte string",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_103_whitespace",
    "tags": ['sve2'],
}

# Data struct definition — parsed by scripts/gen_simd_loop_harness.py
STRUCT_DEF = r"""
struct loop_103_data {
  uint8_t *p;
  uint8_t *end;
  int checksum;
};
"""

# Scalar reference implementation
SCALAR_CODE = r"""
static uint8_t *NOINLINE skip_whitespace(uint8_t *p, uint8_t *end) {
  while (p != end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) {
    p++;
  }
  return p;
}
"""

# Prompt template
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: whitespace
Purpose: Use of MATCH and NMATCH instructions
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
