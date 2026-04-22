"""
loop_040: Clamp operation

Purpose: Use of simd clamp (or min, max and shift) instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_040",
    "num": "040",
    "name": "Clamp operation",
    "description": "Clamp each element of an integer sequence to data-dependent [min, max] bounds",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_040_clamp_operation",
    "tags": ['sve2'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_040_data {
  int32_t count;
  int32_t res;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_040(struct loop_040_data *restrict input) {
  int32_t count = input->count;
  int32_t value = count / 2;
  int32_t result = 0;

  for (int32_t i = 0; i < count; i++) {
    result += CLAMP(value, 2 * i, i);
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Clamp operation
Purpose: Use of simd clamp (or min, max and shift) instructions
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
