"""
loop_010: Conditional reduction (fp)

Purpose: Use of CLAST (SIMD&FP scalar) instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_010",
    "num": "010",
    "name": "Conditional reduction (fp)",
    "description": "Find the last active element of an FP array under a computed predicate",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_010_conditional_reduction_fp",
    "tags": ['sve2'],
}

# Data struct definition — parsed by scripts/gen_simd_loop_harness.py
STRUCT_DEF = r"""
struct loop_010_data {
  float *a;
  uint64_t n;
  int res;
};
"""

# Scalar reference implementation
SCALAR_CODE = r"""
static void inner_loop_010(struct loop_010_data *restrict data) {
  float *a = data->a;
  uint64_t n = data->n;

  bool any = 0;
  bool all = 1;

  for (int i = 0; i < n; i++) {
    if (a[i] < 0.0f) {
      any = 1;
    }
"""

# Prompt template
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Conditional reduction (fp)
Purpose: Use of CLAST (SIMD&FP scalar) instructions
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
EDGE_SIZES = [0, 1, 7, 9999, 10001]

# Input sizes for performance measurement.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [4000000, 16000000]
