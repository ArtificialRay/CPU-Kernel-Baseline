"""
loop_126: conditional update

Purpose: Use of simd loop with conditional update

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_126",
    "num": "126",
    "name": "conditional update",
    "description": "Conditionally update array elements where a per-element predicate holds",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_126_conditional_update",
    "tags": ['sve2'],
}

# Data struct definition — parsed by scripts/gen_simd_loop_harness.py
STRUCT_DEF = r"""
struct loop_126_data {
  uint32_t *restrict a;
  uint32_t *restrict b;
  int n;
  uint32_t res;
};
"""

# Scalar reference implementation
SCALAR_CODE = r"""
static void inner_loop_126(struct loop_126_data *restrict data) {
  uint32_t *a = data->a;
  uint32_t *b = data->b;
  int n = data->n;

  uint32_t res = 0;
  for (int i = 0; i < n; i++) {
    res += a[i] * b[i];
    if (res % 2) {
      res++;
    }
"""

# Prompt template
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: conditional update
Purpose: Use of simd loop with conditional update
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
PERF_SIZES = [1000000, 4000000]
PERF_SIZES_C8G = [4000000, 16000000]  # DRAM-bound on Graviton4 (64MB L3)
