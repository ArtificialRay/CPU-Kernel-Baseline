"""
loop_128: alias in contiguous access

Purpose: Use of simd loop with possible alias in contiguous mem access

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_128",
    "num": "128",
    "name": "alias in contiguous access",
    "description": "Shift array elements in place where source and destination ranges may overlap",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_128_alias_in_contiguous_access",
    "tags": ['sve2'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_128_data {
  uint32_t *restrict a;
  uint32_t *b;
  uint32_t *c;
  int n;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_128(struct loop_128_data *restrict data) {
  uint32_t *a = data->a;
  uint32_t *b = data->b;
  uint32_t *c = data->c;
  int n = data->n;

  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: alias in contiguous access
Purpose: Use of simd loop with possible alias in contiguous mem access
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
EDGE_SIZES = [0, 1, 7, 9999, 10001]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [4000000, 16000000]
