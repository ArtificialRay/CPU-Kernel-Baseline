"""
loop_024: Sum of abs diffs

Purpose: Use of DOT instruction

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_024",
    "num": "024",
    "name": "Sum of abs diffs",
    "description": "Sum of absolute differences between two byte arrays, accumulated into a uint32",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_024_sum_of_abs_diffs",
    "tags": ['sve2', 'dot-product'],
}

# Data struct definition — parsed by scripts/gen_simd_loop_harness.py
STRUCT_DEF = r"""
struct loop_024_data {
  uint8_t *restrict a;
  uint8_t *restrict b;
  int64_t n;
  uint32_t res;
};
"""

# Scalar reference implementation
SCALAR_CODE = r"""
static void inner_loop_024(struct loop_024_data *restrict data) {
  uint8_t *restrict a = data->a;
  uint8_t *restrict b = data->b;
  int64_t n = data->n;

  uint32_t sum = 0;
  for (int i = 0; i < n; i++) {
    sum += __builtin_abs(a[i] - b[i]);
  }
"""

# Prompt template
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Sum of abs diffs
Purpose: Use of DOT instruction
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
EDGE_SIZES = [0, 1, 7, 39999, 40001]

# Input sizes for performance measurement.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [4000000, 16000000]
