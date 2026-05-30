"""
loop_101: Upscale filter

Purpose: Use of top/bottom instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_101",
    "num": "101",
    "name": "Upscale filter",
    "description": "Upscale a pixel buffer by splitting each element into its high and low halves",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_101_upscale_filter",
    "tags": ['sve2'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_101_data {
  uint8_t *restrict a;
  uint8_t *restrict b;
  int n;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_101(struct loop_101_data *restrict input) {
  uint8_t *restrict a = input->a;
  uint8_t *restrict b = input->b;
  int n = input->n;

  for (int i = 0; i < n - 1; i++) {
    uint16_t s1 = b[i];
    uint16_t s2 = b[i + 1];
    a[2 * i] = (3 * s1 + s2 + 2) >> 2;
    a[2 * i + 1] = (3 * s2 + s1 + 2) >> 2;
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Upscale filter
Purpose: Use of top/bottom instructions
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
PERF_SIZES = [2000000, 6000000]
PERF_SIZES_C8G = [8000000, 24000000]  # DRAM-bound on Graviton4 (64MB L3)
