"""
loop_108: Pixel manipulation

Purpose: Use of LD4 with shift-accumulate instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_108",
    "num": "108",
    "name": "Pixel manipulation",
    "description": "Deinterleave RGBA pixel data and accumulate channel values with bit shifts",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_108_pixel_manipulation",
    "tags": ['sve2'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_108_data {
  uint32_t *restrict rgba;
  uint8_t *restrict y;
  int64_t n;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_108(struct loop_108_data *restrict input) {
  uint32_t *restrict rgba = input->rgba;
  uint8_t *restrict y = input->y;
  int64_t n = input->n;

  for (int i = 0; i < n; i++) {
    y[i] = (rgba[i] >> 24) >> 2;
    y[i] += ((rgba[i] >> 16) & 0xff) >> 1;
    y[i] += ((rgba[i] >> 16) & 0xff) >> 3;
    y[i] += ((rgba[i] >> 8) & 0xff) >> 3;
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Pixel manipulation
Purpose: Use of LD4 with shift-accumulate instructions
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
PERF_SIZES = [2000000, 8000000]
PERF_SIZES_C8G = [8000000, 32000000]  # DRAM-bound on Graviton4 (64MB L3)
