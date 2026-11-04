"""
loop_114: Auto-correlation

Purpose: Use of shifts, widening mult and load-replicate instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_114",
    "num": "114",
    "name": "Auto-correlation",
    "description": "Compute auto-correlation of an integer array with widening accumulation",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_114_auto_correlation",
    "tags": ['sve2'],
}

# Data struct definition — parsed by scripts/gen_simd_loop_harness.py
STRUCT_DEF = r"""
struct loop_114_data {
  int16_t *restrict data;
  int16_t *restrict res;
  int32_t n;
  int32_t lags;
  int16_t scale;
};
"""

# Scalar reference implementation
SCALAR_CODE = r"""
static void inner_loop_114(struct loop_114_data *restrict input) {
  int16_t *restrict data = input->data;
  int16_t *restrict res = input->res;
  int32_t n = input->n;
  int32_t lags = input->lags;
  int16_t scale = input->scale;

  for (int lag = 0; lag < lags; lag++) {
    int32_t acc = 0;
    int lmt = n - lag;
    for (int i = 0; i < lmt; i++) {
      acc += ((int32_t)data[i] * (int32_t)data[i + lag]) >> scale;
    }
"""

# Prompt template
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Auto-correlation
Purpose: Use of shifts, widening mult and load-replicate instructions
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
