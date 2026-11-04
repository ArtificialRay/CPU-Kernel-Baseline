"""
loop_124: Radix sort

Purpose: Use of simd instructions in radix sort

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_124",
    "num": "124",
    "name": "Radix sort",
    "description": "Sort integers by digit using a single radix counting pass",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_124_radix_sort",
    "tags": ['sve2', 'sort'],
}

# Data struct definition — parsed by scripts/gen_simd_loop_harness.py
STRUCT_DEF = r"""
struct loop_124_data {
  uint32_t n;
  int32_t *restrict data;
  int32_t *restrict temp;
  uint32_t *restrict hist;
  uint32_t *restrict prfx;
};
"""

# Scalar reference implementation
SCALAR_CODE = r"""
// Implementation
static void NOINLINE do_sort(struct loop_124_data *restrict input) {
  uint32_t n = input->n;
  int32_t *data = input->data;
  int32_t *temp = input->temp;

  com_sort_radix(n, data, temp);
}
"""

# Prompt template
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Radix sort
Purpose: Use of simd instructions in radix sort
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
EDGE_SIZES = [0, 1, 2, 255, 257]

# Input sizes for performance measurement.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [200000, 800000]
PERF_SIZES_C8G = [800000, 3200000]  # DRAM-bound on Graviton4 (64MB L3)
