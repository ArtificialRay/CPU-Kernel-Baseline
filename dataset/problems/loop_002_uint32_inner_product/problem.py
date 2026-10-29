"""
loop_002: UINT32 inner product

Purpose: Use of u32 MLA instruction

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_002",
    "num": "002",
    "name": "UINT32 inner product",
    "description": "Compute the integer dot product of two uint32 arrays",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_002_uint32_inner_product",
    "tags": ['sve2', 'uint'],
}

# Data struct definition — parsed by scripts/gen_simd_loop_harness.py
STRUCT_DEF = r"""
struct loop_002_data {
  uint32_t *restrict a;
  uint32_t *restrict b;
  int n;
  uint32_t res;
};
"""

# Scalar reference implementation
SCALAR_CODE = r"""
static void inner_loop_002(struct loop_002_data *restrict input) {
  uint32_t *restrict a = input->a;
  uint32_t *restrict b = input->b;
  int n = input->n;

  uint32_t res = 0;
  for (int i = 0; i < n; i++) {
    res += a[i] * b[i];
  }
"""

# Prompt template
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: UINT32 inner product
Purpose: Use of u32 MLA instruction
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
