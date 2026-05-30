"""
loop_107: UINT128 multiply

Purpose: Use of ADCL[B/T] instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_107",
    "num": "107",
    "name": "UINT128 multiply",
    "description": "Multiply 64-bit integer pairs producing full 128-bit results with carry",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_107_uint128_multiply",
    "tags": ['sve2', 'uint'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_107_data {
  uint128_t *restrict a;
  uint128_t *restrict b;
  uint256_t *restrict c;
  int64_t n;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_107(struct loop_107_data *restrict input) {
  uint128_t *restrict a = input->a;
  uint128_t *restrict b = input->b;
  uint256_t *restrict c = input->c;
  int64_t n = input->n;

  for (int i = 0; i < n; i++) {
    c[i] = mult256(a[i], b[i]);
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: UINT128 multiply
Purpose: Use of ADCL[B/T] instructions
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
EDGE_SIZES = [0, 1, 3, 999, 1001]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [500000, 2000000]
PERF_SIZES_C8G = [2000000, 8000000]  # DRAM-bound on Graviton4 (64MB L3)
