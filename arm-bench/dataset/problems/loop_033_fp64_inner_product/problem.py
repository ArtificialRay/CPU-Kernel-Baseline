"""
loop_033: FP64 Inner product

Purpose: Use of a tight loop with INC and WHILE instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_033",
    "num": "033",
    "name": "FP64 Inner product",
    "description": "Compute FP64 inner product of two arrays",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_033_fp64_inner_product",
    "tags": ['sve2', 'fp64'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_033_data {
  double *restrict a;
  double *restrict b;
  int64_t n;
  double res;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_033(struct loop_033_data *restrict input) {
  double *a = input->a;
  double *b = input->b;
  int64_t n = input->n;

  double res = 0.0;
  for (int i = 0; i < n; i++) {
    res += a[i] * b[i];
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: FP64 Inner product
Purpose: Use of a tight loop with INC and WHILE instructions
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
EDGE_SIZES = [0, 1, 3, 2999, 3001]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [2000000, 8000000]
PERF_SIZES_C8G = [8000000, 32000000]  # DRAM-bound on Graviton4 (64MB L3)
