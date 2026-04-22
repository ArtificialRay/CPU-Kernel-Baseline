"""
loop_023: Conjugate Gradient

Purpose: Use of gathers load instruction

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_023",
    "num": "023",
    "name": "Conjugate Gradient",
    "description": "Sparse matrix-vector multiply where column indices are stored as an indirect index array",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_023_conjugate_gradient",
    "tags": ['sve2'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_023_data {
  double *restrict a;
  double *restrict b;
  uint32_t *restrict indexes;
  int n;
  double res;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_023(struct loop_023_data *restrict data) {
  double *a = data->a;
  double *b = data->b;
  uint32_t *indexes = data->indexes;
  int n = data->n;

  double res = 0;
  for (int i = 0; i < n; i++) {
    res = res + a[indexes[i]] * b[i];
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Conjugate Gradient
Purpose: Use of gathers load instruction
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
EDGE_SIZES = [0, 1, 3, 4095, 4097]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [1000000, 4000000]
PERF_SIZES_C8G = [4000000, 16000000]  # DRAM-bound on Graviton4 (64MB L3)
