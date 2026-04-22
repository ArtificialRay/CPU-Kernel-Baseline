"""
loop_028: FP64 fast division

Purpose: Use of FRECPE and FRECPS instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_028",
    "num": "028",
    "name": "FP64 fast division",
    "description": "Compute element-wise fast approximate reciprocal division of FP64 arrays",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_028_fp64_fast_division",
    "tags": ['sve2', 'fp64'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_028_data {
  double *restrict input1;
  double *restrict input2;
  double *restrict output;
  int64_t size;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_028(struct loop_028_data *restrict data) {
  double *restrict input1 = data->input1;
  double *restrict input2 = data->input2;
  double *restrict output = data->output;
  int64_t size = data->size;

  for (int64_t i = 0; i < size; i++) {
    output[i] = input1[i] / input2[i];
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: FP64 fast division
Purpose: Use of FRECPE and FRECPS instructions
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
EDGE_SIZES = [0, 1, 3, 9999, 10001]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [2000000, 8000000]
PERF_SIZES_C8G = [8000000, 32000000]  # DRAM-bound on Graviton4 (64MB L3)
