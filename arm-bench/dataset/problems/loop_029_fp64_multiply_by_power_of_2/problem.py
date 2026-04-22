"""
loop_029: FP64 multiply by power of 2

Purpose: Use of FSCALE instruction

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_029",
    "num": "029",
    "name": "FP64 multiply by power of 2",
    "description": "Scale each FP64 element by an integer power of two",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_029_fp64_multiply_by_power_of_2",
    "tags": ['sve2', 'fp64'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_029_data {
  double *restrict input;
  int64_t *restrict scale;
  double *restrict output;
  int64_t size;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_029(struct loop_029_data *restrict data) {
  double *input = data->input;
  int64_t *scale = data->scale;
  double *output = data->output;
  int64_t size = data->size;

  for (int64_t i = 0; i < size; i++) {
    output[i] = __builtin_scalbn(input[i], (int)scale[i]);
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: FP64 multiply by power of 2
Purpose: Use of FSCALE instruction
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
PERF_SIZES = [500000, 2000000]
PERF_SIZES_C8G = [2000000, 8000000]  # DRAM-bound on Graviton4 (64MB L3)
