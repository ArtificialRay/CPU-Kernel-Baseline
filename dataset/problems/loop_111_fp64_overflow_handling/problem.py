"""
loop_111: FP64 overflow handling

Purpose: Use of FLOGB and FSCALE instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_111",
    "num": "111",
    "name": "FP64 overflow handling",
    "description": "Normalise FP64 values that may overflow by detecting and rescaling their exponents",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_111_fp64_overflow_handling",
    "tags": ['sve2', 'fp64'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_111_data {
  double *restrict input1;
  double *restrict input2;
  double *restrict output;
  int64_t *restrict exponent;
  int64_t size;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_111(struct loop_111_data *restrict input) {
  double *input1 = input->input1;
  double *input2 = input->input2;
  double *output = input->output;
  int64_t *exponent = input->exponent;
  int64_t size = input->size;

  for (int64_t i = 0; i < size; i++) {
    output[i] = __builtin_frexp(input1[i], (int *)&exponent[i]);
    output[i] = __builtin_ldexp(output[i], 1);
    output[i] *= input2[i];
    --exponent[i];
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: FP64 overflow handling
Purpose: Use of FLOGB and FSCALE instructions
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
PERF_SIZES = [500000, 1500000]
PERF_SIZES_C8G = [2000000, 8000000]  # DRAM-bound on Graviton4 (64MB L3)
