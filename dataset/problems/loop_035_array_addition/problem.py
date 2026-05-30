"""
loop_035: Array addition

Purpose: Use of WHILE for loop control

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_035",
    "num": "035",
    "name": "Array addition",
    "description": "Element-wise addition of two arrays with arbitrary length (non-multiple-of-vector-width)",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_035_array_addition",
    "tags": ['sve2'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_035_data {
  float *restrict a;
  float *restrict b;
  float *restrict c;
  int64_t n;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_035(struct loop_035_data *restrict input) {
  float *restrict a = input->a;
  float *restrict b = input->b;
  float *restrict c = input->c;
  int64_t n = input->n;

  for (int64_t i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Array addition
Purpose: Use of WHILE for loop control
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
PERF_SIZES = [4000000, 16000000]
