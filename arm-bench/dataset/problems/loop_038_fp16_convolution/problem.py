"""
loop_038: Fp16 convolution

Purpose: Use of fp16 multiply-add instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_038",
    "num": "038",
    "name": "Fp16 convolution",
    "description": "1D convolution of an FP16 signal with an FP16 filter kernel",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_038_fp16_convolution",
    "tags": ['sve2', 'fp16'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_038_data {
  float16_t *restrict a;
  float16_t *restrict b;
  float16_t *restrict c;
  int dim;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_038(struct loop_038_data *restrict data) {
  float16_t *a = data->a;
  float16_t *b = data->b;
  float16_t *c = data->c;
  int dim = data->dim;

  for (int row = 0; row < dim - 1; row++) {
    for (int col = 0; col < dim - 1; col++) {
      FLOAT16_t s0 = fp16_to_native(a[row * dim + col]);
      FLOAT16_t s1 = fp16_to_native(a[row * dim + col + 1]);
      FLOAT16_t s2 = fp16_to_native(a[(row + 1) * dim + col]);
      FLOAT16_t s3 = fp16_to_native(a[(row + 1) * dim + col + 1]);
      FLOAT16_t ac = fp16_to_native(b[row * dim + col]);
      FLOAT16_t k = 0.25f;
      FLOAT16_t r = ac + s0 * k + s1 * k + s2 * k + s3 * k;
      c[row * dim + col] = native_to_fp16(r);
    }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Fp16 convolution
Purpose: Use of fp16 multiply-add instructions
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
EDGE_SIZES = []

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = []
