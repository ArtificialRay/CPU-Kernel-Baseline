"""
loop_037: FP32 complex vector product

Purpose: Use of fp32 FCMLA instruction

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_037",
    "num": "037",
    "name": "FP32 complex vector product",
    "description": "Element-wise complex multiplication of two FP32 complex-number arrays (interleaved re/im)",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_037_fp32_complex_vector_product",
    "tags": ['sve2', 'fp32'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_037_data {
  cfloat32_t *restrict a0;
  cfloat32_t *restrict b0;
  cfloat32_t *restrict c0;
  uint64_t size;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_037(struct loop_037_data *restrict input) {
  cfloat32_t *a = input->a0;
  cfloat32_t *b = input->b0;
  cfloat32_t *c = input->c0;
  uint64_t size = input->size;

  uint64_t i;
  for (i = 0; i < size; i++) {
    c[i].re = (a[i].re * b[i].re) - (a[i].im * b[i].im);
    c[i].im = (a[i].re * b[i].im) + (a[i].im * b[i].re);
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: FP32 complex vector product
Purpose: Use of fp32 FCMLA instruction
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
PERF_SIZES = [2000000, 8000000]
PERF_SIZES_C8G = [8000000, 32000000]  # DRAM-bound on Graviton4 (64MB L3)
