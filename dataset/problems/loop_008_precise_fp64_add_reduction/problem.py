"""
loop_008: Precise fp64 add reduction

Purpose: Use of FADDA instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_008",
    "num": "008",
    "name": "Precise fp64 add reduction",
    "description": "Sum a double array with exact scalar FP ordering — result must be bit-identical to sequential scalar addition",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_008_precise_fp64_add_reduction",
    "tags": ['sve2', 'fp64'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_008_data {
  double *a;
  int n;
  double res;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_008(struct loop_008_data *restrict data) {
  double *a = data->a;
  int n = data->n;
  double res = 0.0;
  for (int i = 0; i < n; i++) {
    res += a[i];
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Precise fp64 add reduction
Purpose: Use of FADDA instructions
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
