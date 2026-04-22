"""
loop_106: Sheep and goats

Purpose: Use of BGRP instruction

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_106",
    "num": "106",
    "name": "Sheep and goats",
    "description": "Partition a vector by bit flag, concentrating set-bit elements to one side",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_106_sheep_and_goats",
    "tags": ['sve2'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_106_data {
  uint32_t *restrict a;
  uint32_t *restrict b;
  uint32_t *restrict perm;
  int64_t n;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_106(struct loop_106_data *restrict input) {
  uint32_t *restrict a = input->a;
  uint32_t *restrict b = input->b;
  uint32_t *restrict p = input->perm;
  int64_t n = input->n;

  for (int i = 0; i < n; i++) {
    b[i] = permute(a[i], p);
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Sheep and goats
Purpose: Use of BGRP instruction
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
EDGE_SIZES = [0, 1, 3, 199, 201]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [30000, 100000]
PERF_SIZES_C8G = [100000, 400000]  # DRAM-bound on Graviton4 (64MB L3)
