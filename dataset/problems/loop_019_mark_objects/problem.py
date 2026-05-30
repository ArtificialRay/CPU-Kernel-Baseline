"""
loop_019: Mark objects

Purpose: Use of scatters store instruction

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_019",
    "num": "019",
    "name": "Mark objects",
    "description": "Write a marker value to a set of scattered (non-contiguous) memory locations",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_019_mark_objects",
    "tags": ['sve2'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_019_data {
  object_t *restrict objects;
  uint32_t *restrict indexes;
  int64_t n;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_019(struct loop_019_data *restrict data) {
  object_t *objects = data->objects;
  uint32_t *indexes = data->indexes;
  int64_t n = data->n;

  for (int i = 0; i < n; i++) {
    objects[indexes[i]].mark = 1;
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Mark objects
Purpose: Use of scatters store instruction
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
EDGE_SIZES = [0, 1, 7, 7999, 8001]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [500000, 2000000]
PERF_SIZES_C8G = [2000000, 8000000]  # DRAM-bound on Graviton4 (64MB L3)
