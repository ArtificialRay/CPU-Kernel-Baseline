"""
loop_120: Insertion sort

Purpose: Use of CMPLT instruction

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_120",
    "num": "120",
    "name": "Insertion sort",
    "description": "Sort a small integer array using insertion sort",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_120_insertion_sort",
    "tags": ['sve2', 'sort'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_120_data {
  uint32_t n;
  int32_t *restrict data;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
// Inner loop
static void NOINLINE do_sort(struct loop_120_data *restrict input) {
  uint32_t n = input->n;
  int32_t *data = input->data;

  com_sort_insertion(n, data);
}
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Insertion sort
Purpose: Use of CMPLT instruction
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
EDGE_SIZES = [0, 1, 2, 255, 257]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [2000, 8000]
