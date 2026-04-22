"""
loop_009: Pointer chasing

Purpose: Use of CTERM and BRK instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_009",
    "num": "009",
    "name": "Pointer chasing",
    "description": "Traverse a linked list and XOR-accumulate the payload from each node",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_009_pointer_chasing",
    "tags": ['sve2'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_009_data {
  node_t *nodes;
  uint64_t res;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void inner_loop_009(struct loop_009_data *restrict data) {
  node_t *nodes = data->nodes;

  uint64_t res = 0;
  for (node_t *p = nodes; p != NULL; p = p->next) {
    res ^= p->payload ^ p->payload2;
  }
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Pointer chasing
Purpose: Use of CTERM and BRK instructions
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
EDGE_SIZES = [2, 7, 999, 1001]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [100000, 400000]
PERF_SIZES_C8G = [400000, 2000000]  # DRAM-bound on Graviton4 (64MB L3)
