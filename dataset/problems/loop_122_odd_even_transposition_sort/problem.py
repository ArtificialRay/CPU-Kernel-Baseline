"""
loop_122: Odd-Even transposition sort

Purpose: Use of CMPLT with SEL instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_122",
    "num": "122",
    "name": "Odd-Even transposition sort",
    "description": "Sort element pairs using odd-even transposition compare-and-swap",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_122_odd_even_transposition_sort",
    "tags": ['sve2', 'sort'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_122_data {
  uint32_t n;
  int32_t *restrict data;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
static void NOINLINE do_sort(struct loop_122_data *restrict input) {
  uint32_t n = input->n;
  int32_t *data = input->data;

  uint32_t i;
  bool sorted;
  do {
    sorted = true;
    for (i = 1; i < n; i += 2) {
      if (data[i - 1] > data[i]) {
        swap_32(&data[i - 1], &data[i]);
        sorted = false;
      }
    }
    for (i = 1; i < n - 1; i += 2) {
      if (data[i + 1] < data[i]) {
        swap_32(&data[i + 1], &data[i]);
        sorted = false;
      }
    }
  } while (!sorted);
}
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Odd-Even transposition sort
Purpose: Use of CMPLT with SEL instructions
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
EDGE_SIZES = [1, 2, 8, 255, 257]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [1000, 4000]
