"""
loop_123: Bitonic mergesort

Purpose: Use of CMPGT with SEL instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_123",
    "num": "123",
    "name": "Bitonic mergesort",
    "description": "Merge two sorted halves using a bitonic compare-and-swap network",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_123_bitonic_mergesort",
    "tags": ['sve2', 'sort'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_123_data {
  uint32_t n;
  int32_t *restrict data;
  int32_t *restrict temp;
  uint32_t *restrict block_sizes;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
// Implementation

static void bitonic_merge(uint32_t n, int32_t *restrict a, int d) {
  if (n <= 1) return;
  uint32_t k = n / 2;
  for (uint32_t i = 0; i < k; i++) {
    if (d == (int)(a[i] > a[i + k])) swap_32(&a[i], &a[i + k]);
  }
  bitonic_merge(k, a, d);
  bitonic_merge(k, a + k, d);
}

static void bitonic_sort(uint32_t n, int32_t *restrict a, int d) {
  if (n <= 1) return;
  uint32_t k = n / 2;
  bitonic_sort(k, a, 1);
  bitonic_sort(k, a + k, 0);
  bitonic_merge(n, a, d);
}

static void NOINLINE do_sort(struct loop_123_data *restrict input) {
  uint32_t n = input->n;
  int32_t *data = input->data;

  bitonic_sort(n, data, 1);
} //Implementation
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Bitonic mergesort
Purpose: Use of CMPGT with SEL instructions
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
EDGE_SIZES = [8, 16, 32, 128, 512]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [8192, 65536]
