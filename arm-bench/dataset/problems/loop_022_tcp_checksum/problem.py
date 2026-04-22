"""
loop_022: TCP checksum

Purpose: Use of simd instructions for misaligned accesses

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_022",
    "num": "022",
    "name": "TCP checksum",
    "description": "Compute one's complement checksum across a buffer of variable-length TCP packets",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_022_tcp_checksum",
    "tags": ['sve2'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_022_data {
  uint8_t *p;
  uint8_t *lmt;
  uint32_t checksum;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
# Note: SCALAR_CODE shows the full inner_loop_022 (what the candidate must replace).
# The helper tcp_checksum can be redefined or inlined in the candidate.
SCALAR_CODE = r"""
// Helper: one's complement checksum over `len` bytes (always even).
// You must define this (or inline it) in your candidate — it is not
// available from the harness when HAVE_CANDIDATE is set.
static uint16_t tcp_checksum(uint8_t *data, uint16_t len) {
  uint64_t sum = 0;
  uint8_t *lmt = data + len;
  for (uint8_t *p = data; p < lmt; p += 2) {
    uint16_t word = *(uint16_t *)p;
    sum += word;
  }
  return ~((sum & 0xffff) + (sum >> 16));
}

// Outer loop: walk a byte buffer of variable-length TCP packets.
// Byte layout per packet: byte[0] is unused, bytes[1..2] hold the packet
// length (uint16_t, little-endian, always even), followed by `length` bytes
// of payload.  p advances by *plength each iteration.
static void inner_loop_022(struct loop_022_data *restrict data) {
  uint8_t *p = data->p;
  uint8_t *lmt = data->lmt;
  uint32_t res = 0;
  while (p < lmt) {
    uint16_t *plength = (void *)(p + 1);
    uint16_t length = *plength & 0xfe;
    uint16_t checksum = tcp_checksum(p, length);
    p += *plength;
    res += 1;
    res ^= checksum << 16;
  }
  data->checksum = res;
}
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: TCP checksum
Purpose: Use of simd instructions for misaligned accesses
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
EDGE_SIZES = [0, 1, 3, 19999, 20001]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [8000000, 32000000]
