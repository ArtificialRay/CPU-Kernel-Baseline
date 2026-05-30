"""
loop_012: Particle motion

Purpose: Use of FP compare instructions

ISA target: SVE2 on Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)
"""

METADATA = {
    "id": "loop_012",
    "num": "012",
    "name": "Particle motion",
    "description": "Advance 3D particle positions by a time step with modular boundary wrapping",
    "isa_target": "sve2",
    "instance_type": "c7g.large",
    "dir_name": "loop_012_particle_motion",
    "tags": ['sve2'],
}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
struct loop_012_data {
  int64_t step;
  double direction[3];
  int64_t magnitude[3];
  double *restrict vx;
  double *restrict vy;
  double *restrict vz;
  double *restrict nx;
  double *restrict ny;
  double *restrict nz;
  uint64_t n;
};
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
// Helper: compute next particle position with modular wrapping.
// step: time step, direction: base position, magnitude: modulus, value: current velocity
static double next_pos(int64_t step, double direction, int64_t magnitude,
                       double value) {
  double pos = direction;
  double vabs = value < 0 ? -value : value;
  double vabsstep =
      (vabs * step) - ((int64_t)(vabs * step) / magnitude) * magnitude;
  double vstep = value < 0 ? -vabsstep : vabsstep;
  pos -= vstep;
  return pos < 0.0 ? pos + magnitude : pos >= magnitude ? pos - magnitude : pos;
}

static void inner_loop_012(struct loop_012_data *restrict data) {
  int64_t step = data->step;
  double *direction = data->direction;
  int64_t *magnitude = data->magnitude;
  double *vx = data->vx;
  double *vy = data->vy;
  double *vz = data->vz;
  double *nx = data->nx;
  double *ny = data->ny;
  double *nz = data->nz;
  uint64_t n = data->n;

  for (int p = 0; p < n; p++) {
    nx[p] = next_pos(step, direction[0], magnitude[0], vx[p]);
    ny[p] = next_pos(step, direction[1], magnitude[1], vy[p]);
    nz[p] = next_pos(step, direction[2], magnitude[2], vz[p]);
  }
}
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {isa_desc}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """Problem: Particle motion
Purpose: Use of FP compare instructions
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
EDGE_SIZES = [0, 1, 3, 999, 1001]

# Input sizes for performance measurement at submit time and via perf() tool.
# Scored against the largest size. Empty list = skip.
PERF_SIZES = [200000, 500000]
PERF_SIZES_C8G = [1000000, 4000000]  # DRAM-bound on Graviton4 (64MB L3)
