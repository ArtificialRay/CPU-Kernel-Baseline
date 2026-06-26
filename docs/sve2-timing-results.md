# SVE2 timing: reference vs autovec (Graviton4)

Real SVE2 latency for both baseline solutions of every simd-loop, measured on a
**Graviton4 `c8g.large`** (Neoverse V2, 128-bit SVE2), clang++-18,
`scripts/bench_timing_sweep.py`. `min` latency on the largest (perf) workload, µs.

- **reference** = scalar, `-O2 -fno-vectorize -fno-slp-vectorize`
- **autovec**   = same source, `-O3 -march=native` (→ `armv9-a+sve2` on Graviton)
- **speedup**   = reference / autovec (how much the compiler's auto-vectorization wins)

Date: 2026-06-26. Instance torn down after the run.

## Headline
**Geomean autovec/reference speedup = 1.42× across 47 loops.** Auto-vectorization
wins big on element-wise / data-parallel kernels and is ~neutral on strict FP
reductions (clang won't reassociate them without `-ffast-math`, so both build to
similar scalar code).

## Top speedups (compiler auto-vectorizes to SVE2)
| Loop | What | speedup |
|------|------|--------:|
| 038 | fp16 stencil convolution | **26.5×** |
| 114 | auto-correlation (int16) | 5.22× |
| 024 | sum of abs diffs (uint8) | 5.16× |
| 217 | int8 row-major GEMV | 4.58× |
| 101 | pixel upscale (uint8) | 4.20× |
| 108 | RGBA→luminance | 3.59× |
| 109 | cuint32 complex add | 3.15× |
| 110 | cint8 complex dot | 2.83× |
| 128 | uint32 aliased add | 2.18× |
| 035 | fp32 add | 2.01× |
| 002 | uint32 inner product | 1.66× |

## Near-neutral (~1.0×)
Strict FP reductions (001/003/008/010/033 dot products, 027 sqrt, 029 ldexp) and
already-tight loops — autovec ≈ reference because the reduction can't be
reassociated without fast-math, or the scalar code is already optimal.

## Full table
```
loop          reference us   autovec us   speedup
loop_001          11496.09     11514.42     1.00x
loop_002           6521.86      3936.37     1.66x
loop_003          23235.07     23053.38     1.01x
loop_004          19760.60     14946.43     1.32x
loop_005             15.30        15.27     1.00x
loop_006             31.05        30.92     1.00x
loop_008          22987.98     22976.74     1.00x
loop_010           9964.61      9958.61     1.00x
loop_024          10292.12      1995.52     5.16x
loop_027         144233.92    140891.20     1.02x
loop_028          29161.96     28676.01     1.02x
loop_029          24891.99     24875.84     1.00x
loop_032           8646.02      8950.07     0.97x
loop_033          23184.93     23065.37     1.01x
loop_034              3.95         3.96     1.00x
loop_035           8451.63      4198.30     2.01x
loop_037             38.67        30.07     1.29x
loop_038            301.14        11.36    26.51x
loop_101              4.66         1.11     4.20x
loop_102             26.73        27.03     0.99x
loop_103             14.27        14.37     0.99x
loop_104             26.17        26.08     1.00x
loop_105          12034.69     12038.72     1.00x
loop_106            307.14       307.18     1.00x
loop_108          36917.49     10280.17     3.59x
loop_109             38.11        12.10     3.15x
loop_110             58.76        20.79     2.83x
loop_112             46.21        34.67     1.33x
loop_113          13257.23      8990.43     1.47x
loop_114           4804.84       920.38     5.22x
loop_120             49.82        50.48     0.99x
loop_121           1229.12      1236.16     0.99x
loop_122             23.75        23.34     1.02x
loop_123            481.00       478.78     1.00x
loop_124          28992.51     28890.24     1.00x
loop_126             14.89        14.90     1.00x
loop_127              5.48         5.48     1.00x
loop_128           9079.79      4158.88     2.18x
loop_130            123.55       124.72     0.99x
loop_135             98.90        98.99     1.00x
loop_216             52.46        52.27     1.00x
loop_217             26.55         5.80     4.58x
loop_218             54.24        53.90     1.01x
loop_219             31.30        30.27     1.03x
loop_220             41.48        37.16     1.12x
loop_221             41.51        39.65     1.05x
loop_223             70.17        70.03     1.00x
geomean: 1.42x over 47 loops
```

## Notes
- Reproduce: `python eval/provision.py --isa sve2 --codebase ""`, then on the
  remote `cd ~/arm-bench && python3 scripts/gen_simd_loop_harness.py &&
  python3 scripts/bench_timing_sweep.py`; tear down with
  `python eval/provision.py --teardown`.
- These are the *baseline* numbers. An agent-written SVE2 kernel should beat both
  — `autovec` is the "free compiler" ceiling to clear.
