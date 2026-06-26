# SVE2 timing: reference vs autovec (Graviton4)

Real SVE2 latency for both baseline solutions of every simd-loop, measured on a
**Graviton4 `c8g.large`** (Neoverse V2, 128-bit SVE2), clang++-18,
`scripts/bench_timing_sweep.py`. `min` latency on the largest (perf) workload, µs.

- **reference** = scalar, `-O2 -fno-vectorize -fno-slp-vectorize`
- **autovec**   = same source, `-O3 -march=native` (→ `armv9-a+sve2` on Graviton)
- **speedup**   = reference / autovec (how much the compiler's auto-vectorization wins)

Date: 2026-06-26 (re-run; instance `c8g.large` still live for follow-up agent runs).

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
loop          reference us   autovec us   speedup  status
loop_001          11586.91     11550.13     1.00x  ok
loop_002           6717.95      3829.74     1.75x  ok
loop_003          23581.25     23359.25     1.01x  ok
loop_004          19024.34     15145.92     1.26x  ok
loop_005             15.30        15.34     1.00x  ok
loop_006             31.04        30.81     1.01x  ok
loop_008          23068.37     22989.93     1.00x  ok
loop_010           9969.95     10013.62     1.00x  ok
loop_024          10305.95      1998.64     5.16x  ok
loop_027         144466.01    141920.11     1.02x  ok
loop_028          29256.82     28793.58     1.02x  ok
loop_029          25037.63     25012.07     1.00x  ok
loop_032           9527.77      8861.44     1.08x  ok
loop_033          23211.42     23061.86     1.01x  ok
loop_034              3.94         3.96     0.99x  ok
loop_035           8494.91      4374.56     1.94x  ok
loop_037             37.78        29.91     1.26x  ok
loop_038            301.11        11.70    25.74x  ok
loop_101              4.69         1.13     4.15x  ok
loop_102             26.65        27.04     0.99x  ok
loop_103             14.49        14.58     0.99x  ok
loop_104             26.18        26.11     1.00x  ok
loop_105          12030.63     12026.87     1.00x  ok
loop_106            307.09       307.03     1.00x  ok
loop_108          36996.22     10217.74     3.62x  ok
loop_109             38.17        12.51     3.05x  ok
loop_110             58.75        20.79     2.83x  ok
loop_112             45.67        32.24     1.42x  ok
loop_113          13159.23      8874.48     1.48x  ok
loop_114           4811.80       916.78     5.25x  ok
loop_120             50.45        50.34     1.00x  ok
loop_121           1270.03      1222.60     1.04x  ok
loop_122             23.52        23.30     1.01x  ok
loop_123            480.70       477.37     1.01x  ok
loop_124          28985.97     31264.44     0.93x  ok
loop_126             14.90        14.92     1.00x  ok
loop_127              5.49         5.51     1.00x  ok
loop_128           9074.30      4327.49     2.10x  ok
loop_130            123.68       124.78     0.99x  ok
loop_135             98.91        98.97     1.00x  ok
loop_216             52.34        52.26     1.00x  ok
loop_217             26.54         5.80     4.58x  ok
loop_218             54.20        54.04     1.00x  ok
loop_219             30.68        29.85     1.03x  ok
loop_220             41.48        37.15     1.12x  ok
loop_221             41.52        39.63     1.05x  ok
loop_223             69.97        70.01     1.00x  ok
geomean: 1.42x over 47 loops
```
(Run-to-run noise is sub-percent on most loops; loop_124 radix sort is the one
mild autovec regression — 0.93×.)

## Notes
- Reproduce: `python eval/provision.py --isa sve2 --codebase ""`, then on the
  remote `cd ~/arm-bench && python3 scripts/gen_simd_loop_harness.py &&
  python3 scripts/bench_timing_sweep.py`; tear down with
  `python eval/provision.py --teardown`.
- These are the *baseline* numbers. An agent-written SVE2 kernel should beat both
  — `autovec` is the "free compiler" ceiling to clear.
