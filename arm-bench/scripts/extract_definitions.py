"""scripts/extract_definitions.py — Migrate conv2d shapes from tests/ to definitions/.

PHASE2.md deliverables #3 + #4 in one pass:

  1. Parse every EXPECT_MATCH(...) call in tests/ncnn/candidate/convolution.cpp
     of the form:
         EXPECT_MATCH(run_conv2d, run_ref_conv2d,
                      in_c, out_c, H, W, kh, kw, sh, sw, pad_h, pad_w
                      [, dh, dw [, with_bias]]);

  2. Bucket by (kh, kw, sh, sw, dh, dw, c_in, c_out) → one Definition per
     unique tuple at definitions/conv/<def_name>.json.

  3. For each bucket, dedupe the (N, H, W, pad_top, pad_left, with_bias) tuples
     and write them as workload points to workloads/conv/<def_name>.jsonl.

  4. For each Definition, stamp one baseline-ncnn-arm Solution at
     solutions/ncnn/baseline-ncnn-arm/conv2d/<def_name>.json whose sources
     embed bench/templates/baseline_ncnn_arm_conv2d_kernel.cpp verbatim.

Idempotent: re-runs overwrite definitions/, workloads/, and baseline solutions/.
Other authors' solutions/ are untouched.

Usage (from arm-bench/):
    python -m scripts.extract_definitions [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Paths ────────────────────────────────────────────────────────────────────

ARM_BENCH_ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE_ROOT = ARM_BENCH_ROOT / "bench-trace"
TESTS_CPP = ARM_BENCH_ROOT / "tests" / "ncnn" / "candidate" / "convolution.cpp"
DEFINITIONS_DIR = WAREHOUSE_ROOT / "definitions" / "conv"
WORKLOADS_DIR = WAREHOUSE_ROOT / "workloads" / "conv"
SOLUTIONS_DIR = (
    WAREHOUSE_ROOT / "solutions" / "ncnn" / "baseline-ncnn-arm" / "conv2d"
)
TEMPLATE_PATH = (
    ARM_BENCH_ROOT / "bench" / "templates" / "baseline_ncnn_arm_conv2d_kernel.cpp"
)

# ── Reference Python (Definition.reference) ─────────────────────────────────

REFERENCE_PY = """\
import torch
import torch.nn.functional as F


def run(input, weight, pad_top, pad_left, activation_type, with_bias):
    x = torch.from_numpy(input)
    w = torch.from_numpy(weight)
    y = F.conv2d(x, w, None,
                 stride=({sh}, {sw}),
                 padding=(int(pad_top), int(pad_left)),
                 dilation=({dh}, {dw}))
    return y
"""

# ── Parser ──────────────────────────────────────────────────────────────────

# Match EXPECT_MATCH(run_conv2d, run_ref_conv2d, ...args...);
EXPECT_RE = re.compile(
    r"EXPECT_MATCH\s*\(\s*run_conv2d\s*,\s*run_ref_conv2d\s*,(.*?)\)\s*;",
    re.DOTALL,
)


@dataclass(frozen=True)
class Shape:
    in_c: int
    out_c: int
    H: int
    W: int
    kh: int
    kw: int
    sh: int
    sw: int
    pad_h: int
    pad_w: int
    dh: int
    dw: int
    with_bias: bool


def parse_args(arglist: str) -> Optional[Shape]:
    """Parse the comma-separated args inside an EXPECT_MATCH(...). Returns None
    if the row can't be parsed (e.g. comment-out, weird literal)."""
    # Strip comments and whitespace, split on commas at top level
    cleaned = re.sub(r"//.*", "", arglist)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    tokens = [t.strip() for t in cleaned.split(",")]
    tokens = [t for t in tokens if t]
    if len(tokens) < 10:
        return None
    try:
        ints = [int(t) for t in tokens[:10]]
    except ValueError:
        return None
    in_c, out_c, H, W, kh, kw, sh, sw, pad_h, pad_w = ints
    dh, dw = 1, 1
    with_bias = False
    if len(tokens) >= 12:
        try:
            dh = int(tokens[10])
            dw = int(tokens[11])
        except ValueError:
            return None
    if len(tokens) >= 13:
        last = tokens[-1].lower()
        if last in ("true", "1"):
            with_bias = True
        elif last in ("false", "0"):
            with_bias = False
        # else: the last token might still be dh/dw; nothing to do
    return Shape(in_c, out_c, H, W, kh, kw, sh, sw, pad_h, pad_w, dh, dw, with_bias)


# ── Definition / Workload / Solution emitters ──────────────────────────────

def def_name(s: Shape) -> str:
    """Naming: conv2d_kh{Kh}_kw{Kw}_sh{Sh}_sw{Sw}_dh{Dh}_dw{Dw}_c{C_in}_c{C_out}."""
    return (
        f"conv2d"
        f"_kh{s.kh}_kw{s.kw}"
        f"_sh{s.sh}_sw{s.sw}"
        f"_dh{s.dh}_dw{s.dw}"
        f"_c{s.in_c}_c{s.out_c}"
    )


def workload_key(s: Shape) -> Tuple[int, int, int, int, int, bool]:
    return (1, s.H, s.W, s.pad_h, s.pad_w, s.with_bias)


def build_definition(s: Shape) -> Dict:
    """Construct the Definition JSON dict for one (kh, kw, sh, sw, dh, dw, c_in, c_out) tuple.

    Pins kernel/stride/dilation/channels as const axes; N/H/W are var axes;
    H_out/W_out are derived var axes (parent=N). Pad / activation_type /
    with_bias are scalar inputs (shape=null) — the reference reads them via
    its kwargs, and the runner forwards them to the harness via _scalar_args_for.
    """
    reference = REFERENCE_PY.format(sh=s.sh, sw=s.sw, dh=s.dh, dw=s.dw)
    return {
        "name": def_name(s),
        "op_type": "conv2d",
        "description": (
            f"2D convolution: {s.kh}x{s.kw} kernel, stride ({s.sh},{s.sw}), "
            f"dilation ({s.dh},{s.dw}), {s.in_c}->{s.out_c} channels. "
            f"Extracted from tests/ncnn/candidate/convolution.cpp."
        ),
        "tags": ["status:phase2", "isa:sve"],
        "axes": {
            "N":     {"type": "var"},
            "H":     {"type": "var", "parent": "N"},
            "W":     {"type": "var", "parent": "N"},
            "H_out": {"type": "var", "parent": "N",
                      "description": "Derived: (H + 2*pad_h - (Kh-1)*Dh - 1) / Sh + 1"},
            "W_out": {"type": "var", "parent": "N",
                      "description": "Derived: (W + 2*pad_w - (Kw-1)*Dw - 1) / Sw + 1"},
            "C_in":  {"type": "const", "value": s.in_c},
            "C_out": {"type": "const", "value": s.out_c},
            "Kh":    {"type": "const", "value": s.kh},
            "Kw":    {"type": "const", "value": s.kw},
            "Sh":    {"type": "const", "value": s.sh},
            "Sw":    {"type": "const", "value": s.sw},
            "Dh":    {"type": "const", "value": s.dh},
            "Dw":    {"type": "const", "value": s.dw},
        },
        "inputs": {
            "input":           {"shape": ["N", "C_in", "H", "W"],       "dtype": "float32"},
            "weight":          {"shape": ["C_out", "C_in", "Kh", "Kw"], "dtype": "float32"},
            "pad_top":         {"shape": None, "dtype": "int32"},
            "pad_left":        {"shape": None, "dtype": "int32"},
            "activation_type": {"shape": None, "dtype": "int32"},
            "with_bias":       {"shape": None, "dtype": "int32"},
        },
        "outputs": {
            "output": {"shape": ["N", "C_out", "H_out", "W_out"], "dtype": "float32"},
        },
        "constraints": [
            "H_out == (H + 2*pad_top - (Kh-1)*Dh - 1) / Sh + 1",
            "W_out == (W + 2*pad_left - (Kw-1)*Dw - 1) / Sw + 1",
        ],
        "reference": reference,
    }


def build_workload(s: Shape, source_tag: str) -> Dict:
    """One workload point for `s`. Pad + with_bias go into scalar_inputs."""
    return {
        "axes": {"N": 1, "H": s.H, "W": s.W},
        "scalar_inputs": {
            "pad_top": s.pad_h,
            "pad_left": s.pad_w,
            "activation_type": 0,
            "with_bias": int(s.with_bias),
        },
        "uuid": f"H{s.H}_W{s.W}_padh{s.pad_h}_padw{s.pad_w}_b{int(s.with_bias)}",
        "tags": {"from": source_tag},
    }


def build_baseline_solution(d_name: str, template_src: str) -> Dict:
    """Build the Solution JSON for `baseline-ncnn-arm` against Definition `d_name`."""
    return {
        "name": f"baseline-ncnn-arm_{d_name}",
        "definition": d_name,
        "dataset": "ncnn",
        "author": "baseline-ncnn-arm",
        "description": (
            f"ncnn::Convolution_arm wrapper for {d_name}. "
            f"Same kernel.cpp content across all Definitions; const params "
            f"come from the Definition. Times create_pipeline + forward "
            f"(matches today's baselines/ncnn.json semantics)."
        ),
        "spec": {
            "language": "cpp",
            "target_hardware": ["graviton3", "aarch64-sve"],
            "entry_point": "kernel.cpp::convolution_kernel",
            "dependencies": ["ncnn", "ncnn_arm_heavy", "openmp"],
            "isa_features": ["sve"],
            "compile_flags": [
                "-O3",
                "-march=armv8.2-a+fp16+dotprod+sve",
                "-fopenmp",
            ],
            "link_flags": [],
        },
        "sources": [
            {"path": "kernel.cpp", "content": template_src},
        ],
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse + report counts only; do not write files.")
    parser.add_argument("--max-defs-warn", type=int, default=30,
                        help="Warn if more than N distinct Definitions are emitted.")
    args = parser.parse_args()

    if not TESTS_CPP.exists():
        print(f"ERROR: {TESTS_CPP} not found", file=sys.stderr)
        return 1
    if not TEMPLATE_PATH.exists():
        print(f"ERROR: kernel template not found: {TEMPLATE_PATH}", file=sys.stderr)
        return 1

    src = TESTS_CPP.read_text()
    shapes: List[Shape] = []
    for m in EXPECT_RE.finditer(src):
        s = parse_args(m.group(1))
        if s is None:
            print(f"WARN: skipped unparseable row: {m.group(0)[:80]}...", file=sys.stderr)
            continue
        shapes.append(s)

    if not shapes:
        print("ERROR: no EXPECT_MATCH(run_conv2d, ...) rows found", file=sys.stderr)
        return 1

    # Bucket by (kh, kw, sh, sw, dh, dw, c_in, c_out)
    buckets: Dict[Tuple, List[Shape]] = defaultdict(list)
    for s in shapes:
        buckets[(s.kh, s.kw, s.sh, s.sw, s.dh, s.dw, s.in_c, s.out_c)].append(s)

    print(f"Parsed {len(shapes)} EXPECT_MATCH rows → {len(buckets)} distinct Definitions")
    if len(buckets) > args.max_defs_warn:
        print(f"WARN: {len(buckets)} Definitions exceeds --max-defs-warn "
              f"({args.max_defs_warn}); revisit specialization keys?",
              file=sys.stderr)

    if args.dry_run:
        for key in sorted(buckets):
            d_name = def_name(buckets[key][0])
            print(f"  {d_name}  ({len(buckets[key])} workload points)")
        return 0

    template_src = TEMPLATE_PATH.read_text()

    DEFINITIONS_DIR.mkdir(parents=True, exist_ok=True)
    WORKLOADS_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

    n_defs = n_wls = n_sols = 0
    for key, group in sorted(buckets.items()):
        # Dedupe workloads in the bucket
        wl_seen = set()
        wl_unique: List[Shape] = []
        for s in group:
            k = workload_key(s)
            if k in wl_seen:
                continue
            wl_seen.add(k)
            wl_unique.append(s)

        d_name = def_name(group[0])
        d_obj = build_definition(group[0])
        (DEFINITIONS_DIR / f"{d_name}.json").write_text(
            json.dumps(d_obj, indent=2) + "\n"
        )
        n_defs += 1

        wl_path = WORKLOADS_DIR / f"{d_name}.jsonl"
        with wl_path.open("w") as f:
            for s in wl_unique:
                f.write(json.dumps(build_workload(s, "tests/ncnn/candidate/convolution.cpp")) + "\n")
                n_wls += 1

        sol_obj = build_baseline_solution(d_name, template_src)
        (SOLUTIONS_DIR / f"{d_name}.json").write_text(
            json.dumps(sol_obj, indent=2) + "\n"
        )
        n_sols += 1

        print(f"  {d_name:<55}  workloads={len(wl_unique):>3}")

    print(f"\nWrote {n_defs} definitions, {n_wls} workloads, {n_sols} baseline solutions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
