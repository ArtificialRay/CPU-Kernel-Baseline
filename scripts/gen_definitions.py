#!/usr/bin/env python3
"""Generate Definition JSON and Workload JSONL files for conv1d, conv2d_depthwise,
deconv2d, and deconv2d_depthwise from the EXPECT_MATCH calls in the candidate test files.

Run from arm-bench/:
    python scripts/gen_definitions.py
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ── Repo layout ───────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO / "tests" / "ncnn" / "candidate"
DEFS_DIR  = REPO / "bench-trace" / "definitions"
WLS_DIR   = REPO / "bench-trace" / "workloads"


# ── Reference code templates ─────────────────────────────────────────────────
# Each template is a Python format string; brace-escaped literal braces use {{/}}.
# The result is stored as a single JSON string with \n separating lines.

def _ref_conv1d(kw: int, sw: int, dw: int, pad: int) -> str:
    return (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def run(input, weight, bias):\n"
        "    x = torch.from_numpy(input).unsqueeze(0)\n"
        "    w = torch.from_numpy(weight)\n"
        "    b = torch.from_numpy(bias)\n"
        f"    return F.conv1d(x, w, b, stride={sw}, padding={pad}, dilation={dw}).squeeze(0).numpy()\n"
    )


def _ref_conv2d_depthwise(kh: int, kw: int, sh: int, sw: int, dh: int, dw: int,
                           pad_top: int, pad_left: int) -> str:
    return (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def run(input, weight, bias):\n"
        "    n, c = input.shape[0], input.shape[1]\n"
        "    x = torch.from_numpy(input)\n"
        "    w = torch.from_numpy(weight).unsqueeze(1)\n"
        "    b = torch.from_numpy(bias)\n"
        f"    return F.conv2d(x, w, b, stride=({sh}, {sw}), padding=({pad_top}, {pad_left}), "
        f"dilation=({dh}, {dw}), groups=c).numpy()\n"
    )


def _ref_deconv2d(kh: int, kw: int, sh: int, sw: int) -> str:
    return (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def run(input, weight, bias):\n"
        "    x = torch.from_numpy(input)\n"
        "    w = torch.from_numpy(weight).permute(1, 0, 2, 3)\n"
        "    b = torch.from_numpy(bias)\n"
        f"    return F.conv_transpose2d(x, w, b, stride=({sh}, {sw}), padding=(0, 0), "
        f"dilation=(1, 1)).numpy()\n"
    )


def _ref_deconv2d_depthwise(kh: int, kw: int, sh: int, sw: int) -> str:
    return (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def run(input, weight, bias):\n"
        "    n, c = input.shape[0], input.shape[1]\n"
        "    x = torch.from_numpy(input)\n"
        "    w = torch.from_numpy(weight).unsqueeze(1)\n"
        "    b = torch.from_numpy(bias)\n"
        f"    return F.conv_transpose2d(x, w, b, stride=({sh}, {sw}), padding=(0, 0), "
        f"dilation=(1, 1), groups=c).numpy()\n"
    )


# ── EXPECT_MATCH parser ───────────────────────────────────────────────────────

def _parse_expect_match(src: str) -> List[Tuple[int, ...]]:
    """Extract the integer argument lists from all EXPECT_MATCH calls in src."""
    # Find everything inside EXPECT_MATCH(run_fn, ref_fn, <args...>)
    pattern = re.compile(
        r'EXPECT_MATCH\s*\(\s*\w+\s*,\s*\w+\s*,\s*([^)]+)\)',
        re.MULTILINE
    )
    results = []
    for m in pattern.finditer(src):
        raw_args = m.group(1)
        # Split on commas, strip whitespace, filter out `true`/`false` bool args
        parts = [p.strip() for p in raw_args.split(',')]
        int_parts = []
        for p in parts:
            if p in ('true', 'false'):
                break  # with_bias is always last and optional; stop here
            try:
                int_parts.append(int(p))
            except ValueError:
                break
        if int_parts:
            results.append(tuple(int_parts))
    return results


# ── Definition / Workload builders ────────────────────────────────────────────

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, lines: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(l) for l in lines) + "\n",
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────
# conv1d
# Args from EXPECT_MATCH: (in_c, out_c, in_w, kw, stride_w, pad_left[, dil_w])
# ─────────────────────────────────────────────────────────────────────────────

def _gen_conv1d() -> None:
    src = (TESTS_DIR / "convolution1d.cpp").read_text()
    calls = _parse_expect_match(src)

    # Group by (out_c, kw, sw, dw, pad_left) → list of (in_c, in_w)
    groups: Dict[Tuple, List[Tuple[int, int, int]]] = {}
    for args in calls:
        in_c, out_c, in_w, kw, sw, pad_left = args[0], args[1], args[2], args[3], args[4], args[5]
        dw = args[6] if len(args) > 6 else 1
        key = (out_c, kw, sw, dw, pad_left)
        groups.setdefault(key, []).append((in_c, in_w))

    for (out_c, kw, sw, dw, pad_left), variants in groups.items():
        # Use first variant's in_c as the const axis (all variants share the same
        # in_c actually? No — they vary. We need per-definition const in_c.
        # Group further by in_c since C_in is const in the Definition.
        by_cin: Dict[int, List[int]] = {}
        for in_c, in_w in variants:
            by_cin.setdefault(in_c, []).append(in_w)

        for in_c, widths in by_cin.items():
            name = f"conv1d_kw{kw}_sw{sw}_dw{dw}_p{pad_left}_cin{in_c}_cout{out_c}"
            ref = _ref_conv1d(kw, sw, dw, pad_left)

            defn = {
                "name": name,
                "op_type": "conv1d",
                "description": (
                    f"1D convolution: kw={kw} stride={sw} dilation={dw} pad={pad_left} "
                    f"C_in={in_c} C_out={out_c}"
                ),
                "tags": ["status:draft"],
                "axes": {
                    "N":     {"type": "var"},
                    "W":     {"type": "var", "parent": "N"},
                    "W_out": {"type": "var", "parent": "N",
                              "description": "Derived: (W + 2*pad - dw*(kw-1) - 1) / sw + 1"},
                    "C_in":  {"type": "const", "value": in_c},
                    "C_out": {"type": "const", "value": out_c},
                    "Kw":    {"type": "const", "value": kw},
                    "Sw":    {"type": "const", "value": sw},
                    "Dw":    {"type": "const", "value": dw},
                },
                "inputs": {
                    # conv1d is modeled natively 2D (ncnn::Convolution1D has no
                    # batch dim): input is (C_in, W), output (C_out, W_out). The
                    # reference adds/removes the batch dim via unsqueeze/squeeze.
                    "input":  {"shape": ["C_in", "W"], "dtype": "float32"},
                    "weight": {"shape": ["C_out", "C_in", "Kw"], "dtype": "float32"},
                    "bias":   {"shape": ["C_out"], "dtype": "float32"},
                },
                "outputs": {
                    "output": {"shape": ["C_out", "W_out"], "dtype": "float32"},
                },
                "constraints": [
                    f"W_out == (W + 2*{pad_left} - {dw}*(Kw-1) - 1) // Sw + 1",
                ],
                "reference": ref,
            }

            workload_lines = []
            for w in sorted(set(widths)):
                wl = {
                    "axes": {"N": 1, "W": w},
                    "scalar_inputs": {
                        "pad_left": pad_left,
                        "activation_type": 0,
                    },
                    "uuid": uuid.uuid4().hex,
                    "tags": {"from": "gen_definitions"},
                }
                workload_lines.append(wl)

            _write_json(DEFS_DIR / "conv1d" / f"{name}.json", defn)
            _write_jsonl(WLS_DIR / "conv1d" / f"{name}.jsonl", workload_lines)
            print(f"  [conv1d] {name}  ({len(workload_lines)} workloads)")


# ─────────────────────────────────────────────────────────────────────────────
# conv2d_depthwise
# Args: (c, in_h, in_w, kh, kw, stride_h, stride_w, pad_top, pad_left[, dil_h, dil_w])
# ─────────────────────────────────────────────────────────────────────────────

def _gen_conv2d_depthwise() -> None:
    src = (TESTS_DIR / "convolutiondepwise.cpp").read_text()
    calls = _parse_expect_match(src)

    # Group by (kh, kw, sh, sw, dh, dw, pad_top, pad_left) → list of (c, h, w)
    groups: Dict[Tuple, List[Tuple[int, int, int]]] = {}
    for args in calls:
        c, in_h, in_w, kh, kw, sh, sw, pad_top, pad_left = (
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]
        )
        dh = args[9]  if len(args) > 9  else 1
        dw = args[10] if len(args) > 10 else 1
        key = (kh, kw, sh, sw, dh, dw, pad_top, pad_left)
        groups.setdefault(key, []).append((c, in_h, in_w))

    for (kh, kw, sh, sw, dh, dw, pad_top, pad_left), variants in groups.items():
        # C is a const axis per definition (each unique C gets its own def here)
        by_c: Dict[int, List[Tuple[int, int]]] = {}
        for c, h, w in variants:
            by_c.setdefault(c, []).append((h, w))

        for c, hw_pairs in by_c.items():
            name = (
                f"conv2d_depthwise_kh{kh}_kw{kw}_sh{sh}_sw{sw}"
                f"_dh{dh}_dw{dw}_p{pad_top}_c{c}"
            )
            ref = _ref_conv2d_depthwise(kh, kw, sh, sw, dh, dw, pad_top, pad_left)

            defn = {
                "name": name,
                "op_type": "conv2d_depthwise",
                "description": (
                    f"Depthwise 2D conv: kh={kh} kw={kw} stride=({sh},{sw}) "
                    f"dilation=({dh},{dw}) pad=({pad_top},{pad_left}) C={c}"
                ),
                "tags": ["status:draft"],
                "axes": {
                    "N":     {"type": "var"},
                    "H":     {"type": "var", "parent": "N"},
                    "W":     {"type": "var", "parent": "N"},
                    "H_out": {"type": "var", "parent": "N",
                              "description": "Derived: (H + 2*pad_top - dh*(kh-1) - 1) / sh + 1"},
                    "W_out": {"type": "var", "parent": "N",
                              "description": "Derived: (W + 2*pad_left - dw*(kw-1) - 1) / sw + 1"},
                    "C":     {"type": "const", "value": c},
                    "Kh":    {"type": "const", "value": kh},
                    "Kw":    {"type": "const", "value": kw},
                    "Sh":    {"type": "const", "value": sh},
                    "Sw":    {"type": "const", "value": sw},
                    "Dh":    {"type": "const", "value": dh},
                    "Dw":    {"type": "const", "value": dw},
                },
                "inputs": {
                    "input":  {"shape": ["N", "C", "H", "W"], "dtype": "float32"},
                    "weight": {"shape": ["C", "Kh", "Kw"], "dtype": "float32"},
                    "bias":   {"shape": ["C"], "dtype": "float32"},
                },
                "outputs": {
                    "output": {"shape": ["N", "C", "H_out", "W_out"], "dtype": "float32"},
                },
                "constraints": [
                    f"H_out == (H + 2*{pad_top} - {dh}*(Kh-1) - 1) // Sh + 1",
                    f"W_out == (W + 2*{pad_left} - {dw}*(Kw-1) - 1) // Sw + 1",
                ],
                "reference": ref,
            }

            workload_lines = []
            for h, w in sorted(set(hw_pairs)):
                wl = {
                    "axes": {"N": 1, "H": h, "W": w},
                    "scalar_inputs": {
                        "pad_left": pad_left,
                        "pad_top": pad_top,
                        "activation_type": 0,
                    },
                    "uuid": uuid.uuid4().hex,
                    "tags": {"from": "gen_definitions"},
                }
                workload_lines.append(wl)

            _write_json(DEFS_DIR / "conv2d_depthwise" / f"{name}.json", defn)
            _write_jsonl(WLS_DIR / "conv2d_depthwise" / f"{name}.jsonl", workload_lines)
            print(f"  [conv2d_depthwise] {name}  ({len(workload_lines)} workloads)")


# ─────────────────────────────────────────────────────────────────────────────
# deconv2d
# Args: (in_c, out_c, in_h, in_w, kh, kw, stride_h, stride_w)
# dilation always 1 in these tests.
# ─────────────────────────────────────────────────────────────────────────────

def _gen_deconv2d() -> None:
    src = (TESTS_DIR / "deconvolution.cpp").read_text()
    calls = _parse_expect_match(src)

    # Group by (in_c, out_c, kh, kw, sh, sw) — dilation=1 always
    groups: Dict[Tuple, List[Tuple[int, int]]] = {}
    for args in calls:
        in_c, out_c, in_h, in_w, kh, kw, sh, sw = (
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]
        )
        key = (in_c, out_c, kh, kw, sh, sw)
        groups.setdefault(key, []).append((in_h, in_w))

    for (in_c, out_c, kh, kw, sh, sw), hw_pairs in groups.items():
        name = f"deconv2d_kh{kh}_kw{kw}_sh{sh}_sw{sw}_cin{in_c}_cout{out_c}"
        ref = _ref_deconv2d(kh, kw, sh, sw)

        defn = {
            "name": name,
            "op_type": "deconv2d",
            "description": (
                f"Transposed 2D conv: kh={kh} kw={kw} stride=({sh},{sw}) "
                f"dilation=(1,1) pad=0 C_in={in_c} C_out={out_c}"
            ),
            "tags": ["status:draft"],
            "axes": {
                "N":     {"type": "var"},
                "H":     {"type": "var", "parent": "N"},
                "W":     {"type": "var", "parent": "N"},
                "H_out": {"type": "var", "parent": "N",
                          "description": "Derived: (H - 1) * sh + kh"},
                "W_out": {"type": "var", "parent": "N",
                          "description": "Derived: (W - 1) * sw + kw"},
                "C_in":  {"type": "const", "value": in_c},
                "C_out": {"type": "const", "value": out_c},
                "Kh":    {"type": "const", "value": kh},
                "Kw":    {"type": "const", "value": kw},
                "Sh":    {"type": "const", "value": sh},
                "Sw":    {"type": "const", "value": sw},
                "Dh":    {"type": "const", "value": 1},
                "Dw":    {"type": "const", "value": 1},
            },
            "inputs": {
                "input":  {"shape": ["N", "C_in", "H", "W"], "dtype": "float32"},
                "weight": {"shape": ["C_out", "C_in", "Kh", "Kw"], "dtype": "float32"},
                "bias":   {"shape": ["C_out"], "dtype": "float32"},
            },
            "outputs": {
                "output": {"shape": ["N", "C_out", "H_out", "W_out"], "dtype": "float32"},
            },
            "constraints": [
                "H_out == (H - 1) * Sh + Kh",
                "W_out == (W - 1) * Sw + Kw",
            ],
            "reference": ref,
        }

        workload_lines = []
        for h, w in sorted(set(hw_pairs)):
            wl = {
                "axes": {"N": 1, "H": h, "W": w},
                "scalar_inputs": {
                    "activation_type": 0,
                },
                "uuid": uuid.uuid4().hex,
                "tags": {"from": "gen_definitions"},
            }
            workload_lines.append(wl)

        _write_json(DEFS_DIR / "deconv2d" / f"{name}.json", defn)
        _write_jsonl(WLS_DIR / "deconv2d" / f"{name}.jsonl", workload_lines)
        print(f"  [deconv2d] {name}  ({len(workload_lines)} workloads)")


# ─────────────────────────────────────────────────────────────────────────────
# deconv2d_depthwise
# Args: (c, in_h, in_w, kh, kw, stride_h, stride_w)
# ─────────────────────────────────────────────────────────────────────────────

def _gen_deconv2d_depthwise() -> None:
    src = (TESTS_DIR / "deconvolutiondepwise.cpp").read_text()
    calls = _parse_expect_match(src)

    # Group by (kh, kw, sh, sw) — dilation=1 always
    groups: Dict[Tuple, List[Tuple[int, int, int]]] = {}
    for args in calls:
        c, in_h, in_w, kh, kw, sh, sw = (
            args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        )
        key = (kh, kw, sh, sw)
        groups.setdefault(key, []).append((c, in_h, in_w))

    for (kh, kw, sh, sw), variants in groups.items():
        # C varies per workload for depthwise deconv (channels == in_c == out_c)
        # All spatial sizes and channels become workloads in the same definition
        name = f"deconv2d_depthwise_kh{kh}_kw{kw}_sh{sh}_sw{sw}"
        ref = _ref_deconv2d_depthwise(kh, kw, sh, sw)

        # Collect unique channel values to create one definition per channel count
        by_c: Dict[int, List[Tuple[int, int]]] = {}
        for c, h, w in variants:
            by_c.setdefault(c, []).append((h, w))

        for c, hw_pairs in by_c.items():
            def_name = f"{name}_c{c}"
            ref_code = _ref_deconv2d_depthwise(kh, kw, sh, sw)

            defn = {
                "name": def_name,
                "op_type": "deconv2d_depthwise",
                "description": (
                    f"Depthwise transposed 2D conv: kh={kh} kw={kw} stride=({sh},{sw}) "
                    f"dilation=(1,1) pad=0 C={c}"
                ),
                "tags": ["status:draft"],
                "axes": {
                    "N":     {"type": "var"},
                    "H":     {"type": "var", "parent": "N"},
                    "W":     {"type": "var", "parent": "N"},
                    "H_out": {"type": "var", "parent": "N",
                              "description": "Derived: (H - 1) * sh + kh"},
                    "W_out": {"type": "var", "parent": "N",
                              "description": "Derived: (W - 1) * sw + kw"},
                    "C":     {"type": "const", "value": c},
                    "Kh":    {"type": "const", "value": kh},
                    "Kw":    {"type": "const", "value": kw},
                    "Sh":    {"type": "const", "value": sh},
                    "Sw":    {"type": "const", "value": sw},
                    "Dh":    {"type": "const", "value": 1},
                    "Dw":    {"type": "const", "value": 1},
                },
                "inputs": {
                    "input":  {"shape": ["N", "C", "H", "W"], "dtype": "float32"},
                    "weight": {"shape": ["C", "Kh", "Kw"], "dtype": "float32"},
                    "bias":   {"shape": ["C"], "dtype": "float32"},
                },
                "outputs": {
                    "output": {"shape": ["N", "C", "H_out", "W_out"], "dtype": "float32"},
                },
                "constraints": [
                    "H_out == (H - 1) * Sh + Kh",
                    "W_out == (W - 1) * Sw + Kw",
                ],
                "reference": ref_code,
            }

            workload_lines = []
            for h, w in sorted(set(hw_pairs)):
                wl = {
                    "axes": {"N": 1, "H": h, "W": w},
                    "scalar_inputs": {
                        "activation_type": 0,
                    },
                    "uuid": uuid.uuid4().hex,
                    "tags": {"from": "gen_definitions"},
                }
                workload_lines.append(wl)

            _write_json(DEFS_DIR / "deconv2d_depthwise" / f"{def_name}.json", defn)
            _write_jsonl(WLS_DIR / "deconv2d_depthwise" / f"{def_name}.jsonl", workload_lines)
            print(f"  [deconv2d_depthwise] {def_name}  ({len(workload_lines)} workloads)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Generating definitions and workloads...")
    _gen_conv1d()
    _gen_conv2d_depthwise()
    _gen_deconv2d()
    _gen_deconv2d_depthwise()
    print("Done.")


if __name__ == "__main__":
    main()
