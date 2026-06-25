#!/usr/bin/env python3
"""Generate Definition JSON and Workload JSONL files for all 5 ncnn op types.

Run from arm-bench/:
    python scripts/gen_definitions.py

Produces 17 definitions total:
  conv2d (6)               bench-trace/definitions/conv/
  conv1d (2)               bench-trace/definitions/conv1d/
  conv2d_depthwise (3)     bench-trace/definitions/conv2d_depthwise/
  deconv2d (4)             bench-trace/definitions/deconv2d/
  deconv2d_depthwise (2)   bench-trace/definitions/deconv2d_depthwise/

Channel (C_in / C) is a workload-level var axis — one definition covers all
channel widths.  Odd spatial inputs (H=113, H=57, etc.) account for ~17-20 % of
total workloads to stress stride=2 / deconv boundary logic.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Repo layout ───────────────────────────────────────────────────────────────

REPO     = Path(__file__).resolve().parent.parent
DEFS_DIR = REPO / "bench-trace" / "definitions"
WLS_DIR  = REPO / "bench-trace" / "workloads"


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, lines: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n",
        encoding="utf-8",
    )


def _wl(axes: Dict[str, int], inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "axes": axes,
        "inputs": inputs,
        "uuid": uuid.uuid4().hex,
        "tags": {"from": "gen_definitions"},
    }


def _rand(*names: str) -> Dict[str, Any]:
    return {n: {"type": "random"} for n in names}


# ── Reference code builders ───────────────────────────────────────────────────

def _ref_conv2d(sh: int, sw: int, dh: int, dw: int,
                pad_top: int, pad_left: int) -> str:
    return (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def run(input, weight, activation_type, with_bias):\n"
        "    x = torch.from_numpy(input)\n"
        "    w = torch.from_numpy(weight)\n"
        f"    return F.conv2d(x, w, None, stride=({sh}, {sw}), "
        f"padding=({pad_top}, {pad_left}), dilation=({dh}, {dw})).numpy()\n"
    )


def _ref_conv1d(sw: int, dw: int, pad: int) -> str:
    return (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def run(input, weight, bias):\n"
        "    x = torch.from_numpy(input).unsqueeze(0)\n"
        "    w = torch.from_numpy(weight)\n"
        "    b = torch.from_numpy(bias)\n"
        f"    return F.conv1d(x, w, b, stride={sw}, padding={pad}, "
        f"dilation={dw}).squeeze(0).numpy()\n"
    )


def _ref_conv2d_depthwise(sh: int, sw: int, dh: int, dw: int, pad: int) -> str:
    return (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def run(input, weight, bias):\n"
        "    c = input.shape[1]\n"
        "    x = torch.from_numpy(input)\n"
        "    w = torch.from_numpy(weight).unsqueeze(1)\n"
        "    b = torch.from_numpy(bias)\n"
        f"    return F.conv2d(x, w, b, stride=({sh}, {sw}), padding=({pad}, {pad}), "
        f"dilation=({dh}, {dw}), groups=c).numpy()\n"
    )


def _ref_deconv2d(sh: int, sw: int) -> str:
    return (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def run(input, weight, bias):\n"
        "    x = torch.from_numpy(input)\n"
        "    w = torch.from_numpy(weight).permute(1, 0, 2, 3)\n"
        "    b = torch.from_numpy(bias)\n"
        f"    return F.conv_transpose2d(x, w, b, stride=({sh}, {sw}), "
        f"padding=(0, 0), dilation=(1, 1)).numpy()\n"
    )


def _ref_deconv2d_depthwise(sh: int, sw: int) -> str:
    return (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def run(input, weight, bias):\n"
        "    c = input.shape[1]\n"
        "    x = torch.from_numpy(input)\n"
        "    w = torch.from_numpy(weight).unsqueeze(1)\n"
        "    b = torch.from_numpy(bias)\n"
        f"    return F.conv_transpose2d(x, w, b, stride=({sh}, {sw}), "
        f"padding=(0, 0), dilation=(1, 1), groups=c).numpy()\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# conv2d — 6 definitions; C_in is a workload-level var axis
# ─────────────────────────────────────────────────────────────────────────────

# (Kh, Kw, Sh, Sw, Dh, Dw, C_out, pad_top, pad_left)
_CONV2D_PARAMS: List[Tuple] = [
    (1, 1, 1, 1, 1, 1, 256, 0, 0),
    (3, 3, 1, 1, 1, 1, 128, 1, 1),
    (3, 3, 1, 1, 2, 2, 256, 2, 2),
    (3, 3, 2, 2, 1, 1, 128, 1, 1),
    (5, 5, 1, 1, 1, 1,  64, 2, 2),
    (7, 7, 2, 2, 1, 1,  64, 3, 3),
]

# Workloads per definition: list of (C_in, H, W)
_CONV2D_WORKLOADS: Dict[Tuple, List[Tuple[int, int, int]]] = {
    (1, 1, 1, 1, 1, 1, 256, 0, 0): [
        ( 64, 224, 224),
        (128,  56,  56),
        (512,  14,  14),
        (512,   7,   7),
        ( 11,  28,  28),   # non-divisible channels
        ( 64, 113, 113),   # odd spatial
    ],
    (3, 3, 1, 1, 1, 1, 128, 1, 1): [
        ( 64, 224, 224),
        (128,  56,  56),
        ( 64,  28,  28),
        ( 64,  14,  14),
        ( 11,  28,  28),   # non-divisible channels
        ( 64, 113, 113),   # odd spatial
    ],
    (3, 3, 1, 1, 2, 2, 256, 2, 2): [
        ( 64,  56,  56),
        (128,  28,  28),
        (256,  14,  14),
        ( 11,  28,  28),   # non-divisible channels
        ( 64,  29,  29),   # odd spatial
    ],
    (3, 3, 2, 2, 1, 1, 128, 1, 1): [
        ( 64, 224, 224),
        (128,  56,  56),
        ( 64,  28,  28),
        ( 11,  56,  56),   # non-divisible channels
        ( 64, 113, 113),   # odd spatial (stride=2 boundary)
        ( 64,  57,  57),   # odd spatial (stride=2 boundary)
    ],
    (5, 5, 1, 1, 1, 1, 64, 2, 2): [
        ( 32, 224, 224),
        ( 64,  56,  56),
        ( 32,  28,  28),
        ( 32,  14,  14),
        ( 11,  28,  28),   # non-divisible channels
        ( 32, 113, 113),   # odd spatial
    ],
    (7, 7, 2, 2, 1, 1, 64, 3, 3): [
        (  3, 224, 224),   # ResNet first-layer (cin=3)
        ( 32, 224, 224),
        (  3, 112, 112),
        ( 11, 224, 224),   # non-divisible channels
        (  3, 113, 113),   # odd spatial + cin=3
        (  3,  56,  56),
    ],
}


def _gen_conv2d() -> None:
    for params in _CONV2D_PARAMS:
        kh, kw, sh, sw, dh, dw, cout, pad_top, pad_left = params
        name = f"conv2d_kh{kh}_kw{kw}_sh{sh}_sw{sw}_dh{dh}_dw{dw}_cout{cout}"

        defn = {
            "name": name,
            "op_type": "conv2d",
            "description": (
                f"2D conv {kh}x{kw} stride=({sh},{sw}) dilation=({dh},{dw}) "
                f"pad=({pad_top},{pad_left}) C_out={cout}. C_in varies per workload."
            ),
            "tags": ["status:active"],
            "axes": {
                "N":      {"type": "var"},
                "H":      {"type": "var", "parent": "N"},
                "W":      {"type": "var", "parent": "N"},
                "H_out":  {"type": "var", "parent": "N"},
                "W_out":  {"type": "var", "parent": "N"},
                "C_in":   {"type": "var"},
                "C_out":  {"type": "const", "value": cout},
                "Kh":     {"type": "const", "value": kh},
                "Kw":     {"type": "const", "value": kw},
                "Sh":     {"type": "const", "value": sh},
                "Sw":     {"type": "const", "value": sw},
                "Dh":     {"type": "const", "value": dh},
                "Dw":     {"type": "const", "value": dw},
                "pad_top":  {"type": "const", "value": pad_top},
                "pad_left": {"type": "const", "value": pad_left},
            },
            "inputs": {
                "input":           {"shape": ["N", "C_in", "H", "W"], "dtype": "float32"},
                "weight":          {"shape": ["C_out", "C_in", "Kh", "Kw"], "dtype": "float32"},
                "activation_type": {"shape": None, "dtype": "int32"},
                "with_bias":       {"shape": None, "dtype": "int32"},
            },
            "outputs": {
                "output": {"shape": ["N", "C_out", "H_out", "W_out"], "dtype": "float32"},
            },
            "constraints": [
                "H_out == (H + 2*pad_top - Dh*(Kh-1) - 1) // Sh + 1",
                "W_out == (W + 2*pad_left - Dw*(Kw-1) - 1) // Sw + 1",
            ],
            "reference": _ref_conv2d(sh, sw, dh, dw, pad_top, pad_left),
        }

        scalar_in = {
            "activation_type": {"type": "scalar", "value": 0},
            "with_bias":       {"type": "scalar", "value": 0},
        }
        workload_lines = [
            _wl({"N": 1, "C_in": cin, "H": h, "W": w},
                {**_rand("input", "weight"), **scalar_in})
            for cin, h, w in _CONV2D_WORKLOADS[params]
        ]

        _write_json(DEFS_DIR / "conv" / f"{name}.json", defn)
        _write_jsonl(WLS_DIR / "conv" / f"{name}.jsonl", workload_lines)
        print(f"  [conv2d] {name}  ({len(workload_lines)} workloads)")


# ─────────────────────────────────────────────────────────────────────────────
# conv1d — 2 definitions; C_in is a workload-level var axis
# ─────────────────────────────────────────────────────────────────────────────

# (Kw, Sw, Dw, C_out, pad)
_CONV1D_PARAMS: List[Tuple] = [
    (1, 1, 1, 512, 0),
    (3, 1, 1, 512, 1),
]

# Workloads per definition: list of (C_in, W)
_CONV1D_WORKLOADS: Dict[Tuple, List[Tuple[int, int]]] = {
    (1, 1, 1, 512, 0): [
        ( 64, 512),
        (128, 256),
        (256, 128),
        (512,  64),
        ( 11, 256),   # non-divisible channels
    ],
    (3, 1, 1, 512, 1): [
        ( 64, 512),
        (128, 256),
        (256, 128),
        (  3, 256),   # cin=3 input-specialisation path
        ( 11, 256),   # non-divisible channels
    ],
}


def _gen_conv1d() -> None:
    for params in _CONV1D_PARAMS:
        kw, sw, dw, cout, pad = params
        name = f"conv1d_kw{kw}_sw{sw}_dw{dw}_cout{cout}_p{pad}"

        defn = {
            "name": name,
            "op_type": "conv1d",
            "description": (
                f"1D conv kw={kw} stride={sw} dilation={dw} pad={pad} "
                f"C_out={cout}. C_in varies per workload."
            ),
            "tags": ["status:active"],
            "axes": {
                "W":     {"type": "var"},
                "W_out": {"type": "var"},
                "C_in":  {"type": "var"},
                "C_out": {"type": "const", "value": cout},
                "Kw":    {"type": "const", "value": kw},
                "Sw":    {"type": "const", "value": sw},
                "Dw":    {"type": "const", "value": dw},
                "pad":   {"type": "const", "value": pad},
            },
            "inputs": {
                "input":  {"shape": ["C_in", "W"], "dtype": "float32"},
                "weight": {"shape": ["C_out", "C_in", "Kw"], "dtype": "float32"},
                "bias":   {"shape": ["C_out"], "dtype": "float32"},
            },
            "outputs": {
                "output": {"shape": ["C_out", "W_out"], "dtype": "float32"},
            },
            "constraints": [
                "W_out == (W + 2*pad - Dw*(Kw-1) - 1) // Sw + 1",
            ],
            "reference": _ref_conv1d(sw, dw, pad),
        }

        workload_lines = [
            _wl({"C_in": cin, "W": w}, _rand("input", "weight", "bias"))
            for cin, w in _CONV1D_WORKLOADS[params]
        ]

        _write_json(DEFS_DIR / "conv1d" / f"{name}.json", defn)
        _write_jsonl(WLS_DIR / "conv1d" / f"{name}.jsonl", workload_lines)
        print(f"  [conv1d] {name}  ({len(workload_lines)} workloads)")


# ─────────────────────────────────────────────────────────────────────────────
# conv2d_depthwise — 3 definitions; C is a workload-level var axis
# ─────────────────────────────────────────────────────────────────────────────

# (Kh, Kw, Sh, Sw, Dh, Dw, pad)
_DW_PARAMS: List[Tuple] = [
    (3, 3, 1, 1, 1, 1, 1),
    (5, 5, 1, 1, 1, 1, 2),
    (3, 3, 2, 2, 1, 1, 1),
]

# Workloads per definition: list of (C, H, W)
_DW_WORKLOADS: Dict[Tuple, List[Tuple[int, int, int]]] = {
    (3, 3, 1, 1, 1, 1, 1): [
        ( 64, 112, 112),
        (128,  56,  56),
        (256,  28,  28),
        ( 11,  56,  56),   # non-divisible channels
        ( 64, 113, 113),   # odd spatial
    ],
    (5, 5, 1, 1, 1, 1, 2): [
        ( 64,  56,  56),
        (128,  28,  28),
        ( 11,  28,  28),   # non-divisible channels
        ( 64,  57,  57),   # odd spatial
    ],
    (3, 3, 2, 2, 1, 1, 1): [
        ( 64, 112, 112),
        (128,  56,  56),
        ( 64,  28,  28),
        ( 64, 113, 113),   # odd spatial (stride=2 boundary)
        ( 11,  57,  57),   # non-divisible + odd spatial
    ],
}


def _gen_conv2d_depthwise() -> None:
    for params in _DW_PARAMS:
        kh, kw, sh, sw, dh, dw, pad = params
        name = f"conv2d_depthwise_kh{kh}_kw{kw}_sh{sh}_sw{sw}_dh{dh}_dw{dw}_p{pad}"

        defn = {
            "name": name,
            "op_type": "conv2d_depthwise",
            "description": (
                f"Depthwise 2D conv {kh}x{kw} stride=({sh},{sw}) "
                f"dilation=({dh},{dw}) pad={pad}. C varies per workload."
            ),
            "tags": ["status:active"],
            "axes": {
                "N":     {"type": "var"},
                "H":     {"type": "var", "parent": "N"},
                "W":     {"type": "var", "parent": "N"},
                "H_out": {"type": "var", "parent": "N"},
                "W_out": {"type": "var", "parent": "N"},
                "C":     {"type": "var"},
                "Kh":    {"type": "const", "value": kh},
                "Kw":    {"type": "const", "value": kw},
                "Sh":    {"type": "const", "value": sh},
                "Sw":    {"type": "const", "value": sw},
                "Dh":    {"type": "const", "value": dh},
                "Dw":    {"type": "const", "value": dw},
                "pad":   {"type": "const", "value": pad},
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
                "H_out == (H + 2*pad - Dh*(Kh-1) - 1) // Sh + 1",
                "W_out == (W + 2*pad - Dw*(Kw-1) - 1) // Sw + 1",
            ],
            "reference": _ref_conv2d_depthwise(sh, sw, dh, dw, pad),
        }

        workload_lines = [
            _wl({"N": 1, "C": c, "H": h, "W": w}, _rand("input", "weight", "bias"))
            for c, h, w in _DW_WORKLOADS[params]
        ]

        _write_json(DEFS_DIR / "conv2d_depthwise" / f"{name}.json", defn)
        _write_jsonl(WLS_DIR / "conv2d_depthwise" / f"{name}.jsonl", workload_lines)
        print(f"  [conv2d_depthwise] {name}  ({len(workload_lines)} workloads)")


# ─────────────────────────────────────────────────────────────────────────────
# deconv2d — 4 definitions; C_in is a workload-level var axis
# ─────────────────────────────────────────────────────────────────────────────

# (Kh, Kw, Sh, Sw, C_out)
_DECONV2D_PARAMS: List[Tuple] = [
    (3, 3, 1, 1, 256),
    (3, 3, 2, 2, 256),
    (4, 4, 1, 1, 128),
    (4, 4, 2, 2, 128),
]

# Workloads per definition: list of (C_in, H, W)
_DECONV2D_WORKLOADS: Dict[Tuple, List[Tuple[int, int, int]]] = {
    (3, 3, 1, 1, 256): [
        ( 64,  56,  56),
        (128,  28,  28),
        ( 11,  28,  28),   # non-divisible channels
        ( 64,  57,  57),   # odd spatial input
    ],
    (3, 3, 2, 2, 256): [
        ( 64,  56,  56),
        (128,  28,  28),
        ( 64,  14,  14),
        ( 11,  28,  28),   # non-divisible channels
        ( 64,  29,  29),   # odd spatial (output=(29-1)*2+3=59)
    ],
    (4, 4, 1, 1, 128): [
        ( 64,  56,  56),
        (128,  28,  28),
        (256,  14,  14),
        ( 11,  28,  28),   # non-divisible channels
    ],
    (4, 4, 2, 2, 128): [
        ( 64,  56,  56),
        (128,  28,  28),
        ( 64,  14,  14),
        ( 11,  28,  28),   # non-divisible channels
        ( 64,  29,  29),   # odd spatial (output=(29-1)*2+4=60)
    ],
}


def _gen_deconv2d() -> None:
    for params in _DECONV2D_PARAMS:
        kh, kw, sh, sw, cout = params
        name = f"deconv2d_kh{kh}_kw{kw}_sh{sh}_sw{sw}_cout{cout}"

        defn = {
            "name": name,
            "op_type": "deconv2d",
            "description": (
                f"Transposed 2D conv {kh}x{kw} stride=({sh},{sw}) "
                f"dilation=(1,1) pad=0 C_out={cout}. C_in varies per workload."
            ),
            "tags": ["status:active"],
            "axes": {
                "N":     {"type": "var"},
                "H":     {"type": "var", "parent": "N"},
                "W":     {"type": "var", "parent": "N"},
                "H_out": {"type": "var", "parent": "N"},
                "W_out": {"type": "var", "parent": "N"},
                "C_in":  {"type": "var"},
                "C_out": {"type": "const", "value": cout},
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
            "reference": _ref_deconv2d(sh, sw),
        }

        workload_lines = [
            _wl({"N": 1, "C_in": cin, "H": h, "W": w},
                _rand("input", "weight", "bias"))
            for cin, h, w in _DECONV2D_WORKLOADS[params]
        ]

        _write_json(DEFS_DIR / "deconv2d" / f"{name}.json", defn)
        _write_jsonl(WLS_DIR / "deconv2d" / f"{name}.jsonl", workload_lines)
        print(f"  [deconv2d] {name}  ({len(workload_lines)} workloads)")


# ─────────────────────────────────────────────────────────────────────────────
# deconv2d_depthwise — 2 definitions; C is a workload-level var axis
# ─────────────────────────────────────────────────────────────────────────────

# (Kh, Kw, Sh, Sw)
_DECONV2D_DW_PARAMS: List[Tuple] = [
    (2, 2, 2, 2),
    (3, 3, 1, 1),
]

# Workloads per definition: list of (C, H, W)
_DECONV2D_DW_WORKLOADS: Dict[Tuple, List[Tuple[int, int, int]]] = {
    (2, 2, 2, 2): [
        ( 64,  56,  56),
        (128,  28,  28),
        ( 11,  28,  28),   # non-divisible channels
        ( 64,  29,  29),   # odd spatial (output=(29-1)*2+2=58)
    ],
    (3, 3, 1, 1): [
        ( 64,  56,  56),
        (128,  28,  28),
        (256,  14,  14),
        ( 11,  28,  28),   # non-divisible channels
    ],
}


def _gen_deconv2d_depthwise() -> None:
    for params in _DECONV2D_DW_PARAMS:
        kh, kw, sh, sw = params
        name = f"deconv2d_depthwise_kh{kh}_kw{kw}_sh{sh}_sw{sw}"

        defn = {
            "name": name,
            "op_type": "deconv2d_depthwise",
            "description": (
                f"Depthwise transposed 2D conv {kh}x{kw} stride=({sh},{sw}) "
                f"dilation=(1,1) pad=0. C varies per workload."
            ),
            "tags": ["status:active"],
            "axes": {
                "N":     {"type": "var"},
                "H":     {"type": "var", "parent": "N"},
                "W":     {"type": "var", "parent": "N"},
                "H_out": {"type": "var", "parent": "N"},
                "W_out": {"type": "var", "parent": "N"},
                "C":     {"type": "var"},
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
            "reference": _ref_deconv2d_depthwise(sh, sw),
        }

        workload_lines = [
            _wl({"N": 1, "C": c, "H": h, "W": w},
                _rand("input", "weight", "bias"))
            for c, h, w in _DECONV2D_DW_WORKLOADS[params]
        ]

        _write_json(DEFS_DIR / "deconv2d_depthwise" / f"{name}.json", defn)
        _write_jsonl(WLS_DIR / "deconv2d_depthwise" / f"{name}.jsonl", workload_lines)
        print(f"  [deconv2d_depthwise] {name}  ({len(workload_lines)} workloads)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Generating definitions and workloads...")
    _gen_conv2d()
    _gen_conv1d()
    _gen_conv2d_depthwise()
    _gen_deconv2d()
    _gen_deconv2d_depthwise()
    print("\nDone.")


if __name__ == "__main__":
    main()
