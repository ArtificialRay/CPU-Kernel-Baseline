#!/usr/bin/env python3
"""
Collect CV-model kernel workloads (conv2d, conv2d_depthwise, gemm, pooling)
via torchvision forward hooks.

Runs ResNet50 and MobileNetV3-Large across 14 standard CV inference resolutions
(analogous to ShareGPT prompt lengths for LLM ops) and captures shapes for
whichever op_type/definition is requested on the command line.

Each op_type is backed by a small capture spec (which nn.Module type(s) to
hook, how to turn a forward pass into a flat record, and which records
belong to which definition) registered in CV_CAPTURE_SPECS. Definitions
themselves are read directly from bench-trace/definitions/<op_type>/ — adding
a new definition to an already-registered op_type needs no script changes.

Usage
-----
    python scripts/collect_workloads_conv.py --op-type conv2d \\
        --definition conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0        # write JSONL
    python scripts/collect_workloads_conv.py --op-type gemm \\
        --definition gemm_fp32_n1000_k1280 --dry-run               # preview only
    python scripts/collect_workloads_conv.py --list-op-types       # show supported op_types
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFS_DIR  = REPO_ROOT / "bench-trace" / "definitions"

# Standard CV inference resolutions (N, H, W) — analogous to ShareGPT prompt lengths.
# Each resolution drives all conv/gemm/pooling layers to produce different shapes.
CV_RESOLUTIONS: list[tuple[int, int, int]] = [
    # Mobile / edge  (N=1 only — CV baselines are from edge frameworks, no batch>1)
    (1,  96,  96),    # face recognition, edge deployment
    (1, 160, 160),    # YOLOv5-nano default
    (1, 224, 224),    # ImageNet standard (MobileNet / ResNet)
    # Medium-resolution detection
    (1, 320, 320),    # YOLOv5-small
    (1, 384, 384),    # ViT-B/16, EfficientNet-L
    (1, 416, 416),    # YOLOv3 / YOLOv4 standard
    # High-resolution detection / segmentation
    (1, 512, 512),    # SSD, medical imaging
    (1, 640, 640),    # YOLOv5/v8 default
    (1, 1024, 1024),  # instance segmentation, satellite imagery
    # Non-square (video frames)
    (1, 360, 640),    # 360p landscape
    (1, 480, 640),    # VGA / COCO standard eval size
]


# ── Shape capture via forward hooks ───────────────────────────────────────────

def _to_pair(v) -> tuple[int, int]:
    if isinstance(v, int):
        return (v, v)
    t = tuple(v)
    return (t[0], t[1]) if len(t) >= 2 else (t[0], t[0])


def _extract_conv(mod, inp, out) -> dict:
    """nn.Conv2d forward pass -> flat record. Feeds both conv2d and
    conv2d_depthwise definitions (discriminated by the `depthwise` flag)."""
    N, C_in, H, W = inp[0].shape
    _, C_out, H_out, W_out = out.shape
    Kh, Kw = _to_pair(mod.kernel_size)
    Sh, Sw = _to_pair(mod.stride)
    Dh, Dw = _to_pair(mod.dilation)
    ph, pw = _to_pair(mod.padding)
    # Depthwise: groups == in_channels and in_channels > 1
    # (excludes trivial 1-channel convs that satisfy groups==in_channels==1)
    is_dw = (mod.groups == mod.in_channels) and (mod.in_channels > 1)
    return {
        "N": int(N), "H": int(H), "W": int(W),
        "H_out": int(H_out), "W_out": int(W_out),
        "C_in": int(C_in), "C_out": int(C_out),
        "C": int(C_in),                           # depthwise: C_in == C_out == C
        "C_group": mod.in_channels // mod.groups,  # depthwise: always 1
        "Kh": Kh, "Kw": Kw, "Sh": Sh, "Sw": Sw, "Dh": Dh, "Dw": Dw,
        "pad_top": ph, "pad_left": pw,
        "depthwise": is_dw,
    }


def _extract_linear(mod, inp, out) -> dict:
    """nn.Linear forward pass -> flat record for gemm definitions."""
    batch, K = inp[0].shape
    _, N_out = out.shape
    return {"M": int(batch), "K": int(K), "N": int(N_out)}


def _extract_pool(mod, inp, out) -> dict:
    """nn.AdaptiveAvgPool2d / nn.MaxPool2d forward pass -> flat record for
    pooling definitions. `pooling_fp32_global_avg` has zero const axes (fed
    by AdaptiveAvgPool2d, `_kind="avg"`); the `pooling_fp32_max_*` family has
    the same Kh/Kw/Sh/Sw/Dh/Dw/pad const-axis convention as conv2d (fed by
    MaxPool2d, `_kind="max"`)."""
    import torch.nn as nn

    N, C, H, W = inp[0].shape
    if isinstance(mod, nn.AdaptiveAvgPool2d):
        return {"N": int(N), "C": int(C), "H": int(H), "W": int(W), "_kind": "avg"}
    _, _, H_out, W_out = out.shape
    Kh, Kw = _to_pair(mod.kernel_size)
    Sh, Sw = _to_pair(mod.stride if mod.stride is not None else mod.kernel_size)
    Dh, Dw = _to_pair(mod.dilation)
    ph, pw = _to_pair(mod.padding)
    return {
        "N": int(N), "C": int(C), "H": int(H), "W": int(W),
        "H_out": int(H_out), "W_out": int(W_out),
        "Kh": Kh, "Kw": Kw, "Sh": Sh, "Sw": Sw, "Dh": Dh, "Dw": Dw,
        "pad_top": ph, "pad_left": pw,
        "_kind": "max",
    }


def _accepts_conv2d(rec: dict, const_axes: dict) -> bool:
    return not rec["depthwise"]


def _accepts_conv2d_depthwise(rec: dict, const_axes: dict) -> bool:
    return rec["depthwise"]


def _accepts_always(rec: dict, const_axes: dict) -> bool:
    return True


def _accepts_pooling(rec: dict, const_axes: dict) -> bool:
    # Which pooling kind a definition wants is derived from its own const
    # axes (max-pool definitions declare Kh; global-avg declares none) —
    # not a hardcoded name string, so this generalizes to future pooling
    # kinds without touching this function.
    return rec["_kind"] == ("max" if "Kh" in const_axes else "avg")


@dataclass
class CVCaptureSpec:
    module_types: tuple
    extract: Callable[[object, tuple, object], dict]
    accepts: Callable[[dict, dict], bool]
    batch_key: str


CV_CAPTURE_SPECS: dict[str, CVCaptureSpec] = {}


def _register_specs() -> None:
    import torch.nn as nn

    CV_CAPTURE_SPECS["conv2d"] = CVCaptureSpec(
        (nn.Conv2d,), _extract_conv, _accepts_conv2d, batch_key="C_in")
    CV_CAPTURE_SPECS["conv2d_depthwise"] = CVCaptureSpec(
        (nn.Conv2d,), _extract_conv, _accepts_conv2d_depthwise, batch_key="C")
    CV_CAPTURE_SPECS["gemm"] = CVCaptureSpec(
        (nn.Linear,), _extract_linear, _accepts_always, batch_key="M")
    CV_CAPTURE_SPECS["pooling"] = CVCaptureSpec(
        (nn.AdaptiveAvgPool2d, nn.MaxPool2d), _extract_pool, _accepts_pooling, batch_key="C")


def _capture(spec: CVCaptureSpec, resolutions: list[tuple[int, int, int]]) -> list[dict]:
    """Register forward hooks for spec.module_types on both reference CV
    models, run them across `resolutions`, and return the flat records."""
    import torch
    import torchvision.models as tvm

    records: list[dict] = []

    def _hook(mod, inp, out):
        records.append(spec.extract(mod, inp, out))

    for model_fn in (tvm.resnet50, tvm.mobilenet_v3_large):
        model = model_fn(weights=None).eval()
        hooks = [m.register_forward_hook(_hook) for m in model.modules()
                 if isinstance(m, spec.module_types)]
        with torch.no_grad():
            for N, H, W in resolutions:
                try:
                    model(torch.zeros(N, 3, H, W))
                except Exception as exc:
                    print(f"  [warn] {model_fn.__name__}({N},{H},{W}): {exc}")
        for h in hooks:
            h.remove()

    return records


# ── Definition loading and generic matching ───────────────────────────────────

def _load_definition(op_type: str, def_name: str) -> dict:
    path = DEFS_DIR / op_type / f"{def_name}.json"
    if not path.exists():
        sys.exit(f"[conv] Definition not found: {path}")
    return json.loads(path.read_text())


def _axes_split(defn: dict) -> tuple[dict, list[str]]:
    """Return (const_axes: name->value, var_axis_names: list) from a definition."""
    const_axes = {k: v["value"] for k, v in defn["axes"].items() if v["type"] == "const"}
    var_axis_names = [k for k, v in defn["axes"].items() if v["type"] == "var"]
    return const_axes, var_axis_names


def _matches(rec: dict, const_axes: dict) -> bool:
    return all(rec.get(k) == v for k, v in const_axes.items())


def _var_axes(rec: dict, var_axis_names: list[str]) -> dict:
    return {k: rec[k] for k in var_axis_names}


def _w8a8ch_scalars(def_name: str) -> Optional[dict]:
    """w8a8ch definitions need a per-tensor activation input_scale that isn't
    provided by torchvision shape capture. Reference impls (see
    bench-trace/definitions/*/*_w8a8ch_*.json) dequantize int8 accumulation
    as acc * input_scale * weight_scales before clipping to int8 range —
    scale must be small enough that outputs don't uniformly saturate.
    Seeded by def_name so repeated collection runs are reproducible.
    """
    if "w8a8ch" not in def_name:
        return None
    scale = random.Random(def_name).uniform(0.005, 0.05)
    return {"input_scale": round(scale, 5)}


# ── gen_workload.py CLI helper ─────────────────────────────────────────────────

def _gen_workload(
    def_name: str,
    axes_list: list[dict],
    dry_run: bool,
    batch_key: str,
    max_count: int = 20,
    scalars: Optional[dict] = None,
) -> None:
    cmd = ["python", "scripts/gen-workload/gen_workload.py", def_name]
    for axes in axes_list:
        cmd += ["--add", ",".join(f"{k}={v}" for k, v in axes.items())]
    for k, v in (scalars or {}).items():
        cmd += ["--scalar", f"{k}={v}"]
    cmd += ["--max-count", str(max_count), "--batch-key", batch_key]
    if dry_run:
        cmd += ["--dry-run"]
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"  [warn] gen_workload.py returned {result.returncode} for {def_name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect CV-model kernel workloads (conv2d, conv2d_depthwise, "
                    "gemm, pooling) via torchvision forward hooks."
    )
    parser.add_argument("--op-type", help="Op type to collect for (see --list-op-types).")
    parser.add_argument("--definition", help="Definition name to write workloads for.")
    parser.add_argument("--list-op-types", action="store_true",
                        help="Print supported op_types and exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without touching JSONL files.")
    args = parser.parse_args()

    _register_specs()

    if args.list_op_types:
        print("\n".join(sorted(CV_CAPTURE_SPECS)))
        return

    if not args.op_type or not args.definition:
        parser.error("--op-type and --definition are required (or pass --list-op-types).")

    if args.op_type not in CV_CAPTURE_SPECS:
        parser.error(
            f"Unsupported op_type '{args.op_type}' for this collector. "
            f"Supported: {sorted(CV_CAPTURE_SPECS)}. If this op_type has no "
            f"torchvision-hookable source, use gen_workload.py --add directly."
        )

    spec = CV_CAPTURE_SPECS[args.op_type]
    defn = _load_definition(args.op_type, args.definition)
    const_axes, var_axis_names = _axes_split(defn)

    print(f"[conv] Capturing shapes for op_type={args.op_type} "
          f"definition={args.definition} × {len(CV_RESOLUTIONS)} resolutions...")
    records = _capture(spec, CV_RESOLUTIONS)
    print(f"  captured {len(records)} raw records")

    axes_list = [
        _var_axes(rec, var_axis_names)
        for rec in records
        if spec.accepts(rec, const_axes) and _matches(rec, const_axes)
    ]

    if not axes_list:
        print(f"[conv] [skip] {args.definition}: no matching captures in ResNet50 / "
              f"MobileNetV3-Large — this shape doesn't occur in either model; "
              f"add it manually via gen_workload.py --add.")
        return

    print(f"[conv] {len(axes_list)} matching candidates (dry_run={args.dry_run})")
    scalars = _w8a8ch_scalars(args.definition)
    _gen_workload(args.definition, axes_list, args.dry_run,
                  batch_key=spec.batch_key, scalars=scalars)
    print("[conv] Done.")


if __name__ == "__main__":
    main()
