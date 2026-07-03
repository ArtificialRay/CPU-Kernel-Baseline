#!/usr/bin/env python3
"""Extract op shape records from real models.

Each record is a dict:
    {
        "op_type":   str,   # "gemm", "rms_norm", "mha", "lstm", "moe", "pooling",
                            # "conv2d", "conv2d_depthwise"
        "dtype":     str,   # "fp32" | "q8_0" | "w8a8ch"
        "axes":      dict,  # const axis name → int value
        "model_tag": str,   # "qwen1.5-moe-a2.7b" etc.
        "description": str,
    }

Called by gen_model_definitions.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_Record = dict[str, Any]


# ── Dedup ─────────────────────────────────────────────────────────────────────

def _dedup(records: list[_Record]) -> list[_Record]:
    """Drop records with duplicate (op_type, dtype, axes) — first model's tag wins."""
    seen: set[tuple] = set()
    out: list[_Record] = []
    for r in records:
        key = (r["op_type"], r["dtype"], tuple(sorted(r["axes"].items())))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gemm(N: int, K: int, dtype: str, model_tag: str, desc: str = "") -> _Record:
    return {
        "op_type": "gemm", "dtype": dtype,
        "axes": {"N": N, "K": K},
        "model_tag": model_tag,
        "description": desc or f"GEMM N={N} K={K}",
    }


def _conv2d(kh, kw, sh, sw, dh, dw, pad_top, pad_left, dtype: str, model_tag: str) -> _Record:
    return {
        "op_type": "conv2d", "dtype": dtype,
        "axes": {
            "Kh": kh, "Kw": kw, "Sh": sh, "Sw": sw, "Dh": dh, "Dw": dw,
            "pad_top": pad_top, "pad_left": pad_left,
        },
        "model_tag": model_tag,
        "description": (
            f"2D conv {kh}x{kw} stride=({sh},{sw}) dilation=({dh},{dw}) "
            f"pad=({pad_top},{pad_left}). C_in and C_out vary per workload."
        ),
    }


def _conv2d_depthwise(kh, kw, sh, sw, dh, dw, pad_top, pad_left, dtype: str, model_tag: str) -> _Record:
    return {
        "op_type": "conv2d_depthwise", "dtype": dtype,
        "axes": {
            "Kh": kh, "Kw": kw, "Sh": sh, "Sw": sw, "Dh": dh, "Dw": dw,
            "pad_top": pad_top, "pad_left": pad_left,
        },
        "model_tag": model_tag,
        "description": (
            f"Depthwise 2D conv {kh}x{kw} stride=({sh},{sw}) dilation=({dh},{dw}) "
            f"pad=({pad_top},{pad_left}). C (=C_in=C_out=groups) varies per workload."
        ),
    }


# ── Method A: HuggingFace config.json ─────────────────────────────────────────

def extract_hf_shapes(
    model_id: str,
    model_tag: str,
    dtypes: list[str] = ["fp32", "q8_0"],
) -> list[_Record]:
    """Download config.json only (no weights) and emit shape records."""
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(model_id, "config.json")
    cfg: dict = json.loads(Path(cfg_path).read_text())

    hidden    = cfg["hidden_size"]
    n_heads   = cfg["num_attention_heads"]
    n_kv      = cfg.get("num_key_value_heads", n_heads)
    head_dim  = hidden // n_heads
    q_dim     = n_heads * head_dim   # == hidden for full-rank Q
    kv_dim    = n_kv * head_dim

    n_expert  = cfg.get("num_experts") or cfg.get("num_local_experts")
    top_k     = cfg.get("num_experts_per_tok") or cfg.get("top_k")
    expert_ff = (cfg.get("moe_intermediate_size") or cfg.get("intermediate_size")) if n_expert else None
    is_moe    = bool(n_expert and top_k and expert_ff)

    records: list[_Record] = []

    for dtype in dtypes:
        # Attention projections — unique (N, K) pairs
        attn_shapes: set[tuple[int, int]] = set()
        attn_shapes.add((q_dim, hidden))   # Q, O
        attn_shapes.add((kv_dim, hidden))  # K, V (may equal Q if GQA absent)
        attn_shapes.add((hidden, q_dim))   # output projection
        for N, K in attn_shapes:
            records.append(_gemm(N, K, dtype, model_tag, f"attn proj N={N} K={K}"))

        if is_moe:
            for N, K in [(expert_ff, hidden), (hidden, expert_ff)]:
                records.append(_gemm(N, K, dtype, model_tag, f"expert ffn N={N} K={K}"))

        if dtype == "fp32":
            records.append({
                "op_type": "rms_norm", "dtype": "fp32",
                "axes": {"D": hidden},
                "model_tag": model_tag,
                "description": f"RMSNorm D={hidden}",
            })
            records.append({
                "op_type": "mha", "dtype": "fp32",
                "axes": {"n_heads": n_heads, "head_dim": head_dim, "kv_heads": n_kv},
                "model_tag": model_tag,
                "description": f"MHA h={n_heads} d={head_dim} kv={n_kv}",
            })
            if is_moe:
                records.append({
                    "op_type": "moe", "dtype": "fp32",
                    "axes": {
                        "n_embd": hidden, "n_ff": expert_ff,
                        "n_expert": n_expert, "n_expert_used": top_k,
                    },
                    "model_tag": model_tag,
                    "description": f"MoE e={n_expert} k={top_k} d={hidden} ff={expert_ff}",
                })
        elif is_moe:
            records.append({
                "op_type": "moe", "dtype": "q8_0",
                "axes": {
                    "n_embd": hidden, "n_ff": expert_ff,
                    "n_expert": n_expert, "n_expert_used": top_k,
                },
                "model_tag": model_tag,
                "description": f"MoE q8_0 e={n_expert} k={top_k} d={hidden} ff={expert_ff}",
            })

    return records


# ── Method B: torchvision forward hooks ────────────────────────────────────────

_MIN_FC_ELEMENTS = 64  # skip Linear layers smaller than this in each dimension

def extract_torchvision_shapes(
    model_fn,
    model_tag: str,
    input_size: tuple = (1, 3, 224, 224),
    dtypes: list[str] = ["fp32", "w8a8ch"],
) -> list[_Record]:
    """Register forward hooks, run a dummy pass, collect Linear / pool / conv2d shapes."""
    import torch
    import torch.nn as nn

    model = model_fn(weights=None).eval()
    records: list[_Record] = []
    linear_shapes: set[tuple[int, int]] = set()
    has_global_avg_pool = False
    conv_regular: set[tuple[int, int, int, int, int, int, int, int]] = set()
    conv_depthwise: set[tuple[int, int, int, int, int, int, int, int]] = set()

    def _lin_hook(m: nn.Linear, inp, out):
        N_out, K = m.out_features, m.in_features
        if min(N_out, K) >= _MIN_FC_ELEMENTS:
            linear_shapes.add((N_out, K))

    def _pool_hook(m, inp, out):
        nonlocal has_global_avg_pool
        has_global_avg_pool = True

    def _conv_hook(m: nn.Conv2d, inp, out):
        kh, kw = m.kernel_size
        sh, sw = m.stride
        dh, dw = m.dilation
        ph, pw = m.padding
        key = (kh, kw, sh, sw, dh, dw, ph, pw)
        if m.groups == m.in_channels and m.groups == m.out_channels and m.groups > 1:
            conv_depthwise.add(key)
        elif m.groups == 1:
            conv_regular.add(key)

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_lin_hook))
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            hooks.append(m.register_forward_hook(_pool_hook))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv_hook))

    with torch.no_grad():
        model(torch.zeros(*input_size))
    for h in hooks:
        h.remove()

    for N, K in sorted(linear_shapes):
        for dtype in dtypes:
            records.append(_gemm(N, K, dtype, model_tag, f"fc N={N} K={K}"))

    if has_global_avg_pool and "fp32" in dtypes:
        records.append({
            "op_type": "pooling", "dtype": "fp32",
            "axes": {},
            "model_tag": model_tag,
            "description": "global average pooling",
        })

    for kh, kw, sh, sw, dh, dw, ph, pw in sorted(conv_regular):
        for dtype in dtypes:
            records.append(_conv2d(kh, kw, sh, sw, dh, dw, ph, pw, dtype, model_tag))

    for kh, kw, sh, sw, dh, dw, ph, pw in sorted(conv_depthwise):
        for dtype in dtypes:
            records.append(_conv2d_depthwise(kh, kw, sh, sw, dh, dw, ph, pw, dtype, model_tag))

    return records


# ── Method C: hardcoded (DeepSpeech2) ─────────────────────────────────────────

def deepspeech2_shapes(dtypes: list[str] = ["fp32"]) -> list[_Record]:
    """Standard DS2: 5-layer bidirectional LSTM, hidden=800, input_size=322 (=161*2)."""
    tag = "deepspeech2"
    records: list[_Record] = []
    for dtype in dtypes:
        if dtype == "fp32":
            records.append({
                "op_type": "lstm", "dtype": "fp32",
                "axes": {"input_size": 322, "hidden_size": 800},
                "model_tag": tag,
                "description": "DeepSpeech2 LSTM input=322 hidden=800",
            })
        # Output FC: N=29 (char vocab), K=800
        records.append(_gemm(29, 800, dtype, tag, "DS2 output FC N=29 K=800"))
    return records


# ── Aggregate ─────────────────────────────────────────────────────────────────

def get_all_shapes(
    dtypes_llm: list[str] = ["fp32", "q8_0"],
    dtypes_cv:  list[str] = ["fp32", "w8a8ch"],
    dtypes_asr: list[str] = ["fp32"],
) -> list[_Record]:
    """Return deduped shape records from all five models."""
    import torchvision

    records: list[_Record] = []

    for model_id, tag in [
        ("Qwen/Qwen1.5-MoE-A2.7B", "qwen1.5-moe-a2.7b"),
        ("allenai/OLMoE-1B-7B-0924", "olmoe-1b-7b"),
    ]:
        try:
            records += extract_hf_shapes(model_id, tag, dtypes_llm)
            print(f"  [hf] {model_id} ok")
        except Exception as e:
            print(f"  [hf] {model_id} FAILED: {e}")

    for model_fn, tag in [
        (torchvision.models.resnet50,           "resnet50"),
        (torchvision.models.mobilenet_v3_large, "mobilenetv3-large"),
    ]:
        try:
            records += extract_torchvision_shapes(model_fn, tag, dtypes=dtypes_cv)
            print(f"  [tv] {tag} ok")
        except Exception as e:
            print(f"  [tv] {tag} FAILED: {e}")

    records += deepspeech2_shapes(dtypes_asr)

    return _dedup(records)
