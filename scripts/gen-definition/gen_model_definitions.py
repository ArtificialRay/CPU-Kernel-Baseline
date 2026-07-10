#!/usr/bin/env python3
"""Generate Definition JSONs for new op types from real model shapes.

Run from cpu-kernel-baseline/:
    python scripts/gen-definition/gen_model_definitions.py

Writes to bench-trace/definitions/<op_type>/<name>.json.
No workload generation — add workloads separately via /gen-workload.

Op types covered (not in gen_definitions.py):
  gemm             fp32 / q8_0 (llama.cpp Q8_0) / w8a8ch (ncnn per-channel)
  rms_norm         fp32
  mha              fp32
  lstm             fp32
  moe              fp32 / q8_0
  pooling          fp32
  conv2d           fp32 / w8a8ch (ncnn per-output-channel weight, per-tensor activation)
  conv2d_depthwise fp32 / w8a8ch (ncnn per-channel weight AND per-channel activation)
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Callable

REPO     = Path(__file__).resolve().parent.parent.parent
DEFS_DIR = REPO / "bench-trace" / "definitions"

_Record = dict[str, Any]


# ── I/O ───────────────────────────────────────────────────────────────────────

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


# ── Reference functions ────────────────────────────────────────────────────────

def _ref_gemm_fp32() -> str:
    return (
        "import numpy as np\n"
        "def run(A, B):\n"
        "    return A @ B.T\n"
    )


def _ref_gemm_q8_0(N: int, K: int) -> str:
    return textwrap.dedent(f"""\
        import numpy as np
        def run(A, A_scales, B, B_scales):
            M = A.shape[0]; blk = 32
            A_f = (A.reshape(M, -1, blk).astype(np.float32)
                   * A_scales.astype(np.float32)[:, :, np.newaxis]).reshape(M, {K})
            B_f = (B.reshape({N}, -1, blk).astype(np.float32)
                   * B_scales.astype(np.float32)[:, :, np.newaxis]).reshape({N}, {K})
            return A_f @ B_f.T
        """)


def _ref_gemm_w8a8ch(N: int) -> str:
    # Full-precision reference: dequantize A/B to float32 first (real = int8 * scale),
    # then a plain float32 matmul — matches gemm_q4_k_m's dequant-first style rather
    # than exact-integer accumulation + a single closing rescale.
    return textwrap.dedent(f"""\
        import numpy as np
        def run(A, B, input_scale, weight_scales):
            A_f = A.astype(np.float32) * np.float32(input_scale)
            B_f = B.astype(np.float32) * weight_scales.astype(np.float32)[:, np.newaxis]
            return (A_f @ B_f.T).astype(np.float32)
        """)


def _ref_rms_norm() -> str:
    return (
        "import numpy as np\n"
        "def run(x, weight):\n"
        "    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)\n"
        "    return (x / rms) * weight\n"
    )


def _ref_mha(n_heads: int, head_dim: int, kv_heads: int) -> str:
    scale = head_dim ** -0.5
    groups = n_heads // kv_heads
    expand = f"np.repeat(K, {groups}, axis=1)" if groups > 1 else "K"
    expand_v = f"np.repeat(V, {groups}, axis=1)" if groups > 1 else "V"
    return textwrap.dedent(f"""\
        import numpy as np
        def run(Q, K, V):
            M = Q.shape[0]; S = K.shape[0]
            K_e = {expand}
            V_e = {expand_v}
            Qt = Q.transpose(1, 0, 2)
            Kt = K_e.transpose(1, 0, 2)
            Vt = V_e.transpose(1, 0, 2)
            scores = np.matmul(Qt, Kt.transpose(0, 2, 1)) * {scale}
            scores -= scores.max(-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(-1, keepdims=True)
            return np.matmul(probs, Vt).transpose(1, 0, 2)
        """)


def _ref_lstm(hidden_size: int) -> str:
    H = hidden_size
    return textwrap.dedent(f"""\
        import numpy as np
        def run(x, h0, c0, W_ih, W_hh, b):
            T = x.shape[0]; H = {H}
            h, c = h0.copy(), c0.copy()
            out = np.empty((T, H), dtype=np.float32)
            for t in range(T):
                gates = W_ih @ x[t] + W_hh @ h + b
                i = 1 / (1 + np.exp(-gates[:H]))
                f = 1 / (1 + np.exp(-gates[H:2*H]))
                g = np.tanh(gates[2*H:3*H])
                o = 1 / (1 + np.exp(-gates[3*H:]))
                c = f * c + i * g
                h = o * np.tanh(c)
                out[t] = h
            return out
        """)


def _ref_moe_fp32(n_embd: int, n_expert_used: int) -> str:
    return textwrap.dedent(f"""\
        import numpy as np
        def run(hidden_states, router_weight, gate_proj, up_proj, down_proj):
            T = hidden_states.shape[0]
            logits = hidden_states @ router_weight.T
            probs = np.exp(logits - logits.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            top_idx = np.argsort(probs, axis=-1)[:, -{n_expert_used}:]
            top_w = np.take_along_axis(probs, top_idx, axis=-1)
            top_w /= top_w.sum(-1, keepdims=True)
            out = np.zeros((T, {n_embd}), dtype=np.float32)
            for t in range(T):
                for ki in range({n_expert_used}):
                    e = top_idx[t, ki]
                    g = gate_proj[e] @ hidden_states[t]
                    u = up_proj[e] @ hidden_states[t]
                    silu = g / (1 + np.exp(-g))
                    out[t] += top_w[t, ki] * (down_proj[e] @ (silu * u))
            return out
        """)


def _ref_moe_q8_0(n_embd: int, n_expert_used: int) -> str:
    return textwrap.dedent(f"""\
        import numpy as np
        def run(hidden_states, hs_scales, router_weight,
                gate_proj, gate_scales, up_proj, up_scales,
                down_proj, down_scales):
            T = hidden_states.shape[0]; blk = 32
            def dq(x, s):
                r, K = x.shape
                return (x.reshape(r, -1, blk).astype(np.float32)
                        * s.astype(np.float32)[:, :, np.newaxis]).reshape(r, K)
            hs_f = dq(hidden_states, hs_scales)
            logits = hs_f @ router_weight.T
            probs = np.exp(logits - logits.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            top_idx = np.argsort(probs, axis=-1)[:, -{n_expert_used}:]
            top_w = np.take_along_axis(probs, top_idx, axis=-1)
            top_w /= top_w.sum(-1, keepdims=True)
            out = np.zeros((T, {n_embd}), dtype=np.float32)
            for t in range(T):
                for ki in range({n_expert_used}):
                    e = top_idx[t, ki]
                    gp = dq(gate_proj[e], gate_scales[e])
                    up = dq(up_proj[e], up_scales[e])
                    dn = dq(down_proj[e], down_scales[e])
                    g = gp @ hs_f[t]; u = up @ hs_f[t]
                    silu = g / (1 + np.exp(-g))
                    out[t] += top_w[t, ki] * (dn @ (silu * u))
            return out
        """)


def _ref_pooling() -> str:
    return (
        "import numpy as np\n"
        "def run(input):\n"
        "    return input.mean(axis=(-2, -1))\n"
    )


def _ref_conv2d_fp32(sh: int, sw: int, dh: int, dw: int, pad_top: int, pad_left: int) -> str:
    return textwrap.dedent(f"""\
        import torch
        import torch.nn.functional as F

        def run(input, weight, bias):
            x = torch.from_numpy(input)
            w = torch.from_numpy(weight)
            b = torch.from_numpy(bias)
            return F.conv2d(x, w, b, stride=({sh}, {sw}), padding=({pad_top}, {pad_left}),
                             dilation=({dh}, {dw})).numpy()
        """)


def _ref_conv2d_w8a8ch(kh: int, kw: int, sh: int, sw: int, dh: int, dw: int,
                        pad_top: int, pad_left: int) -> str:
    # Full-precision reference: dequantize input/weight to float32 first (real =
    # int8 * scale), then accumulate the convolution in plain float32 — matches
    # gemm_q4_k_m's dequant-first style rather than exact-integer accumulation +
    # a single closing rescale.
    return textwrap.dedent(f"""\
        import numpy as np

        def run(input, weight, bias, input_scale, weight_scales):
            N, Cin, H, W = input.shape
            Cout = weight.shape[0]
            Kh, Kw = {kh}, {kw}
            Sh, Sw = {sh}, {sw}
            Dh, Dw = {dh}, {dw}
            Ph, Pw = {pad_top}, {pad_left}
            Hout = (H + 2 * Ph - Dh * (Kh - 1) - 1) // Sh + 1
            Wout = (W + 2 * Pw - Dw * (Kw - 1) - 1) // Sw + 1
            input_f = input.astype(np.float32) * np.float32(input_scale)
            weight_f = weight.astype(np.float32) * weight_scales.astype(np.float32)[:, np.newaxis, np.newaxis, np.newaxis]
            xp = np.pad(input_f, ((0, 0), (0, 0), (Ph, Ph), (Pw, Pw)))
            acc = np.zeros((N, Cout, Hout, Wout), dtype=np.float32)
            for kh in range(Kh):
                for kw in range(Kw):
                    patch = xp[:, :, kh * Dh: kh * Dh + Sh * Hout: Sh,
                                     kw * Dw: kw * Dw + Sw * Wout: Sw]
                    acc += np.einsum('ncHW,oc->noHW', patch, weight_f[:, :, kh, kw])
            acc += bias.astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]
            return acc.astype(np.float32)
        """)


def _ref_conv2d_depthwise_fp32(sh: int, sw: int, dh: int, dw: int,
                                pad_top: int, pad_left: int) -> str:
    return textwrap.dedent(f"""\
        import torch
        import torch.nn.functional as F

        def run(input, weight, bias):
            x = torch.from_numpy(input)
            w = torch.from_numpy(weight)
            b = torch.from_numpy(bias)
            return F.conv2d(x, w, b, stride=({sh}, {sw}), padding=({pad_top}, {pad_left}),
                             dilation=({dh}, {dw}), groups=x.shape[1]).numpy()
        """)


def _ref_conv2d_depthwise_w8a8ch(kh: int, kw: int, sh: int, sw: int, dh: int, dw: int,
                                  pad_top: int, pad_left: int) -> str:
    # Full-precision reference: dequantize input/weight to float32 first (real =
    # int8 * scale, both per-channel here), then accumulate in plain float32 — matches
    # gemm_q4_k_m's dequant-first style rather than exact-integer accumulation + a
    # single closing rescale.
    return textwrap.dedent(f"""\
        import numpy as np

        def run(input, weight, bias, input_scales, weight_scales):
            N, C, H, W = input.shape
            Kh, Kw = {kh}, {kw}
            Sh, Sw = {sh}, {sw}
            Dh, Dw = {dh}, {dw}
            Ph, Pw = {pad_top}, {pad_left}
            Hout = (H + 2 * Ph - Dh * (Kh - 1) - 1) // Sh + 1
            Wout = (W + 2 * Pw - Dw * (Kw - 1) - 1) // Sw + 1
            input_f = input.astype(np.float32) * input_scales.astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]
            weight_f = weight.astype(np.float32)[:, 0, :, :] * weight_scales.astype(np.float32)[:, np.newaxis, np.newaxis]
            xp = np.pad(input_f, ((0, 0), (0, 0), (Ph, Ph), (Pw, Pw)))
            acc = np.zeros((N, C, Hout, Wout), dtype=np.float32)
            for kh in range(Kh):
                for kw in range(Kw):
                    patch = xp[:, :, kh * Dh: kh * Dh + Sh * Hout: Sh,
                                     kw * Dw: kw * Dw + Sw * Wout: Sw]
                    acc += patch * weight_f[:, kh, kw][np.newaxis, :, np.newaxis, np.newaxis]
            acc += bias.astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]
            return acc.astype(np.float32)
        """)


# ── Builders ──────────────────────────────────────────────────────────────────

def _tags(model_tag: str) -> list[str]:
    return ["status:active", f"model:{model_tag}"]


def _build_gemm_fp32(r: _Record) -> dict:
    N, K = r["axes"]["N"], r["axes"]["K"]
    return {
        "name": f"gemm_fp32_n{N}_k{K}",
        "op_type": "gemm",
        "description": r["description"],
        "tags": _tags(r["model_tag"]),
        "axes": {
            "M": {"type": "var"},
            "N": {"type": "const", "value": N},
            "K": {"type": "const", "value": K},
        },
        "inputs": {
            "A": {"shape": ["M", "K"], "dtype": "float32"},
            "B": {"shape": ["N", "K"], "dtype": "float32"},
        },
        "outputs": {
            "C": {"shape": ["M", "N"], "dtype": "float32"},
        },
        "constraints": [],
        "reference": _ref_gemm_fp32(),
    }


def _build_gemm_q8_0(r: _Record) -> dict:
    N, K = r["axes"]["N"], r["axes"]["K"]
    assert K % 32 == 0, f"Q8_0 requires K divisible by 32, got K={K}"
    K_blk = K // 32
    return {
        "name": f"gemm_q8_0_n{N}_k{K}",
        "op_type": "gemm",
        "description": r["description"],
        "tags": _tags(r["model_tag"]),
        "axes": {
            "M":     {"type": "var"},
            "N":     {"type": "const", "value": N},
            "K":     {"type": "const", "value": K},
            "K_blk": {"type": "const", "value": K_blk},
        },
        "inputs": {
            "A":        {"shape": ["M", "K"],     "dtype": "int8"},
            "A_scales": {"shape": ["M", "K_blk"], "dtype": "float16"},
            "B":        {"shape": ["N", "K"],     "dtype": "int8"},
            "B_scales": {"shape": ["N", "K_blk"], "dtype": "float16"},
        },
        "outputs": {
            "C": {"shape": ["M", "N"], "dtype": "float32"},
        },
        "constraints": [],
        "reference": _ref_gemm_q8_0(N, K),
    }


def _build_gemm_w8a8ch(r: _Record) -> dict:
    N, K = r["axes"]["N"], r["axes"]["K"]
    return {
        "name": f"gemm_w8a8ch_n{N}_k{K}",
        "op_type": "gemm",
        "description": r["description"],
        "tags": _tags(r["model_tag"]),
        "axes": {
            "M": {"type": "var"},
            "N": {"type": "const", "value": N},
            "K": {"type": "const", "value": K},
        },
        "inputs": {
            "A":             {"shape": ["M", "K"], "dtype": "int8"},
            "B":             {"shape": ["N", "K"], "dtype": "int8"},
            "input_scale":   {"shape": None,       "dtype": "float32"},
            "weight_scales": {"shape": ["N"],      "dtype": "float32"},
        },
        "outputs": {
            "C": {"shape": ["M", "N"], "dtype": "int8"},
        },
        "constraints": [],
        "reference": _ref_gemm_w8a8ch(N),
    }


def _build_rms_norm(r: _Record) -> dict:
    D = r["axes"]["D"]
    return {
        "name": f"rms_norm_fp32_d{D}",
        "op_type": "rms_norm",
        "description": r["description"],
        "tags": _tags(r["model_tag"]),
        "axes": {
            "M": {"type": "var"},
            "D": {"type": "const", "value": D},
        },
        "inputs": {
            "x":      {"shape": ["M", "D"], "dtype": "float32"},
            "weight": {"shape": ["D"],      "dtype": "float32"},
        },
        "outputs": {
            "output": {"shape": ["M", "D"], "dtype": "float32"},
        },
        "constraints": [],
        "reference": _ref_rms_norm(),
    }


def _build_mha(r: _Record) -> dict:
    n_heads  = r["axes"]["n_heads"]
    head_dim = r["axes"]["head_dim"]
    kv_heads = r["axes"]["kv_heads"]
    return {
        "name": f"mha_fp32_h{n_heads}_d{head_dim}_kvh{kv_heads}",
        "op_type": "mha",
        "description": r["description"],
        "tags": _tags(r["model_tag"]),
        "axes": {
            "M":        {"type": "var"},
            "S":        {"type": "var"},
            "n_heads":  {"type": "const", "value": n_heads},
            "head_dim": {"type": "const", "value": head_dim},
            "kv_heads": {"type": "const", "value": kv_heads},
        },
        "inputs": {
            "Q": {"shape": ["M", "n_heads",  "head_dim"], "dtype": "float32"},
            "K": {"shape": ["S", "kv_heads", "head_dim"], "dtype": "float32"},
            "V": {"shape": ["S", "kv_heads", "head_dim"], "dtype": "float32"},
        },
        "outputs": {
            "output": {"shape": ["M", "n_heads", "head_dim"], "dtype": "float32"},
        },
        "constraints": [],
        "reference": _ref_mha(n_heads, head_dim, kv_heads),
    }


def _build_lstm(r: _Record) -> dict:
    input_size  = r["axes"]["input_size"]
    hidden_size = r["axes"]["hidden_size"]
    hidden_x4   = 4 * hidden_size
    return {
        "name": f"lstm_fp32_i{input_size}_h{hidden_size}",
        "op_type": "lstm",
        "description": r["description"],
        "tags": _tags(r["model_tag"]),
        "axes": {
            "T":           {"type": "var"},
            "input_size":  {"type": "const", "value": input_size},
            "hidden_size": {"type": "const", "value": hidden_size},
            "hidden_x4":   {"type": "const", "value": hidden_x4},
        },
        "inputs": {
            "x":    {"shape": ["T",         "input_size"],  "dtype": "float32"},
            "h0":   {"shape": ["hidden_size"],              "dtype": "float32"},
            "c0":   {"shape": ["hidden_size"],              "dtype": "float32"},
            "W_ih": {"shape": ["hidden_x4", "input_size"],  "dtype": "float32"},
            "W_hh": {"shape": ["hidden_x4", "hidden_size"], "dtype": "float32"},
            "b":    {"shape": ["hidden_x4"],                "dtype": "float32"},
        },
        "outputs": {
            "output": {"shape": ["T", "hidden_size"], "dtype": "float32"},
        },
        "constraints": [],
        "reference": _ref_lstm(hidden_size),
    }


def _build_moe_fp32(r: _Record) -> dict:
    ax = r["axes"]
    n_embd, n_ff = ax["n_embd"], ax["n_ff"]
    n_expert, n_expert_used = ax["n_expert"], ax["n_expert_used"]
    return {
        "name": f"moe_fp32_e{n_expert}_k{n_expert_used}_d{n_embd}_ff{n_ff}",
        "op_type": "moe",
        "description": r["description"],
        "tags": _tags(r["model_tag"]),
        "axes": {
            "n_tokens":      {"type": "var"},
            "n_embd":        {"type": "const", "value": n_embd},
            "n_ff":          {"type": "const", "value": n_ff},
            "n_expert":      {"type": "const", "value": n_expert},
            "n_expert_used": {"type": "const", "value": n_expert_used},
        },
        "inputs": {
            "hidden_states": {"shape": ["n_tokens", "n_embd"],            "dtype": "float32"},
            "router_weight": {"shape": ["n_expert", "n_embd"],            "dtype": "float32"},
            "gate_proj":     {"shape": ["n_expert", "n_ff",   "n_embd"],  "dtype": "float32"},
            "up_proj":       {"shape": ["n_expert", "n_ff",   "n_embd"],  "dtype": "float32"},
            "down_proj":     {"shape": ["n_expert", "n_embd", "n_ff"],    "dtype": "float32"},
        },
        "outputs": {
            "output": {"shape": ["n_tokens", "n_embd"], "dtype": "float32"},
        },
        "constraints": [],
        "reference": _ref_moe_fp32(n_embd, n_expert_used),
    }


def _build_moe_q8_0(r: _Record) -> dict:
    ax = r["axes"]
    n_embd, n_ff = ax["n_embd"], ax["n_ff"]
    n_expert, n_expert_used = ax["n_expert"], ax["n_expert_used"]
    assert n_embd % 32 == 0, f"Q8_0 requires n_embd divisible by 32, got {n_embd}"
    assert n_ff   % 32 == 0, f"Q8_0 requires n_ff divisible by 32, got {n_ff}"
    n_embd_blk = n_embd // 32
    n_ff_blk   = n_ff   // 32
    return {
        "name": f"moe_q8_0_e{n_expert}_k{n_expert_used}_d{n_embd}_ff{n_ff}",
        "op_type": "moe",
        "description": r["description"],
        "tags": _tags(r["model_tag"]),
        "axes": {
            "n_tokens":      {"type": "var"},
            "n_embd":        {"type": "const", "value": n_embd},
            "n_ff":          {"type": "const", "value": n_ff},
            "n_expert":      {"type": "const", "value": n_expert},
            "n_expert_used": {"type": "const", "value": n_expert_used},
            "n_embd_blk":    {"type": "const", "value": n_embd_blk},
            "n_ff_blk":      {"type": "const", "value": n_ff_blk},
        },
        "inputs": {
            "hidden_states": {"shape": ["n_tokens", "n_embd"],                    "dtype": "int8"},
            "hs_scales":     {"shape": ["n_tokens", "n_embd_blk"],                "dtype": "float16"},
            "router_weight": {"shape": ["n_expert", "n_embd"],                    "dtype": "float32"},
            "gate_proj":     {"shape": ["n_expert", "n_ff",   "n_embd"],          "dtype": "int8"},
            "gate_scales":   {"shape": ["n_expert", "n_ff",   "n_embd_blk"],      "dtype": "float16"},
            "up_proj":       {"shape": ["n_expert", "n_ff",   "n_embd"],          "dtype": "int8"},
            "up_scales":     {"shape": ["n_expert", "n_ff",   "n_embd_blk"],      "dtype": "float16"},
            "down_proj":     {"shape": ["n_expert", "n_embd", "n_ff"],            "dtype": "int8"},
            "down_scales":   {"shape": ["n_expert", "n_embd", "n_ff_blk"],        "dtype": "float16"},
        },
        "outputs": {
            "output": {"shape": ["n_tokens", "n_embd"], "dtype": "float32"},
        },
        "constraints": [],
        "reference": _ref_moe_q8_0(n_embd, n_expert_used),
    }


def _build_pooling(r: _Record) -> dict:
    return {
        "name": "pooling_fp32_global_avg",
        "op_type": "pooling",
        "description": r["description"],
        "tags": _tags(r["model_tag"]),
        "axes": {
            "N": {"type": "var"},
            "C": {"type": "var"},
            "H": {"type": "var", "parent": "N"},
            "W": {"type": "var", "parent": "N"},
        },
        "inputs": {
            "input": {"shape": ["N", "C", "H", "W"], "dtype": "float32"},
        },
        "outputs": {
            "output": {"shape": ["N", "C"], "dtype": "float32"},
        },
        "constraints": [],
        "reference": _ref_pooling(),
    }


def _conv2d_axes(ax: dict) -> tuple[int, int, int, int, int, int, int, int]:
    return (ax["Kh"], ax["Kw"], ax["Sh"], ax["Sw"], ax["Dh"], ax["Dw"],
            ax["pad_top"], ax["pad_left"])


def _build_conv2d_fp32(r: _Record) -> dict:
    kh, kw, sh, sw, dh, dw, pt, pl = _conv2d_axes(r["axes"])
    return {
        "name": f"conv2d_fp32_kh{kh}_kw{kw}_sh{sh}_sw{sw}_dh{dh}_dw{dw}_p{pt}",
        "op_type": "conv2d",
        "description": r["description"],
        "tags": _tags(r["model_tag"]),
        "axes": {
            "N": {"type": "var"},
            "H": {"type": "var", "parent": "N"},
            "W": {"type": "var", "parent": "N"},
            "H_out": {"type": "var", "parent": "N"},
            "W_out": {"type": "var", "parent": "N"},
            "C_in": {"type": "var"},
            "C_out": {"type": "var"},
            "Kh": {"type": "const", "value": kh},
            "Kw": {"type": "const", "value": kw},
            "Sh": {"type": "const", "value": sh},
            "Sw": {"type": "const", "value": sw},
            "Dh": {"type": "const", "value": dh},
            "Dw": {"type": "const", "value": dw},
            "pad_top": {"type": "const", "value": pt},
            "pad_left": {"type": "const", "value": pl},
        },
        "inputs": {
            "input": {"shape": ["N", "C_in", "H", "W"], "dtype": "float32"},
            "weight": {"shape": ["C_out", "C_in", "Kh", "Kw"], "dtype": "float32"},
            "bias": {"shape": ["C_out"], "dtype": "float32"},
        },
        "outputs": {
            "output": {"shape": ["N", "C_out", "H_out", "W_out"], "dtype": "float32"},
        },
        "constraints": [
            "H_out == (H + 2*pad_top - Dh*(Kh-1) - 1) // Sh + 1",
            "W_out == (W + 2*pad_left - Dw*(Kw-1) - 1) // Sw + 1",
        ],
        "reference": _ref_conv2d_fp32(sh, sw, dh, dw, pt, pl),
    }


def _build_conv2d_w8a8ch(r: _Record) -> dict:
    kh, kw, sh, sw, dh, dw, pt, pl = _conv2d_axes(r["axes"])
    return {
        "name": f"conv2d_w8a8ch_kh{kh}_kw{kw}_sh{sh}_sw{sw}_dh{dh}_dw{dw}_p{pt}",
        "op_type": "conv2d",
        "description": r["description"],
        "tags": _tags(r["model_tag"]),
        "axes": {
            "N": {"type": "var"},
            "H": {"type": "var", "parent": "N"},
            "W": {"type": "var", "parent": "N"},
            "H_out": {"type": "var", "parent": "N"},
            "W_out": {"type": "var", "parent": "N"},
            "C_in": {"type": "var"},
            "C_out": {"type": "var"},
            "Kh": {"type": "const", "value": kh},
            "Kw": {"type": "const", "value": kw},
            "Sh": {"type": "const", "value": sh},
            "Sw": {"type": "const", "value": sw},
            "Dh": {"type": "const", "value": dh},
            "Dw": {"type": "const", "value": dw},
            "pad_top": {"type": "const", "value": pt},
            "pad_left": {"type": "const", "value": pl},
        },
        "inputs": {
            "input": {"shape": ["N", "C_in", "H", "W"], "dtype": "int8"},
            "weight": {"shape": ["C_out", "C_in", "Kh", "Kw"], "dtype": "int8"},
            "bias": {"shape": ["C_out"], "dtype": "float32"},
            "input_scale": {"shape": None, "dtype": "float32"},
            "weight_scales": {"shape": ["C_out"], "dtype": "float32"},
        },
        "outputs": {
            "output": {"shape": ["N", "C_out", "H_out", "W_out"], "dtype": "int8"},
        },
        "constraints": [
            "H_out == (H + 2*pad_top - Dh*(Kh-1) - 1) // Sh + 1",
            "W_out == (W + 2*pad_left - Dw*(Kw-1) - 1) // Sw + 1",
        ],
        "reference": _ref_conv2d_w8a8ch(kh, kw, sh, sw, dh, dw, pt, pl),
    }


def _build_conv2d_depthwise_fp32(r: _Record) -> dict:
    kh, kw, sh, sw, dh, dw, pt, pl = _conv2d_axes(r["axes"])
    return {
        "name": f"conv2d_depthwise_fp32_kh{kh}_kw{kw}_sh{sh}_sw{sw}_dh{dh}_dw{dw}_p{pt}",
        "op_type": "conv2d_depthwise",
        "description": r["description"],
        "tags": _tags(r["model_tag"]),
        "axes": {
            "N": {"type": "var"},
            "H": {"type": "var", "parent": "N"},
            "W": {"type": "var", "parent": "N"},
            "H_out": {"type": "var", "parent": "N"},
            "W_out": {"type": "var", "parent": "N"},
            "C": {"type": "var"},
            "C_group": {"type": "const", "value": 1},
            "Kh": {"type": "const", "value": kh},
            "Kw": {"type": "const", "value": kw},
            "Sh": {"type": "const", "value": sh},
            "Sw": {"type": "const", "value": sw},
            "Dh": {"type": "const", "value": dh},
            "Dw": {"type": "const", "value": dw},
            "pad_top": {"type": "const", "value": pt},
            "pad_left": {"type": "const", "value": pl},
        },
        "inputs": {
            "input": {"shape": ["N", "C", "H", "W"], "dtype": "float32"},
            "weight": {"shape": ["C", "C_group", "Kh", "Kw"], "dtype": "float32"},
            "bias": {"shape": ["C"], "dtype": "float32"},
        },
        "outputs": {
            "output": {"shape": ["N", "C", "H_out", "W_out"], "dtype": "float32"},
        },
        "constraints": [
            "H_out == (H + 2*pad_top - Dh*(Kh-1) - 1) // Sh + 1",
            "W_out == (W + 2*pad_left - Dw*(Kw-1) - 1) // Sw + 1",
        ],
        "reference": _ref_conv2d_depthwise_fp32(sh, sw, dh, dw, pt, pl),
    }


def _build_conv2d_depthwise_w8a8ch(r: _Record) -> dict:
    kh, kw, sh, sw, dh, dw, pt, pl = _conv2d_axes(r["axes"])
    return {
        "name": f"conv2d_depthwise_w8a8ch_kh{kh}_kw{kw}_sh{sh}_sw{sw}_dh{dh}_dw{dw}_p{pt}",
        "op_type": "conv2d_depthwise",
        "description": r["description"],
        "tags": _tags(r["model_tag"]),
        "axes": {
            "N": {"type": "var"},
            "H": {"type": "var", "parent": "N"},
            "W": {"type": "var", "parent": "N"},
            "H_out": {"type": "var", "parent": "N"},
            "W_out": {"type": "var", "parent": "N"},
            "C": {"type": "var"},
            "C_group": {"type": "const", "value": 1},
            "Kh": {"type": "const", "value": kh},
            "Kw": {"type": "const", "value": kw},
            "Sh": {"type": "const", "value": sh},
            "Sw": {"type": "const", "value": sw},
            "Dh": {"type": "const", "value": dh},
            "Dw": {"type": "const", "value": dw},
            "pad_top": {"type": "const", "value": pt},
            "pad_left": {"type": "const", "value": pl},
        },
        "inputs": {
            "input": {"shape": ["N", "C", "H", "W"], "dtype": "int8"},
            "weight": {"shape": ["C", "C_group", "Kh", "Kw"], "dtype": "int8"},
            "bias": {"shape": ["C"], "dtype": "float32"},
            "input_scales": {"shape": ["C"], "dtype": "float32"},
            "weight_scales": {"shape": ["C"], "dtype": "float32"},
        },
        "outputs": {
            "output": {"shape": ["N", "C", "H_out", "W_out"], "dtype": "int8"},
        },
        "constraints": [
            "H_out == (H + 2*pad_top - Dh*(Kh-1) - 1) // Sh + 1",
            "W_out == (W + 2*pad_left - Dw*(Kw-1) - 1) // Sw + 1",
        ],
        "reference": _ref_conv2d_depthwise_w8a8ch(kh, kw, sh, sw, dh, dw, pt, pl),
    }


# ── Builder registry ──────────────────────────────────────────────────────────

BUILDERS: dict[str, Callable[[_Record], dict]] = {
    "gemm/fp32":               _build_gemm_fp32,
    "gemm/q8_0":               _build_gemm_q8_0,
    "gemm/w8a8ch":             _build_gemm_w8a8ch,
    "rms_norm/fp32":           _build_rms_norm,
    "mha/fp32":                _build_mha,
    "lstm/fp32":               _build_lstm,
    "moe/fp32":                _build_moe_fp32,
    "moe/q8_0":                _build_moe_q8_0,
    "pooling/fp32":            _build_pooling,
    "conv2d/fp32":             _build_conv2d_fp32,
    "conv2d/w8a8ch":           _build_conv2d_w8a8ch,
    "conv2d_depthwise/fp32":   _build_conv2d_depthwise_fp32,
    "conv2d_depthwise/w8a8ch": _build_conv2d_depthwise_w8a8ch,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "gen-definition"))
    from model_shapes import get_all_shapes

    print("Extracting shapes from models...")
    records = get_all_shapes()
    print(f"  {len(records)} unique shape records\n")

    written = skipped = errors = 0
    for r in records:
        key = f"{r['op_type']}/{r['dtype']}"
        builder = BUILDERS.get(key)
        if builder is None:
            print(f"  [skip]  {key}  (no builder)")
            skipped += 1
            continue
        try:
            defn = builder(r)
        except Exception as e:
            print(f"  [error] {key} axes={r['axes']}: {e}")
            errors += 1
            continue

        name     = defn["name"]
        op_type  = defn["op_type"]
        out_path = DEFS_DIR / op_type / f"{name}.json"

        # Skip if file already exists with matching structure (tags may differ — first model wins)
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            if existing == defn:
                print(f"  [same]  {name}")
                continue
            ex_notags = {k: v for k, v in existing.items() if k != "tags"}
            new_notags = {k: v for k, v in defn.items() if k != "tags"}
            if ex_notags == new_notags:
                print(f"  [keep]  {name}  (tags differ, first model wins)")
                continue

        _write_json(out_path, defn)
        print(f"  [write] {name}")
        written += 1

    print(f"\nDone: {written} written, {skipped} skipped, {errors} errors.")


if __name__ == "__main__":
    main()
