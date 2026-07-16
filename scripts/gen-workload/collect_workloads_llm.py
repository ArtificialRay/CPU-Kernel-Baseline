#!/usr/bin/env python3
"""
Collect LLM-model kernel workloads via llama.cpp on a remote Graviton
instance, using ShareGPT prompts as traffic.

Flow
----
1. Provision (or reuse) Graviton4 via a subprocess call to eval/provision.py
2. Rsync cpu-kernel-baseline/llama.cpp → remote ~/llama.cpp/
3. Upload scripts/collect_ggml_shapes.cpp and a CMakeLists.txt stub;
   patch remote examples/CMakeLists.txt to include the new target
4. Build collect-ggml-shapes on the remote
5. Sample ShareGPT prompts locally and rsync prompts.txt to remote
6. Run collect-ggml-shapes → ~/llm_shapes.json on the remote
7. rsync_from ~/llm_shapes.json to local scratch directory
8. Extract M-value distribution (M = token count per prompt, model-agnostic)
9. Call gen_workload.py for the requested --op-type/--definition

Note on model choice: any GGUF LLM works for M-value collection — we only
extract token counts (M) from ShareGPT prompts, which are independent of
model architecture. K/N shapes are fixed in the definitions.

Recommended models (already on remote, or download via huggingface_hub):
    ~/models/Llama-3.2-1B-Instruct-Q8_0.gguf       (already present)
    ~/models/OLMoE-1B-7B-0924-Instruct-Q8_0.gguf   (covers olmoe definitions)
    ~/models/Qwen1.5-MoE-A2.7B-Chat-Q8_0.gguf      (covers qwen1.5-moe defs)

Each op_type is backed by a small axes-builder registered in
LLM_AXES_BUILDERS (how to turn the collected M-value / per-expert-token
distribution into that op_type's workload axes). Definitions themselves are
read directly from bench-trace/definitions/<op_type>/ — adding a new
definition to an already-registered op_type needs no script changes.

Usage
-----
    python scripts/collect_workloads_llm.py \\
        --model ~/models/Llama-3.2-1B-Instruct-Q8_0.gguf \\
        --op-type gemm --definition gemm_fp32_n2048_k2048 \\
        --num-prompts 200

    python scripts/collect_workloads_llm.py \\
        --model ~/models/Llama-3.2-1B-Instruct-Q8_0.gguf \\
        --op-type mha --definition mha_fp32_h16_d128_kvh16 \\
        --num-prompts 50 --dry-run

    python scripts/collect_workloads_llm.py --list-op-types
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
LLAMA_LOCAL = REPO_ROOT.parent / "llama.cpp"
DEFS_DIR   = REPO_ROOT / "bench-trace" / "definitions"
CPP_SRC    = REPO_ROOT / "scripts" / "gen-workload" / "collect_ggml_shapes.cpp"

_EXAMPLE_CMAKE = """\
set(TARGET collect-ggml-shapes)
add_executable(${TARGET} collect-ggml-shapes.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
"""


# ── ShareGPT prompt sampling ───────────────────────────────────────────────────

_SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered"
    "/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
_SHAREGPT_CACHE = Path(tempfile.gettempdir()) / "ShareGPT_V3_unfiltered_cleaned_split.json"


def _load_sharegpt_prompts(n: int) -> list[str]:
    import urllib.request

    if not _SHAREGPT_CACHE.exists():
        print(f"[llm] Downloading ShareGPT → {_SHAREGPT_CACHE} ...")
        urllib.request.urlretrieve(_SHAREGPT_URL, _SHAREGPT_CACHE)
        print("[llm] Download complete.")
    else:
        print(f"[llm] Using cached ShareGPT at {_SHAREGPT_CACHE}")

    data = json.loads(_SHAREGPT_CACHE.read_text())
    prompts: list[str] = []
    for ex in data:
        if len(prompts) >= n:
            break
        for turn in ex.get("conversations", []):
            if turn.get("from") != "human":
                continue
            text = " ".join(turn["value"].split())   # collapse whitespace → single line
            if 10 < len(text.split()) < 1500:
                prompts.append(text)
                break

    if not prompts:
        sys.exit("[llm] No prompts extracted from ShareGPT dataset.")
    print(f"[llm] Extracted {len(prompts)} prompts")
    return prompts[:n]


# ── Per-op-type axes builders ──────────────────────────────────────────────────

def _build_m_axes(m_values: list[int], moe_m_values: Optional[list[int]]) -> list[dict]:
    return [{"M": m} for m in sorted(set([1] + [m for m in m_values if m > 0]))]


def _build_mha_axes(m_values: list[int], moe_m_values: Optional[list[int]]) -> list[dict]:
    # Prefill: M=S=prompt_len; decode: M=1,S=max_m (full KV cache)
    all_m = sorted(set([1] + [m for m in m_values if m > 0]))
    max_m = max(all_m)
    return [{"M": m, "S": m} for m in all_m] + [{"M": 1, "S": max_m}]


def _build_moe_axes(m_values: list[int], moe_m_values: Optional[list[int]]) -> list[dict]:
    # Use per-expert token counts from MoE "exps" tensors when available.
    # These give real routing distribution (M/n_expert_used) vs dense-model overestimate.
    effective = moe_m_values if moe_m_values else m_values
    return [{"n_tokens": m} for m in sorted(set([1] + [m for m in effective if m > 0]))]


LLM_AXES_BUILDERS: dict[str, Callable[[list[int], Optional[list[int]]], list[dict]]] = {
    "gemm":     _build_m_axes,
    "rms_norm": _build_m_axes,
    "mha":      _build_mha_axes,
    "moe":      _build_moe_axes,
}

LLM_BATCH_KEYS: dict[str, str] = {
    "gemm": "M", "rms_norm": "M", "mha": "M", "moe": "n_tokens",
}


def _w8a8ch_scalars(def_name: str) -> Optional[dict]:
    """w8a8ch definitions need a per-tensor activation input_scale that isn't
    provided by llama.cpp shape capture. Reference impls (see
    bench-trace/definitions/*/*_w8a8ch_*.json) dequantize int8 accumulation
    as acc * input_scale * weight_scales before clipping to int8 range —
    scale must be small enough that outputs don't uniformly saturate.
    Seeded by def_name so repeated collection runs are reproducible.
    """
    if "w8a8ch" not in def_name:
        return None
    scale = random.Random(def_name).uniform(0.005, 0.05)
    return {"input_scale": round(scale, 5)}


# ── Definition lookup ──────────────────────────────────────────────────────────

def _load_definition(op_type: str, def_name: str) -> dict:
    path = DEFS_DIR / op_type / f"{def_name}.json"
    if not path.exists():
        sys.exit(f"[llm] Definition not found: {path}")
    return json.loads(path.read_text())


# ── gen_workload.py subprocess helper ─────────────────────────────────────────

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
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


# ── Remote build / run ─────────────────────────────────────────────────────────

def _setup_remote(handle) -> str:
    """Rsync llama.cpp, upload C++ source, patch CMakeLists, build. Returns binary path."""

    if not LLAMA_LOCAL.exists():
        sys.exit(
            f"[llm] llama.cpp not found at {LLAMA_LOCAL}\n"
            f"      Clone it:  git clone --depth=1 https://github.com/ggml-org/llama.cpp.git"
            f" {LLAMA_LOCAL}"
        )

    print("[llm] Rsyncing llama.cpp to remote ~/llama.cpp/ ...")
    handle.rsync_to(
        str(LLAMA_LOCAL),
        "~/llama.cpp",
        excludes=["build", ".git", "/models", "*.gguf", "__pycache__"],
    )

    # Place collect_ggml_shapes.cpp as a new example subdir.
    handle.run("mkdir -p ~/llama.cpp/examples/collect-ggml-shapes")
    handle.upload_file(str(CPP_SRC),
                       "~/llama.cpp/examples/collect-ggml-shapes/collect-ggml-shapes.cpp")

    # Write CMakeLists.txt for the new example via heredoc.
    cmake_upload = tempfile.NamedTemporaryFile(
        mode="w", suffix=".cmake", delete=False, prefix="ggml_cmake_"
    )
    cmake_upload.write(_EXAMPLE_CMAKE)
    cmake_upload.flush()
    handle.upload_file(cmake_upload.name,
                       "~/llama.cpp/examples/collect-ggml-shapes/CMakeLists.txt")
    Path(cmake_upload.name).unlink(missing_ok=True)

    # Idempotently add add_subdirectory(collect-ggml-shapes) to examples CMakeLists.
    handle.run(
        "grep -qF 'collect-ggml-shapes' ~/llama.cpp/examples/CMakeLists.txt || "
        "sed -i 's|add_subdirectory(eval-callback)|add_subdirectory(eval-callback)\\n"
        "add_subdirectory(collect-ggml-shapes)|' ~/llama.cpp/examples/CMakeLists.txt"
    )

    print("[llm] Building collect-ggml-shapes on remote (may take a few minutes)...")
    build_cmd = (
        "set -e; cd ~/llama.cpp && "
        "cmake -B build "
        "  -DLLAMA_CURL=OFF "
        "  -DCMAKE_C_COMPILER=clang-18 "
        "  -DCMAKE_CXX_COMPILER=clang++-18 "
        "  -DCMAKE_BUILD_TYPE=Release "
        "  -DLLAMA_BUILD_TESTS=OFF "
        "  -DLLAMA_BUILD_TOOLS=OFF "
        "  -DLLAMA_BUILD_EXAMPLES=ON "
        "  2>&1 | tail -5 && "
        "cmake --build build --target collect-ggml-shapes -j$(nproc) 2>&1 | tail -10"
    )
    rc, out, err = handle.run(build_cmd, timeout=1200)
    if rc != 0:
        print(f"[llm] Build failed (rc={rc}):\nstdout: {out[-2000:]}\nstderr: {err[-500:]}")
        sys.exit(1)
    # Locate the binary (path differs across llama.cpp versions).
    rc2, bin_path, _ = handle.run(
        "find ~/llama.cpp/build -type f -name 'collect-ggml-shapes' 2>/dev/null | head -1"
    )
    bin_path = bin_path.strip()
    if not bin_path:
        sys.exit("[llm] collect-ggml-shapes binary not found after build.")
    print(f"[llm] Build succeeded → {bin_path}")
    return bin_path


def _run_collection(handle, bin_path: str, model_remote: str, prompts: list[str]) -> list[dict]:
    """Upload prompts, run collect-ggml-shapes, download and parse shapes."""
    # Upload prompts as a newline-separated text file.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="prompts_"
    )
    tmp.write("\n".join(prompts))
    tmp.flush()
    handle.upload_file(tmp.name, "~/llm_prompts.txt")
    Path(tmp.name).unlink(missing_ok=True)

    print(f"[llm] Running collect-ggml-shapes with model {model_remote} ...")
    run_cmd = (
        f"{bin_path} "
        f"-m {model_remote} "
        "--prompts-file ~/llm_prompts.txt "
        "--output ~/llm_shapes.json "
        "-ngl 0 -c 4096 -b 4096 "
        "2>&1 | tail -6"
    )
    rc, out, err = handle.run(run_cmd, timeout=7200)  # up to 2 h for large models
    print(f"[llm] Collection rc={rc}:\n{out}")
    if rc != 0:
        sys.exit(f"[llm] Collection failed:\n{err[-500:]}")

    # Download shapes JSON to a local temp dir.
    local_tmp = Path(tempfile.mkdtemp(prefix="ggml_shapes_"))
    handle.rsync_from("~/llm_shapes.json", str(local_tmp))
    shapes_file = local_tmp / "llm_shapes.json"
    if not shapes_file.exists():
        sys.exit("[llm] shapes file not found after rsync_from")

    records = json.loads(shapes_file.read_text())
    print(f"[llm] Downloaded {len(records)} shape records")
    return records


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect LLM-model kernel workloads via llama.cpp on remote Graviton."
    )
    parser.add_argument("--op-type", help="Op type to collect for (see --list-op-types).")
    parser.add_argument("--definition", help="Definition name to write workloads for.")
    parser.add_argument("--list-op-types", action="store_true",
                        help="Print supported op_types and exit.")
    parser.add_argument(
        "--model",
        help="REMOTE path to GGUF model on the Graviton instance "
             "(e.g. ~/models/llama-3.2-1b.Q8_0.gguf). "
             "Download the model to the remote first.",
    )
    parser.add_argument("--num-prompts", type=int, default=200,
                        help="Number of ShareGPT prompts to sample (default: 200)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print workloads without writing JSONL files")
    args = parser.parse_args()

    if args.list_op_types:
        print("\n".join(sorted(LLM_AXES_BUILDERS)))
        return

    if not args.op_type or not args.definition or not args.model:
        parser.error("--op-type, --definition, and --model are required "
                     "(or pass --list-op-types).")

    if args.op_type not in LLM_AXES_BUILDERS:
        parser.error(
            f"Unsupported op_type '{args.op_type}' for this collector. "
            f"Supported: {sorted(LLM_AXES_BUILDERS)}. If this op_type has no "
            f"ggml-hookable source, use gen_workload.py --add directly."
        )

    # Validates the definition exists before provisioning/building anything remote.
    _load_definition(args.op_type, args.definition)

    # Step 1 — provision. eval/provision.py is a standalone script (see its
    # module docstring) — invoke it via subprocess, then read the shared
    # eval/eval_config.json it wrote, rather than importing its internals.
    print("[llm] Provisioning/reusing Graviton4 instance...")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "eval" / "provision.py"), "--isa", "sve2"],
        check=True,
    )
    eval_config = json.loads((REPO_ROOT / "eval" / "eval_config.json").read_text())
    c8g = eval_config.get("instances", {}).get("c8g", {})
    if not c8g.get("host"):
        raise RuntimeError("eval/provision.py exited successfully but wrote no c8g instance")

    sys.path.insert(0, str(REPO_ROOT))
    from eval.remote import InstanceHandle  # noqa: PLC0415

    handle = InstanceHandle(
        host=c8g["host"], user=c8g.get("user", "ubuntu"),
        key_file=c8g.get("key_file", "~/.ssh/id_rsa"), instance_type="c8g.large",
    )

    # Step 2–4 — sync + build.
    bin_path = _setup_remote(handle)

    # Step 5 — ShareGPT prompts.
    prompts = _load_sharegpt_prompts(args.num_prompts)

    # Step 6–7 — collect shapes on remote, download results.
    records = _run_collection(handle, bin_path, args.model, prompts)

    # Step 8 — split records: general M values (non-MoE) vs per-expert token counts (MoE).
    # OLMoE "exps" tensors (ffn_gate_exps / ffn_up_exps / ffn_down_exps) give real
    # per-expert token counts; non-exps records give sequence-level M for gemm/mha/rms_norm.
    m_values = [r["M"] for r in records if r["M"] > 0 and "exps" not in r.get("name", "")]
    moe_m_values = [r["M"] for r in records if r["M"] > 0 and "exps" in r.get("name", "")]
    print(f"[llm] M values (non-MoE): {len(m_values)} (unique: {len(set(m_values))})")
    print(f"[llm] M values (MoE exps): {len(moe_m_values)} (unique: {len(set(moe_m_values))})")
    if not moe_m_values:
        print("[llm] WARNING: no exps records found — model may not be MoE. Using general m_values for moe defs.")

    # Step 9 — generate workloads for the one requested definition.
    print(f"[llm] Generating workloads for {args.op_type}/{args.definition} "
          f"(dry_run={args.dry_run})...")
    axes_list = LLM_AXES_BUILDERS[args.op_type](m_values, moe_m_values or None)
    scalars = _w8a8ch_scalars(args.definition)
    _gen_workload(args.definition, axes_list, args.dry_run,
                  batch_key=LLM_BATCH_KEYS[args.op_type], scalars=scalars)
    print("[llm] Done.")


if __name__ == "__main__":
    main()
