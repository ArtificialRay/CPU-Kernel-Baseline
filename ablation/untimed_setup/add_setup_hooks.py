#!/usr/bin/env python3
"""Give the expert baselines an untimed setup hook, in place, in a dataset root.

The harness on this branch calls armbench_setup_<op_type> once per workload,
outside the timing (see config/kernel_contracts.yaml::setup_weights). This
script adds that hook to the baselines whose timed calls still do one-time
preparation:

- baseline-ncnn-arm, conv2d and conv2d_depthwise: every call builds a
  Convolution(DepthWise)_arm layer and runs create_pipeline (weight
  transform / packing) before forward. Setup now builds the layer once and
  keeps it; every later call only runs forward and copies the output.
- baseline-kleidiai-arm (all 10): every call checks its cached packed weights
  against the input with memcmp (refresh_cache). Setup now packs once and the
  check is skipped until teardown.

Both hooks work the same way: setup calls the baseline's own entry once with
the arguments the harness gives setup (real weights, zero-filled activations
and output), with a flag set that makes the entry keep what it prepared.

Not touched: llama.cpp baselines (each already caches its ggml graph after the
first, untimed correctness call), ncnn pooling (no weights; setup is never
called for it), SIMD loops (no weights).

Usage:
    python3 ablation/untimed_setup/add_setup_hooks.py --root bench-trace          # rewrite
    python3 ablation/untimed_setup/add_setup_hooks.py --root bench-trace --check  # report only

Idempotent: a solution that already exports armbench_setup_<op_type> is left
as is.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ENTRY_RE = re.compile(r"int\s+(armbench_entry_(\w+))\s*\(([^)]*)\)", re.S)


def _param_names(params: str) -> list[str]:
    """`const float* A, int M` -> ['A', 'M']."""
    names = []
    for p in params.split(","):
        m = re.search(r"(\w+)\s*(\[\s*\])?\s*$", p.strip())
        if not m:
            raise ValueError(f"cannot parse parameter {p!r}")
        names.append(m.group(1))
    return names


def _entry(binding: str):
    m = ENTRY_RE.search(binding)
    if not m:
        raise ValueError("no armbench_entry_<op_type> in binding.cpp")
    name, op, params = m.group(1), m.group(2), " ".join(m.group(3).split())
    return name, op, params, _param_names(params)


# ── KleidiAI ──────────────────────────────────────────────────────────────────

KAI_CACHE_RE = re.compile(
    r"(bool refresh_cache\(uint8_t\*& snapshot, size_t& snapshot_size, "
    r"const void\* data, size_t size\) \{\n)")


def kleidiai(binding: str) -> str:
    name, op, params, args = _entry(binding)
    if f"armbench_setup_{op}" in binding:
        return binding
    if len(KAI_CACHE_RE.findall(binding)) != 1:
        raise ValueError("expected exactly one refresh_cache definition")
    out = KAI_CACHE_RE.sub(
        "// Set by armbench_setup_<op_type> once the weights are packed: until\n"
        "// teardown the inputs given to refresh_cache (weights only) cannot change,\n"
        "// so the per-call memcmp against the snapshot is skipped.\n"
        "bool armbench_weights_ready = false;\n\n"
        r"\1"
        "    if (armbench_weights_ready && snapshot != nullptr && snapshot_size == size) return false;\n",
        binding)
    out = out.rstrip("\n") + f"""

// Untimed setup: pack the weights once. The harness passes the real weights
// and zero-filled activations/output; running the entry fills the caches.
extern "C" int armbench_setup_{op}({params})
{{
    armbench_weights_ready = false;
    const int rc = {name}({", ".join(args)});
    armbench_weights_ready = (rc == 0);
    return rc;
}}

extern "C" int armbench_teardown_{op}(void)
{{
    armbench_weights_ready = false;
    return 0;
}}
"""
    return out


# ── ncnn ──────────────────────────────────────────────────────────────────────

LAYER_RE = re.compile(r"^([ \t]*)(\w+_arm) (\w+);\n", re.M)
NCNN_STATE = """
// Untimed setup (see binding.cpp): while armbench_setup_keep is set, the kernel
// keeps the layer it builds (with its pipeline) in a slot instead of a local,
// and later calls reuse it and only run forward.
namespace {
void* armbench_setup_layers[4] = {nullptr, nullptr, nullptr, nullptr};
void (*armbench_setup_destroy[4])(void*) = {nullptr, nullptr, nullptr, nullptr};
bool armbench_setup_keep = false;
} // namespace

void armbench_layers_begin_setup() { armbench_setup_keep = true; }
void armbench_layers_end_setup() { armbench_setup_keep = false; }
void armbench_layers_teardown()
{
    for (int i = 0; i < 4; ++i) {
        if (armbench_setup_layers[i]) armbench_setup_destroy[i](armbench_setup_layers[i]);
        armbench_setup_layers[i] = nullptr;
        armbench_setup_destroy[i] = nullptr;
    }
}
"""


def ncnn_kernel(kernel: str) -> str:
    if "armbench_setup_layers" in kernel:
        return kernel
    out, pos, slot = [], 0, 0
    for m in LAYER_RE.finditer(kernel):
        ind, typ, var = m.group(1), m.group(2), m.group(3)
        end = kernel.find("create_pipeline(", m.end())
        line_end = kernel.find("\n", end) + 1
        if end < 0:
            raise ValueError(f"no create_pipeline after {typ} {var}")
        if slot >= 4:
            raise ValueError("more than 4 layers")
        body = kernel[m.end():line_end]
        body = "".join(ind + "    " + l[len(ind):] if l.strip() else l for l in body.splitlines(True))
        out.append(kernel[pos:m.start()])
        out.append(
            f"{ind}{typ} {var}_local{slot};\n"
            f"{ind}{typ}* {var}_ptr{slot} = static_cast<{typ}*>(armbench_setup_layers[{slot}]);\n"
            f"{ind}if ({var}_ptr{slot} == nullptr) {{\n"
            f"{ind}    {var}_ptr{slot} = armbench_setup_keep ? new {typ} : &{var}_local{slot};\n"
            f"{ind}    {typ}& {var} = *{var}_ptr{slot};\n"
            f"{body}"
            f"{ind}    if (armbench_setup_keep) {{\n"
            f"{ind}        armbench_setup_layers[{slot}] = {var}_ptr{slot};\n"
            f"{ind}        armbench_setup_destroy[{slot}] = [](void* p) {{\n"
            f"{ind}            {typ}* l = static_cast<{typ}*>(p);\n"
            f"{ind}            ncnn::Option o;\n"
            f"{ind}            l->destroy_pipeline(o);\n"
            f"{ind}            delete l;\n"
            f"{ind}        }};\n"
            f"{ind}    }}\n"
            f"{ind}}}\n"
            f"{ind}{typ}& {var} = *{var}_ptr{slot};\n")
        pos = line_end
        slot += 1
    if slot == 0:
        raise ValueError("no ncnn layer in kernel.cpp")
    out.append(kernel[pos:])
    text = "".join(out)
    includes = list(re.finditer(r"^#include .*\n", text, re.M))
    cut = includes[-1].end()
    return text[:cut] + NCNN_STATE + text[cut:]


def ncnn_binding(binding: str) -> str:
    name, op, params, args = _entry(binding)
    if f"armbench_setup_{op}" in binding:
        return binding
    return binding.rstrip("\n") + f"""

void armbench_layers_begin_setup();
void armbench_layers_end_setup();
void armbench_layers_teardown();

// Untimed setup: build the layer and its pipeline once from the real weights
// (the activations and output the harness passes here are zero-filled
// stand-ins) and keep them for every later call.
extern "C" int armbench_setup_{op}({params})
{{
    armbench_layers_teardown();
    armbench_layers_begin_setup();
    const int rc = {name}({", ".join(args)});
    armbench_layers_end_setup();
    if (rc != 0) armbench_layers_teardown();
    return rc;
}}

extern "C" int armbench_teardown_{op}(void)
{{
    armbench_layers_teardown();
    return 0;
}}
"""


# ── driver ────────────────────────────────────────────────────────────────────

TARGETS = [
    ("ncnn", "baseline-ncnn-arm", ("conv2d", "conv2d_depthwise")),
    ("kleidiai", "baseline-kleidiai-arm", None),
]


def rewrite(sol: dict, dataset: str) -> dict:
    files = {s["path"]: s for s in sol["sources"]}
    if dataset == "kleidiai":
        files["binding.cpp"]["content"] = kleidiai(files["binding.cpp"]["content"])
    else:
        files["kernel.cpp"]["content"] = ncnn_kernel(files["kernel.cpp"]["content"])
        files["binding.cpp"]["content"] = ncnn_binding(files["binding.cpp"]["content"])
    return sol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="bench-trace", help="dataset root (default: bench-trace)")
    ap.add_argument("--check", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args()
    root = Path(args.root)
    changed = skipped = 0
    for dataset, author, op_types in TARGETS:
        for path in sorted((root / "solutions" / dataset / author).glob("*/*.json")):
            if op_types is not None and path.parent.name not in op_types:
                continue
            before = path.read_text()
            sol = rewrite(json.loads(before), dataset)
            after = json.dumps(sol, indent=2, ensure_ascii=False) + "\n"
            if json.loads(before) == sol and "armbench_setup_" in before:
                skipped += 1
                continue
            changed += 1
            print(("would update " if args.check else "updated ") + str(path.relative_to(root)))
            if not args.check:
                path.write_text(after)
    print(f"{changed} {'to update' if args.check else 'updated'}, {skipped} already had the hook")
    return 0


if __name__ == "__main__":
    sys.exit(main())
