"""Agent-facing MCP tool schemas — identical across every dataset.
"""

from __future__ import annotations

_CHECK_PROGRESS_SCHEMA = {
    "name": "check_progress",
    "description": (
        "Check whether `definition` already has a submitted best result on "
        "disk from an earlier session. Call this BEFORE your first "
        "compile() for a definition. Returns {\"has_prior_progress\": "
        "false} if nothing exists yet — proceed with the reference-scalar "
        "starting point as normal. Otherwise returns "
        "{\"has_prior_progress\": true, \"best_version\": M, "
        "\"best_metrics\": {...}, \"best_code\": \"...\"}. When it does: "
        "your FIRST two calls for this definition must be compile(best_code) "
        "then evaluate() on the result — even though you already know the "
        "score — before doing anything else, to help you re-establish this"
        "session's own notion of \"best so far\",You don't need to track version "
        "numbers yourself — compile() always tells you which version it "
        "assigned."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "definition": {
                "type": "string",
                "description": "Definition to check for prior progress.",
            },
        },
        "required": ["definition"],
    },
}

_COMPILE_SCHEMA = {
    "name": "compile",
    "description": (
        "Compile your kernel.cpp for the given definition. The harness/binding "
        "files are provided automatically — you only write the kernel. "
        "You can call this with different `definition` values across the "
        "session; each definition keeps its own compile/evaluate history. "
        "Returns {\"status\": \"OK\", \"definition\": ..., \"version\": N} on "
        "success — pass both `definition` and `version` back into evaluate()/"
        "disassemble() to confirm you're acting on this exact compile, not one "
        "from another definition or a later recompile. Or "
        "{\"status\": \"COMPILE_ERROR\", \"error\": \"...\"} on failure."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "definition": {
                "type": "string",
                "description": (
                    "Name of the bench-trace definition to compile against "
                    "(e.g. 'conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0'). Must "
                    "belong to this server's dataset."
                ),
            },
            "code": {
                "type": "string",
                "description": (
                    "Full C++ source for kernel.cpp. Before writing this, read "
                    "the `reference-scalar-kernel.cpp` resource for this "
                    "definition — it's a working scalar reference implementation "
                    "showing the exact function name and signature you must "
                    "implement (naming convention varies by dataset, e.g. "
                    "`inner_<op_type>` vs `armbench_llamacpp_<op_type>(...)`). "
                    "Replace its body with an optimized SIMD version; keep the "
                    "same signature. Harness/binding files are provided "
                    "automatically."
                ),
            },
        },
        "required": ["definition", "code"],
    },
}

_EVALUATE_SCHEMA = {
    "name": "evaluate",
    "description": (
        "Run the compiled kernel identified by (`definition`, `version`) "
        "against all workloads: checks correctness (fail-fast on the first "
        "failing workload) and, if that passes, measures wall-time/cycle "
        "counts in the same pass — always both, one call. Both args are "
        "required and must match your own last compile() call for that "
        "definition exactly — errors instead of silently evaluating a "
        "different definition's compile, or a version that's since been "
        "superseded by another compile() (e.g. from a concurrent call in "
        "the same turn). "
        "Whenever this beats the best cycle speedup seen so far this "
        "session, it's immediately persisted to bench-trace — that result "
        "already counts even if you never call submit(). "
        "Returns {\"status\": \"PASSED\", \"correctness\": {...}, "
        "\"performance\": {...}} or {\"status\": \"<error>\", "
        "\"failed_workload\": \"...\", \"log\": \"...\"}."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "definition": {
                "type": "string",
                "description": "Must match the `definition` from your last compile() call.",
            },
            "version": {
                "type": "integer",
                "description": "Must match the `version` from your last compile() call.",
            },
        },
        "required": ["definition", "version"],
    },
}

_DISASSEMBLE_SCHEMA = {
    "name": "disassemble",
    "description": (
        "Disassemble the compiled .so identified by (`definition`, `version`) "
        "(up to 300 lines of AArch64 assembly). Defaults to this definition's "
        "own kernel entry symbol (the function you implemented); pass `fn` to "
        "inspect a different symbol. `definition`/`version` are required and "
        "validated the same way as evaluate()'s — see its description."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "definition": {
                "type": "string",
                "description": "Must match the `definition` from your last compile() call.",
            },
            "version": {
                "type": "integer",
                "description": "Must match the `version` from your last compile() call.",
            },
            "fn": {
                "type": "string",
                "description": "Symbol to disassemble. Omit to use this definition's own kernel entry symbol.",
            },
        },
        "required": ["definition", "version"],
    },
}


def standard_tool_schemas() -> list[dict]:
    """The standard agent tool schemas, identical across datasets."""
    return [
        _CHECK_PROGRESS_SCHEMA,
        _COMPILE_SCHEMA,
        _EVALUATE_SCHEMA,
        _DISASSEMBLE_SCHEMA,
    ]


__all__ = ["standard_tool_schemas"]
