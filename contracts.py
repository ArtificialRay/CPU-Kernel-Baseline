"""Kernel-session contracts shared by eval/, mcp_app/, skills/, and bench/,
loaded from config/kernel_contracts.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
_YAML_PATH = REPO_ROOT / "config" / "kernel_contracts.yaml"


@dataclass(frozen=True)
class IsaSpec:
    march: str
    features: list[str]
    labels: list[str]
    instance_type: str


@lru_cache(maxsize=1)
def _load() -> dict:
    with _YAML_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _contracts() -> dict:
    return _load()["contracts"]


def _isa_table() -> dict[str, IsaSpec]:
    return {
        isa: IsaSpec(
            march=spec["march"],
            features=list(spec["features"]),
            labels=list(spec["labels"]),
            instance_type=spec["instance_type"],
        )
        for isa, spec in _load()["isa"].items()
    }


AGENT_KERNEL_FILENAME: str = _contracts()["agent_kernel_filename"]
REFERENCE_SCALAR_FILENAME: str = _contracts()["reference_scalar_filename"]
REFERENCE_SCALAR_AUTHORS: dict[str, str] = dict(_load()["reference_scalar_authors"])
BASELINE_AUTHORS: dict[str, str] = dict(_load()["baseline_authors_default"])
BASELINE_AUTHORS_BY_ISA: dict[str, dict[str, str]] = {
    ds: dict(m) for ds, m in (_load().get("baseline_authors_by_isa") or {}).items()
}


def baseline_author_for(dataset: str, isa: str | None) -> str:
    """The speedup-baseline author for `dataset` on an `isa` box: the
    tier-specific author from baseline_authors_by_isa when one is declared
    else BASELINE_AUTHORS[dataset]."""
    return BASELINE_AUTHORS_BY_ISA.get(dataset, {}).get(isa or "", BASELINE_AUTHORS[dataset])


ISA_TABLE: dict[str, IsaSpec] = _isa_table()

# isa -> EC2 instance type, the subset of ISA_TABLE most callers actually need.
ISA_INSTANCE_MAP: dict[str, str] = {isa: spec.instance_type for isa, spec in ISA_TABLE.items()}

# bench/config.py's BenchmarkConfig/EvalConfig field defaults, and its
# per-op-type / per-definition tolerance overrides. Raw dicts — bench/config.py
# owns the EvalOverride dataclass these get wrapped in, since that shape is
# bench-specific.
EVAL_DEFAULTS: dict = dict(_load()["eval_defaults"])
EVAL_OP_TYPE_OVERRIDES: dict[str, dict] = {
    op: dict(cfg) for op, cfg in _load()["eval_op_type_overrides"].items()
}
EVAL_DEFINITION_OVERRIDES: dict[str, dict] = {
    name: dict(cfg) for name, cfg in _load()["eval_definition_overrides"].items()
}

# Disallowed-source-pattern policy for agent-submitted kernel.cpp — consumed
# by mcp_app/agent_tools/base.py::_disallowed_source_patterns(), never by bench/.
_DISALLOWED_SOURCE_PATTERNS: dict = _load()["disallowed_source_patterns"]
DISALLOWED_SOURCE_PATTERNS_DEFAULT: list[str] = list(_DISALLOWED_SOURCE_PATTERNS["default"])
DISALLOWED_SOURCE_PATTERNS_BY_OP_TYPE: dict[str, list[str]] = {
    op: list(patterns) for op, patterns in _DISALLOWED_SOURCE_PATTERNS["by_op_type"].items()
}
DISALLOWED_SOURCE_PATTERNS_BY_ISA: dict[str, list[str]] = {
    isa: list(patterns) for isa, patterns in _DISALLOWED_SOURCE_PATTERNS["by_isa"].items()
}
DISALLOWED_SOURCE_PATTERNS_BY_DATASET: dict[str, list[str]] = {
    ds: list(patterns) for ds, patterns in _DISALLOWED_SOURCE_PATTERNS.get("by_dataset", {}).items()
}

# eval/'s litellm turn loop (completion timeout, temperature, retry budget) and
# eval/mcp_client.py's MCP session (per-call timeouts) — raw dicts, like EVAL_DEFAULTS.
AGENT_LOOP_DEFAULTS: dict = dict(_load()["tool_call_loop"])
MCP_CLIENT_DEFAULTS: dict = dict(_load()["tool_call_loop"]["mcp_client"])

# The MCP server's evaluate() budget, and the shortest wait any harness's MCP
# client may use for a tool call (see mcp_tool_timeouts in kernel_contracts.yaml).
EVALUATE_TIMEOUT_S: float = float(_load()["mcp_tool_timeouts"]["evaluate_s"])
CLIENT_TOOL_TIMEOUT_S: float = EVALUATE_TIMEOUT_S + float(_load()["mcp_tool_timeouts"]["client_margin_s"])
if not CLIENT_TOOL_TIMEOUT_S > EVALUATE_TIMEOUT_S:
    raise ValueError("mcp_tool_timeouts.client_margin_s in config/kernel_contracts.yaml must be positive")


def check_client_tool_timeout(harness: str, timeout_s: float) -> None:
    """Raise unless `harness`'s MCP client waits at least CLIENT_TOOL_TIMEOUT_S
    for a tool call. A shorter wait lets the client give up on an evaluate()
    the server is still running, and the agent's next evaluate() is then timed
    on a busy core. `timeout_s` is in seconds, whatever unit the harness's own
    config uses."""
    if not timeout_s >= CLIENT_TOOL_TIMEOUT_S:
        raise RuntimeError(
            f"{harness}: MCP tool-call timeout {timeout_s:g}s is shorter than the server's evaluate() "
            f"budget {EVALUATE_TIMEOUT_S:g}s plus margin ({CLIENT_TOOL_TIMEOUT_S:g}s); see "
            f"mcp_tool_timeouts in config/kernel_contracts.yaml."
        )


__all__ = [
    "IsaSpec",
    "AGENT_KERNEL_FILENAME",
    "REFERENCE_SCALAR_FILENAME",
    "REFERENCE_SCALAR_AUTHORS",
    "BASELINE_AUTHORS",
    "BASELINE_AUTHORS_BY_ISA",
    "baseline_author_for",
    "ISA_TABLE",
    "ISA_INSTANCE_MAP",
    "EVAL_DEFAULTS",
    "EVAL_OP_TYPE_OVERRIDES",
    "EVAL_DEFINITION_OVERRIDES",
    "DISALLOWED_SOURCE_PATTERNS_DEFAULT",
    "DISALLOWED_SOURCE_PATTERNS_BY_OP_TYPE",
    "DISALLOWED_SOURCE_PATTERNS_BY_ISA",
    "DISALLOWED_SOURCE_PATTERNS_BY_DATASET",
    "AGENT_LOOP_DEFAULTS",
    "MCP_CLIENT_DEFAULTS",
    "EVALUATE_TIMEOUT_S",
    "CLIENT_TOOL_TIMEOUT_S",
    "check_client_tool_timeout",
]
