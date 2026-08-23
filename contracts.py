"""Kernel-session contracts shared by eval/, mcp_app/, skills/, and bench/.
config is in <REPO-ROOT>/config/kernel_contracts.yaml with 
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
BASELINE_AUTHORS: dict[str, str] = dict(_load()["baseline_authors"])
ISA_TABLE: dict[str, IsaSpec] = _isa_table()

# isa -> EC2 instance type, the subset of ISA_TABLE most callers actually need.
ISA_INSTANCE_MAP: dict[str, str] = {isa: spec.instance_type for isa, spec in ISA_TABLE.items()}

# bench/config.py's BenchmarkConfig/EvalConfig field defaults, and its
# per-op-type tolerance overrides. Raw dicts — bench/config.py owns the
# EvalOverride dataclass these get wrapped in, since that shape is bench-specific.
EVAL_DEFAULTS: dict = dict(_load()["eval_defaults"])
EVAL_OP_TYPE_OVERRIDES: dict[str, dict] = {
    op: dict(cfg) for op, cfg in _load()["eval_op_type_overrides"].items()
}

# Disallowed-source-pattern policy for agent-submitted kernel.cpp — consumed
# by mcp_app/agent_tools/base.py::_disallowed_source_patterns(), never by
# bench/ (see config/kernel_contracts.yaml's disallowed_source_patterns
# comment for why this lives here and not in bench/config.py).
_DISALLOWED_SOURCE_PATTERNS: dict = _load()["disallowed_source_patterns"]
DISALLOWED_SOURCE_PATTERNS_DEFAULT: list[str] = list(_DISALLOWED_SOURCE_PATTERNS["default"])
DISALLOWED_SOURCE_PATTERNS_BY_OP_TYPE: dict[str, list[str]] = {
    op: list(patterns) for op, patterns in _DISALLOWED_SOURCE_PATTERNS["by_op_type"].items()
}
DISALLOWED_SOURCE_PATTERNS_BY_ISA: dict[str, list[str]] = {
    isa: list(patterns) for isa, patterns in _DISALLOWED_SOURCE_PATTERNS["by_isa"].items()
}

# eval/evaluator.py::run_agentic_eval's litellm turn loop (completion timeout,
# temperature, retry budget) and eval/mcp_client.py's MCP session (per-call timeouts) — raw dicts, same treatment as EVAL_DEFAULTS above.
AGENT_LOOP_DEFAULTS: dict = dict(_load()["tool_call_loop"])
MCP_CLIENT_DEFAULTS: dict = dict(_load()["mcp_client"])

__all__ = [
    "IsaSpec",
    "AGENT_KERNEL_FILENAME",
    "REFERENCE_SCALAR_FILENAME",
    "REFERENCE_SCALAR_AUTHORS",
    "BASELINE_AUTHORS",
    "ISA_TABLE",
    "ISA_INSTANCE_MAP",
    "EVAL_DEFAULTS",
    "EVAL_OP_TYPE_OVERRIDES",
    "DISALLOWED_SOURCE_PATTERNS_DEFAULT",
    "DISALLOWED_SOURCE_PATTERNS_BY_OP_TYPE",
    "DISALLOWED_SOURCE_PATTERNS_BY_ISA",
    "AGENT_LOOP_DEFAULTS",
    "MCP_CLIENT_DEFAULTS",
]
