"""Docs-ablation arms: everything that differs between control / docs / nudge.

Nothing here is imported by the main harness; run.py wires these into
test_scripts/bench_fleet.py at its two seams (see README.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "test_scripts", REPO_ROOT):   # bench_fleet.py's own import idiom
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from harness_adapters import ClaudeCodeAdapter, Job  # noqa: E402

ARMS = ("control", "docs", "nudge")

# `nudge` arm: a strong, explicit per-job instruction to read the Arm Software
# Optimization Guide (exposed as MCP resources) BEFORE optimizing. SKILL.md
# already mentions the docs mildly; this makes it an unmissable first step.
DOC_NUDGE = (
    " IMPORTANT — before writing or compiling ANY kernel, FIRST call list_resources() "
    "and read the Arm Neoverse Software Optimization Guide for the target hardware in "
    "full (docs/neoverse-v2-swog.md for Graviton4/SVE2; docs/neoverse-v1-swog.md for "
    "Graviton3/SVE). Ground every optimization decision — instruction selection, vector "
    "width, unroll factor, and scheduling — in its per-instruction latency/throughput "
    "tables, and briefly note which guidance you applied. Do not begin optimizing until "
    "you have read it."
)

# `control` arm: SKILL.md's hardware-docs bullet, located by its first and last
# lines so the canonical SKILL.md needs no markers. Drift → loud failure.
_DOCS_BLOCK_BEGIN = "- For instruction-level cost when scheduling SVE2/NEON/FP code"
_DOCS_BLOCK_END = "These are large — read the relevant §3.x section on demand, not wholesale."


def strip_docs_section(skill_text: str) -> str:
    """Remove the hardware-docs bullet (begin line .. end line, inclusive)."""
    start = skill_text.find(_DOCS_BLOCK_BEGIN)
    end = skill_text.find(_DOCS_BLOCK_END, start if start != -1 else 0)
    if start == -1 or end == -1:
        raise RuntimeError(
            "SKILL.md's hardware-docs section anchors not found — SKILL.md changed; "
            "update _DOCS_BLOCK_BEGIN/_END in ablation/docs_ablation/arms.py "
            "(refusing to run a 'control' arm that may still mention the docs)."
        )
    end += len(_DOCS_BLOCK_END)
    if end < len(skill_text) and skill_text[end] == "\n":
        end += 1
    return skill_text[:start] + skill_text[end:]


class DocsAblationClaudeCodeAdapter(ClaudeCodeAdapter):
    """ClaudeCodeAdapter with one arm applied. Same constructor, same
    run_job contract — bench_fleet.py can't tell the difference."""

    arm: str = "docs"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.arm not in ARMS:
            raise ValueError(f"unknown docs-ablation arm {self.arm!r}; expected one of {ARMS}")
        if self.arm == "control":
            self.system_prompt = strip_docs_section(self.system_prompt)

    def run_job(self, job: Job, **kwargs) -> int:
        if self.arm == "nudge":
            job = Job(name=job.name, prompt=job.prompt + DOC_NUDGE)
        return super().run_job(job, **kwargs)


def adapter_class_for(arm: str) -> type[DocsAblationClaudeCodeAdapter]:
    if arm not in ARMS:
        raise ValueError(f"unknown docs-ablation arm {arm!r}; expected one of {ARMS}")
    return type(f"{arm.capitalize()}ArmClaudeCodeAdapter", (DocsAblationClaudeCodeAdapter,), {"arm": arm})


def hide_remote_docs(target, remote_root: str, author: str) -> None:
    """`control` arm: delete the SWOG copies the server just wrote into this
    author's run dir on the box, and verify they're gone. Safe to run right
    after prepare_session() returns: mcp_app/server.py builds the session
    (which writes docs/) before it starts listening, and
    mcp_app/resources.py globs run_dir/docs/*.md per list_resources() call,
    so nothing caches the pre-delete listing."""
    run_dir = f"{remote_root}/agent-runs-mcp/{author}"
    rc, out, err = target.run(f"rm -rf {run_dir}/docs && test ! -e {run_dir}/docs && echo DOCS_HIDDEN")
    if rc != 0 or "DOCS_HIDDEN" not in out:
        raise RuntimeError(f"could not hide {run_dir}/docs on {target.host}: rc={rc} {err.strip()}")
    print(f"[docs_ablation] control arm: removed {run_dir}/docs on {target.host}")
