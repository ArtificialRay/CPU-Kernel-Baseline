"""harness_adapters — per-harness HarnessAdapter implementations used by
bench_fleet.py's shared driver. Split out from bench_fleet.py itself so the
orchestration (compute author/label once, provision, prepare_session,
retry/log/sync loop) stays separate from what's genuinely harness-specific:
how each harness is invoked, how it connects to the MCP endpoint, its own
retry quirks. One module per harness; this package re-exports them all, so
`from harness_adapters import ClaudeCodeAdapter, Job, ...` keeps working.
"""

from .base import HarnessAdapter, Job, run_and_tee
from .claude_code import ClaudeCodeAdapter
from .cline import ClineAdapter
from .codex import CodexAdapter
from .nanobot import NanobotAdapter
from .own import OwnHarnessAdapter
from .single_shot import SingleShotAdapter

__all__ = [
    "HarnessAdapter", "Job", "run_and_tee",
    "ClaudeCodeAdapter", "ClineAdapter", "CodexAdapter", "NanobotAdapter",
    "OwnHarnessAdapter", "SingleShotAdapter",
]
