"""DispatcherKernelSession — routes tool calls to the right per-dataset
KernelSession by looking up which dataset a call's `definition` belongs to.

Composition, not inheritance: see KernelSessionLike's docstring (base.py) for
why this does NOT subclass KernelSession. `mcp_app/server.py` needs no
changes to use this — it only ever calls the methods declared on
KernelSessionLike, which this class satisfies structurally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import KernelSession, standard_tool_schemas


class DispatcherKernelSession:
    """Wraps one KernelSession per dataset; resolves and delegates each call.

    Each sub-session was built by its own `session.py::build_tools()` call,
    so each has its own independent `TraceSet` instance (not shared) — that's
    fine, `_session_for` only ever reads a sub-session's own `_trace_set`,
    never cross-references between them.
    """

    def __init__(self, sessions: dict[str, KernelSession], run_dir: Path) -> None:
        self._sessions = sessions
        self._run_dir = run_dir

    def _session_for(self, definition: str) -> KernelSession:
        """Resolve which sub-session owns `definition`, mirroring the same
        membership check KernelSession._get_or_create_definition already
        does internally (base.py) — `any(s.dataset.value == dataset for s in
        solutions)` — just evaluated against every configured dataset
        instead of a single one.
        """
        for dataset, sub in self._sessions.items():
            solutions = sub._trace_set.solutions.get(definition, [])
            if any(s.dataset.value == dataset for s in solutions):
                return sub
        raise ValueError(
            f"Unknown definition {definition!r}, or it has no solution in "
            f"any of this session's datasets ({sorted(self._sessions)})."
        )

    # ── shared tool implementations — same signatures as KernelSession's ──

    def compile(self, definition: str, code: str) -> dict:
        return self._session_for(definition).compile(definition, code)

    def evaluate(self, definition: str, version: int) -> dict:
        return self._session_for(definition).evaluate(definition, version)

    def disassemble(self, definition: str, version: int, fn: str | None = None) -> dict:
        return self._session_for(definition).disassemble(definition, version, fn)

    def submit(self, definition: str, explanation: str = "") -> dict:
        return self._session_for(definition).submit(definition, explanation)

    def dispatch_tool_call(self, name: str, args: dict) -> dict:
        """Route a tool name to its method; return error dict on unknown name.

        Identical shape to KernelSession.dispatch_tool_call (base.py) — kept
        as its own copy rather than a shared helper since it's this small.
        """
        method = getattr(self, name, None)
        if method is None or name.startswith("_"):
            return {"error": f"unknown tool: {name!r}"}
        try:
            return method(**args)
        except Exception as e:
            return {"error": str(e)}

    def tool_schemas(self) -> list[dict]:
        return standard_tool_schemas()

    def note_session_definition(self, session: Any, definition: str) -> None:
        self._session_for(definition).note_session_definition(session, definition)

    def session_definitions(self, session: Any) -> frozenset[str]:
        """Union across every sub-session — resource visibility (see
        mcp_app/resources.py) must cover whichever dataset(s) `session`
        actually compiled from, not just one.
        """
        return frozenset().union(
            *(s.session_definitions(session) for s in self._sessions.values())
        )

    def cleanup(self) -> None:
        for s in self._sessions.values():
            s.cleanup()


__all__ = ["DispatcherKernelSession"]
