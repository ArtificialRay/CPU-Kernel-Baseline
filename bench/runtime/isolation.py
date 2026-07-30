"""Subprocess isolation for running untrusted candidate-kernel code.

`bench/runner.py::run_solution_on_workloads` and
`mcp_app/agent_tools/ops.py::evaluate_kernel` both dlopen an agent-written
`.so` and call straight into it via ctypes. That candidate code can be
pathologically slow, hang outright, or crash (e.g. a stack-overflowing
`alloca`) — none of which a same-process caller can defend against: a Python
signal handler can't preempt a tight native loop, and a native crash takes
the whole interpreter down with it, including a long-lived caller like
mcp_app's MCP server (see the incident that motivated this module —
harness_trajs/nanobot/ncnn_sve_conv2d_w8a8ch_kh1_kw1_sh1_sw1_dh1_dw1_p0.log).

`run_in_subprocess` is the one generic primitive both callers wrap
themselves in: run `target` in a spawned child, enforce a hard wall-clock
timeout the parent can always make good on (SIGKILL, if needed), and turn a
crashed/hung child into a raised exception instead of a dead process. Mirrors
`bench/runtime/timing.py`'s `WatchdogTimeout` / `bench/compile/builder.py`'s
`CompileError` — plain `RuntimeError` subclasses, raised on failure, meant to
be caught by name at call sites exactly like those already are.
"""

from __future__ import annotations

import multiprocessing
import resource
import signal
from typing import Any, Callable, Optional, Tuple

DEFAULT_ISOLATION_TIMEOUT_S = 600.0
"""Comfortably under the MCP client's 900s toolTimeout
(skills/nanobot/nanobot-kernel-session/config.json) so an agent gets a clean
structured result before the client's own wait expires."""

_CHILD_STACK_LIMIT_BYTES = 128 * 1024 * 1024
"""Best-effort RLIMIT_STACK bump for the child's main thread before running
`target` — the OS default (~8MB) is smaller than some plausible (if
wasteful) candidate-kernel scratch allocations; this gives those a chance to
actually succeed instead of immediately SIGSEGV'ing, while still bounding
truly runaway allocations."""


class SubprocessTimeout(RuntimeError):
    """Raised when the isolated subprocess didn't finish within `timeout_s`."""


class SubprocessCrashed(RuntimeError):
    """Raised when the isolated subprocess was killed by a signal (segfault,
    OOM-kill, etc.) rather than returning normally."""


def _bump_stack_limit() -> None:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        new_soft = _CHILD_STACK_LIMIT_BYTES
        if hard != resource.RLIM_INFINITY:
            new_soft = min(new_soft, hard)
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_STACK, (new_soft, hard))
    except (ValueError, OSError):
        pass  # best-effort — proceed with whatever limit is already in place


def _child_trampoline(
    queue: "multiprocessing.Queue[Tuple[str, Any]]",
    target: Callable[..., Any],
    args: tuple,
    kwargs: dict,
) -> None:
    _bump_stack_limit()
    try:
        queue.put(("ok", target(*args, **kwargs)))
    except BaseException as exc:  # noqa: BLE001 — deliberately broad: re-raised as-is in the parent
        queue.put(("error", exc))


def run_in_subprocess(
    target: Callable[..., Any],
    args: tuple = (),
    kwargs: Optional[dict] = None,
    *,
    timeout_s: float = DEFAULT_ISOLATION_TIMEOUT_S,
) -> Any:
    """Run `target(*args, **kwargs)` in a spawned child process; return its
    result directly, or raise.

    `target` must be a module-level, picklable callable — spawn re-imports it
    by reference in the child, it never gets a copy of the parent's live
    state via fork-style COW (deliberate: `mcp_app.server` is multi-threaded
    via `asyncio.to_thread`, and forking a multi-threaded process risks
    inherited-but-never-released locks in the child).

    On success, returns whatever `target` returned, unchanged. On failure:
    - `SubprocessTimeout` if the child is still running after `timeout_s`
      (escalates SIGTERM then SIGKILL to reclaim it).
    - `SubprocessCrashed` if the child was killed by a signal.
    - whatever `target` itself raised, re-raised with its original type —
      so a caller's existing `except SomeError` around a direct call to
      `target` keeps working unchanged around the isolated call too.
    """
    kwargs = kwargs or {}
    ctx = multiprocessing.get_context("spawn")
    queue: "multiprocessing.Queue[Tuple[str, Any]]" = ctx.Queue()
    proc = ctx.Process(target=_child_trampoline, args=(queue, target, args, kwargs))
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        raise SubprocessTimeout(f"exceeded {timeout_s}s (subprocess killed)")

    if proc.exitcode is not None and proc.exitcode < 0:
        sig = signal.Signals(-proc.exitcode)
        raise SubprocessCrashed(
            f"kernel crashed the evaluation subprocess with signal {sig.name} ({sig.value})"
        )

    kind, payload = queue.get()
    if kind == "error":
        raise payload
    return payload


__all__ = [
    "DEFAULT_ISOLATION_TIMEOUT_S",
    "SubprocessTimeout",
    "SubprocessCrashed",
    "run_in_subprocess",
]
