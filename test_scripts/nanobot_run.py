#!/usr/bin/env python3
"""nanobot_run.py — one single-message nanobot run, with token-usage capture.

Drop-in for what NanobotAdapter used to spawn:

    nanobot agent --logs -m MSG -w WORKSPACE -c CONFIG --session KEY

but driven through nanobot's public Python SDK (`nanobot.nanobot.Nanobot`,
identical signature in the pinned 0.2.2 and the 0.3.x on PATH) so a lifecycle
hook can see what the CLI never persists anywhere: per-iteration token usage
and timing. That is written to --usage-out as JSON:

    {"model": ..., "session_key": ..., "wall_time_s": ..., "stop_reason": ...,
     "error": ..., "usage": {"prompt_tokens": N, "completion_tokens": N, ...},
     "iterations": [{"iteration", "ts", "llm_s", "total_s", "tools": [...],
                     "prompt_tokens", "completion_tokens", ...}, ...]}

analysis/wandb_log_run.py reads it (by convention `<job log stem>.usage.json`,
next to the job log) to fill the tokens/cost/turn columns that claude-code
runs get from the CLI's stream-json — so nanobot runs land in the same W&B
columns. The nanobot runtime log lines ("Tool call: ...") still go to
stderr exactly as with `--logs`, so bench_fleet's log-tee and
is_benign_failure() keep working unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _usage_hook_class():
    """Built lazily so `--help` works without nanobot importable."""
    from nanobot.agent.hook import AgentHook

    class UsageHook(AgentHook):
        """Per-iteration usage + latency. before_iteration → model request
        starts; before_execute_tools → model responded (llm_s); after_iteration
        → tools done (total_s). context.usage is THIS iteration's usage
        (the runner resets it per response), so rows are additive."""

        def __init__(self) -> None:
            super().__init__()
            self.iterations: list[dict] = []
            self._t0: float | None = None
            self._t_llm: float | None = None

        async def before_iteration(self, context) -> None:
            self._t0 = time.monotonic()
            self._t_llm = None

        async def before_execute_tools(self, context) -> None:
            if self._t0 is not None:
                self._t_llm = time.monotonic() - self._t0

        async def after_iteration(self, context) -> None:
            now = time.monotonic()
            total = (now - self._t0) if self._t0 is not None else None
            llm = self._t_llm if self._t_llm is not None else total
            row = {
                "iteration": context.iteration,
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "llm_s": round(llm, 3) if llm is not None else None,
                "total_s": round(total, 3) if total is not None else None,
                "tools": [getattr(c, "name", str(c)) for c in (context.tool_calls or [])],
            }
            for k, v in (context.usage or {}).items():
                try:
                    row[k] = int(v)
                except (TypeError, ValueError):
                    pass
            self.iterations.append(row)
            self._t0 = None

    return UsageHook


def _totals(iterations: list[dict]) -> dict:
    tot: dict[str, int] = {}
    for it in iterations:
        for k, v in it.items():
            if k.endswith("_tokens") and isinstance(v, int):
                tot[k] = tot.get(k, 0) + v
    return tot


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--message", required=True)
    p.add_argument("-w", "--workspace", required=True)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--session", default="cli:direct")
    p.add_argument("--usage-out", required=True, help="JSON sidecar to write usage/timing to.")
    p.add_argument("--logs", action="store_true", help="Show nanobot runtime logs (like `nanobot agent --logs`).")
    args = p.parse_args()

    from loguru import logger
    from nanobot.nanobot import Nanobot

    (logger.enable if args.logs else logger.disable)("nanobot")
    try:  # mirrors the CLI's _load_runtime_config(); harmless if absent
        from nanobot.config.loader import set_config_path
        set_config_path(Path(args.config).expanduser().resolve())
    except ImportError:
        pass

    hook = _usage_hook_class()()
    out = {
        "harness": "nanobot", "session_key": args.session, "model": None,
        "wall_time_s": None, "stop_reason": None, "error": None,
        "usage": {}, "iterations": [],
    }
    t0 = time.monotonic()
    rc = 0

    async def _run():
        bot = Nanobot.from_config(args.config, workspace=args.workspace)
        cfg = getattr(bot, "_config", None)
        try:
            out["model"] = cfg.agents.defaults.model if cfg else None
        except AttributeError:
            pass
        try:
            return await bot.run(args.message, session_key=args.session, hooks=[hook])
        finally:
            close = getattr(bot, "aclose", None) or getattr(getattr(bot, "_loop", None), "close_mcp", None)
            if close is not None:
                await close()

    try:
        result = asyncio.run(_run())
        out["stop_reason"] = getattr(result, "stop_reason", None)
        out["error"] = getattr(result, "error", None)
        if getattr(result, "usage", None):
            out["usage"] = dict(result.usage)
        content = getattr(result, "content", "") or ""
        if content:
            print(f"\n🐈 nanobot\n{content}")
    except Exception as e:  # noqa: BLE001 — surfaced via rc + sidecar, never swallowed
        out["error"] = f"{type(e).__name__}: {e}"
        rc = 1
        import traceback
        traceback.print_exc()
    finally:
        out["wall_time_s"] = round(time.monotonic() - t0, 1)
        out["iterations"] = hook.iterations
        if not out["usage"]:
            out["usage"] = _totals(hook.iterations)
        try:
            Path(args.usage_out).write_text(json.dumps(out, indent=1))
        except OSError as e:
            print(f"[nanobot_run] could not write {args.usage_out}: {e}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
