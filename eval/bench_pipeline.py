"""eval/bench_pipeline.py — Shared execution layer for compile/evaluate/disassemble.

BenchPipeline manages a pool of InstanceHandles and routes SSH calls to
eval/agent_tools/remote_runner.py on each remote instance. Callers
(AgentTools, CLI) never touch InstanceHandle directly.

Thread safety: each thread gets a sticky handle on its first call (round-robin
assignment). Subsequent calls in that thread reuse the same handle because the
compiled .so lives on the assigned instance.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bench.config import BenchmarkConfig
    from bench.data.solution import Solution
    from eval.provision import InstanceHandle


class PipelineInstanceError(RuntimeError):
    """Raised when an instance is unreachable after all retries."""

    def __init__(self, host: str, op: str, detail: str) -> None:
        super().__init__(f"Instance {host!r} unreachable during {op!r}: {detail}")
        self.host = host
        self.op = op


class BenchPipeline:
    """Execution interface for compile / evaluate / disassemble on a remote instance pool.

    Takes a list of InstanceHandles; caller never accesses handles directly.
    Handles are assigned to threads in round-robin order on the first call and
    remain sticky for the thread's lifetime (session affinity).

    Instance failure recovery (three-tier):
      1. SSH retry ×3 (backoff 2s / 4s / 8s) on TimeoutExpired / OSError.
      2. boto3 EC2 stop+start (best-effort) if instance_id is known.
      3. Raise PipelineInstanceError — caller aborts the definition.
    """

    _REMOTE_ROOT = "~/arm-bench"

    def __init__(self, handles: list[InstanceHandle]) -> None:
        if not handles:
            raise ValueError("BenchPipeline requires at least one InstanceHandle")
        self._handles = handles
        self._idx = 0
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def instance_type(self) -> str:
        """EC2 instance type shared by all handles (e.g. 'c8g.large').

        All handles in one pipeline run must be the same instance family so
        ISA flags (march, isa_features) are consistent across the pool.
        """
        return self._handles[0].instance_type

    def _get_handle(self) -> InstanceHandle:
        """Return this thread's sticky handle; assign one round-robin on first call."""
        if not hasattr(self._local, "handle"):
            with self._lock:
                self._local.handle = self._handles[self._idx % len(self._handles)]
                self._idx += 1
        return self._local.handle

    # ── public API ────────────────────────────────────────────────────────────

    def compile(self, definition: str, solution: Solution, timeout: int = 120) -> dict:
        """Compile solution on this thread's assigned handle."""
        handle = self._get_handle()
        return self._run_remote(handle, "compile", {
            "solution": solution.model_dump(mode="json"),
            "definition": definition,
        }, timeout=timeout)

    def evaluate(
        self,
        definition: str,
        so_path: str,
        solution_name: str = "agent",
        bench_cfg: BenchmarkConfig | None = None,
        timeout: int = 300,
    ) -> dict:
        """Run all workloads for definition on this thread's handle."""
        handle = self._get_handle()
        args: dict = {
            "so_path": so_path,
            "definition": definition,
            "solution_name": solution_name,
        }
        if bench_cfg is not None:
            args["benchmark_config"] = dataclasses.asdict(bench_cfg)
        return self._run_remote(handle, "evaluate", args, timeout=timeout)

    def disassemble(
        self,
        so_path: str,
        op_type: str,
        symbol: str | None = None,
        timeout: int = 30,
    ) -> dict:
        """Disassemble so_path on this thread's handle."""
        handle = self._get_handle()
        args: dict = {"so_path": so_path, "op_type": op_type}
        if symbol is not None:
            args["symbol"] = symbol
        return self._run_remote(handle, "disassemble", args, timeout=timeout)

    def run_shell(self, cmd: str, timeout: int = 120) -> tuple[int, str, str]:
        """Run an arbitrary shell command on this thread's assigned handle."""
        return self._get_handle().run(cmd, timeout=timeout)

    # ── internal SSH + recovery ───────────────────────────────────────────────

    def _run_remote(
        self, handle: InstanceHandle, op: str, args: dict, timeout: int = 120
    ) -> dict:
        """SSH to handle, pipe JSON to remote_runner.py, return parsed result.

        Retries up to 3 times on connection/timeout errors (backoff 2s/4s/8s).
        On sustained failure: attempts EC2 stop+start, then raises
        PipelineInstanceError so the caller can abort the current definition.
        """
        remote_cmd = (
            f"cd {self._REMOTE_ROOT} && "
            f"python3 -u eval/agent_tools/remote_runner.py {op}"
        )
        last_exc: Exception | None = None

        for attempt in range(3):
            try:
                proc = subprocess.run(
                    handle.ssh_cmd(remote_cmd),
                    input=json.dumps(args),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                stdout = proc.stdout.strip()
                if not stdout:
                    raise OSError(
                        f"no output (rc={proc.returncode}): {proc.stderr[:300]}"
                    )
                try:
                    return json.loads(stdout)
                except json.JSONDecodeError:
                    return {
                        "status": "RUNTIME_ERROR",
                        "error": f"Remote {op} returned non-JSON: {stdout[:200]}",
                        "stderr": proc.stderr[:300],
                    }
            except subprocess.TimeoutExpired as e:
                last_exc = e
            except OSError as e:
                last_exc = e

            wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
            time.sleep(wait)

        # All retries exhausted — attempt EC2 restart as best-effort recovery.
        if handle.instance_id:
            self._attempt_restart(handle)

        raise PipelineInstanceError(handle.host, op, str(last_exc))

    def _attempt_restart(self, handle: InstanceHandle) -> None:
        """Stop+start the EC2 instance via boto3; update handle.host with new IP.

        Best-effort: errors are silently swallowed. The primary purpose is to
        bring the instance back online for future definitions after this one
        is aborted via PipelineInstanceError.
        """
        try:
            import boto3
            from eval.provision import _wait_for_ssh
        except ImportError:
            return
        try:
            ec2 = boto3.client("ec2")
            ec2.stop_instances(InstanceIds=[handle.instance_id])
            ec2.get_waiter("instance_stopped").wait(InstanceIds=[handle.instance_id])
            ec2.start_instances(InstanceIds=[handle.instance_id])
            ec2.get_waiter("instance_running").wait(InstanceIds=[handle.instance_id])
            resp = ec2.describe_instances(InstanceIds=[handle.instance_id])
            new_ip = resp["Reservations"][0]["Instances"][0]["PublicIpAddress"]
            handle.host = new_ip
            _wait_for_ssh(handle)
        except Exception:
            pass


__all__ = ["BenchPipeline", "PipelineInstanceError"]
