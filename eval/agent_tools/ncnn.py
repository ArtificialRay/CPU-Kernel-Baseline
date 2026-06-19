"""NCNNAgentTools — AgentTools implementation for the NCNN dataset.

Agent writes `kernel.cpp` containing `inner_<op_type>(...)`. The harness
files (`<op_type>.h` + `<op_type>.cpp`) are lifted from the existing
reference-scalar solution for the definition, so the agent only needs to
write the inner kernel — not the binding boilerplate.

Supported op_type: conv2d (primary target; extensible to other ncnn ops).
"""

from __future__ import annotations

from typing import Optional

from bench.data.solution import (
    Solution,
    SourceFile,
    SolutionSpec,
    SupportedDatasets,
    SupportedLanguages,
)

from .base import AgentTools

# Map EC2 instance type → (march flag, isa_features, target_hardware label)
_INSTANCE_ISA: dict[str, tuple[str, list[str], list[str]]] = {
    "c7g.large":  ("-march=armv8.2-a+sve",  ["sve"],  ["graviton3", "aarch64-sve"]),
    "c8g.large":  ("-march=armv9-a+sve2",   ["sve2"], ["graviton4", "aarch64-sve2"]),
    "c7g.xlarge": ("-march=armv8.2-a+sve",  ["sve"],  ["graviton3", "aarch64-sve"]),
    "c8g.xlarge": ("-march=armv9-a+sve2",   ["sve2"], ["graviton4", "aarch64-sve2"]),
}
_FALLBACK_ISA = ("-march=armv8-a", [], ["aarch64"])


class NCNNAgentTools(AgentTools):
    """Tool surface for NCNN-dataset definitions (conv2d, etc.)."""

    dataset = "ncnn"

    @classmethod
    def can_handle(cls, dataset: str) -> bool:
        return dataset == "ncnn"

    def make_solution(self, code: str) -> Solution:
        """Wrap agent-written kernel.cpp into a Solution alongside harness files.

        Harness files (e.g. conv2d.h + conv2d.cpp) are lifted from the
        reference-scalar solution for this definition. compile_flags, isa_features,
        and target_hardware are derived from the remote instance type so they
        always match the actual hardware — not hardcoded.
        """
        ref = self._trace_set.get_baseline_solution(
            self._definition.name, "reference-scalar"
        )
        if ref is None:
            raise ValueError(
                f"No reference-scalar solution for definition {self._definition.name!r} "
                "in TraceSet — cannot build NCNN candidate harness. "
                "Run scripts/gen_definitions.py to generate reference solutions."
            )

        # Lift all harness files except kernel.cpp from the reference solution.
        harness = [s for s in ref.sources if s.path != "kernel.cpp"]
        if not harness:
            raise ValueError(
                f"reference-scalar solution for {self._definition.name!r} has no "
                "harness files besides kernel.cpp — unexpected layout"
            )

        agent_kernel = SourceFile(path="kernel.cpp", content=code)

        # Derive ISA-specific flags from the remote instance type.
        march, isa_features, target_hardware = _INSTANCE_ISA.get(
            self._handle.instance_type, _FALLBACK_ISA
        )

        # Inherit non-optimization, non-march flags from the reference spec
        # (e.g. -std=c++14, any definition-specific flags), then apply -O3 + march.
        base_flags = [
            f for f in ref.spec.compile_flags
            if not f.startswith("-O") and not f.startswith("-march=")
        ]
        compile_flags = ["-O3", march, *base_flags]

        return Solution(
            name=f"{self._author}_{self._definition.name}",
            definition=self._definition.name,
            dataset=SupportedDatasets.NCNN,
            author=self._author,
            spec=SolutionSpec(
                language=ref.spec.language,
                target_hardware=target_hardware,
                entry_point=ref.spec.entry_point,
                dependencies=list(ref.spec.dependencies),
                isa_features=isa_features,
                compile_flags=compile_flags,
                link_flags=list(ref.spec.link_flags),
            ),
            sources=[*harness, agent_kernel],
        )

    def disassemble(self, fn: Optional[str] = None) -> dict:
        """Disassemble the agent's inner kernel (not the binding wrapper)."""
        if fn is None:
            fn = f"inner_{self._definition.op_type}"
        return super().disassemble(fn=fn)

    @classmethod
    def tool_schemas(cls) -> list[dict]:
        return [
            {
                "name": "compile",
                "description": (
                    "Compile your kernel.cpp on the remote ARM (Graviton4 SVE2) instance. "
                    "Your file must define `inner_<op_type>(...)` — the harness files "
                    "(`<op_type>.h` and `<op_type>.cpp`) are provided automatically. "
                    "Returns {\"status\": \"OK\", \"version\": N} on success, or "
                    "{\"status\": \"COMPILE_ERROR\", \"error\": \"...\"} on failure."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "Full C++ source for kernel.cpp. Must implement the "
                                "`inner_<op_type>` function declared in `<op_type>.h`. "
                                "Use `extern \"C\"` if needed; the symbol does NOT need to be "
                                "exported (the harness calls it internally)."
                            ),
                        }
                    },
                    "required": ["code"],
                },
            },
            {
                "name": "evaluate",
                "description": (
                    "Run the last compiled kernel against all workloads on the remote. "
                    "Checks correctness first (fail-fast on the first failing workload). "
                    "If `measure=true` (default) also collects wall-time and cycle counts. "
                    "Returns {\"status\": \"PASSED\", \"performance\": {...}} or "
                    "{\"status\": \"<error>\", \"failed_workload\": \"...\", \"log\": \"...\"}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "measure": {
                            "type": "boolean",
                            "description": (
                                "Whether to measure timing and cycle counts (true, default) "
                                "or only run correctness checks (false, faster)."
                            ),
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "disassemble",
                "description": (
                    "Disassemble the last compiled .so on the remote (up to 300 lines of "
                    "AArch64 assembly). Defaults to `inner_<op_type>` (your kernel). "
                    "Pass `fn` to inspect a different symbol (e.g. the entry wrapper)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fn": {
                            "type": "string",
                            "description": (
                                "Symbol to disassemble. Omit to use `inner_<op_type>` "
                                "(the function you wrote)."
                            ),
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "read_code",
                "description": (
                    "Read a source file or disassembly saved during this session. "
                    "Compiled versions are saved as v1.cpp, v2.cpp, ... (N from compile() result). "
                    "Disassembled versions are saved as v1.s, v2.s, ... (written by disassemble()). "
                    "On error, returns the list of available files so you can pick the right one."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "File to read, e.g. 'v2.cpp' or 'v1.s'.",
                        }
                    },
                    "required": ["filename"],
                },
            },
            {
                "name": "submit",
                "description": (
                    "Compile, run correctness + timing on all workloads, score, and "
                    "persist to the solution warehouse. Use this when you are confident "
                    "in your implementation. "
                    "Returns {\"status\": \"PASSED\", \"time_speedup\": X, "
                    "\"cycle_speedup\": Y} on success."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Final kernel.cpp source to submit.",
                        },
                        "explanation": {
                            "type": "string",
                            "description": (
                                "Brief description of the optimization approach "
                                "(e.g. 'SVE2 fmla tiling with 4×8 register blocking')."
                            ),
                        },
                    },
                    "required": ["code"],
                },
            },
        ]


__all__ = ["NCNNAgentTools"]
