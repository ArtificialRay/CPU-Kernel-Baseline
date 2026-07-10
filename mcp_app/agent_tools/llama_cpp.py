"""LlamaCppKernelSession — KernelSession implementation for the llama.cpp dataset.

Adapted from eval/agent_tools/llama_cpp.py: agent writes `kernel.cpp`
implementing `armbench_llamacpp_<op_type>(...)` (the contract declared in
`<op_type>.h`). The contract header + `binding.cpp` (which bakes the
Definition's const axes and exposes `armbench_entry_<op_type>` over the slim
raw-pointer ABI `RawDataset` expects — the same convention every other
candidate uses) are lifted from the `reference-scalar` solution, so the agent
only writes the kernel body.
"""

from __future__ import annotations

from typing import Optional

from bench.data.solution import Solution, SourceFile, SolutionSpec, SupportedDatasets

from . import isa
from .base import KernelSession, standard_tool_schemas


class LlamaCppKernelSession(KernelSession):
    """Tool surface for llama.cpp dataset definitions (gemm / moe / mha / rms_norm)."""

    dataset = "llama.cpp"

    def make_solution(self, code: str) -> Solution:
        """Wrap agent-written kernel.cpp into a Solution alongside the harness.

        The harness (everything but kernel.cpp) is lifted from the
        'reference-scalar' solution — the slim raw-pointer ABI RawDataset
        expects (mirrors NCNNKernelSession.make_solution()) — not from the
        real ggml baseline. compile_flags/isa_features/target_hardware are
        derived from the session's explicit isa (-O3 + the requested isa's
        march), inheriting reference-scalar's language standard.
        """
        ref = self._trace_set.get_baseline_solution(self._definition.name, "reference-scalar")
        if ref is None:
            raise ValueError(
                f"No 'reference-scalar' solution for definition "
                f"{self._definition.name!r} in TraceSet — cannot build llama.cpp "
                "candidate harness. Run scripts/gen_candidate_solution.py."
            )

        harness = [s for s in ref.sources if s.path != "kernel.cpp"]
        if not harness:
            raise ValueError(
                f"'reference-scalar' solution for {self._definition.name!r} has no harness "
                "files besides kernel.cpp — unexpected layout"
            )

        agent_kernel = SourceFile(path="kernel.cpp", content=code)

        march, isa_features, target_hardware = isa.march_for_isa(
            self._isa, instance_label=self._instance_label
        )

        base_flags = [
            f for f in ref.spec.compile_flags
            if not f.startswith("-O") and not f.startswith("-march=")
        ]
        compile_flags = ["-O3", march, *base_flags]

        return Solution(
            name=f"{self._author}_{self._definition.name}",
            definition=self._definition.name,
            dataset=SupportedDatasets.LLAMA_CPP,
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
        """Disassemble the agent's kernel function (not the entry wrapper)."""
        if fn is None:
            fn = f"armbench_llamacpp_{self._definition.op_type}"
        return super().disassemble(fn=fn)

    @classmethod
    def tool_schemas(cls) -> list[dict]:
        return standard_tool_schemas(
            code_description=(
                "Full C++ source for kernel.cpp. Must implement the "
                "`armbench_llamacpp_<op_type>(...)` function declared in `<op_type>.h`. "
                "You may use ggml (headers/libs are linked) or hand-written SVE/SVE2 "
                "intrinsics. binding.cpp (the void* ABI shim) is provided automatically."
            ),
            disasm_hint="armbench_llamacpp_<op_type>",
        )


__all__ = ["LlamaCppKernelSession"]
