"""
eval/tools.py — Tool implementations for the simd-loops LLM benchmark.

Each function executes via SSH on a provisioned Arm EC2 instance.
These are the tools exposed to the LLM agent (compile, run, perf, disassemble, submit).

The agent writes C code only — it never sees SSH commands or bash.
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from eval.config import ISA_MAKE_TARGET, REPO_ROOT, ISA_TIER

CANDIDATE_START = "// CANDIDATE_INJECT_START"
CANDIDATE_END = "// CANDIDATE_INJECT_END"
BASELINE_START = "// BASELINE_INJECT_START"
BASELINE_END = "// BASELINE_INJECT_END"
CANDIDATE_TESTCASE_START = "// CANDIDATE_TESTCASE_START"
CANDIDATE_TESTCASE_END = "// CANDIDATE_TESTCASE_END"
BASELINE_TESTCASE_START = "// BASELINE_TESTCASE_START"
BASELINE_TESTCASE_END = "// BASELINE_TESTCASE_END"

# Mapping: starter .cpp filename → (cmake library targets, extra ncnn include subdirs).
# Used by _compile_ncnn to link the correct pre-built cmake libraries.
NCNN_STARTER_DEPS: dict[str, tuple[list[str], list[str]]] = {
    "convolution.cpp":            (["mapped_conv_arm", "mapped_conv_base"], ["arm-heavy-optimized/conv"]),
    "convolution1d.cpp":          (["mapped_conv_arm", "mapped_conv_base"], ["arm-heavy-optimized/conv"]),
    "convolutiondepthwise.cpp":   (["mapped_conv_arm", "mapped_conv_base"], ["arm-heavy-optimized/conv"]),
    "decovolution.cpp":           (["mapped_conv_arm", "mapped_conv_base"], ["arm-heavy-optimized/conv"]),
    "deconvolutiondepthwise.cpp": (["mapped_conv_arm", "mapped_conv_base"], ["arm-heavy-optimized/conv"]),
}


@dataclass
class CompileResult:
    success: bool
    errors: str = ""
    warnings: str = ""

    def to_tool_result(self) -> dict:
        if self.success:
            return {"success": True, "warnings": self.warnings or "(none)"}
        return {"success": False, "errors": self.errors}


@dataclass
class RunResult:
    correct: bool
    candidate_runtime_ms: float | None = None
    baseline_runtime_ms:float | None = None
    output: str = ""

    def to_tool_result(self) -> dict:
        return {
            "correct": self.correct,
            "candidate runtime_ms": self.candidate_runtime_ms,
            "baseline runtime_ms": self.baseline_runtime_ms,
            "output": self.output.strip(),
        }


@dataclass
class PerfResult:
    cycles: int | None = None
    instructions: int | None = None
    ipc: float | None = None
    l1d_miss_pct: float | None = None
    raw_output: str = ""

    def to_tool_result(self) -> dict:
        return {
            "cycles": self.cycles,
            "instructions": self.instructions,
            "ipc": self.ipc,
            "l1d_miss_pct": self.l1d_miss_pct,
            "raw_output": self.raw_output.strip(),
        }


@dataclass
class DisasmResult:
    asm: str = ""
    bytes: int = 0

    def to_tool_result(self) -> dict:
        return {"asm": self.asm, "bytes": self.bytes}


@dataclass
class EvalResult:
    correct: bool
    speedup_vs_scalar: float | None = None
    speedup_vs_autovec: float | None = None
    speedup_vs_ref: float | None = None
    level: int = 0
    compile_error: str = ""
    runtime_ms: float | None = None
    tool_calls: int = 0
    # Timing at each PERF_SIZE: {size: runtime_ms}. Populated at submit time.
    perf_by_size: dict | None = None

    def to_dict(self) -> dict:
        return {
            "correct": self.correct,
            "speedup_vs_scalar": self.speedup_vs_scalar,
            "speedup_vs_autovec": self.speedup_vs_autovec,
            "speedup_vs_ref": self.speedup_vs_ref,
            "level": self.level,
            "compile_error": self.compile_error,
            "runtime_ms": self.runtime_ms,
            "tool_calls": self.tool_calls,
            "perf_by_size": self.perf_by_size,
        }


class SIMDTools:
    """
    SSH-backed tools for compiling and running SIMD kernels on a remote Arm instance.

    Used both by the agentic eval loop (as LLM tool calls) and by the
    single-shot eval harness (eval_from_generations.py).
    """

    def __init__(self, handle, problem_id: str, isa: str):
        self.handle = handle
        self.problem_id = problem_id
        self.isa = isa
        self.make_target = ISA_MAKE_TARGET[isa]
        self._last_compile_ok = False
        self._tool_calls = 0
        self._last_candidate_code: str | None = None

        # Remote paths for ncnn starter files
        #TODO: User name for remote instance may vary, hard-code ubuntu here
        self.remote_project_root = "/home/ubuntu/Remote/CPU-Kernel-Baseline" # in local test, don't pollute main working dir; 
        self.remote_ncnn_root = f"{self.remote_project_root}/ncnn"
        self.remote_ncnn_build = f"{self.remote_ncnn_root}/mapped/tests/build"
        self.remote_starter_dir = f"{self.remote_project_root}/arm-bench/starter"
        self.remote_binary: str | None = None           # set by compile()
        self.remote_baseline_binary: str | None = None  # set by compile()


    # ─── SSH / local execution helpers ──────────────────────────────────────

    def _run(self, cmd: str, timeout: int = 120) -> tuple[int, str, str]:
        """Run a shell command via SSH if a handle is set, else locally."""
        if self.handle is not None:
            return self._run(cmd, timeout=timeout)
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr

    def _upload(self, local_path: str, remote_path: str) -> None:
        """Upload a file via SSH if a handle is set, else copy locally."""
        if self.handle is not None:
            self.handle.upload_file(local_path, remote_path)
        else:
            # remote_path may use ~ — expand relative to home
            dst = os.path.expanduser(remote_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(local_path, dst)

    # ─── Upload helpers ────────────────────────────────────────────────────────

    def upload_ncnn_tree(self) -> None:
        """
        Sync the local ncnn source tree and arm-bench/starter directory to the
        remote instance under ``self.remote_project_root``.

        Uses rsync when an SSH handle is available (incremental, fast on
        subsequent calls).  Falls back to shutil.copytree for local-only
        testing (handle is None).
        """
        local_ncnn = str(REPO_ROOT.parent / "ncnn")            # CPU-Kernel-Baseline/ncnn
        local_starter = str(REPO_ROOT / "starter")              # arm-bench/starter

        remote_ncnn = self.remote_ncnn_root                     # ~/Remote/.../ncnn
        remote_starter = self.remote_starter_dir                # ~/Remote/.../arm-bench/starter

        rsync_excludes = ["build", "__pycache__", "*.o", "*.d", ".git"]

        if self.handle is not None:
            # Ensure remote directories exist
            self.handle.run(f"mkdir -p {remote_ncnn} {remote_starter}")
            self.handle.rsync_to(local_ncnn, remote_ncnn, excludes=rsync_excludes)
            self.handle.rsync_to(local_starter, remote_starter, excludes=rsync_excludes)
        else:
            # Local-only: expand ~ and copy trees
            for src, dst in [(local_ncnn, remote_ncnn), (local_starter, remote_starter)]:
                dst_expanded = os.path.expanduser(dst)
                if os.path.exists(dst_expanded):
                    shutil.rmtree(dst_expanded)
                shutil.copytree(src, dst_expanded,
                                ignore=shutil.ignore_patterns(*rsync_excludes))

    # ─── Tool: compile ───────────────────────────────────────────────────────

    @staticmethod
    def _strip_block(source: str, start_marker: str, end_marker: str) -> str:
        """Remove everything between (and including) *start_marker* … *end_marker*."""
        return re.sub(
            re.escape(start_marker) + ".*?" + re.escape(end_marker),
            "",
            source,
            flags=re.DOTALL,
        )

    @staticmethod
    def _extract_test_functions(source: str, start_marker: str, end_marker: str) -> list[str]:
        """Extract test function names from a TESTCASE block."""
        m = re.search(
            re.escape(start_marker) + r"(.*?)" + re.escape(end_marker),
            source, flags=re.DOTALL,
        )
        if not m:
            return []
        block = m.group(1)
        return re.findall(r"void\s+(test_\w+)\s*\(\s*\)", block)

    @staticmethod
    def _generate_main(test_funcs: list[str], suite_name: str) -> str:
        """Generate a main() function that runs each test with RUN_TEST."""
        lines = ["\n// ── Auto-generated main ──────────────────────────────────────"]
        for fn in test_funcs:
            lines.append(f"void {fn}();")
        lines.append("")
        lines.append("int main() {")
        for fn in test_funcs:
            lines.append(f"    RUN_TEST({fn});")
        lines.append(f'    print_summary("{suite_name}");')
        lines.append("    return g_failed ? 1 : 0;")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def compile(self, file: str) -> CompileResult:
        """
        Compile an ncnn starter .cpp file into **two** binaries:

          <stem>       – candidate code only  (BASELINE block stripped)
          <stem>_arm   – baseline code only   (CANDIDATE block stripped)

        For example ``file="convolution.cpp"`` produces::

            starter/build/convolution      (candidate)
            starter/build/convolution_arm  (baseline / ARM-optimised)

        ``self.remote_binary`` is set to the candidate binary and
        ``self.remote_baseline_binary`` to the baseline binary so that
        ``run()`` / ``perf()`` / ``disassemble()`` work with the candidate
        by default.

        Args:
            file: Starter filename, e.g. "convolution.cpp".

        Returns:
            CompileResult with success flag and any errors/warnings.
        """
        self._tool_calls += 1

        local_file = REPO_ROOT / "starter" / file
        if not local_file.exists():
            return CompileResult(success=False, errors=f"Starter file not found: {local_file}")

        source = local_file.read_text()
        if CANDIDATE_START not in source:
            return CompileResult(
                success=False,
                errors=f"CANDIDATE_INJECT_START marker missing from {file}.",
            )

        stem = file.rsplit(".", 1)[0]  # "convolution.cpp" → "convolution"

        # ── Generate two source variants ─────────────────────────────
        # candidate-only: keep CANDIDATE block, strip BASELINE block
        candidate_src = source
        if BASELINE_START in source:
            candidate_src = self._strip_block(source, BASELINE_START, BASELINE_END)
        if BASELINE_TESTCASE_START in candidate_src:
            candidate_src = self._strip_block(candidate_src, BASELINE_TESTCASE_START, BASELINE_TESTCASE_END)

        # baseline-only: keep BASELINE block, strip CANDIDATE block
        baseline_src = self._strip_block(source, CANDIDATE_START, CANDIDATE_END)
        if CANDIDATE_TESTCASE_START in baseline_src:
            baseline_src = self._strip_block(baseline_src, CANDIDATE_TESTCASE_START, CANDIDATE_TESTCASE_END)

        # ── Auto-generate main() for each variant ───────────────────
        candidate_tests = self._extract_test_functions(source, CANDIDATE_TESTCASE_START, CANDIDATE_TESTCASE_END)
        baseline_tests = self._extract_test_functions(source, BASELINE_TESTCASE_START, BASELINE_TESTCASE_END)

        if candidate_tests:
            candidate_src += self._generate_main(candidate_tests, f"{stem}_candidate")
        if baseline_tests:
            baseline_src += self._generate_main(baseline_tests, f"{stem}_baseline")

        # ── Upload both variants ─────────────────────────────────────
        candidate_remote = f"{self.remote_starter_dir}/{stem}_candidate.cpp"
        baseline_remote = f"{self.remote_starter_dir}/{stem}_baseline.cpp"

        for src_text, remote_path in [
            (candidate_src, candidate_remote),
            (baseline_src, baseline_remote),
        ]:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as f:
                f.write(src_text)
                tmp_path = f.name
            try:
                self._upload(tmp_path, remote_path)
            finally:
                os.unlink(tmp_path)

        # # ── Ensure include symlink ───────────────────────────────────
        # # arm-bench/ncnn → ncnn/ so #include "../ncnn/mapped/..." resolves
        # symlink_cmd = (
        #     f"ln -sfn {self.remote_ncnn_root} "
        #     f"{self.remote_project_root}/arm-bench/ncnn"
        # )
        # self._run(symlink_cmd)

        # ── Ensure cmake libraries are built ─────────────────────────
        lib_targets, extra_inc_dirs = NCNN_STARTER_DEPS.get(
            file, (["mapped_conv_arm", "mapped_conv_base"], [])
        )
        all_cmake_targets = ["ncnn_stub"] + lib_targets
        cmake_build_cmd = (
            f"mkdir -p {self.remote_ncnn_build} && "
            f"cd {self.remote_ncnn_build} && "
            f"cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 && "
            f"make -j$(nproc) {' '.join(all_cmake_targets)} 2>&1"
        )
        rc, output, _ = self._run(cmake_build_cmd, timeout=180)
        if rc != 0:
            self._last_compile_ok = False
            errors = "\n".join(
                l for l in output.splitlines() if "error:" in l.lower()
            )
            return CompileResult(
                success=False, errors=errors or f"ncnn cmake build failed:\n{output}"
            )

        # ── Shared compiler / linker flags ───────────────────────────
        include_flags = (
            f"-I {self.remote_starter_dir} "
            f"-I {self.remote_ncnn_root} "
            f"-I {self.remote_ncnn_root}/framework"
        )
        for subdir in extra_inc_dirs:
            include_flags += f" -I {self.remote_ncnn_root}/{subdir}"

        lib_flags = " ".join(
            f"{self.remote_ncnn_build}/lib{t}.a" for t in all_cmake_targets
        )

        binary_dir = f"{self.remote_starter_dir}/build"
        cxx_base = (
            f"clang++ -O2 -std=c++14 -march=armv8.2-a+fp16+dotprod -fopenmp "
            f"{include_flags}"
        )
        link_tail = f"{lib_flags} -lm -lstdc++"

        all_warnings: list[str] = []

        # ── Compile candidate binary → <stem> ────────────────────────
        candidate_bin = f"{binary_dir}/{stem}"
        candidate_cmd = (
            f"mkdir -p {binary_dir} && "
            f"{cxx_base} {candidate_remote} {link_tail} "
            f"-o {candidate_bin} 2>&1"
        )
        rc, combined, _ = self._run(candidate_cmd, timeout=120)

        all_warnings.extend(
            l for l in combined.splitlines()
            if "warning:" in l.lower() and "error:" not in l.lower()
        )
        if rc != 0:
            self._last_compile_ok = False
            errors = "\n".join(
                l for l in combined.splitlines() if "error:" in l.lower()
            )
            return CompileResult(
                success=False,
                errors=f"[{stem}] " + (errors or combined),
            )

        # ── Compile baseline binary → <stem>_arm ─────────────────────
        baseline_bin = f"{binary_dir}/{stem}_arm"
        baseline_cmd = (
            f"{cxx_base} {baseline_remote} {link_tail} "
            f"-o {baseline_bin} 2>&1"
        )
        rc, combined, _ = self._run(baseline_cmd, timeout=120)

        all_warnings.extend(
            l for l in combined.splitlines()
            if "warning:" in l.lower() and "error:" not in l.lower()
        )
        if rc != 0:
            self._last_compile_ok = False
            errors = "\n".join(
                l for l in combined.splitlines() if "error:" in l.lower()
            )
            return CompileResult(
                success=False,
                errors=f"[{stem}_arm] " + (errors or combined),
            )

        # ── Success — update state ───────────────────────────────────
        self._last_compile_ok = True
        self._last_candidate_code = source
        self.remote_binary = candidate_bin           # candidate for run()/perf()/disassemble()
        self.remote_baseline_binary = baseline_bin   # baseline for comparison
        return CompileResult(
            success=True,
            warnings="\n".join(all_warnings) if all_warnings else "",
        )

    # ─── Tool: run ───────────────────────────────────────────────────────────

    def run(self, n: int = 1) -> RunResult:
        """
        Run both the candidate and baseline binaries and report correctness + timing.

        Each binary contains fixed test cases (no size parameter needed).
        The binary exits 0 if all tests pass, non-zero if any fail.

        Args:
            n:    Number of iterations to run for timing (the binary is invoked
                  n times and the total wall-clock time is measured).

        Returns:
            RunResult with correct flag, runtime_ms (candidate), and output
            showing results from both candidate and baseline runs.
        """
        self._tool_calls += 1
        if not self._last_compile_ok:
            return RunResult(correct=False, output="No compiled binary — run compile() first.")

        outputs: list[str] = []
        all_correct = True

        # Run candidate binary
        candidate_time_cmd = (
            f"t0=$(date +%s%N); "
            f"for i in $(seq 1 {n}); do {self.remote_binary}; rc=$?; "
            f"if [ $rc -ne 0 ]; then exit $rc; fi; done; "
            f"t1=$(date +%s%N); "
            f'echo "TIME_NS=$((t1-t0))"'
        )
        rc, stdout, _ = self._run(candidate_time_cmd, timeout=300)
        candidate_correct = (rc == 0) and ("FAIL" not in stdout)
        candidate_ms = None
        m = re.search(r"TIME_NS=(\d+)", stdout)
        if m:
            candidate_ms = round(int(m.group(1)) / 1e6, 3)
        output_clean = re.sub(r"TIME_NS=\d+", "", stdout).strip()
        # Show only the last iteration's output (avoid n repetitions)
        last_summary = ""
        for line in output_clean.splitlines():
            if line.strip().startswith("[") and "passed" in line:
                last_summary = line.strip()
        outputs.append(f"[candidate] {'PASS' if candidate_correct else 'FAIL'}  {last_summary}  time={candidate_ms}ms")
        if not candidate_correct:
            outputs.append(f"  output: {output_clean[-500:]}")
            all_correct = False

        # Run baseline binary
        baseline_ms=None
        if self.remote_baseline_binary:
            baseline_time_cmd = (
                f"t0=$(date +%s%N); "
                f"for i in $(seq 1 {n}); do {self.remote_baseline_binary}; rc=$?; "
                f"if [ $rc -ne 0 ]; then exit $rc; fi; done; "
                f"t1=$(date +%s%N); "
                f'echo "TIME_NS=$((t1-t0))"'
            )
            rc, stdout, _ = self._run(baseline_time_cmd, timeout=300)
            baseline_correct = (rc == 0) and ("FAIL" not in stdout)
            baseline_ms = None
            m = re.search(r"TIME_NS=(\d+)", stdout)
            if m:
                baseline_ms = round(int(m.group(1)) / 1e6, 3)
            output_clean = re.sub(r"TIME_NS=\d+", "", stdout).strip()
            last_summary = ""
            for line in output_clean.splitlines():
                if line.strip().startswith("[") and "passed" in line:
                    last_summary = line.strip()
            outputs.append(f"[baseline]  {'PASS' if baseline_correct else 'FAIL'}  {last_summary}  time={baseline_ms}ms")
            if not baseline_correct:
                outputs.append(f"  output: {output_clean[-500:]}")
                all_correct = False

        return RunResult(
            correct=all_correct,
            candidate_runtime_ms=candidate_ms,
            baseline_runtime_ms=baseline_ms,
            output="\n".join(outputs),
        )

    # ─── Tool: perf ──────────────────────────────────────────────────────────

    def perf(self, n: int = 1) -> PerfResult:
        """
        Run perf stat on both candidate and baseline binaries to collect
        hardware PMU counters and compare performance.

        Available on Graviton3/4 via Nitro:
          - cycles, instructions, IPC
          - r04 = L1D_CACHE accesses, r03 = L1D_CACHE_REFILL (misses)

        Note: L2/L3 counters are not exposed by the Nitro hypervisor.

        Args:
            n:    Number of times to invoke the binary under perf stat.

        Returns:
            PerfResult with cycles, instructions, IPC, L1D miss % for the
            candidate, plus raw_output containing both candidate and baseline
            results for comparison.
        """
        self._tool_calls += 1
        if not self._last_compile_ok:
            return PerfResult(raw_output="No compiled binary — run compile() first.")

        perf_probe = (
            "PERF=$(ls /usr/lib/linux-aws-*-tools-*/perf 2>/dev/null | head -1); "
            "PERF=${PERF:-perf}; "
        )

        # ── Candidate perf ──
        run_loop = f"for i in $(seq 1 {n}); do {self.remote_binary}; done" if n > 1 else self.remote_binary
        candidate_cmd = (
            f"{perf_probe}"
            f"sudo $PERF stat "
            f"-e cycles,instructions,r04,r03 "
            f"bash -c '{run_loop}' "
            f"2>&1"
        )
        rc, candidate_output, _ = self._run(candidate_cmd, timeout=300)
        candidate_perf = self._parse_perf_output(candidate_output)
        candidate_perf.raw_output = f"=== CANDIDATE ===\n{candidate_output}"
        #raw_parts = [f"=== CANDIDATE ===\n{candidate_output}"]

        # ── Baseline perf ──
        if self.remote_baseline_binary:
            run_loop_bl = f"for i in $(seq 1 {n}); do {self.remote_baseline_binary}; done" if n > 1 else self.remote_baseline_binary
            baseline_cmd = (
                f"{perf_probe}"
                f"sudo $PERF stat "
                f"-e cycles,instructions,r04,r03 "
                f"bash -c '{run_loop_bl}' "
                f"2>&1"
            )
            rc, baseline_output, _ = self._run(baseline_cmd, timeout=300)
            baseline_perf = self._parse_perf_output(baseline_output)
            baseline_perf.raw_output = f"=== BASELINE ===\n{baseline_output}"
            #raw_parts.append(f"\n=== BASELINE ===\n{baseline_output}")

        #candidate_perf.raw_output = "\n".join(raw_parts)
        return candidate_perf,baseline_perf

    @staticmethod
    def _parse_perf_output(output: str) -> PerfResult:
        cycles = _parse_perf_counter(output, "cycles")
        instructions = _parse_perf_counter(output, "instructions")

        ipc = None
        m = re.search(r"([\d.]+)\s+insn per cycle", output)
        if m:
            ipc = float(m.group(1))
        elif cycles and instructions:
            ipc = round(instructions / cycles, 2) if cycles > 0 else None

        l1d_accesses = _parse_perf_counter(output, r"r04")
        l1d_misses = _parse_perf_counter(output, r"r03")
        l1d_miss_pct = None
        if l1d_accesses and l1d_misses and l1d_accesses > 0:
            l1d_miss_pct = round(100.0 * l1d_misses / l1d_accesses, 2)

        return PerfResult(
            cycles=cycles,
            instructions=instructions,
            ipc=ipc,
            l1d_miss_pct=l1d_miss_pct,
            raw_output=output,
        )

    # ─── Tool: disassemble ───────────────────────────────────────────────────

    def disassemble(self, fn: str | None = None) -> DisasmResult:
        """
        Disassemble the compiled binary, optionally filtered to a function.

        Args:
            fn: Function name to filter to (e.g. "inner_loop_001").
                If None, returns the full disassembly (may be large).

        Returns:
            DisasmResult with assembly text and approximate byte count.
        """
        self._tool_calls += 1
        if not self._last_compile_ok:
            return DisasmResult(asm="No compiled binary — run compile() first.")

        def _objdump_fn(name: str) -> str:
            return (
                f"llvm-objdump-18 -d {self.remote_binary} "
                f"| awk '/<{name}>:/ {{p=1}} p && /<[a-zA-Z_].*>:/ && !/<{name}>:/ {{p=0}} p'"
            )

        if fn:
            rc, output, stderr = self._run(_objdump_fn(fn), timeout=60)
            if rc != 0:
                return DisasmResult(asm=f"objdump failed: {stderr}")
            # If the requested symbol was inlined, fall back to main
            if not output.strip():
                rc, output, stderr = self._run(_objdump_fn("main"), timeout=60)
                if rc != 0:
                    return DisasmResult(asm=f"objdump failed: {stderr}")
        else:
            rc, output, stderr = self._run(
                f"llvm-objdump-18 -d {self.remote_binary}", timeout=60
            )
            if rc != 0:
                return DisasmResult(asm=f"objdump failed: {stderr}")

        # Truncate to first 500 lines to avoid flooding context
        lines = output.splitlines()
        truncated = False
        if len(lines) > 500:
            lines = lines[:500]
            truncated = True
        asm = "\n".join(lines)
        if truncated:
            asm += "\n... (truncated at 500 lines)"

        return DisasmResult(asm=asm, bytes=len(output.encode()))

    # ─── Tool: submit ─────────────────────────────────────────────────────────

    def submit(self, code: str) -> EvalResult:
        """
        Final submission: compile, run correctness checks, and compare
        candidate vs ARM baseline performance.

        Args:
            code: The starter .cpp filename (e.g. "convolution.cpp").

        Returns:
            EvalResult with correctness, speedup vs baseline, and timing.
        """
        self._tool_calls += 1

        # Compile both candidate and baseline
        cr = self.compile(code)
        if not cr.success:
            return EvalResult(
                correct=False,
                level=0,
                compile_error=cr.errors,
                tool_calls=self._tool_calls,
            )

        # Run correctness check
        rr = self.run(n=1)
        if not rr.correct:
            return EvalResult(
                correct=False,
                level=0,
                tool_calls=self._tool_calls,
            )

        # Authoritative timing: run both candidate and baseline multiple times
        n_iters = 10
        # Candidate timing
        candidate_cmd = (
            f"t0=$(date +%s%N); "
            f"for i in $(seq 1 {n_iters}); do {self.remote_binary} > /dev/null 2>&1; done; "
            f"t1=$(date +%s%N); "
            f'echo "TIME_NS=$((t1-t0))"'
        )
        _, stdout, _ = self._run(candidate_cmd, timeout=600)
        candidate_ms = None
        m = re.search(r"TIME_NS=(\d+)", stdout)
        if m:
            candidate_ms = round(int(m.group(1)) / 1e6, 3)

        # Baseline timing
        baseline_ms = None
        if self.remote_baseline_binary:
            baseline_cmd = (
                f"t0=$(date +%s%N); "
                f"for i in $(seq 1 {n_iters}); do {self.remote_baseline_binary} > /dev/null 2>&1; done; "
                f"t1=$(date +%s%N); "
                f'echo "TIME_NS=$((t1-t0))"'
            )
            _, stdout, _ = self._run(baseline_cmd, timeout=600)
            m = re.search(r"TIME_NS=(\d+)", stdout)
            if m:
                baseline_ms = round(int(m.group(1)) / 1e6, 3)

        # Compute speedup: baseline / candidate (>1 means candidate is faster)
        speedup_vs_ref = None
        level = 1  # correct
        if candidate_ms and baseline_ms and candidate_ms > 0:
            speedup_vs_ref = round(baseline_ms / candidate_ms, 2)
            if speedup_vs_ref > 1.0:
                level = 2  # faster than ARM baseline

        return EvalResult(
            correct=True,
            speedup_vs_ref=speedup_vs_ref,
            level=level,
            runtime_ms=candidate_ms,
            tool_calls=self._tool_calls,
        )


    # ─── OpenAI-compatible tool schemas ──────────────────────────────────────

    @staticmethod
    def tool_schemas() -> list[dict]:
        """Return OpenAI-compatible function tool definitions for LiteLLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "compile",
                    "description": (
                        "Compile your SIMD implementation on the target Arm instance. "
                        "Returns whether compilation succeeded and any errors/warnings."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": (
                                    "Your complete C implementation of the inner_loop function. "
                                    "Must preserve the exact function signature from the scalar version."
                                ),
                            },
                        },
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run",
                    "description": (
                        "Run both candidate and baseline binaries and check correctness + timing. "
                        "Must call compile() successfully first. "
                        "Test cases are fixed in the starter file."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n": {
                                "type": "integer",
                                "description": "Number of iterations (default 1; more = more stable timing).",
                                "default": 1,
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "perf",
                    "description": (
                        "Run perf stat on both candidate and baseline binaries to collect "
                        "hardware PMU counters: cycles, instructions, IPC, L1D cache miss rate. "
                        "Compares performance between candidate and ARM baseline. "
                        "Note: L2/L3 counters are not available on Nitro-based Graviton instances."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n": {
                                "type": "integer",
                                "description": "Number of iterations.",
                                "default": 1,
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "disassemble",
                    "description": (
                        "Disassemble the compiled binary. Filter to a specific function "
                        "to see the generated AArch64 instructions. Useful for checking "
                        "whether the compiler vectorized correctly."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fn": {
                                "type": "string",
                                "description": (
                                    "Function name to filter to, e.g. 'inner_loop_001'. "
                                    "If omitted, returns full disassembly (may be large)."
                                ),
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "submit",
                    "description": (
                        "Submit your final implementation for scoring. "
                        "Compiles, runs 1000 iterations, and computes speedup vs scalar and autovec baselines. "
                        "Call this when you are satisfied with your implementation."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Your final optimized C implementation.",
                            },
                        },
                        "required": ["code"],
                    },
                },
            },
        ]

    def dispatch_tool_call(self, name: str, args: dict) -> dict:
        """Dispatch a tool call by name and return a serialisable result dict."""
        if name == "compile":
            return self.compile(args["code"]).to_tool_result()
        elif name == "run":
            return self.run(args.get("n", 1)).to_tool_result()
        elif name == "perf":
            return self.perf(args.get("n", 1)).to_tool_result()
        elif name == "disassemble":
            return self.disassemble(args.get("fn")).to_tool_result()
        elif name == "submit":
            result = self.submit(args["code"])
            return result.to_dict()
        else:
            return {"error": f"Unknown tool: {name}"}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_perf_counter(text: str, event: str) -> int | None:
    """Parse a numeric counter value from perf stat output."""
    # perf stat output format: "   1,234,567      cycles  ..."
    pattern = rf"([\d,]+)\s+{re.escape(event)}"
    m = re.search(pattern, text)
    if m:
        return int(m.group(1).replace(",", ""))
    return None
