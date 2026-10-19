#!/usr/bin/env python3
"""kernel_set: solutions in, manifest out -- with a fake compiler, so no ARM box is needed.

Run:  .venv/bin/python -m unittest e2e.adapters.llamacpp.test_kernel_set   (from the repo root)
"""
import contextlib
import io
import json
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from e2e.adapters.llamacpp import kernel_set
from e2e.errors import StepFailed

DATASET = "llama.cpp"
GEMM_H = "namespace gemm_def {{ constexpr int N = {N}; }}\n"
FAKE_CXX = """#!/bin/sh
# stands in for clang++: "builds" the -o file unless the kernel asks it not to
out=""; kernel=""; prev=""
for a in "$@"; do
  [ "$prev" = "-o" ] && out="$a"
  case "$a" in *kernel.cpp) kernel="$a" ;; esac
  prev="$a"
done
grep -q COMPILE_FAIL "$kernel" && { echo "error: boom" >&2; exit 1; }
# only fails once the rows-ABI harness (which defines N_FULL) is in place
grep -q ROWS_FAIL "$kernel" && grep -q N_FULL "$(dirname "$kernel")/gemm.cpp" && { echo "error: rows" >&2; exit 1; }
: > "$out"
"""


def solution(name: str, N: int, kernel: str, gemm_h: str | None = None) -> dict:
    return {"name": f"a_{name}", "definition": name, "dataset": DATASET, "author": "a",
            "sources": [{"path": "gemm.h", "content": gemm_h or GEMM_H.format(N=N)},
                        {"path": "gemm.cpp", "content": "// harness\n"},
                        {"path": "kernel.cpp", "content": kernel}]}


class KernelSetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "solutions"
        self.cxx = self.tmp / "fake-cxx"
        self.cxx.write_text(FAKE_CXX)
        self.cxx.chmod(self.cxx.stat().st_mode | stat.S_IEXEC)
        self.out = self.tmp / "out" / "manifest.json"

    def put(self, author: str, name: str, sol: dict) -> None:
        d = self.root / DATASET / author / "gemm"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps(sol))

    def run_kernel_set(self, author: str = "a", **kw) -> tuple[int, str]:
        """(1 when a guard failed else 0, everything it printed)."""
        kw.setdefault("march", "-march=armv8-a")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = kernel_set.build_kernel_set(
                author, self.out, solutions_root=self.root, cxx=str(self.cxx), **kw)
        if result.guard_failures:
            buf.write("FAILED:\n" + "\n".join("  " + f for f in result.guard_failures))
        return (1 if result.guard_failures else 0), buf.getvalue()

    def manifest(self) -> dict:
        return json.loads(self.out.read_text())

    def test_builds_the_solutions_it_was_given(self):
        self.put("a", "gemm_ggml_q4_K_n1024_k2560", solution("gemm_ggml_q4_K_n1024_k2560", 1024, "// k\n"))
        self.put("a", "gemm_ggml_q6_K_n2560_k9216", solution("gemm_ggml_q6_K_n2560_k9216", 2560, "// k\n"))
        rc, _ = self.run_kernel_set()
        self.assertEqual(rc, 0)
        kernels = self.manifest()["kernels"]
        self.assertEqual([(k["type"], k["K"], k["N"], k["abi"]) for k in kernels],
                         [("q4_K", 2560, 1024, "entry"), ("q6_K", 9216, 2560, "entry")])
        self.assertEqual(self.manifest()["author"], "a")
        # the solution's own harness was compiled, nothing borrowed from elsewhere
        built = self.out.parent / "build" / "gemm_ggml_q4_K_n1024_k2560"
        self.assertEqual((built / "gemm.cpp").read_text(), "// harness\n")

    def test_intermediate_files_and_other_names_do_not_become_kernels(self):
        name = "gemm_ggml_q4_K_n1024_k2560"
        self.put("a", name, solution(name, 1024, "// the submitted best\n"))
        self.put("a", name + "_current", solution(name, 1024, "// the last compile\n"))
        self.put("a", name + "_best", solution(name, 1024, "// mid-rename\n"))
        self.put("a", "gemm_q4_k_m_n1024_k2048", solution("gemm_q4_k_m_n1024_k2048", 1024, "// legacy\n"))
        self.run_kernel_set()
        m = self.manifest()
        self.assertEqual([k["definition"] for k in m["kernels"]], [name])
        self.assertEqual((self.out.parent / "build" / name / "kernel.cpp").read_text(), "// the submitted best\n")
        self.assertEqual([s["definition"] for s in m["skipped"]], ["gemm_q4_k_m_n1024_k2048"])

    def test_rows_abi_rewrites_and_falls_back_when_it_cannot_compile(self):
        ok, bad = "gemm_ggml_q4_K_n1024_k2560", "gemm_ggml_q5_K_n1024_k2560"
        self.put("a", ok, solution(ok, 1024, "static float scratch[16];\n"))
        self.put("a", bad, solution(bad, 1024, "ROWS_FAIL\n"))
        self.run_kernel_set(rows_abi=True)
        abis = {k["definition"]: k["abi"] for k in self.manifest()["kernels"]}
        self.assertEqual(abis, {ok: "entry_rows", bad: "entry"})
        built = self.out.parent / "build" / ok
        self.assertIn("thread_local", (built / "kernel.cpp").read_text())
        self.assertIn("N_FULL", (built / "gemm.h").read_text())
        # the fallback recompiled the solution as submitted, not the rewritten harness
        self.assertEqual((self.out.parent / "build" / bad / "gemm.cpp").read_text(), "// harness\n")

    def test_embeddable_says_why_not(self):
        self.assertEqual(kernel_set.embeddable("gemm_ggml_q4_K_n1024_k2560"), (True, ""))
        for name, fragment in (("gemm_ggml_q8_0_n1024_k2560", "no kernel for"),
                               ("gemm_q4_k_m_n1024_k2048", "flat-quant"),
                               ("rms_norm_fp32_d2048", "not a packed-ABI gemm")):
            ok, why = kernel_set.embeddable(name)
            self.assertFalse(ok, name)
            self.assertIn(fragment, why)

    def test_manifest_records_framework_and_embeddability(self):
        good = "gemm_ggml_q4_K_n1024_k2560"
        self.put("a", good, solution(good, 1024, "// k\n"))
        self.put("a", "gemm_q4_k_m_n1024_k2048", solution("gemm_q4_k_m_n1024_k2048", 1024, "// legacy\n"))
        self.put("a", "gemm_ggml_q5_K_n1024_k2560", solution("gemm_ggml_q5_K_n1024_k2560", 1024, "COMPILE_FAIL\n"))
        self.run_kernel_set(only_types=["q4_K", "q5_K"])
        m = self.manifest()
        self.assertEqual(m["framework"], "llamacpp")
        self.assertEqual([(k["definition"], k["embeddable"]) for k in m["kernels"]], [(good, True)])
        self.assertEqual({s["definition"]: s["embeddable"] for s in m["skipped"]},
                         {"gemm_q4_k_m_n1024_k2048": False, "gemm_ggml_q5_K_n1024_k2560": False})

    def test_compile_failure_is_skipped_not_fatal(self):
        good, broken = "gemm_ggml_q4_K_n1024_k2560", "gemm_ggml_q5_K_n1024_k2560"
        self.put("a", good, solution(good, 1024, "// k\n"))
        self.put("a", broken, solution(broken, 1024, "COMPILE_FAIL\n"))
        self.run_kernel_set()
        m = self.manifest()
        self.assertEqual([k["definition"] for k in m["kernels"]], [good])
        self.assertIn("compile failed", m["skipped"][0]["reason"])

    def test_kernel_calling_ggml_is_skipped(self):
        name = "gemm_ggml_q4_K_n1024_k2560"
        self.put("a", name, solution(name, 1024, "// k\n"))
        original = kernel_set.undefined_ggml_symbols
        kernel_set.undefined_ggml_symbols = lambda so: ["ggml_vec_dot_q4_K_q8_K"]
        self.addCleanup(setattr, kernel_set, "undefined_ggml_symbols", original)
        self.run_kernel_set()
        m = self.manifest()
        self.assertEqual(m["kernels"], [])
        self.assertIn("ggml_vec_dot_q4_K_q8_K", m["skipped"][0]["reason"])

    @unittest.skipUnless(shutil.which("gcc") and shutil.which("nm"), "needs gcc and nm")
    def test_undefined_ggml_symbols_reads_the_real_object(self):
        src = self.tmp / "k.c"
        src.write_text("extern int ggml_vec_dot_q4_K_q8_K(void); int f(void){return ggml_vec_dot_q4_K_q8_K();}\n")
        so = self.tmp / "k.so"
        subprocess.run(["gcc", "-shared", "-fPIC", str(src), "-o", str(so)], check=True)
        self.assertEqual(kernel_set.undefined_ggml_symbols(so), ["ggml_vec_dot_q4_K_q8_K"])
        src.write_text("int f(void){return 1;}\n")
        subprocess.run(["gcc", "-shared", "-fPIC", str(src), "-o", str(so)], check=True)
        self.assertEqual(kernel_set.undefined_ggml_symbols(so), [])

    def test_guards_catch_a_stale_set(self):
        a, b = "gemm_ggml_q4_K_n1024_k2560", "gemm_ggml_q5_K_n1024_k2560"
        self.put("a", a, solution(a, 1024, "// new\n"))
        self.put("a", b, solution(b, 1024, "// same\n"))
        self.put("old", a, solution(a, 1024, "// new\n"))        # the "new" kernel is the old one
        self.put("old", b, solution(b, 1024, "// same\n"))
        rc, text = self.run_kernel_set(reference_author="old", expect_changed=[a], expect_same=[b])
        self.assertEqual(rc, 1)
        self.assertIn(f"{a}: BYTE-IDENTICAL to the reference", text)
        self.assertNotIn(f"{b}:", text)

        self.put("a", a, solution(a, 1024, "// really new\n"))
        rc, _ = self.run_kernel_set(reference_author="old", expect_changed=[a], expect_same=[b], expect_count=2)
        self.assertEqual(rc, 0)
        rc, text = self.run_kernel_set(expect_count=3)
        self.assertEqual(rc, 1)
        self.assertIn("expected 3", text)
        rc, text = self.run_kernel_set(forbid=["really"])
        self.assertEqual(rc, 1)
        self.assertIn("forbidden pattern", text)

    def test_guards_compare_through_the_rows_abi_rewrite(self):
        name = "gemm_ggml_q4_K_n1024_k2560"
        self.put("a", name, solution(name, 1024, "static float scratch[16];\n"))
        self.put("old", name, solution(name, 1024, "static thread_local float scratch[16];\n"))
        rc, _ = self.run_kernel_set(reference_author="old", expect_same=[name])
        self.assertEqual(rc, 0)

    def test_missing_author_dir_is_an_error(self):
        with self.assertRaises(StepFailed):
            self.run_kernel_set("nobody")


if __name__ == "__main__":
    unittest.main()
