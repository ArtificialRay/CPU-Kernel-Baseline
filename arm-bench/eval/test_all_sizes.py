"""
eval/test_all_sizes.py — Cross-kernel integration test for variable-size testing.

Phase 1 (fast): verify EDGE_SIZES / PERF_SIZES are loaded for all 43 problems
               that have size constraints.

Phase 2 (slow): for a diverse sample covering different data types and loop
               archetypes, inject the scalar reference as the candidate, run
               _check_edge_sizes() and _collect_perf_sizes(), and assert the
               scalar always passes its own edge checks and collects timing.

Usage:
    python -m eval.test_all_sizes [--isa sve] [--phase1-only]
"""

import argparse
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Sample problems covering different archetypes
# ---------------------------------------------------------------------------
# Each entry: (problem_id, description)
# Using scalar_code from problem.py where available; falling back to a direct
# extract from the loop .c file for the 3 problems that lack it.
SAMPLE = [
    # fp32 / uint32 standard array
    ("loop_001", "fp32 inner product"),
    ("loop_002", "uint32 inner product"),
    # fp64 standard array
    ("loop_003", "fp64 inner product"),
    ("loop_008", "precise fp64 add reduction"),
    # uint8 / uint16
    ("loop_022", "tcp checksum (uint16)"),
    ("loop_024", "sum of abs diffs (uint8)"),
    # multiple arrays with different strides
    ("loop_012", "particle motion (6×fp64)"),
    ("loop_037", "fp32 complex vector product"),
    # string / byte buffers
    ("loop_005", "strlen short strings"),
    ("loop_034", "short string compares"),
    # cascade sum (power-of-2 sizes)
    ("loop_105", "cascade summation (pow2)"),
    # scatter/gather
    ("loop_106", "sheep and goats (SAG)"),
    # sorting (O(n²) — small sizes only)
    ("loop_120", "insertion sort"),
    ("loop_124", "radix sort"),
    # fixed-dimension (should gracefully skip all size tests)
    ("loop_025", "small matmul (fixed dims, skip)"),
    ("loop_104", "byte histogram (fixed, skip)"),
]


def _extract_branch_content(src: str, branch_start: int) -> str | None:
    """
    Given the character offset where a #elif/#else branch starts (right after
    the directive line), extract all C source up to the next top-level
    #elif / #else / #endif (at depth 0).  Returns the trimmed content.
    """
    lines = src[branch_start:].splitlines(keepends=True)
    depth = 0
    collected = []
    for line in lines:
        s = line.strip()
        if depth == 0 and s.startswith(("#elif", "#else", "#endif")):
            break
        if s.startswith("#if"):
            depth += 1
        elif s.startswith("#endif"):
            depth -= 1
        collected.append(line)
    return "".join(collected).strip() or None


def extract_scalar_from_source(problem_id: str) -> str | None:
    """
    Extract a self-contained scalar implementation from the loop .c file.

    Strategy:
    1. Extract the HAVE_AUTOVEC/HAVE_NATIVE branch content (includes helpers).
    2. If that content doesn't define inner_loop_NNN itself, also extract the
       #if !defined(HAVE_CANDIDATE) block (which has the real inner_loop_NNN
       calling those helpers) and append it.
    3. Fall back to the #else branch after HAVE_CANDIDATE for loops with only
       a single scalar implementation.
    """
    num = problem_id.split("_")[1]
    src_path = REPO_ROOT / "loops" / f"loop_{num}.c"
    if not src_path.exists():
        return None
    src = src_path.read_text()

    fn_name = f"inner_loop_{num}"

    # 1. Find the HAVE_AUTOVEC || HAVE_NATIVE branch
    branch_m = re.search(
        r"#elif\s+defined\(HAVE_AUTOVEC\)\s+\|\|\s+defined\(HAVE_NATIVE\)"
        r"|#elif\s+\(defined\(HAVE_AUTOVEC\)",
        src,
    )
    if branch_m:
        # Skip continuation lines of the #elif directive (lines ending with \)
        pos = branch_m.end()
        while True:
            nl = src.find("\n", pos)
            if nl == -1:
                break
            line = src[pos:nl]
            if line.rstrip().endswith("\\"):
                pos = nl + 1  # skip this continuation line
            else:
                pos = nl + 1
                break
        if branch_m:
            autovec_content = _extract_branch_content(src, pos) or ""
            if fn_name in autovec_content:
                return autovec_content  # complete: has the inner_loop function

            # HAVE_AUTOVEC only has helpers; inner_loop is in #if !HAVE_CANDIDATE.
            # Extract the nested #if HAVE_AUTOVEC branch WITHIN !HAVE_CANDIDATE
            # (avoids picking up SVE/NEON asm variants with nested preprocessor).
            not_cand_m = re.search(
                r"^#if\s+!defined\(HAVE_CANDIDATE\)", src, re.MULTILINE
            )
            if not_cand_m:
                nl_nc = src.find("\n", not_cand_m.end())
                after_not_cand = src[not_cand_m.end():]
                # Try to find a nested #if HAVE_AUTOVEC within !HAVE_CANDIDATE
                nested_m = re.search(
                    r"^#if\s+defined\(HAVE_AUTOVEC\)\s+\|\|\s+defined\(HAVE_NATIVE\)",
                    after_not_cand,
                    re.MULTILINE,
                )
                if nested_m:
                    nl2 = after_not_cand.find("\n", nested_m.end())
                    if nl2 != -1:
                        scalar_part = _extract_branch_content(after_not_cand, nl2 + 1) or ""
                        # Also grab inner_loop_NNN which may follow the #endif
                        combined_search = after_not_cand[nested_m.start():]
                        inner_m = re.search(
                            rf"(static void {fn_name}.*?^}})",
                            combined_search,
                            re.DOTALL | re.MULTILINE,
                        )
                        inner_part = inner_m.group(0) if inner_m else ""
                        parts = [autovec_content, scalar_part, inner_part]
                        combined = "\n\n".join(p for p in parts if p).strip()
                        if combined:
                            return combined
                else:
                    # No nested #if HAVE_AUTOVEC: !HAVE_CANDIDATE block is simple,
                    # extract its entire content (typically just inner_loop_NNN).
                    if nl_nc != -1:
                        nc_content = _extract_branch_content(src, nl_nc + 1) or ""
                        parts = [autovec_content, nc_content]
                        combined = "\n\n".join(p for p in parts if p).strip()
                        if combined:
                            return combined
            if autovec_content:
                return autovec_content

    # 2. Fall back to the #else branch after HAVE_CANDIDATE (no HAVE_AUTOVEC)
    else_m = re.search(r"^#else\b", src, re.MULTILINE)
    if else_m:
        nl = src.find("\n", else_m.end())
        if nl != -1:
            content = _extract_branch_content(src, nl + 1)
            if content:
                return content

    return None


def get_scalar_candidate(problem_id: str, problems: dict) -> str | None:
    """Return the best available scalar implementation for this problem."""
    sc = problems.get(problem_id, {}).get("scalar_code", "").strip()

    # Many SCALAR_CODE entries in problem.py are incomplete fragments (missing
    # closing brace, data->res assignment, etc.).  Always prefer extracting the
    # complete function from the loop .c source file when available.
    src = extract_scalar_from_source(problem_id)
    if src:
        return src

    # Fall back to problem.py SCALAR_CODE only if source extraction failed,
    # and only when it looks like a complete function (not a fragment).
    if sc and not sc.startswith("#") and not sc.startswith("//Do not"):
        return sc

    return None


# ---------------------------------------------------------------------------
# Phase 1: metadata check
# ---------------------------------------------------------------------------

def phase1_metadata(problems: dict) -> bool:
    print("\n── Phase 1: metadata check (all problems) ──────────────────────")
    all_ok = True
    problems_with_sizes = {
        pid: m for pid, m in problems.items()
        if m.get("edge_sizes") or m.get("perf_sizes")
    }
    print(f"  Problems with EDGE_SIZES or PERF_SIZES: {len(problems_with_sizes)}")

    for pid, m in sorted(problems_with_sizes.items()):
        edge = m.get("edge_sizes", [])
        perf = m.get("perf_sizes", [])
        # Sanity: sizes are lists of ints, no duplicates within each list
        ok = (
            isinstance(edge, list)
            and isinstance(perf, list)
            and all(isinstance(s, int) and s >= 0 for s in edge)
            and all(isinstance(s, int) and s > 0 for s in perf)
            and len(edge) == len(set(edge))
            and len(perf) == len(set(perf))
        )
        if not ok:
            print(f"  [FAIL] {pid}: bad sizes  edge={edge} perf={perf}")
            all_ok = False

    if all_ok:
        print(f"  [PASS] all {len(problems_with_sizes)} problems have valid size lists")

    # Show the ones that are intentionally empty (fixed-dim loops)
    empty = [pid for pid, m in problems.items()
             if not m.get("edge_sizes") and not m.get("perf_sizes")]
    print(f"  Intentionally empty (fixed dims / no SIZE): {len(empty)} problems")
    return all_ok


# ---------------------------------------------------------------------------
# Phase 2: compile + edge + perf per sample problem
# ---------------------------------------------------------------------------

PASS_S = "\033[32mPASS\033[0m"
FAIL_S = "\033[31mFAIL\033[0m"
SKIP_S = "\033[33mSKIP\033[0m"


def fmt(ok: bool | None, label: str, detail: str = "") -> str:
    marker = PASS_S if ok else (SKIP_S if ok is None else FAIL_S)
    return f"  [{marker}] {label}" + (f"  ({detail})" if detail else "")


def phase2_sample(handle, isa: str, problems: dict) -> bool:
    from eval.tools import SIMDTools

    print("\n── Phase 2: compile + edge + perf check (sample problems) ──────")

    # Sync local loop .c files to remote to ensure no stale candidates from
    # previous runs interfere with builds (compile() builds all loops together).
    print("  Syncing loop source files to remote...")
    handle.rsync_to(str(REPO_ROOT / "loops"), "~/simd-loops/loops")
    print("  Sync done.\n")

    all_ok = True

    for problem_id, description in SAMPLE:
        if problem_id not in problems:
            print(f"\n  [{SKIP_S}] {problem_id} — not in problems index")
            continue

        meta = problems[problem_id]
        edge_sizes = meta.get("edge_sizes", [])
        perf_sizes = meta.get("perf_sizes", [])

        print(f"\n  {'─'*56}")
        print(f"  {problem_id}  {description}")
        print(f"    edge_sizes={edge_sizes}")
        print(f"    perf_sizes={perf_sizes}")

        # Problems with no sizes — verify they're gracefully skipped
        if not edge_sizes and not perf_sizes:
            tools = SIMDTools(handle=handle, problem_id=problem_id, isa=isa)
            candidate = get_scalar_candidate(problem_id, problems)
            if candidate:
                cr = tools.compile(candidate)
                if cr.success:
                    ef = tools._check_edge_sizes()
                    pf = tools._collect_perf_sizes()
                    print(fmt(ef is None, "edge check skipped gracefully",
                              f"returned {ef!r}"))
                    print(fmt(pf == {}, "perf check skipped gracefully",
                              f"returned {pf!r}"))
                else:
                    print(fmt(None, "compile failed — skipping", cr.errors[:60]))
            else:
                print(fmt(None, "no scalar candidate available — skipping"))
            continue

        # Problems with sizes — inject scalar, run edge+perf checks
        candidate = get_scalar_candidate(problem_id, problems)
        if not candidate:
            print(fmt(None, "no scalar candidate available — skipping"))
            continue

        tools = SIMDTools(handle=handle, problem_id=problem_id, isa=isa)

        # Compile
        t0 = time.time()
        cr = tools.compile(candidate)
        compile_ms = (time.time() - t0) * 1000
        print(fmt(cr.success, f"compile  ({compile_ms:.0f} ms)",
                  cr.errors[:80] if not cr.success else ""))
        if not cr.success:
            all_ok = False
            continue

        # Default-size correctness (n=1 iterations)
        rr = tools.run(n=1)
        print(fmt(rr.correct, f"run (default size)",
                  rr.output[:60] if not rr.correct else f"runtime_ms={rr.runtime_ms}"))
        if not rr.correct:
            all_ok = False
            # Don't skip — edge checks might still be informative

        # Edge sizes
        if edge_sizes:
            t0 = time.time()
            ef = tools._check_edge_sizes()
            edge_ms = (time.time() - t0) * 1000
            ok = ef is None
            print(fmt(ok, f"edge check  ({edge_ms:.0f} ms, {len(edge_sizes)} sizes)",
                      ef or "all passed"))
            if not ok:
                all_ok = False

        # Perf sizes — collect timing for the first PERF_SIZE only to save time
        if perf_sizes:
            first_size = perf_sizes[0]
            t0 = time.time()
            # Temporarily override to just test the first perf size
            tools._perf_sizes = [first_size]
            pf = tools._collect_perf_sizes()
            perf_ms = (time.time() - t0) * 1000
            ms_per_iter = pf.get(first_size)
            ok = ms_per_iter is not None
            print(fmt(ok,
                      f"perf_by_size[{first_size}]  ({perf_ms:.0f} ms)",
                      f"{ms_per_iter} ms/iter" if ok else "None (failed)"))
            if not ok:
                all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isa", default="sve", choices=["neon", "sve", "sve2"])
    parser.add_argument("--phase1-only", action="store_true",
                        help="Only run the fast metadata check")
    args = parser.parse_args()

    from eval.provision import get_or_provision
    from eval.config import load_problems

    problems = load_problems(with_code=True)

    p1_ok = phase1_metadata(problems)

    if args.phase1_only:
        sys.exit(0 if p1_ok else 1)

    print(f"\n  Provisioning instance for isa={args.isa}...")
    handle = get_or_provision(args.isa)
    print(f"  Instance: {handle.host}")

    p2_ok = phase2_sample(handle, args.isa, problems)

    overall = p1_ok and p2_ok
    print(f"\n{'='*60}")
    print(f"  Overall: " + ("\033[32mPASS\033[0m" if overall else "\033[31mFAIL\033[0m"))
    print(f"{'='*60}\n")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
