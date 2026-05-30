"""
eval/config.py — Configuration loader for the arm-bench benchmark.

Reads eval_config.json (or eval_config.json.example) and exposes helpers
for loading problem metadata and baseline timings.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
EVAL_CONFIG_PATH = REPO_ROOT / "eval" / "eval_config.json"
DATASET_PATH = REPO_ROOT / "dataset"
PROBLEMS_JSON = DATASET_PATH / "problems.json"
BASELINES_DIR = REPO_ROOT / "baselines"
STARTER_PATH = REPO_ROOT / "starter"
NCNN_PROBLEMS_JSON = STARTER_PATH / "ncnn" / "problems.json"
NCNN_BASELINES_JSON = BASELINES_DIR / "ncnn.json"

# ISA → instance tier mapping
ISA_TIER = {
    "neon": "c7g",
    "sve": "c7g",
    "sve2": "c8g",
    "sme2": "c8g",
}

# ISA → make target name
ISA_MAKE_TARGET = {
    "neon": "neon",
    "sve": "sve",
    "sve2": "sve2",
    "sme2": "sme2",
}

# ISA → human-readable instance description for prompts
ISA_INSTANCE_DESC = {
    "neon": "Arm Neoverse V1 (AWS Graviton3, NEON 128-bit)",
    "sve": "Arm Neoverse V1 (AWS Graviton3, SVE 256-bit)",
    "sve2": "Arm Neoverse V2 (AWS Graviton4, SVE2 128-bit)",
    "sme2": "Arm Neoverse V2 (AWS Graviton4, SVE2 128-bit)",
}


def load_config() -> dict:
    """Load eval_config.json. Raises if not found."""
    if not EVAL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"eval_config.json not found at {EVAL_CONFIG_PATH}\n"
            f"Copy eval_config.json.example to eval_config.json and fill in your instance IPs,\n"
            f"or run: python eval/provision.py --instance c7g.large"
        )
    return json.loads(EVAL_CONFIG_PATH.read_text())


def load_problems(with_code: bool = True) -> dict:
    """
    Load dataset/problems.json. Returns dict keyed by problem ID.

    Args:
        with_code: If True, also load struct_def and scalar_code from
                   each problem's problem.py (needed by evaluator/generate_samples).
    """
    if not PROBLEMS_JSON.exists():
        raise FileNotFoundError(
            f"problems.json not found. Run: python scripts/extract_dataset.py"
        )
    problems = json.loads(PROBLEMS_JSON.read_text())
    result = {p["id"]: p for p in problems}

    if with_code:
        for pid, meta in result.items():
            dir_name = meta.get("dir_name", "")
            problem_py = DATASET_PATH / "problems" / dir_name / "problem.py"
            if not problem_py.exists():
                continue
            # Parse SCALAR_CODE and STRUCT_DEF from problem.py as raw text
            text = problem_py.read_text()

            struct_m = _extract_triple_quoted(text, "STRUCT_DEF")
            scalar_m = _extract_triple_quoted(text, "SCALAR_CODE")
            if struct_m:
                meta["struct_def"] = struct_m.strip()
            if scalar_m:
                meta["scalar_code"] = scalar_m.strip()

            # Load per-problem test size lists
            edge = _extract_int_list(text, "EDGE_SIZES")
            perf = _extract_int_list(text, "PERF_SIZES")
            meta["edge_sizes"] = edge if edge is not None else []
            meta["perf_sizes"] = perf if perf is not None else []

            # Extract NEON implementation from the loop source file (if present)
            loop_num = pid.split("_")[1]
            loop_src = REPO_ROOT / "loops" / f"loop_{loop_num}.c"
            neon = _extract_neon_code(loop_src)
            if neon:
                meta["neon_code"] = neon.strip()

    return result


def _extract_int_list(text: str, var_name: str) -> list[int] | None:
    """Extract an integer list literal (e.g. EDGE_SIZES = [0, 1, 7]) from Python source."""
    import ast, re
    pattern = re.compile(rf"^{var_name}\s*=\s*(\[.*?\])", re.MULTILINE | re.DOTALL)
    m = pattern.search(text)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None


def _extract_triple_quoted(text: str, var_name: str) -> str | None:
    """Extract the content of a triple-quoted r-string variable from Python source."""
    import re
    # Match: VAR_NAME = r"""\n...\n"""
    pattern = re.compile(
        rf'{var_name}\s*=\s*r"""(.*?)"""',
        re.DOTALL,
    )
    m = pattern.search(text)
    return m.group(1) if m else None


def _extract_neon_code(loop_src: Path) -> str | None:
    """
    Extract the `#elif defined(__ARM_NEON)` implementation from a loop source
    file. Returns the full function text (from `static void ...` to closing `}`),
    or None if no NEON branch exists.
    """
    if not loop_src.exists():
        return None
    lines = loop_src.read_text().splitlines(keepends=True)

    # Find the start of the NEON branch
    neon_start = next(
        (i for i, l in enumerate(lines) if "__ARM_NEON" in l and l.strip().startswith("#elif")),
        None,
    )
    if neon_start is None:
        return None

    # Collect lines from neon_start+1 until the next top-level #elif/#else/#endif
    depth = 0
    func_lines = []
    in_func = False
    for i in range(neon_start + 1, len(lines)):
        s = lines[i].strip()
        # Stop at the next preprocessor branch at depth 0
        if depth == 0 and s.startswith(("#elif", "#else", "#endif")):
            break
        if s.startswith("#if"):
            depth += 1
        elif s.startswith("#endif"):
            depth -= 1
        func_lines.append(lines[i])

    return "".join(func_lines) if func_lines else None


def load_problem_sizes(problem_id: str, isa: str = "") -> tuple[list[int], list[int]]:
    """
    Return (edge_sizes, perf_sizes) for a single problem without loading the
    entire problems index. Reads only that problem's problem.py.

    If isa is "sve2" (c8g tier) and the problem defines PERF_SIZES_C8G, that
    list is used instead of PERF_SIZES so scoring is DRAM-bound on Graviton4's
    larger 64MB L3 cache.
    """
    if not PROBLEMS_JSON.exists():
        return [], []
    problems_raw = json.loads(PROBLEMS_JSON.read_text())
    meta = next((p for p in problems_raw if p["id"] == problem_id), None)
    if meta is None:
        return [], []
    problem_py = DATASET_PATH / "problems" / meta.get("dir_name", "") / "problem.py"
    if not problem_py.exists():
        return [], []
    text = problem_py.read_text()
    edge = _extract_int_list(text, "EDGE_SIZES") or []

    # Use c8g-specific sizes when available and targeting c8g
    perf = None
    if isa in ("sve2", "sme2"):
        perf = _extract_int_list(text, "PERF_SIZES_C8G")
    if perf is None:
        perf = _extract_int_list(text, "PERF_SIZES") or []
    return edge, perf


def load_baselines(tier: str) -> dict:
    """
    Load baselines/{tier}.json. Returns dict:
      { "loop_001": { "scalar_ms": 156.3, "autovec_ms": 42.1 }, ... }
    """
    path = BASELINES_DIR / f"{tier}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_ncnn_problems() -> dict:
    """
    Load starter/ncnn/problems.json and augment each entry with:
      - scalar_code: current candidates_src/ncnn/*.cpp (what the agent optimizes)
      - struct_def:  starter/ncnn/candidate/*.h       (class + kernel decl)

    Paths in problems.json are expected to be relative to the arm-bench repo root.

    Returns dict keyed by problem ID.
    """
    if not NCNN_PROBLEMS_JSON.exists():
        raise FileNotFoundError(
            f"starter/ncnn/problems.json not found at {NCNN_PROBLEMS_JSON}. "
            f"Make sure arm-bench/starter/ncnn/ is populated."
        )
    problems_raw = json.loads(NCNN_PROBLEMS_JSON.read_text())
    result = {}

    for meta in problems_raw:
        pid = meta["id"]
        entry = dict(meta)

        candidate_rel = meta.get("candidate_source", "")
        candidate_path = REPO_ROOT / candidate_rel
        if candidate_path.exists():
            entry["scalar_code"] = candidate_path.read_text()
        else:
            entry["scalar_code"] = f"// Source not found locally at {candidate_rel}"

        header_rel = meta.get("starter_header", "")
        header_path = REPO_ROOT / header_rel
        if header_path.exists():
            entry["struct_def"] = header_path.read_text()
        else:
            entry["struct_def"] = f"// Header not found locally at {header_rel}"

        result[pid] = entry

    return result


def load_ncnn_baselines() -> dict:
    """
    Load baselines/ncnn.json. Returns dict:
      { "conv2d": { "scalar_ms": 123.4, "autovec_ms": 45.6, "ref_ms": null }, ... }
    """
    if not NCNN_BASELINES_JSON.exists():
        return {}
    return json.loads(NCNN_BASELINES_JSON.read_text())


def problem_path(problem_id: str) -> Path:
    """Return the path to a problem directory, e.g. dataset/problems/loop_001_saxpy/"""
    problems = load_problems()
    if problem_id not in problems:
        raise KeyError(f"Problem {problem_id!r} not found in problems.json")
    meta = problems[problem_id]
    return DATASET_PATH / "problems" / meta["dir_name"]


def loop_source_path(problem_id: str) -> Path:
    """Return the path to the original loop_NNN.c source file."""
    loop_num = problem_id.split("_")[1]  # "loop_001" → "001"
    return REPO_ROOT / "loops" / f"loop_{loop_num}.c"
