"""
scripts/extract_dataset.py — Generate the benchmark dataset from loop source files.

Reads loops/loops.inc for metadata and loops/loop_NNN.c for scalar code.
Outputs dataset/problems/{loop_NNN_slug}/problem.py + dataset/problems.json.

Also patches loop_NNN.c files with #ifdef HAVE_CANDIDATE injection points
(required by the eval harness — run with --add-candidate-blocks).

Usage:
    python scripts/extract_dataset.py
    python scripts/extract_dataset.py --add-candidate-blocks
    python scripts/extract_dataset.py --loop 001         # single loop
"""

import argparse
import json
import os
import re
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
LOOPS_INC = REPO_ROOT / "loops" / "loops.inc"
LOOPS_DIR = REPO_ROOT / "loops"
DATASET_DIR = REPO_ROOT / "dataset"
PROBLEMS_DIR = DATASET_DIR / "problems"

CANDIDATE_START = "// CANDIDATE_INJECT_START"
CANDIDATE_END = "// CANDIDATE_INJECT_END"


# ─── Parse loops.inc ─────────────────────────────────────────────────────────

def parse_loops_inc() -> list[dict]:
    """
    Parse loops/loops.inc and return a list of loop metadata dicts.

    Each entry:
        { "num": "001", "name": "FP32 inner product",
          "purpose": "Use of fp32 MLA...", "streaming": "STREAMING_COMPATIBLE" }
    """
    content = LOOPS_INC.read_text()
    # Match: LOOP(NNN, "name", "purpose") or LOOP(NNN, "name", "purpose", ATTR)
    pattern = re.compile(
        r'LOOP\(\s*(\w+)\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"(?:\s*,\s*(\w+))?\s*\)'
    )
    loops = []
    for m in pattern.finditer(content):
        loops.append({
            "num": m.group(1).zfill(3),
            "name": m.group(2).strip(),
            "purpose": m.group(3).strip(),
            "streaming": m.group(4) or "",
        })
    return loops


def isa_from_streaming(num: str, streaming: str) -> str:
    """
    Determine ISA target from loop number and streaming attribute.

    STREAMING → sme2  (SME2, requires Graviton4 c8g.large)
    Otherwise → sve2  (SVE/SVE2, works on Graviton3 c7g.large)
    """
    if streaming == "STREAMING":
        return "sme2"
    return "sve2"


def instance_from_isa(isa: str) -> str:
    return "c8g.large" if isa == "sme2" else "c7g.large"


# ─── Parse loop_NNN.c ────────────────────────────────────────────────────────

def extract_struct(source: str, num: str) -> str:
    """Extract 'struct loop_NNN_data { ... };' from source."""
    pattern = re.compile(
        rf'(struct\s+loop_{num}_data\s*\{{[^}}]*\}}\s*;)',
        re.DOTALL,
    )
    m = pattern.search(source)
    return m.group(1).strip() if m else ""


def extract_scalar_impl(source: str, num: str) -> str:
    """
    Extract the scalar implementation of inner_loop_NNN.

    Three structural patterns appear in the source tree:

    (a) Most loops have an explicit autovec block right after the CANDIDATE
        placeholder:
            #if defined(HAVE_CANDIDATE) { placeholder }
            #elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE) { scalar }
            #elif (HAVE_SVE_INTRINSICS ...)
            #elif (NEON) ...
            #else { ABORT }
            #endif
        Pre-existing dataset uses this 72/75 of the time. Note the AUTOVEC
        branch sits behind `#elif`, NOT `#if`, because it's never the first
        branch — CANDIDATE always comes first.

    (b) A few loops (loop_005/006/034) use non-SIMD-able primitives
        (strlen_opt, strcmp_opt) and don't bother defining a separate
        autovec block. Scalar is the only non-CANDIDATE implementation
        and lives directly in the #else of CANDIDATE:
            #if defined(HAVE_CANDIDATE) { placeholder }
            #else                       { scalar }
            #endif

    (c) Hypothetical: the file *starts* with HAVE_AUTOVEC as its #if (no
        CANDIDATE wrapper). Old extractor only handled this — none of the
        current 75 loops use it, but we keep the matcher for forward compat.

    Returns the function text including signature, or "" if no match.
    """
    after = None

    # (a) Common case: `#elif defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)`.
    start_marker = re.search(
        r'#elif\s+defined\(HAVE_AUTOVEC\)\s*\|\|\s*defined\(HAVE_NATIVE\)',
        source
    )
    if start_marker:
        after = source[start_marker.end():]

    # (c) Legacy form `#if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)`.
    if after is None:
        start_marker = re.search(
            r'#if\s+defined\(HAVE_AUTOVEC\)\s*\|\|\s*defined\(HAVE_NATIVE\)',
            source
        )
        if start_marker:
            after = source[start_marker.end():]

    # (b) Fallback: scalar sits in the #else of HAVE_CANDIDATE, with no
    # intervening #elif (otherwise we'd land on the trailing ABORT block).
    if after is None:
        else_marker = re.search(
            r'#if\s+defined\(HAVE_CANDIDATE\)(?:(?!\n#elif)[\s\S])*?\n(#else)',
            source
        )
        if not else_marker:
            return ""
        # Block starts right after the #else line.
        after = source[else_marker.end(1):]

    # Find the next preprocessor directive (#elif / #else / #endif).
    next_directive = re.search(r'\n#(?:elif|else|endif)', after)
    block = after[:next_directive.start()] if next_directive else after

    # Return the entire block — some loops (e.g. loop_012, loop_022) define
    # helper functions before `inner_loop_NNN` in the same AUTOVEC block, and
    # the agent needs to see those to understand the algorithm. Earlier the
    # function-pattern shortcut here silently dropped helpers; the cost of
    # always returning the block is just a few extra blank lines for simple
    # loops, which _clean_code trims.
    return _clean_code(block.strip())


def _clean_code(code: str) -> str:
    """Remove excessive blank lines and normalize indentation."""
    lines = code.splitlines()
    # Remove leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _make_slug(name: str) -> str:
    """Convert loop name to a filesystem-safe slug."""
    slug = name.lower()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    slug = slug.strip('_')
    return slug[:40]


# ─── Write problem.py ────────────────────────────────────────────────────────

PROBLEM_PY_TEMPLATE = '''\
"""
{problem_id}: {name}

Purpose: {purpose}

ISA target: {isa_upper} on {instance_desc}
"""

METADATA = {{
    "id": "{problem_id}",
    "num": "{num}",
    "name": "{name}",
    "description": "{purpose}",
    "isa_target": "{isa}",
    "instance_type": "{instance_type}",
    "dir_name": "{dir_name}",
    "tags": {tags},
}}

# Data struct definition — shown to the LLM for context
STRUCT_DEF = r"""
{struct_def}
"""

# Scalar reference implementation — the LLM's task is to optimize this
SCALAR_CODE = r"""
{scalar_code}
"""

# Prompt template (used by generate_samples.py and run_benchmark.py)
SYSTEM_PROMPT = """\
You are an expert AArch64 SIMD programmer. Your task is to write an optimized
implementation of a given loop kernel for {{isa_desc}}.
Preserve the exact function signature. The `res` checksum field must match
the scalar output. Output only the C function — no markdown, no explanation.
"""

USER_PROMPT_TEMPLATE = """\
Problem: {name}
Purpose: {purpose}
Target: {{isa_upper}} on {{isa_desc}}

Struct definition:
```c
{{struct_def}}
```

Scalar implementation to optimize:
```c
{{scalar_code}}
```

Write an optimized {{isa_upper}} implementation. Output only the C function.
"""
'''

ISA_DESC = {
    "neon": "Arm Neoverse V1 (AWS Graviton3, NEON 128-bit)",
    "sve": "Arm Neoverse V1 (AWS Graviton3, SVE 256-bit)",
    "sve2": "Arm Neoverse V1 (AWS Graviton3, SVE2 256-bit)",
    "sme2": "Arm Neoverse V2 (AWS Graviton4, SME2 128-bit)",
}

ISA_TAGS = {
    "neon": ["neon"],
    "sve": ["sve"],
    "sve2": ["sve2"],
    "sme2": ["sme2", "streaming"],
}


def _build_metadata(loop: dict) -> tuple[str, str, str, list[str], str]:
    """Compute the per-loop metadata derived purely from loops.inc + naming."""
    num = loop["num"]
    name = loop["name"]
    purpose = loop["purpose"]
    isa = isa_from_streaming(num, loop["streaming"])
    instance_type = instance_from_isa(isa)
    problem_id = f"loop_{num}"
    slug = _make_slug(name)
    dir_name = f"{problem_id}_{slug}"

    tags = list(ISA_TAGS.get(isa, []))
    for keyword, tag in [
        ("matrix", "matmul"), ("sort", "sort"), ("dot", "dot-product"),
        ("fp32", "fp32"), ("fp64", "fp64"), ("fp16", "fp16"),
        ("bf16", "bf16"), ("int8", "int8"), ("uint", "uint"),
        ("strlen", "string"), ("utf", "string"), ("histogram", "histogram"),
    ]:
        if keyword.lower() in name.lower() or keyword.lower() in purpose.lower():
            tags.append(tag)
    tags = list(dict.fromkeys(tags))

    return problem_id, dir_name, isa, tags, instance_type


def _render_problem_py(loop: dict, struct_def: str, scalar_code: str) -> tuple[str, str]:
    """Render problem.py content. Returns (content, dir_name)."""
    num = loop["num"]
    name = loop["name"]
    purpose = loop["purpose"]
    problem_id, dir_name, isa, tags, instance_type = _build_metadata(loop)

    content = PROBLEM_PY_TEMPLATE.format(
        problem_id=problem_id,
        num=num,
        name=name,
        purpose=purpose,
        isa=isa,
        isa_upper=isa.upper(),
        instance_type=instance_type,
        instance_desc=ISA_DESC.get(isa, isa),
        dir_name=dir_name,
        tags=repr(tags),
        struct_def=struct_def,
        scalar_code=scalar_code,
    )
    return content, dir_name


def write_problem(loop: dict, struct_def: str, scalar_code: str,
                  *, force: bool = False, fill_missing_scalar: bool = False) -> tuple[dict, str]:
    """Write dataset/problems/loop_NNN_slug/problem.py.

    Returns (metadata_dict, action) where action is one of:
      "written"  : file did not exist, or --force given; full re-render
      "skipped"  : file exists, safe-default skip (preserves hand-curated content)
      "filled"   : file exists with empty SCALAR_CODE; injected scalar in-place
      "skipped_has_scalar" : file exists with non-empty SCALAR_CODE; --fill-missing-scalar
                              ignored because content already present

    The metadata_dict is always built from loops.inc + naming (independent of
    the file on disk), so problems.json regenerates consistently regardless
    of whether we wrote.
    """
    content, dir_name = _render_problem_py(loop, struct_def, scalar_code)
    num = loop["num"]
    name = loop["name"]
    purpose = loop["purpose"]
    problem_id, _, isa, tags, instance_type = _build_metadata(loop)

    problem_dir = PROBLEMS_DIR / dir_name
    problem_dir.mkdir(parents=True, exist_ok=True)
    target = problem_dir / "problem.py"

    action = "written"
    if target.exists() and not force:
        existing = target.read_text()
        if fill_missing_scalar and scalar_code:
            # Inject scalar_code into existing SCALAR_CODE block only if currently empty
            sc_re = re.compile(r'(SCALAR_CODE\s*=\s*r?""")(.*?)(""")', re.DOTALL)
            m = sc_re.search(existing)
            if m and m.group(2).strip() == "":
                new_text, n = sc_re.subn(
                    rf'\g<1>\n{scalar_code}\n\g<3>', existing, count=1)
                if n == 1:
                    target.write_text(new_text)
                    action = "filled"
                else:
                    action = "skipped"
            else:
                action = "skipped_has_scalar"
        else:
            action = "skipped"
    else:
        target.write_text(content)
        action = "written"

    meta = {
        "id": problem_id,
        "num": num,
        "name": name,
        "description": purpose,
        "isa_target": isa,
        "instance_type": instance_type,
        "dir_name": dir_name,
        "tags": tags,
        "struct_def": struct_def,
        "scalar_code": scalar_code,
    }
    return meta, action


# ─── Add HAVE_CANDIDATE blocks ────────────────────────────────────────────────

CANDIDATE_PLACEHOLDER = '''\
{start}
static void inner_loop_{num}(struct loop_{num}_data *restrict data) {{
    /* CANDIDATE: the eval harness injects the LLM implementation here */
    (void)data;
}}
{end}'''


def add_candidate_block(loop_file: Path, num: str) -> bool:
    """
    Add #if defined(HAVE_CANDIDATE) block to loop_NNN.c if not already present.
    Returns True if modified, False if already patched.
    """
    source = loop_file.read_text()

    if CANDIDATE_START in source:
        return False  # already patched

    # Find the first #if defined(HAVE_AUTOVEC) line to insert before it
    marker_pattern = re.compile(
        r'(#if\s+defined\(HAVE_AUTOVEC\).*)'
    )
    m = marker_pattern.search(source)
    if not m:
        print(f"  SKIP {loop_file.name}: no HAVE_AUTOVEC marker found")
        return False

    placeholder = CANDIDATE_PLACEHOLDER.format(
        num=num, start=CANDIDATE_START, end=CANDIDATE_END
    )
    insertion = f"#if defined(HAVE_CANDIDATE)\n{placeholder}\n#elif "
    patched = source[:m.start()] + insertion + source[m.start() + len("#if "):]
    loop_file.write_text(patched)
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract arm-bench benchmark dataset from loops/loop_NNN.c into "
            "dataset/problems/. By default, existing problem.py files are NOT "
            "overwritten — many were hand-curated after extraction (e.g. helpers "
            "merged into SCALAR_CODE, comments added). Use --force to overwrite, "
            "or --fill-missing-scalar to surgically inject SCALAR_CODE only where "
            "it is currently empty."
        ),
    )
    parser.add_argument("--loop", help="Process only this loop number, e.g. 001")
    parser.add_argument("--add-candidate-blocks", action="store_true",
                        help="Also patch loop_NNN.c files with HAVE_CANDIDATE injection points")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing problem.py files. Default is to skip them "
                             "to preserve hand-curated content (helpers in SCALAR_CODE, "
                             "tuned EDGE_SIZES/PERF_SIZES, custom descriptions, etc.).")
    parser.add_argument("--fill-missing-scalar", action="store_true",
                        help="For problem.py files that already exist with an empty "
                             "SCALAR_CODE block, surgically inject the extracted scalar "
                             "without touching the rest of the file. Useful after "
                             "improving extract_scalar_impl().")
    args = parser.parse_args()

    PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)

    loops = parse_loops_inc()
    if args.loop:
        loops = [l for l in loops if l["num"] == args.loop.zfill(3)]
        if not loops:
            print(f"Loop {args.loop} not found in loops.inc")
            return

    print(f"Processing {len(loops)} loops "
          f"(mode: {'force-overwrite' if args.force else 'fill-missing-scalar' if args.fill_missing_scalar else 'skip-existing'})")

    problems = []
    counts = {"written": 0, "skipped": 0, "filled": 0, "skipped_has_scalar": 0}
    for loop in loops:
        num = loop["num"]
        loop_file = LOOPS_DIR / f"loop_{num}.c"

        if not loop_file.exists():
            print(f"  SKIP loop_{num}: file not found")
            continue

        source = loop_file.read_text()
        struct_def = extract_struct(source, num)
        scalar_code = extract_scalar_impl(source, num)

        if not scalar_code:
            print(f"  WARN loop_{num}: no scalar implementation found in loops/loop_{num}.c")

        if args.dry_run:
            print(f"  [dry] loop_{num}: {loop['name']} → {isa_from_streaming(num, loop['streaming'])}")
            continue

        meta, action = write_problem(
            loop, struct_def, scalar_code,
            force=args.force, fill_missing_scalar=args.fill_missing_scalar,
        )
        problems.append(meta)
        counts[action] = counts.get(action, 0) + 1
        tag = {
            "written": "WROTE",
            "filled":  "FILLED SCALAR",
            "skipped": "skip (exists)",
            "skipped_has_scalar": "skip (already has scalar)",
        }[action]
        print(f"  loop_{num}: {tag}: {loop['name']} ({meta['isa_target']})")

        if args.add_candidate_blocks:
            modified = add_candidate_block(loop_file, num)
            if modified:
                print(f"    ✓ Added HAVE_CANDIDATE block to loop_{num}.c")

    summary_bits = [f"{v} {k}" for k, v in counts.items() if v]
    print(f"\nActions: {', '.join(summary_bits) if summary_bits else 'nothing written'}")

    if args.dry_run:
        return

    # Write problems.json (without scalar_code/struct_def to keep it compact).
    # The index is regenerated only when --force is used, for two reasons:
    #
    #   1. Targeted operations (--loop, --fill-missing-scalar) shouldn't
    #      churn the global index. --loop would shrink it; --fill-missing
    #      shouldn't touch metadata at all.
    #
    #   2. The existing problems.json may carry hand-curated fields that
    #      are richer than what loops.inc provides. For example:
    #        loops.inc:   "Use of fp32 MLA instruction"
    #        problems.json: "Compute the FP32 dot product of two float arrays"
    #      Default regeneration would clobber the longer description.
    #
    # `--force` is the explicit signal that the caller knows they are
    # blowing away any hand-curated content and wants a full re-render.
    out = DATASET_DIR / "problems.json"
    if not args.force:
        print(f"\nSkipped {out} regeneration (use --force to overwrite; "
              f"existing index may contain hand-curated descriptions).")
    elif args.loop:
        print(f"\nSkipped {out} regeneration (--loop is targeted; the full "
              f"index would shrink to one entry).")
    else:
        index = []
        for p in problems:
            entry = {k: v for k, v in p.items() if k not in ("scalar_code", "struct_def")}
            index.append(entry)
        out.write_text(json.dumps(index, indent=2))
        print(f"\nWrote {len(problems)} problems to {out}")
    print(f"Problem files: {PROBLEMS_DIR}")

    # Print ISA breakdown
    by_isa = {}
    for p in problems:
        by_isa[p["isa_target"]] = by_isa.get(p["isa_target"], 0) + 1
    print("\nISA breakdown:")
    for isa, count in sorted(by_isa.items()):
        print(f"  {isa}: {count} problems")


if __name__ == "__main__":
    main()
