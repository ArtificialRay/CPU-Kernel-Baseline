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
    Extract the scalar (HAVE_AUTOVEC || HAVE_NATIVE) implementation of inner_loop_NNN.

    Returns the full function text including signature.
    """
    # Find the #if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE) block
    # Strategy: find the function body between this #if and the next #elif / #else / #endif
    pattern = re.compile(
        r'#if\s+defined\(HAVE_AUTOVEC\)[^#]*?#elif|'
        r'#if\s+defined\(HAVE_AUTOVEC\)[^#]*?#else|'
        r'#if\s+defined\(HAVE_AUTOVEC\)[^#]*?#endif',
        re.DOTALL
    )
    # Simpler approach: extract the function that appears right after
    # #if defined(HAVE_AUTOVEC) || defined(HAVE_NATIVE)
    start_marker = re.search(
        r'#if\s+defined\(HAVE_AUTOVEC\)\s*\|\|\s*defined\(HAVE_NATIVE\)',
        source
    )
    if not start_marker:
        return ""

    after = source[start_marker.end():]
    # Find the next preprocessor directive (#elif, #else, #endif)
    next_directive = re.search(r'\n#(?:elif|else|endif)', after)
    if next_directive:
        block = after[:next_directive.start()]
    else:
        block = after

    # Extract the function definition
    fn_pattern = re.compile(
        rf'(static\s+void\s+inner_loop_{num}\s*\([^)]*\)[^{{]*\{{.*?\}})',
        re.DOTALL
    )
    m = fn_pattern.search(block)
    if m:
        return _clean_code(m.group(1))

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


def write_problem(loop: dict, struct_def: str, scalar_code: str) -> dict:
    """Write dataset/problems/loop_NNN_slug/problem.py and return metadata."""
    num = loop["num"]
    name = loop["name"]
    purpose = loop["purpose"]
    isa = isa_from_streaming(num, loop["streaming"])
    instance_type = instance_from_isa(isa)
    problem_id = f"loop_{num}"
    slug = _make_slug(name)
    dir_name = f"{problem_id}_{slug}"

    # Infer tags from ISA + name
    tags = list(ISA_TAGS.get(isa, []))
    for keyword, tag in [
        ("matrix", "matmul"), ("sort", "sort"), ("dot", "dot-product"),
        ("fp32", "fp32"), ("fp64", "fp64"), ("fp16", "fp16"),
        ("bf16", "bf16"), ("int8", "int8"), ("uint", "uint"),
        ("strlen", "string"), ("utf", "string"), ("histogram", "histogram"),
    ]:
        if keyword.lower() in name.lower() or keyword.lower() in purpose.lower():
            tags.append(tag)
    tags = list(dict.fromkeys(tags))  # deduplicate

    problem_dir = PROBLEMS_DIR / dir_name
    problem_dir.mkdir(parents=True, exist_ok=True)

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
    (problem_dir / "problem.py").write_text(content)

    return {
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
    parser = argparse.ArgumentParser(description="Extract simd-loops benchmark dataset")
    parser.add_argument("--loop", help="Process only this loop number, e.g. 001")
    parser.add_argument("--add-candidate-blocks", action="store_true",
                        help="Also patch loop_NNN.c files with HAVE_CANDIDATE injection points")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files")
    args = parser.parse_args()

    PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)

    loops = parse_loops_inc()
    if args.loop:
        loops = [l for l in loops if l["num"] == args.loop.zfill(3)]
        if not loops:
            print(f"Loop {args.loop} not found in loops.inc")
            return

    print(f"Processing {len(loops)} loops...")

    problems = []
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
            print(f"  WARN loop_{num}: no scalar implementation found")

        if args.dry_run:
            print(f"  [dry] loop_{num}: {loop['name']} → {isa_from_streaming(num, loop['streaming'])}")
            continue

        meta = write_problem(loop, struct_def, scalar_code)
        problems.append(meta)
        print(f"  loop_{num}: {loop['name']} ({meta['isa_target']})")

        if args.add_candidate_blocks:
            modified = add_candidate_block(loop_file, num)
            if modified:
                print(f"    ✓ Added HAVE_CANDIDATE block to loop_{num}.c")

    if args.dry_run:
        return

    # Write problems.json (without scalar_code/struct_def to keep it compact)
    index = []
    for p in problems:
        entry = {k: v for k, v in p.items() if k not in ("scalar_code", "struct_def")}
        index.append(entry)

    out = DATASET_DIR / "problems.json"
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
