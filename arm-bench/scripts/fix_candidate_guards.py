"""
scripts/fix_candidate_guards.py

Fixes the structural bug in loop_NNN.c files where an unconditional
`inner_loop_NNN` definition (and any preceding helper functions) sits
after the #endif that closes the HAVE_CANDIDATE block. This causes a
redefinition error when HAVE_CANDIDATE is defined globally.

Fix: wrap everything between that #endif and the LOOP_DECL line in
     #if !defined(HAVE_CANDIDATE) / #endif

Usage:
    python scripts/fix_candidate_guards.py [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path

LOOPS_DIR = Path(__file__).parent.parent / "loops"


def find_candidate_endif(lines: list[str], inject_end_idx: int) -> int | None:
    """
    Starting just after CANDIDATE_INJECT_END, find the #endif that closes
    the top-level HAVE_CANDIDATE #if block (ignoring nested #if/#endif pairs).
    """
    depth = 0
    for i in range(inject_end_idx + 1, len(lines)):
        s = lines[i].strip()
        if s.startswith("#if"):
            depth += 1
        elif s.startswith("#endif"):
            if depth == 0:
                return i
            depth -= 1
    return None


def find_inner_loop_close(lines: list[str], start_idx: int) -> int | None:
    """
    Starting at the line containing `static void inner_loop_NNN`, track brace
    depth and return the index of the line containing the closing `}`.
    """
    depth = 0
    found_open = False
    for i in range(start_idx, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if "{" in lines[i]:
            found_open = True
        if found_open and depth == 0:
            return i
    return None


def find_candidate_if(lines: list[str]) -> int | None:
    """Return the line index of #if defined(HAVE_CANDIDATE)."""
    return next(
        (i for i, l in enumerate(lines)
         if l.strip().startswith("#if") and "HAVE_CANDIDATE" in l),
        None,
    )


def find_preceding_statics(lines: list[str], candidate_if_idx: int) -> int | None:
    """
    Look back from #if defined(HAVE_CANDIDATE) for any static variable/array
    declarations (not function defs) that sit between the struct/define block
    and the candidate guard. Return the index of the first such line, or None.
    These are file-scope data used only by the unconditional inner_loop wrapper
    and will cause -Wunused-variable when HAVE_CANDIDATE is set.
    """
    # Scan up to 20 lines back, stop at struct/define/comment blocks
    start = None
    for i in range(candidate_if_idx - 1, max(candidate_if_idx - 20, -1), -1):
        s = lines[i].strip()
        if not s or s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            continue
        # A static variable/array declaration (not a function: no `(`)
        if s.startswith("static") and "(" not in s:
            start = i
        elif s.startswith("static") and "(" in s:
            break  # it's a static function, stop
        else:
            break
    return start


def fix_file(path: Path, dry_run: bool) -> bool:
    """Return True if a fix was applied."""
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    num = re.search(r"loop_(\d+)", path.name).group(1)

    # Already fixed?
    if "#if !defined(HAVE_CANDIDATE)" in text:
        return False

    # Find CANDIDATE_INJECT_END
    inject_end = next(
        (i for i, l in enumerate(lines) if "CANDIDATE_INJECT_END" in l), None
    )
    if inject_end is None:
        return False

    # Find the #endif closing the HAVE_CANDIDATE block
    endif_idx = find_candidate_endif(lines, inject_end)
    if endif_idx is None:
        print(f"  WARNING: no closing #endif found in {path.name}", file=sys.stderr)
        return False

    # Find the unconditional inner_loop_NNN definition after #endif.
    # Skip files where the first inner_loop after #endif is already inside
    # another nested #if block (those need a different, whole-chain wrapping).
    inner_loop_start = None
    depth_after_endif = 0
    for i in range(endif_idx + 1, len(lines)):
        s = lines[i].strip()
        if s.startswith("#if"):
            depth_after_endif += 1
        elif s.startswith("#endif"):
            depth_after_endif -= 1
        if depth_after_endif == 0 and re.match(
            rf"\s*static\s+\S+\s+inner_loop_{num}\s*\(", lines[i]
        ):
            inner_loop_start = i
            break
        # Stop search at LOOP_DECL
        if "LOOP_DECL" in lines[i]:
            break

    if inner_loop_start is None:
        return False  # No unconditional definition — already fine or needs manual fix

    # Find the closing brace of inner_loop_NNN
    inner_loop_end = find_inner_loop_close(lines, inner_loop_start)
    if inner_loop_end is None:
        print(f"  WARNING: could not find closing brace in {path.name}", file=sys.stderr)
        return False

    # Also check for static data declarations just before #if HAVE_CANDIDATE
    # that are only used in the wrapped block (e.g. static size_t count[] in loop_031)
    candidate_if_idx = find_candidate_if(lines)
    preceding_static = (
        find_preceding_statics(lines, candidate_if_idx)
        if candidate_if_idx is not None else None
    )

    if preceding_static is not None:
        # Move the static declaration into the !HAVE_CANDIDATE guard by removing
        # it from its current location and prepending it inside the guard.
        static_lines = lines[preceding_static:candidate_if_idx]
        # Remove from original position
        lines = lines[:preceding_static] + lines[candidate_if_idx:]
        # Recalculate indices after removal
        shift = candidate_if_idx - preceding_static
        inject_end -= shift
        endif_idx -= shift
        inner_loop_start -= shift
        inner_loop_end -= shift

        guard_start = endif_idx + 1
        guard_end = inner_loop_end
        new_lines = (
            lines[:guard_start]
            + ["\n#if !defined(HAVE_CANDIDATE)\n"]
            + static_lines
            + lines[guard_start : guard_end + 1]
            + ["#endif /* !HAVE_CANDIDATE */\n"]
            + lines[guard_end + 1 :]
        )
    else:
        # Wrap from (endif_idx+1) through inner_loop_end in the guard
        guard_start = endif_idx + 1
        guard_end = inner_loop_end
        new_lines = (
            lines[:guard_start]
            + ["\n#if !defined(HAVE_CANDIDATE)\n"]
            + lines[guard_start : guard_end + 1]
            + ["#endif /* !HAVE_CANDIDATE */\n"]
            + lines[guard_end + 1 :]
        )

    if dry_run:
        extra = f" (+ static data from line {preceding_static+1})" if preceding_static else ""
        print(f"  [dry-run] {path.name}: would wrap lines {guard_start+1}–{guard_end+1}{extra}")
    else:
        path.write_text("".join(new_lines))
        extra = f" (+ static data from line {preceding_static+1})" if preceding_static else ""
        print(f"  Fixed {path.name}: wrapped lines {guard_start+1}–{guard_end+1}{extra}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fixed = 0
    skipped = 0
    for path in sorted(LOOPS_DIR.glob("loop_*.c")):
        if fix_file(path, args.dry_run):
            fixed += 1
        else:
            skipped += 1

    print(f"\n{'[dry-run] Would fix' if args.dry_run else 'Fixed'} {fixed} files, {skipped} already OK or skipped.")


if __name__ == "__main__":
    main()
