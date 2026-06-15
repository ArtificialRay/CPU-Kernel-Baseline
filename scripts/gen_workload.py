#!/usr/bin/env python3
"""Generate and append new workloads for an existing Definition.

Reads the definition JSON to determine which inputs are tensors (shape != null)
vs scalars (shape == null). Inherits scalar values from the first existing workload.
New workloads use the `inputs` dict format; existing workloads in the old
`scalar_inputs` format are migrated in-place when --migrate is passed.

Usage examples:

    # Add two workloads to a conv2d definition
    python scripts/gen_workload.py conv2d_kh1_kw1_sh1_sw1_dh1_dw1_c64_c256 \\
        --add N=1,H=28,W=28 --add N=1,H=112,W=112

    # Preview without writing
    python scripts/gen_workload.py conv2d_kh3_kw3_sh1_sw1_dh1_dw1_c64_c128 \\
        --add N=1,H=80,W=80 --dry-run
"""

from __future__ import annotations

import argparse
import json
import uuid as _uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Repo layout ───────────────────────────────────────────────────────────────

REPO     = Path(__file__).resolve().parent.parent
DEFS_DIR = REPO / "bench-trace" / "definitions"
WLS_DIR  = REPO / "bench-trace" / "workloads"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stable_uuid(def_name: str, axes: Dict[str, int]) -> str:
    """Deterministic UUID from definition name + axes (reproducible across runs)."""
    key = json.dumps({"def": def_name, "axes": axes}, sort_keys=True)
    return _uuid.uuid5(_uuid.NAMESPACE_DNS, key).hex


def _find_definition(name: str) -> Optional[Path]:
    """Search bench-trace/definitions/**/ for <name>.json."""
    for p in DEFS_DIR.rglob(f"{name}.json"):
        return p
    return None


def _find_workload_file(def_path: Path) -> Path:
    """Derive workload JSONL path from definition JSON path."""
    rel = def_path.relative_to(DEFS_DIR)
    return WLS_DIR / rel.parent / rel.with_suffix(".jsonl").name


def _make_inputs(defn_inputs: Dict, scalar_vals: Dict[str, Any]) -> Dict:
    """Build workload inputs dict from definition's inputs spec + inherited scalar values.

    - shape != null  →  {"type": "random"}
    - shape == null  →  {"type": "scalar", "value": scalar_vals[name]}
    """
    result = {}
    for name, spec in defn_inputs.items():
        if spec.get("shape") is not None:
            result[name] = {"type": "random"}
        else:
            if name not in scalar_vals:
                raise KeyError(
                    f"Scalar input '{name}' required by definition but not found in "
                    f"existing workloads. Pass --scalar {name}=<value> to provide it."
                )
            result[name] = {"type": "scalar", "value": scalar_vals[name]}
    return result


def _read_workloads(wl_path: Path) -> List[Dict]:
    if not wl_path.exists():
        return []
    lines = []
    for line in wl_path.read_text().splitlines():
        line = line.strip()
        if line:
            lines.append(json.loads(line))
    return lines


def _extract_scalar_vals(workloads: List[Dict], defn_inputs: Dict) -> Dict[str, Any]:
    """Pull scalar values from the first workload that has them (old or new format)."""
    scalar_names = {n for n, s in defn_inputs.items() if s.get("shape") is None}
    if not scalar_names:
        return {}
    for wl in workloads:
        # New format: wl["inputs"][name] = {"type": "scalar", "value": v}
        if "inputs" in wl:
            result = {}
            for n in scalar_names:
                entry = wl["inputs"].get(n)
                if entry and entry.get("type") == "scalar":
                    result[n] = entry["value"]
            if len(result) == len(scalar_names):
                return result
        # Old format: wl["scalar_inputs"] = {name: value}
        if "scalar_inputs" in wl:
            si = wl["scalar_inputs"]
            if scalar_names.issubset(si.keys()):
                return {n: si[n] for n in scalar_names}
    return {}


def _migrate_workload(wl: Dict, defn_inputs: Dict, scalar_vals: Dict[str, Any],
                      def_name: str) -> Dict:
    """Convert an old scalar_inputs workload to new inputs format."""
    if "inputs" in wl and "scalar_inputs" not in wl:
        return wl
    axes = wl["axes"]
    return {
        "axes": axes,
        "inputs": _make_inputs(defn_inputs, scalar_vals),
        "uuid": _stable_uuid(def_name, axes),
        "tags": wl.get("tags", {}),
    }


def _axes_key(axes: Dict[str, int]) -> tuple:
    return tuple(sorted(axes.items()))


def _write_jsonl(path: Path, workloads: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(w) for w in workloads) + "\n",
        encoding="utf-8",
    )


def _parse_axes(axes_str: str) -> Dict[str, int]:
    """Parse 'N=1,H=28,W=28' into {'N': 1, 'H': 28, 'W': 28}."""
    result = {}
    for part in axes_str.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        result[k.strip()] = int(v.strip())
    return result


# ── Core operation ────────────────────────────────────────────────────────────

def process_definition(
    def_name: str,
    extra_nhw: List[Tuple[int, int, int]],
    extra_axes_list: List[Dict[str, int]],
    *,
    migrate: bool,
    dry_run: bool,
    extra_scalars: Dict[str, Any],
) -> None:
    def_path = _find_definition(def_name)
    if def_path is None:
        print(f"  [ERROR] Definition not found: {def_name}")
        return

    defn = json.loads(def_path.read_text())
    defn_inputs = defn["inputs"]
    wl_path = _find_workload_file(def_path)
    existing = _read_workloads(wl_path)

    scalar_vals = _extract_scalar_vals(existing, defn_inputs)
    scalar_vals.update(extra_scalars)

    migrated = (
        [_migrate_workload(w, defn_inputs, scalar_vals, def_name) for w in existing]
        if migrate else list(existing)
    )

    seen = {_axes_key(w["axes"]) for w in migrated}

    new_wls: List[Dict] = []
    for n, h, w in extra_nhw:
        axes = {"N": n, "H": h, "W": w}
        if _axes_key(axes) in seen:
            continue
        seen.add(_axes_key(axes))
        new_wls.append({
            "axes": axes,
            "inputs": _make_inputs(defn_inputs, scalar_vals),
            "uuid": _stable_uuid(def_name, axes),
            "tags": {"from": "gen_workload"},
        })

    for axes in extra_axes_list:
        if _axes_key(axes) in seen:
            print(f"  [skip] {def_name} axes={axes} already present")
            continue
        seen.add(_axes_key(axes))
        new_wls.append({
            "axes": axes,
            "inputs": _make_inputs(defn_inputs, scalar_vals),
            "uuid": _stable_uuid(def_name, axes),
            "tags": {"from": "gen_workload"},
        })

    final = migrated + new_wls
    n_migrated = sum(1 for o, n in zip(existing, migrated) if o != n)
    label = (
        f"{len(existing)} existing"
        + (f", {n_migrated} migrated" if n_migrated else "")
        + (f", +{len(new_wls)} new" if new_wls else "")
        + f"  →  {len(final)} total"
    )
    print(f"  {def_name}  ({label})")

    if not dry_run:
        _write_jsonl(wl_path, final)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate / append workloads for an existing Definition."
    )
    parser.add_argument(
        "definition",
        nargs="?",
        help="Definition name. Required unless --migrate-all is passed.",
    )
    parser.add_argument(
        "--add",
        metavar="N=1,H=28,W=28",
        action="append",
        default=[],
        help="Axes for a new workload (comma-separated key=value). Can be repeated.",
    )
    parser.add_argument(
        "--scalar",
        metavar="NAME=VALUE",
        action="append",
        default=[],
        help="Override a scalar input value (e.g. --scalar pad_top=1).",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Rewrite existing workloads from old scalar_inputs format to new inputs format.",
    )
    parser.add_argument(
        "--migrate-all",
        action="store_true",
        help="Migrate every definition in bench-trace/definitions/ to the new inputs format.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without touching any files.",
    )
    args = parser.parse_args()

    if not args.migrate_all and not args.definition:
        parser.error("Provide a definition name or --migrate-all.")

    extra_scalars: Dict[str, Any] = {}
    for kv in args.scalar:
        k, _, v = kv.partition("=")
        k = k.strip()
        v = v.strip()
        try:
            extra_scalars[k] = int(v)
        except ValueError:
            try:
                extra_scalars[k] = float(v)
            except ValueError:
                extra_scalars[k] = v

    extra_axes_list = [_parse_axes(s) for s in args.add]

    if args.dry_run:
        print("[dry-run] no files will be written")

    if args.migrate_all:
        print("Migrating all definitions in bench-trace/definitions/ ...")
        for def_path in sorted(DEFS_DIR.rglob("*.json")):
            def_name = def_path.stem
            process_definition(
                def_name,
                extra_nhw=[],
                extra_axes_list=[],
                migrate=True,
                dry_run=args.dry_run,
                extra_scalars=extra_scalars,
            )
    else:
        process_definition(
            args.definition,
            extra_nhw=[],
            extra_axes_list=extra_axes_list,
            migrate=args.migrate,
            dry_run=args.dry_run,
            extra_scalars=extra_scalars,
        )

    print("Done.")


if __name__ == "__main__":
    main()
