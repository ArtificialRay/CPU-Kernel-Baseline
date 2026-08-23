"""ISA handling for mcp_app — explicit compile-flag table + a runtime safety check.

Two deliberately different roles:

  march_for_isa(isa)        — the source of truth for compile flags. `isa` is
                               always supplied explicitly by the caller (never
                               inferred), so a given experiment deterministically
                               compiles with the same flags every time it asks
                               for a given ISA.
  verify_isa_available(isa) — a runtime /proc/cpuinfo safety check, run once at
                               session startup. Catches a caller's provisioning
                               mistake (e.g. asked for sve2, landed on an
                               sve-only box) — it never decides march flags.

No instance-type table lives here (or anywhere in mcp_app): which EC2 instance
type satisfies a given ISA is a provisioning concern, out of scope for
mcp_app.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from contracts import ISA_TABLE

# isa name -> (march flag, isa_features, target_hardware label), from contracts.py
# (shared with eval/provision.py etc.) plus a local
# "portable" alias: same -march as neon (armv8-a mandates NEON); the "no
# hand-written SIMD" constraint is a PROMPT concern (nanobot skill), not a
# compile flag, so "portable" isn't a real hardware tier in contracts.yaml.
_ISA_MARCH: dict[str, tuple[str, list[str], list[str]]] = {
    isa: (spec.march, spec.features, spec.labels) for isa, spec in ISA_TABLE.items()
}
# Same compile flags as neon (armv8-a mandates NEON), but its own generic
# target_hardware label — "portable" solutions aren't claiming NEON-specific
# intent, just whatever the compiler auto-vectorizes.
_ISA_MARCH["portable"] = (ISA_TABLE["neon"].march, list(ISA_TABLE["neon"].features), ["aarch64"])

# isa name -> /proc/cpuinfo "Features" tokens that must ALL be present.
# TODO: exact token names for Graviton3 (sve) vs. Graviton4 (sve2) should be
# confirmed against real hardware; this only affects the precision of the
# safety check below, never the compiled output.
_ISA_CPUINFO_TOKENS: dict[str, list[str]] = {
    "portable": ["asimd"],
    "neon": ["asimd"],
    "sve": ["sve"],
    "sve2": ["sve2"],
    "sme2": ["sme2"],
}

SUPPORTED_ISAS = tuple(_ISA_MARCH)


class MarchInfo(NamedTuple):
    march_flag: str
    isa_features: list[str]
    target_hardware: list[str]


def march_for_isa(isa: str, *, instance_label: str | None = None) -> MarchInfo:
    """Return (march flag, isa_features, target_hardware) for an explicit isa string.

    `instance_label` (e.g. "c8g.large"), if given, is appended to
    target_hardware purely for provenance/tagging — it is never used to pick
    compile flags.
    """
    if isa not in _ISA_MARCH:
        raise ValueError(f"Unknown isa {isa!r}. Supported: {sorted(_ISA_MARCH)}")
    march_flag, isa_features, target_hardware = _ISA_MARCH[isa]
    target_hardware = list(target_hardware)
    if instance_label:
        target_hardware.append(instance_label)
    return MarchInfo(march_flag, list(isa_features), target_hardware)


def _read_cpuinfo_features(cpuinfo_path: Path = Path("/proc/cpuinfo")) -> set[str]:
    """Return the set of tokens in /proc/cpuinfo's first "Features" line."""
    if not cpuinfo_path.exists():
        return set()
    for line in cpuinfo_path.read_text().splitlines():
        if line.startswith("Features"):
            _, _, rest = line.partition(":")
            return set(rest.split())
    return set()


def verify_isa_available(isa: str, *, cpuinfo_path: Path = Path("/proc/cpuinfo")) -> None:
    """Raise RuntimeError if this machine doesn't actually support `isa`.

    A fail-fast sanity check run once at server startup — not the mechanism
    that decides compile flags (see march_for_isa above).
    """
    if isa not in _ISA_CPUINFO_TOKENS:
        raise ValueError(f"Unknown isa {isa!r}. Supported: {sorted(_ISA_CPUINFO_TOKENS)}")
    required = set(_ISA_CPUINFO_TOKENS[isa])
    detected = _read_cpuinfo_features(cpuinfo_path)
    missing = required - detected
    if missing:
        raise RuntimeError(
            f"Requested isa={isa!r} but this machine's /proc/cpuinfo Features "
            f"line is missing {sorted(missing)} (detected: {sorted(detected)}). "
            "Did whatever provisioned this instance pick the wrong instance type?"
        )


__all__ = ["MarchInfo", "SUPPORTED_ISAS", "march_for_isa", "verify_isa_available"]
