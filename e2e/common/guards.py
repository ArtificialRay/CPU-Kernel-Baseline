"""Checks that a built kernel set is the one you meant to build.

A manifest built from hand-assembled or stale solution files can silently hold the *old*
kernels: every later check still passes (the shapes are all intercepted, the perplexity looks
plausible) because nothing about the splice is wrong -- it is faithfully running the wrong
sources, and that costs a whole measurement run to notice. These fail fast instead."""
from __future__ import annotations

import re
from typing import Callable

from .solutions import source_text


def check_guards(
    solutions: dict, built: list[str], reference: dict, *, digest: Callable[[str], str],
    expect_count: int | None = None, expect_changed=(), expect_same=(), forbid=(),
) -> list[str]:
    """The failures (empty = every guard holds), judged on the sources that went into the set.

    solutions   definition -> solution JSON the set was built from
    built       names of the definitions that made it into the set
    reference   definition -> solution JSON of the set to compare against (may be empty)
    digest      source text -> comparable hash; lets the framework normalise before comparing
    expect_changed   each of these MUST differ from the reference
    expect_same      each of these MUST be identical to it (deliberately reused)
    forbid           no source may match any of these regexes
    """
    got = {name: source_text(solutions[name]) for name in built}
    ref = {name: source_text(sol) for name, sol in reference.items()}
    fail = []
    if expect_count is not None and len(got) != expect_count:
        fail.append(f"manifest has {len(got)} kernels, expected {expect_count}")
    for name in expect_changed:
        if name not in got:
            fail.append(f"{name}: expected in the manifest, absent")
        elif name not in ref:
            fail.append(f"{name}: no reference to compare against")
        elif digest(got[name]) == digest(ref[name]):
            fail.append(f"{name}: BYTE-IDENTICAL to the reference -- this is the old kernel, not the new one")
    for name in expect_same:
        if name not in got:
            fail.append(f"{name}: expected in the manifest, absent")
        elif name in ref and digest(got[name]) != digest(ref[name]):
            fail.append(f"{name}: expected to be reused unchanged but differs from the reference")
    for pattern in forbid:
        rx = re.compile(pattern)
        for name, text in sorted(got.items()):
            if rx.search(text):
                fail.append(f"{name}: source matches forbidden pattern /{pattern}/")
    return fail
