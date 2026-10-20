"""Workload uuids.

Three schemes exist because three generators grew their own, and a uuid is the key that traces and
results are joined on, so none can change without orphaning what was recorded under it. They are
kept as separate, named functions; new generators should use `uuid_def_axes`.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Dict


def uuid_def_axes(def_name: str, axes: Dict[str, int]) -> str:
    """uuid5 of the definition name and its axes (gen_workload)."""
    key = json.dumps({"def": def_name, "axes": axes}, sort_keys=True)
    return uuid.uuid5(uuid.NAMESPACE_DNS, key).hex


def uuid_md5_def_axes(def_name: str, axes: dict) -> str:
    """md5 of "<name>:<axes json>" (the llama.cpp e2e gemm workloads)."""
    return hashlib.md5(f"{def_name}:{json.dumps(axes, sort_keys=True)}".encode()).hexdigest()


def uuid_simd_loop(loop_id: str, n: int, source: str) -> str:
    """uuid5 of a simd-loop id, its N and where the workload came from (gen_simd_loop_harness)."""
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"{loop_id}_N{n}_{source}").hex
