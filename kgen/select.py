"""Choosing which workloads of a candidate pool to keep."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List


def filter_axes(candidates: List[Dict[str, int]], batch_key: str, max_count: int,
                max_dups: int = 2) -> List[Dict[str, int]]:
    """Two-layer workload filtering: dedup + greedy farthest-point diversity.

    Layer 1: same axes combo kept at most max_dups times.
    Layer 2: normalized Euclidean greedy farthest-point selection.
      Phase 1 — one seed per distinct batch_key value (ensures coverage across batch sizes).
                With more distinct values than max_count the seeds are spread evenly over the
                sorted values, smallest and largest included (taking the first max_count would
                keep only the smallest batches and never reach Phase 2).
      Phase 2 — greedy fill to reach max_count.
    """
    seen_count: Dict[str, int] = defaultdict(int)
    pool: List[Dict[str, int]] = []
    for ax in candidates:
        key = json.dumps(ax, sort_keys=True)
        if seen_count[key] < max_dups:
            pool.append(ax)
            seen_count[key] += 1

    if len(pool) <= max_count:
        return pool

    all_keys = list(candidates[0].keys())

    def _norm_dist(a: Dict[str, int], b: Dict[str, int]) -> float:
        return sum(
            ((a[k] - b[k]) / max(abs(a[k]), abs(b[k]), 1.0)) ** 2
            for k in all_keys
        ) ** 0.5

    def _min_dist(entry: Dict[str, int], sel: List[Dict[str, int]]) -> float:
        return min(_norm_dist(entry, s) for s in sel)

    by_batch: Dict[int, List[Dict[str, int]]] = defaultdict(list)
    for ax in pool:
        by_batch[ax[batch_key]].append(ax)

    batch_values = sorted(by_batch)
    if len(batch_values) > max_count:
        last = len(batch_values) - 1
        picks = {round(i * last / max(max_count - 1, 1)) for i in range(max_count)}
        batch_values = [batch_values[i] for i in sorted(picks)]

    selected: List[Dict[str, int]] = []
    for bs in batch_values:
        grp = by_batch[bs]
        pick = grp[0] if not selected else max(grp, key=lambda e: _min_dist(e, selected))
        selected.append(pick)
        if len(selected) >= max_count:
            break

    sel_ids = {id(e) for e in selected}
    remaining = [ax for ax in pool if id(ax) not in sel_ids]
    while len(selected) < max_count and remaining:
        best = max(remaining, key=lambda e: _min_dist(e, selected))
        selected.append(best)
        remaining.remove(best)

    return selected
