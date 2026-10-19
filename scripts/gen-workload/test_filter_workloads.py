#!/usr/bin/env python3
"""_filter_workloads keeps batch sizes from the whole traced range (issue #90).

Run:  python -m unittest scripts/gen-workload/test_filter_workloads.py   (or python <this file>)
"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("gen_workload", Path(__file__).with_name("gen_workload.py"))
gen_workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_workload)
filter_workloads = gen_workload._filter_workloads


class FilterWorkloadsTest(unittest.TestCase):
    def test_single_axis_covers_the_whole_range_not_only_the_smallest_values(self):
        candidates = [{"M": m} for m in range(1, 513)]           # prompts of 1..512 tokens
        picked = [w["M"] for w in filter_workloads(candidates, "M", 24)]
        self.assertEqual(len(picked), 24)
        self.assertEqual((min(picked), max(picked)), (1, 512))
        self.assertGreater(sum(m >= 128 for m in picked), 12)    # not 1..31 any more
        self.assertEqual(len(set(picked)), 24)

    def test_few_distinct_values_are_all_kept(self):
        candidates = [{"M": m} for m in (1, 4, 16, 64)] * 3
        self.assertEqual(sorted(w["M"] for w in filter_workloads(candidates, "M", 20)), [1, 1, 4, 4, 16, 16, 64, 64])

    def test_max_count_one_keeps_the_smallest(self):
        self.assertEqual(filter_workloads([{"M": m} for m in (5, 9, 300)], "M", 1), [{"M": 5}])

    def test_two_axes_still_fill_up_to_max_count(self):
        candidates = [{"M": m, "S": s} for m in range(1, 100) for s in (128, 512)]
        picked = filter_workloads(candidates, "M", 20)
        self.assertEqual(len(picked), 20)
        self.assertEqual((min(w["M"] for w in picked), max(w["M"] for w in picked)), (1, 99))


if __name__ == "__main__":
    unittest.main()
