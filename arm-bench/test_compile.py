"""Script to smoke-test SIMDTools.compile() locally (handle=None)."""
from pathlib import Path

from eval.tools import SIMDTools
from eval.evaluator import load_starter_problems

PROBLEM_ID = "conv2d"
ISA = "sve"

problems = load_starter_problems()
problem = problems[PROBLEM_ID]

tools = SIMDTools(handle=None, problem_id=PROBLEM_ID, isa=ISA)

print(f"[1] Uploading (copying) ncnn tree to {tools.remote_project_root} ...")
tools.upload_ncnn_tree()

print(f"[2] Calling compile('{problem['starter']}') ...")
result = tools.compile(problem["starter"])

print(f"\n[3] success = {result.success}")
if not result.success:
    print("----- errors -----")
    print(result.errors[:4000])
else:
    print(f"candidate bin: {tools.remote_binary}")
    print(f"baseline bin:  {tools.remote_baseline_binary}")
    print(f"candidate exists: {Path(tools.remote_binary).exists()}")
    print(f"baseline  exists: {Path(tools.remote_baseline_binary).exists()}")
    if result.warnings:
        print(f"warnings (first 500 chars):\n{result.warnings[:500]}")
