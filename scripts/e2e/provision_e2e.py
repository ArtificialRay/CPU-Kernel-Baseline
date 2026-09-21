#!/usr/bin/env python3
"""Bring up the end-to-end measurement tree on an already-provisioned box.

Reuses the launch skill's machinery: the instance is looked up by its
eval/eval_config.json label (provision it first with eval/provision.py or
bench_fleet.py --skip-final-teardown), then config/dataset_builds.json's
"llama.cpp-e2e" entry is run through ensure_dataset_ready (pinned clone,
stock tools build); --model <key> then fetches that config/e2e_models.json
GGUF into ~/models on the box. With --agent-build it then derives the
override-enabled "agent" build via scripts/e2e/build_agent_llamacpp.sh,
which is rsynced to the box together with scripts/e2e/override/.

  python3 scripts/e2e/provision_e2e.py --label e2e-c8g-4xl --model qwen3.8-27b [--agent-build]

Nothing here launches instances or spends money; it only builds on a box
that already exists in the registry.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "skills" / "launch"))
sys.path.insert(0, str(REPO_ROOT))

from launch_session import _read_config_instance, ensure_dataset_ready  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="eval/eval_config.json instance label")
    ap.add_argument("--model", default=None, help="config/e2e_models.json key to fetch on the box (e.g. qwen3.8-27b)")
    ap.add_argument("--agent-build", action="store_true",
                    help="also build the override-enabled tree (build-agent) on the box")
    args = ap.parse_args()

    inst = _read_config_instance(args.label)
    if inst is None:
        raise SystemExit(f"no instance under label {args.label!r} in eval/eval_config.json")
    target = inst.target
    print(f"[e2e] {args.label}: {target.user}@{target.host}")

    ensure_dataset_ready(target, "llama.cpp-e2e")

    if args.model:
        models = json.loads((REPO_ROOT / "config" / "e2e_models.json").read_text())
        if args.model not in models:
            raise SystemExit(f"unknown model {args.model!r}; known: {[k for k in models if not k.startswith('_')]}")
        m = models[args.model]
        cmd = (f"mkdir -p ~/models && cd ~/models && (test -f {m['file']} || (curl -sSL -C - -o {m['file']}.part "
               f"https://huggingface.co/{m['repo']}/resolve/main/{m['file']} && mv -f {m['file']}.part {m['file']})) && ls -la {m['file']}")
        print(f"[e2e] fetching {m['repo']}/{m['file']} ({m['size_gb']} GB) on the box")
        rc, out, err = target.run(cmd, timeout=3600)
        print(out.strip()[-300:])
        if rc != 0:
            raise SystemExit(f"model fetch failed: {err[-500:]}")

    if args.agent_build:
        src = REPO_ROOT / "scripts" / "e2e"
        target.run("mkdir -p ~/arm-bench-e2e/scripts/e2e", timeout=30)   # rsync only creates the last path component
        rc = subprocess.run(["rsync", "-az", "--exclude", "__pycache__", f"{src}/",
                             f"{target.user}@{target.host}:arm-bench-e2e/scripts/e2e/",
                             "-e", f"ssh -i {target.key_file} -o StrictHostKeyChecking=accept-new"]).returncode
        if rc != 0:
            raise SystemExit("rsync of scripts/e2e failed")
        rc, out, err = target.run("bash ~/arm-bench-e2e/scripts/e2e/build_agent_llamacpp.sh ~/llama.cpp-e2e", timeout=1800)
        print(out[-2000:]); print(err[-2000:], file=sys.stderr)
        if rc != 0:
            raise SystemExit("agent build failed")
    print("[e2e] ready")


if __name__ == "__main__":
    main()
