#!/usr/bin/env python3
"""Bring up the end-to-end measurement tree on an already-provisioned box.

Reuses the launch skill's machinery: the instance is looked up by its
provisioning/eval_config.json label (provision it first with provisioning/provision.py or
bench_fleet.py --skip-final-teardown), then config/dataset_builds.json's
"llama.cpp-e2e" entry is run through ensure_dataset_ready (pinned clone,
stock tools build); --model <key> then fetches that config/e2e_models.json
GGUF into ~/models on the box. With --agent-build it then derives the
override-enabled "agent" build via e2e/adapters/llamacpp/build_agent.sh,
which is rsynced to the box together with e2e/adapters/llamacpp/override/.

  deploy("e2e-c8g-4xl", "qwen3.8-27b", agent_build=True)      # or: python -m e2e evaluate --steps deploy --label ...

Nothing here launches instances or spends money; it only builds on a box
that already exists in the registry.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "skills" / "launch"))
sys.path.insert(0, str(REPO_ROOT))

from launch_session import _read_config_instance, ensure_dataset_ready  # noqa: E402
from e2e.errors import StepFailed  # noqa: E402


def deploy(label: str, model: str | None = None, agent_build: bool = False) -> None:
    """Bring up the end-to-end measurement tree on the already-provisioned box `label`
    (a provisioning/eval_config.json label), driving it over SSH.

    model        config/e2e_models.json key whose GGUF to fetch into ~/models on the box
    agent_build  also build the override-enabled "agent" tree (and the no-repack control)
    """
    inst = _read_config_instance(label)
    if inst is None:
        raise StepFailed(f"no instance under label {label!r} in provisioning/eval_config.json")
    target = inst.target
    print(f"[e2e] {label}: {target.user}@{target.host}")

    ensure_dataset_ready(target, "llama.cpp-e2e")

    if model:
        models = json.loads((REPO_ROOT / "config" / "e2e_models.json").read_text())
        if model not in models:
            raise StepFailed(f"unknown model {model!r}; known: {[k for k in models if not k.startswith('_')]}")
        m = models[model]
        cmd = (f"mkdir -p ~/models && cd ~/models && (test -f {m['file']} || (curl -sSL -C - -o {m['file']}.part "
               f"https://huggingface.co/{m['repo']}/resolve/main/{m['file']} && mv -f {m['file']}.part {m['file']})) && ls -la {m['file']}")
        print(f"[e2e] fetching {m['repo']}/{m['file']} ({m['size_gb']} GB) on the box")
        rc, out, err = target.run(cmd, timeout=3600)
        print(out.strip()[-300:])
        if rc != 0:
            raise StepFailed(f"model fetch failed: {err[-500:]}")

    if agent_build:
        src = REPO_ROOT / "e2e" / "adapters" / "llamacpp"
        remote_dir = "arm-bench-e2e/e2e/adapters/llamacpp"
        target.run(f"mkdir -p ~/{remote_dir}", timeout=30)   # rsync only creates the last path component
        rc = subprocess.run(["rsync", "-az", "--exclude", "__pycache__", f"{src}/",
                             f"{target.user}@{target.host}:{remote_dir}/",
                             "-e", f"ssh -i {target.key_file} -o StrictHostKeyChecking=accept-new"]).returncode
        if rc != 0:
            raise StepFailed(f"rsync of {src} failed")
        rc, out, err = target.run("bash ~/arm-bench-e2e/e2e/adapters/llamacpp/build_agent.sh ~/llama.cpp-e2e", timeout=1800)
        print(out[-2000:]); print(err[-2000:], file=sys.stderr)
        if rc != 0:
            raise StepFailed("agent build failed")
    print("[e2e] ready")
