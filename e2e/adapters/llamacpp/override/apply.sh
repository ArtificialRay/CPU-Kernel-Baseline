#!/usr/bin/env bash
# apply.sh <llama.cpp dir> - install the armbench kernel-override hook into a
# llama.cpp checkout (tested against tag v0.4.1 / commit b29c606). Idempotent.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dst="${1:?usage: apply.sh <llama.cpp dir>}"
dst="$(cd "$dst" && pwd)"
[ -f "$dst/ggml/src/ggml-cpu/ggml-cpu.c" ] || { echo "apply.sh: $dst is not a llama.cpp checkout" >&2; exit 1; }

cp "$here/armbench_override.h" "$here/armbench_override.c" "$dst/ggml/src/ggml-cpu/"
echo "apply.sh: copied armbench_override.{h,c} -> $dst/ggml/src/ggml-cpu/"

cd "$dst"
if git apply --check --reverse "$here/ggml-cpu-override.patch" >/dev/null 2>&1; then
    echo "apply.sh: patch already applied"
elif git apply --check "$here/ggml-cpu-override.patch"; then
    git apply "$here/ggml-cpu-override.patch"
    echo "apply.sh: patch applied"
else
    echo "apply.sh: patch does not apply cleanly (expected llama.cpp v0.4.1)" >&2
    exit 1
fi
