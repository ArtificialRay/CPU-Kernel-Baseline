#!/usr/bin/env bash
# build_agent_llamacpp.sh <llama.cpp-e2e dir> — derive the override-enabled "agent" build from the
# pinned e2e tree (config/dataset_builds.json "llama.cpp-e2e" clones it; build-stock is the baseline).
#   build-agent: ggml-cpu override hook applied (scripts/e2e/override/apply.sh) and repack/KleidiAI
#                OFF so every 2-D mul_mat reaches the hook (on NEON+i8mm stock ggml repacks q4_K
#                whenever N % 8 == 0 and the repacked tensors never hit ggml_compute_forward's switch).
#   build-norepack: stock code, repack/KleidiAI OFF — the fair "same kernels ggml would use without
#                its interleaved fast path" reference. Report all three (see docs/e2e_qwen35.md).
# Run on the box (clang-18). Idempotent: apply.sh detects an already-applied patch.
set -euo pipefail
L="${1:?llama.cpp-e2e dir}"; L="${L/#\~/$HOME}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CC_="${CC:-clang-18}"; CXX_="${CXX:-clang++-18}"
COMMON=(-DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF)
NOREPACK=(-DGGML_CPU_REPACK=OFF -DGGML_CPU_KLEIDIAI=OFF)
TARGETS=(llama-bench llama-cli llama-perplexity)

echo "[agent-build] applying override hook to $L"
bash "$HERE/override/apply.sh" "$L"
cd "$L"
echo "[agent-build] configure+build build-agent (hook + no repack)"
CC=$CC_ CXX=$CXX_ cmake -B build-agent "${COMMON[@]}" "${NOREPACK[@]}" > build-agent.configure.log
cmake --build build-agent -j"$(nproc)" --target "${TARGETS[@]}" > build-agent.build.log
# build-norepack must NOT contain the hook: build it from a pristine copy of the ggml-cpu dispatch.
if [ ! -x build-norepack/bin/llama-bench ]; then
  echo "[agent-build] configure+build build-norepack (stock code, no repack) from a pristine checkout"
  rm -rf ../llama.cpp-e2e-norepack && git worktree add -f ../llama.cpp-e2e-norepack HEAD >/dev/null 2>&1 \
    || git clone -q --depth=1 --branch "$(git describe --tags --exact-match 2>/dev/null || echo v0.4.1)" https://github.com/ggml-org/llama.cpp ../llama.cpp-e2e-norepack
  ( cd ../llama.cpp-e2e-norepack && CC=$CC_ CXX=$CXX_ cmake -B build "${COMMON[@]}" "${NOREPACK[@]}" > configure.log \
      && cmake --build build -j"$(nproc)" --target "${TARGETS[@]}" > build.log )
  mkdir -p build-norepack && ln -sfn ../../llama.cpp-e2e-norepack/build/bin build-norepack/bin
fi
for b in build-stock build-agent build-norepack; do printf "%-15s " $b; test -x $b/bin/llama-bench && echo ok || echo MISSING; done
echo "[agent-build] verify the hook is live:  ARMBENCH_OVERRIDES=manifest.json ARMBENCH_OVERRIDE_LOG=1 build-agent/bin/llama-bench -m <gguf> -n 8 -p 0"
