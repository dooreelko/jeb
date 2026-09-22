#!/usr/bin/env bash
# Runs Flappy Bird (scripts/flappy.sh) against the NPU-enabled llama.cpp build at
# /home/doo/projects/llama.cpp instead of this project's own .venv llama-cpp-python build.
# See docs/superpowers/specs/2026-09-22-npu-flappy-bf16-padding-design.md.
# usage: scripts/flappy-npu.sh [flappy options]     (works from any directory)
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"
ensure_nix_shell "$@"
cd "$ROOT"

NPU_LLAMA_BIN=/home/doo/projects/llama.cpp/build/bin
if [[ ! -f "$NPU_LLAMA_BIN/libllama.so" ]]; then
  echo "error: $NPU_LLAMA_BIN/libllama.so not found -- build it first (see the implementation plan's Task 2)" >&2
  exit 1
fi
export LLAMA_CPP_LIB_PATH="$NPU_LLAMA_BIN"
export LD_LIBRARY_PATH="$NPU_LLAMA_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# The kernel JIT compiler (ggml-hsa/kernel-compiler.cpp) attaches to this process's own Python
# interpreter (flappy.sh runs .venv/bin/python) and needs the IRON toolchain reachable from
# there: PEANO_INSTALL_DIR so aiecc can find `opt`, and `opt`/aiecc themselves on PATH. Without
# this, every kernel not already in the shared cache fails to compile ("tool 'opt' not found")
# and silently falls back to CPU for every op -- see src/ggml-hsa/README.md's "Embedding in a
# Python-based consumer" section in the ../ggml repo for the general requirement.
PEANO_DIR="$ROOT/.venv/lib/python3.12/site-packages/llvm-aie"
if [[ ! -d "$PEANO_DIR" ]]; then
  echo "error: $PEANO_DIR not found -- install requirements-iron.txt into .venv first" >&2
  exit 1
fi
export PEANO_INSTALL_DIR="$PEANO_DIR"
export PATH="$PEANO_DIR/bin:$PATH"

# The NPU GEMM kernel requires bf16 weights (Q4_K_M, Flappy's default --model, can't dispatch
# to it at all -- every MUL_MAT kernel creation fails and silently falls back to CPU). Default to
# the bf16 model this issue built (models/Qwen3.5-0.8B-bf16.gguf) unless the caller already passed
# --model explicitly.
args=("$@")
has_model_flag=0
for a in "$@"; do
  if [[ "$a" == "--model" || "$a" == --model=* ]]; then
    has_model_flag=1
    break
  fi
done
if [[ "$has_model_flag" -eq 0 ]]; then
  args=(--model models/Qwen3.5-0.8B-bf16.gguf "${args[@]}")
fi

exec scripts/flappy.sh "${args[@]}"
