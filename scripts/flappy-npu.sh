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
exec scripts/flappy.sh "$@"
