#!/usr/bin/env bash
# Run the comparison inside the nix shell (ROCm libs + gfx override), teeing raw output to logs/.
# usage: scripts/run.sh [model.gguf] [n_examples]     (works from any directory)
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"

model="${1:-models/Qwen3.5-4B-Q4_K_M.gguf}"
n="${2:-200}"
# a path that exists relative to where you are is taken as-is; otherwise it is relative to the project root
[[ -e "$model" ]] && model="$(readlink -f "$model")"

ensure_nix_shell "$model" "$n"
cd "$ROOT"

log="logs/$(date +%Y%m%d-%H%M%S)-$(basename "$model" .gguf)-n${n}.log"
mkdir -p logs
echo "logging to $ROOT/$log"
# PIPESTATUS: report the python exit code, not tee's
.venv/bin/python -u -m src.eval "$model" "$n" 2>&1 | tee "$log"
exit "${PIPESTATUS[0]}"
