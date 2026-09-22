#!/usr/bin/env bash
# vizdoom defend_the_center (src/doom.py, the baseline; DOOM_MODULE=src.doom-experiments for the
# state-description ladder, or use scripts/doom-experiments.sh) inside the nix shell. --watch
# prints one line per decision, no escape codes, so output is always teed to logs/ (unlike
# scripts/flappy.sh).
# usage: scripts/doom.sh [doom options]     (works from any directory)
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"
ensure_nix_shell "$@"
cd "$ROOT"
MODULE="${DOOM_MODULE:-src.doom}"
mkdir -p logs
tag="$(printf -- "-%s" "$@" | tr -c 'A-Za-z0-9\n' - | sed 's/--*/-/g')"
log="logs/$(date +%Y%m%d-%H%M%S)-doom${tag}.log"
echo "logging to $ROOT/$log"
.venv/bin/python -u -m "$MODULE" "$@" 2>&1 | tee "$log"
exit "${PIPESTATUS[0]}"
