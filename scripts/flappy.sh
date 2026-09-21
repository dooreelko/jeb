#!/usr/bin/env bash
# Flappy Bird (src/flappy.py, the winning setup; FLAPPY_MODULE=src.flappy-experiments for the
# experiments, or use scripts/flappy-experiments.sh) inside the nix shell. Output is teed to logs/ unless the
# terminal animation is on (it would fill the log with escape codes): that is the default for src.flappy
# (turn it off with --no-watch) and opt-in with --watch for the experiments.
# usage: scripts/flappy.sh [flappy options]     (works from any directory)
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"
ensure_nix_shell "$@"
cd "$ROOT"
MODULE="${FLAPPY_MODULE:-src.flappy}"
if [[ " $* " == *" --watch "* || ( "$MODULE" == src.flappy && " $* " != *" --no-watch "* ) ]]; then
  exec .venv/bin/python -u -m "$MODULE" "$@"
fi
mkdir -p logs
tag="$(printf -- "-%s" "$@" | tr -c 'A-Za-z0-9\n' - | sed 's/--*/-/g')"
log="logs/$(date +%Y%m%d-%H%M%S)-flappy${tag}.log"
echo "logging to $ROOT/$log"
.venv/bin/python -u -m "$MODULE" "$@" 2>&1 | tee "$log"
exit "${PIPESTATUS[0]}"
