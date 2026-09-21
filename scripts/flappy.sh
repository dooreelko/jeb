#!/usr/bin/env bash
# Flappy Bird benchmark (src/flappy.py) inside the nix shell. Output is teed to logs/ unless --watch
# is given (the terminal animation would fill the log with escape codes).
# usage: scripts/flappy.sh [flappy options]     (works from any directory)
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"
ensure_nix_shell "$@"
cd "$ROOT"
if [[ " $* " == *" --watch "* ]]; then
  exec .venv/bin/python -u -m src.flappy "$@"
fi
mkdir -p logs
tag="$(printf -- "-%s" "$@" | tr -c 'A-Za-z0-9\n' - | sed 's/--*/-/g')"
log="logs/$(date +%Y%m%d-%H%M%S)-flappy${tag}.log"
echo "logging to $ROOT/$log"
.venv/bin/python -u -m src.flappy "$@" 2>&1 | tee "$log"
exit "${PIPESTATUS[0]}"
