#!/usr/bin/env bash
# The full Flappy Bird experiment runner (src/flappy-experiments.py): all prompt variants, option styles
# and policies. usage: scripts/flappy-experiments.sh [options]     (works from any directory)
set -euo pipefail
export FLAPPY_MODULE=src.flappy-experiments
exec "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/flappy.sh" "$@"
