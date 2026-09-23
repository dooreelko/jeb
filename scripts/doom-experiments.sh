#!/usr/bin/env bash
# The doom state-description ladder (src/doom-experiments.py): --variant {1,2,3,4,5}.
# usage: scripts/doom-experiments.sh --variant N [options]     (works from any directory)
set -euo pipefail
export DOOM_MODULE=src.doom-experiments
exec "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/doom.sh" "$@"
