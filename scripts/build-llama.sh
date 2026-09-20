#!/usr/bin/env bash
# Rebuild llama-cpp-python with the HIP (ROCm) backend; re-enters the nix shell by itself.
# gfx1150 because the 860M (gfx1152) is presented as 11.5.0 via HSA_OVERRIDE_GFX_VERSION.
# usage: scripts/build-llama.sh                        (works from any directory)
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"
ensure_nix_shell "$@"
cd "$ROOT"

export HIPCXX="$(hipconfig -l)/clang"
export HIP_PATH="$(hipconfig -R)"
export CMAKE_ARGS="-DGGML_HIP=on -DAMDGPU_TARGETS=gfx1150 -DCMAKE_BUILD_TYPE=Release"
export FORCE_CMAKE=1
unset VIRTUAL_ENV
uv pip install --python .venv/bin/python --reinstall --no-cache --no-binary llama-cpp-python "llama-cpp-python==0.3.35"
