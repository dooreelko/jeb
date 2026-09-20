# Sourced by the other scripts (not run directly). Keeps them independent of the caller's cwd.
# Sets ROOT (project root) and SCRIPT_PATH (the calling script's real path), and provides
# ensure_nix_shell to re-exec the caller inside the project's nix shell.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[1]}")"
ROOT="$(dirname "$(dirname "$SCRIPT_PATH")")"

# usage: ensure_nix_shell "$@"   (pass absolute paths: the re-exec happens from $ROOT)
ensure_nix_shell() {
  if [[ -z "${IN_NIX_SHELL:-}" ]]; then
    cd "$ROOT"  # shell.nix lives here
    exec nix-shell --run "$(printf '%q ' "$SCRIPT_PATH" "$@")"
  fi
}
