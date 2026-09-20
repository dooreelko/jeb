
secrets are managed by secret-tool

project-specific dependencies managed by nix in shell.nix

first choice for scripting language is bash

# Moth Agent Guide

This guide helps LLM agents work effectively with moth, a git-based file issue tracker.

## Overview

Moth stores issues as markdown files in `.moth/` directories organized by status (ready, doing, done). Each issue has a unique ID, severity, and slug derived from the title.
NEVER manupulate the moth files directly, ALWAYS use `moth` cli for any changes.
`moth update` replaces the whole description from stdin — when adding to an
issue (e.g. a decision record), always read the current description first
(`moth show`) and pass it back through unchanged plus the new content
appended. Never drop existing text.

**Note**: Moth automatically recreates missing status directories (e.g., if git removes empty directories). As long as `config.yml` exists, moth will recover gracefully.

## Workflow Commands

See `moth --agent-help`

## DOD

never run `moth done`.
never run `moth start`.

you never decide when the task is done.

## Git

Don't use worktrees, but feature branches instead

## Specs

Moth is the primary spec destination for this repo, not `docs/superpowers/specs/`.
A moth issue's description should stay a decision record: what was decided, why,
and what's explicitly out of scope — enough to recreate the plan (same or better)
and tasks later, but no low-level technical detail (exact routes, ports, commands,
etc). If a fuller technical design doc is written (e.g. via the brainstorming
skill), keep it under `docs/superpowers/specs/` and link to it from the moth
issue rather than duplicating its detail there.


