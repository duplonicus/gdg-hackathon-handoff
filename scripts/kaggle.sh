#!/usr/bin/env bash
# Kaggle CLI with this repo's token: scripts/kaggle.sh b t status handoff-decision
# The token comes from the gitignored .env (KAGGLE_API_TOKEN) and is never printed.
# Uses the CLI from GitHub main: the PyPI release (2.2.4) has no `benchmarks quota`.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE_API_TOKEN="$(grep -m1 '^KAGGLE_API_TOKEN=' "$root/.env" | cut -d= -f2-)"
export KAGGLE_API_TOKEN
# The CLI creates its config dir at import; ~/.config is read-only in the Claude sandbox.
export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-${TMPDIR:-/tmp}/kaggle-config}"
mkdir -p "$KAGGLE_CONFIG_DIR"
exec uvx --from git+https://github.com/Kaggle/kaggle-cli kaggle "$@"
