#!/usr/bin/env bash
# Install the pinned CSV-to-NPZ converter in its own environment.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SM="$(dirname "$HERE")/SuperMimic-Reproduced"
REV=ba6105349069bf040f1fdad60873d13138d505a2
command -v uv >/dev/null || { echo 'Install uv first (or run setup/setup_official.sh).'; exit 1; }
if [ ! -d "$SM/.git" ]; then
  git clone https://github.com/CuboiLeo/SuperMimic-Reproduced.git "$SM"
fi
if [ "$(git -C "$SM" rev-parse HEAD)" != "$REV" ]; then
  git -C "$SM" fetch origin
  git -C "$SM" checkout "$REV"
fi
cd "$SM"
env -u PYTHONPATH -u VIRTUAL_ENV uv sync --frozen
