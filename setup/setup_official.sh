#!/usr/bin/env bash
# Official SuperMimic (HybridRobotics/supermimic) + our stand-in for its unreleased mjlab fork.
# Layout (what the official pyproject expects):
#   $WORK/supermimic        official code, pinned, unmodified, its own uv env (.venv, Python 3.13)
#   $WORK/mjlab_research    public mjlab v1.5.0 + setup/mjlab_research.patch (branch fencing-research)
#   $WORK/FencingRobot      this repo
# The official uv.lock pins mjlab_research's dependencies exactly as mjlab v1.5.0 declares them, so
# `uv sync --frozen` installs the official environment unchanged.
# usage: bash setup/setup_official.sh      (from the FencingRobot checkout; WORK = parent directory)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; WORK="$(dirname "$HERE")"
SM_URL=https://github.com/HybridRobotics/supermimic.git
SM_REV=5efbf114e11f725c31d83e217be987c13c0e4ce2     # 2026-11-10 "Add the Apache-2.0 license and acknowledgements"
MJ_URL=https://github.com/mujocolab/mjlab.git
MJ_REV=v1.5.0                                       # 08090e8a7; matches the official uv.lock's mjlab metadata
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

echo "== 1. official supermimic @ $SM_REV"
[ -d "$WORK/supermimic/.git" ] || git clone "$SM_URL" "$WORK/supermimic"
git -C "$WORK/supermimic" fetch -q origin && git -C "$WORK/supermimic" checkout -q "$SM_REV"
git -C "$WORK/supermimic" diff --quiet || { echo "ERROR: $WORK/supermimic has local changes"; exit 1; }

echo "== 2. mjlab_research = mjlab $MJ_REV + setup/mjlab_research.patch"
if [ ! -d "$WORK/mjlab_research/.git" ]; then
  git clone -q "$MJ_URL" "$WORK/mjlab_research"
  git -C "$WORK/mjlab_research" checkout -q -b fencing-research "$MJ_REV"
  git -C "$WORK/mjlab_research" apply "$HERE/setup/mjlab_research.patch"
  git -C "$WORK/mjlab_research" -c user.name=FencingRobot -c user.email=fencingrobot@localhost \
    commit -q -am "FencingRobot: SuperMimic interfaces (setup/mjlab_research.patch)"
else
  echo "   exists; checking it equals $MJ_REV + patch"
  git -C "$WORK/mjlab_research" diff "$MJ_REV" | diff -q - "$HERE/setup/mjlab_research.patch" >/dev/null \
    || echo "WARNING: $WORK/mjlab_research differs from $MJ_REV + patch (regenerate with: git -C ../mjlab_research diff $MJ_REV > setup/mjlab_research.patch)"
fi

echo "== 3. uv env (frozen official lock) + this repo (editable, no deps)"
cd "$WORK/supermimic"
env -u PYTHONPATH -u VIRTUAL_ENV uv sync --frozen
env -u PYTHONPATH -u VIRTUAL_ENV uv pip install --no-deps -e "$HERE"

echo "== 4. checks"
env -u PYTHONPATH .venv/bin/python -c "import torch, mujoco, warp; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), '| mujoco', mujoco.__version__, '| warp', warp.__version__)"
env -u PYTHONPATH .venv/bin/python "$HERE/tools/check_mjlab_research.py"
MUJOCO_GL=disable env -u PYTHONPATH uv run --frozen pytest -q || echo "WARNING: official tests failed - report before training"
echo "done. Next: docs/REPRODUCE.md."
