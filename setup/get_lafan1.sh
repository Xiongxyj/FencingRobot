#!/usr/bin/env bash
# LAFAN1 retargeted to G1 (30 fps CSV), for reproducing SuperMimic before training on our data.
# usage: bash setup/get_lafan1.sh   (run from FencingRobot; writes into ../SuperMimic-Reproduced/data)
set -euo pipefail
SM="$(cd "$(dirname "$0")/../.." && pwd)/SuperMimic-Reproduced"; cd "$SM"
env -u PYTHONPATH uvx --from huggingface_hub hf download lvhaidong/LAFAN1_Retargeting_Dataset \
  --repo-type dataset --include "g1/*" --local-dir data/raw/lafan1
env -u PYTHONPATH uv run python scripts/data/csv_to_npz_batch.py --input-dir data/raw/lafan1/g1 --output-dir data/npz/lafan1 --input-fps 30
