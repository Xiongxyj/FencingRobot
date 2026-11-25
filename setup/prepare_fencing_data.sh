#!/usr/bin/env bash
# Convert the released 30 Hz motion dataset to 50 Hz by time interpolation.
# Usage: bash setup/prepare_fencing_data.sh
set -euo pipefail
if [[ $# -ne 0 ]]; then echo "usage: $0" >&2; exit 2; fi
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SM="$(dirname "$HERE")/SuperMimic-Reproduced"
PKG="$HERE/data/fencing"
OUT="$HERE/build/fencing"
cd "$SM"
env -u PYTHONPATH uv run python "$HERE/tools/verify_data.py" --pkg "$PKG"
env -u PYTHONPATH uv run python "$HERE/tools/export_csv.py" --pkg "$PKG" --out "$OUT"
env -u PYTHONPATH uv run python scripts/data/csv_to_npz_batch.py --input-dir "$OUT/csv" \
  --output-dir "$OUT/npz" --input-fps 30 --output-fps 50 --fps-map "$OUT/fps_map.csv" --overwrite
env -u PYTHONPATH uv run python "$HERE/tools/add_masks_to_npz.py" --npz-dir "$OUT/npz" --pkg "$PKG"
echo "motion npz (50 Hz, with masks): $OUT/npz"
