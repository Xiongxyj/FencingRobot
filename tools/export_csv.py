"""Export data/fencing to SuperMimic CSV (+ per-frame mask sidecars), then convert with SuperMimic's own converter:

  python tools/export_csv.py --out build/fencing                     # csv/, meta/, fps_map.csv
  cd ../SuperMimic-Reproduced && env -u PYTHONPATH uv run python scripts/data/csv_to_npz_batch.py \
      --input-dir ../FencingRobot/build/fencing/csv --output-dir data/npz/fencing \
      --input-fps 30 --output-fps 50 --fps-map ../FencingRobot/build/fencing/fps_map.csv
Converting 30 fps to 50 Hz BY TIME is essential (feeding 30 fps frames as 50 Hz plays the motion 1.67x too fast)."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fencingrobot.data import PKG, export

ap = argparse.ArgumentParser(); ap.add_argument("--pkg", type=Path, default=PKG); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--limit", type=int, default=None); A = ap.parse_args()
print(json.dumps(export(A.pkg, A.out, A.limit), indent=1))
