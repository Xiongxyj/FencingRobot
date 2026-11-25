"""Write the per-frame masks (weapon_arm_weight, no_track_mask, legs_unreliable) on the 50 Hz clock into the converted
motion npz files as fr_* keys (see fencingrobot.data.attach_masks).   usage: python tools/add_masks_to_npz.py --npz-dir build/fencing/npz"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fencingrobot.data import PKG, attach_masks

ap = argparse.ArgumentParser(); ap.add_argument("--npz-dir", type=Path, required=True); ap.add_argument("--pkg", type=Path, default=PKG)
A = ap.parse_args()
info = attach_masks(A.npz_dir, A.pkg); print(json.dumps(info, indent=1))
json.dump(info, open(A.npz_dir.parent / "masks.json", "w"), indent=1)
