"""Check the motion package after a clone / pull: sha256 of every segment vs files.sha256, schema, joint order, fps,
NaN / quaternion norm, frame counts vs manifest.csv.   usage: python tools/verify_data.py [--pkg data/fencing]"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fencingrobot.data import G1_JOINT_NAMES, PKG, load_segment, segments, sha256

ap = argparse.ArgumentParser(); ap.add_argument("--pkg", type=Path, default=PKG); A = ap.parse_args()
want = {l.split()[1]: l.split()[0] for l in open(A.pkg / "files.sha256") if l.strip()}
man = {r["segment_id"]: r for r in csv.DictReader(open(A.pkg / "manifest.csv"))}
segs = segments(A.pkg); bad = []
need = ("qpos", "fps", "joint_names", "weapon_arm_weight", "no_track_mask", "legs_unreliable", "mirrored")
frames = 0
for p in segs:
    rel = f"segments/{p.name}"
    if want.get(rel) != sha256(p):
        bad.append((p.stem, "sha256")); continue
    d = load_segment(p); T = len(d["qpos"]); frames += T
    if any(k not in d for k in need): bad.append((p.stem, "missing keys"))
    if tuple(str(j) for j in d["joint_names"]) != G1_JOINT_NAMES: bad.append((p.stem, "joint order"))
    if float(d["fps"]) != 30.0: bad.append((p.stem, f"fps {d['fps']}"))
    if not np.isfinite(d["qpos"]).all(): bad.append((p.stem, "NaN"))
    if np.abs(np.linalg.norm(d["qpos"][:, 3:7], axis=1) - 1).max() > 1e-3: bad.append((p.stem, "quat norm"))
    if int(man[p.stem]["frames"]) != T: bad.append((p.stem, "frames vs manifest"))
    if len(d["weapon_arm_weight"]) != T: bad.append((p.stem, "mask length"))
print(f"{len(segs)} segments ({len(want)} listed), {frames} frames, {frames / 30 / 3600:.3f} h; problems: {len(bad)}")
for b in bad[:20]:
    print("  ", b)
sys.exit(1 if bad or len(segs) != len(want) else 0)
