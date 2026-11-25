"""Pick the 40-clip evaluation set from the per-clip features (tools/clip_features.py) and a full-set evaluation
(tools/eval_official.py over all clips). The rule below was fixed before the full-set returns were looked at.

Only clips of >= 100 frames (2 s). Groups are filled in this order; a clip already taken is skipped and the next one in
that ranking is used.
  A  reference self-contact (weapon vs own body), 8: 4 deepest penetration, 2 most contact frames, 2 deepest among
     those whose main contact partner is the left hand
  B  reference blade below the floor, 4: 2 deepest, 2 most frames
  C  policy return on the full-set evaluation, 12: 6 lowest and 6 highest norm_return (a failed clip ranks by its
     norm_return like any other)
  D  speed, 6: 3 highest tip_speed_p99, 3 highest joint_speed_p99
  E  posture / travel, 5: 3 lowest pelvis height, 2 largest root travel
  F  length, 2: the 2 longest clips
  G  masks, 2: lowest weapon-arm weight; highest unreliable-legs fraction
  H  typical, 1: the clip closest to the median in tip_speed_p99, seconds and norm_return (sum of absolute z-scores)
usage: python3 tools/select_eval_clips.py --features build/fencing/clip_features.csv --eval ../runs/<eval>.json --out build/fencing/eval40.csv
"""
import argparse
import csv
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--eval", required=True)
ap.add_argument("--out", required=True)
A = ap.parse_args()
R = list(csv.DictReader(open(A.features)))
for r in R:
  for k in r:
    if k not in ("clip", "selfcol_with"):
      r[k] = float(r[k])
ev = {c["clip"] if "clip" in c else c["name"]: c for c in json.load(open(A.eval))["clips"]}
for r in R:
  e = ev[r["clip"]]
  r["norm_return"], r["failed"] = float(e["norm_return"]), int(bool(e["failed"]))
  for k in ("tip_err_m", "wrist_ori_err_deg", "drift_xy_end_m", "wrist_pitch_sat_frac"):
    if k in e:
      r[k] = e[k]
R = [r for r in R if r["frames"] >= 100]
taken: dict[str, tuple[str, str]] = {}


def take(group, why, key, n, reverse=True, pool=None):
  got = 0
  for r in sorted(pool if pool is not None else R, key=lambda r: r[key], reverse=reverse):
    if got == n:
      break
    if r["clip"] not in taken:
      taken[r["clip"]] = (group, why)
      got += 1
  assert got == n, (group, why)


sc = [r for r in R if r["selfcol_frames"] > 0]
take("A", "self-contact: deepest", "selfcol_max_mm", 4, pool=sc)
take("A", "self-contact: most frames", "selfcol_frames", 2, pool=sc)
take("A", "self-contact: left hand", "selfcol_max_mm", 2, pool=[r for r in sc if r["selfcol_with"].split("|")[0] == "left_hand"])
fl = [r for r in R if r["floor_frames"] > 0]
take("B", "blade below floor: deepest", "floor_max_mm", 2, pool=fl)
take("B", "blade below floor: most frames", "floor_frames", 2, pool=fl)
take("C", "return: lowest", "norm_return", 6, reverse=False)
take("C", "return: highest", "norm_return", 6)
take("D", "speed: tip", "tip_speed_p99", 3)
take("D", "speed: joints", "joint_speed_p99", 3)
take("E", "posture: lowest pelvis", "pelvis_z_min", 3, reverse=False)
take("E", "travel: largest", "root_travel_m", 2)
take("F", "length: longest", "frames", 2)
take("G", "mask: lowest weapon-arm weight", "arm_weight_mean", 1, reverse=False)
take("G", "mask: most unreliable legs", "legs_unreliable_frac", 1)
z = lambda k: np.abs((np.array([r[k] for r in R]) - np.median([r[k] for r in R])) / (np.std([r[k] for r in R]) + 1e-9))  # noqa: E731
for r, s in zip(R, z("tip_speed_p99") + z("seconds") + z("norm_return")):
  r["_typ"] = float(s)
take("H", "typical", "_typ", 1, reverse=False)
assert len(taken) == 40
cols = ["group", "why", "clip", "seconds", "norm_return", "failed", "tip_err_m", "wrist_ori_err_deg", "drift_xy_end_m", "wrist_pitch_sat_frac", "selfcol_frames", "selfcol_max_mm", "selfcol_with", "floor_frames",
        "floor_max_mm", "tip_speed_p99", "joint_speed_p99", "pelvis_z_min", "root_travel_m", "arm_weight_mean", "legs_unreliable_frac"]
by = {r["clip"]: r for r in R}
with open(A.out, "w", newline="") as fh:
  w = csv.writer(fh)
  w.writerow(cols)
  for c, (g, why) in taken.items():
    w.writerow([g, why, c] + [by[c][k] for k in cols[3:]])
print("saved", A.out)
