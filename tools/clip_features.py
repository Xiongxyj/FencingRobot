"""Per-clip features of the converted fencing references (CPU only), written to build/fencing/clip_features.csv.

For every clip (all frames unless --stride): length; reference self-contact on the epee robot (fist / guard / blade vs
the robot's other bodies, and blade tip below the floor: frames and worst depth); speeds (joint speed, tip speed, root
speed); posture (lowest pelvis height); masks (weapon-arm weight, untracked joints, unreliable legs); joints beyond the
soft limits. The rigid-blade geometry is used for contact (the soft blade has the same shape).
usage (official env, from ~/fencing/supermimic):
  CUDA_VISIBLE_DEVICES= uv run --frozen python ../FencingRobot/tools/clip_features.py --motion-file ../FencingRobot/build/fencing/npz
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fencingrobot.robot_spec import get_g1_epee_spec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--motion-file", required=True)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--out", default=None)
A = ap.parse_args()
m = get_g1_epee_spec().compile()
d = mujoco.MjData(m)
wrist = m.body("right_wrist_yaw_link").id
weapon = {i for i in range(m.ngeom) if m.geom(i).name in ("right_hand_collision", "epee_guard_collision", "epee_blade_collision")}
col = [i for i in range(m.ngeom) if m.geom(i).name.endswith("_collision")]
for i in range(m.ngeom):   # mjlab FULL_COLLISION: only *_collision geoms collide
  m.geom_contype[i] = m.geom_conaffinity[i] = int(i in col)
lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * 0.9      # soft limits: factor 0.9 as in mjlab's G1 cfg
tip_id = m.site("epee_tip").id
rows = []
files = sorted(glob.glob(str(Path(A.motion_file) / "*.npz")))
for n, f in enumerate(files):
  with np.load(f) as z:
    fps = float(z["fps"][0])
    pos, quat, jp, jv = z["body_pos_w"][:, 0], z["body_quat_w"][:, 0], z["joint_pos"], z["joint_vel"]
    rv = z["body_lin_vel_w"][:, 0]
    w, nt, lu = z["fr_weapon_arm_weight"], z["fr_no_track_mask"], z["fr_legs_unreliable"]
  T = len(jp)
  tip = np.zeros((T, 3)); body_depth = np.zeros(T); partner: dict[str, int] = {}
  for t in range(0, T, A.stride):
    d.qpos[:3], d.qpos[3:7], d.qpos[7:] = pos[t], quat[t], jp[t]
    mujoco.mj_kinematics(m, d)
    tip[t] = d.site_xpos[tip_id]
    mujoco.mj_collision(m, d)
    worst, who = 0.0, None
    for c in d.contact[: d.ncon]:
      if (c.geom1 in weapon) != (c.geom2 in weapon) and c.dist < 0:
        other = c.geom2 if c.geom1 in weapon else c.geom1
        if m.geom_bodyid[other] == wrist:
          continue
        if -c.dist > worst:
          worst, who = -c.dist, m.geom(other).name.replace("_collision", "")
    body_depth[t] = worst
    if who:
      partner[who] = partner.get(who, 0) + 1
  s = slice(0, T, A.stride)
  tipv = np.linalg.norm(np.gradient(tip[s], A.stride / fps, axis=0), axis=1) if T > 2 * A.stride else np.zeros(1)
  floor_depth = np.clip(0.004 - tip[s, 2], 0, None)
  js = np.abs(jv)
  rows.append({
    "clip": Path(f).stem, "frames": T, "seconds": round(T / fps, 2),
    "selfcol_frames": int((body_depth[s] > 0).sum()), "selfcol_frac": round(float((body_depth[s] > 0).mean()), 4),
    "selfcol_max_mm": round(float(body_depth.max() * 1000), 1),
    "selfcol_with": "|".join(k for k, _ in sorted(partner.items(), key=lambda kv: -kv[1])[:3]),
    "floor_frames": int((floor_depth > 0).sum()), "floor_frac": round(float((floor_depth > 0).mean()), 4),
    "floor_max_mm": round(float(floor_depth.max() * 1000), 1),
    "joint_speed_p99": round(float(np.percentile(js.max(1), 99)), 2), "joint_speed_max": round(float(js.max()), 2),
    "arm_speed_p99": round(float(np.percentile(js[:, 22:].max(1), 99)), 2),
    "leg_speed_p99": round(float(np.percentile(js[:, :12].max(1), 99)), 2),
    "tip_speed_p99": round(float(np.percentile(tipv, 99)), 2), "tip_speed_mean": round(float(tipv.mean()), 2),
    "root_speed_p99": round(float(np.percentile(np.linalg.norm(rv[:, :2], axis=1), 99)), 2),
    "root_travel_m": round(float(np.linalg.norm(pos[-1, :2] - pos[0, :2])), 2),
    "pelvis_z_min": round(float(pos[:, 2].min()), 3),
    "arm_weight_mean": round(float(w.mean()), 4), "no_track_frac": round(float(nt.any(1).mean()), 4),
    "legs_unreliable_frac": round(float(lu.mean()), 4),
    "beyond_soft_limit_frac": round(float((np.abs(jp - mid) > half).any(1).mean()), 4),
  })
  if n % 100 == 0:
    print(n, len(files), flush=True)
out = Path(A.out) if A.out else Path(A.motion_file).parent / "clip_features.csv"
with open(out, "w", newline="") as fh:
  wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
  wr.writeheader()
  wr.writerows(rows)
print("saved", out, len(rows), "clips")
