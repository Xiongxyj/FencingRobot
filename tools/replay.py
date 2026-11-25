"""Kinematic replay of one motion segment on the epee robot (MuJoCo viewer, real time).
usage: python tools/replay.py 0428_10_f0_s0 [--speed 0.5] [--g1-xml PATH]   (no display: use --frames-only to print tip path)"""
import argparse, sys, time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fencingrobot.data import PKG, load_segment
from fencingrobot.robot_spec import apply_grasp

ap = argparse.ArgumentParser(); ap.add_argument("segment"); ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--g1-xml", type=Path, default=None); ap.add_argument("--frames-only", action="store_true"); A = ap.parse_args()
if A.g1_xml is None:
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
    A.g1_xml = Path(G1_XML)
spec = mujoco.MjSpec.from_file(str(A.g1_xml)); apply_grasp(spec)
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], rgba=[0.3, 0.3, 0.35, 1])
m = spec.compile(); d = mujoco.MjData(m)
seg = load_segment(PKG / "segments" / f"{A.segment}.npz"); Q, fps = seg["qpos"].astype(float), float(seg["fps"])
tip = m.site("epee_tip").id
if A.frames_only:
    for t in range(0, len(Q), max(1, int(fps) // 5)):
        d.qpos[:] = Q[t]; mujoco.mj_kinematics(m, d); print(t, np.round(d.site_xpos[tip], 3))
    sys.exit()
import mujoco.viewer
with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        for t in range(len(Q)):
            t0 = time.time(); d.qpos[:] = Q[t]; mujoco.mj_forward(m, d); v.sync()
            time.sleep(max(0.0, 1.0 / (fps * A.speed) - (time.time() - t0)))
            if not v.is_running():
                break
