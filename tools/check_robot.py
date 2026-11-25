"""Check the training robot (G1 + Revo2 fixed grasp + epee merged into right_wrist_yaw_link).
 1. build: mjlab's G1 spec (or --g1-xml) + robot/grasp/grasp_spec.json; body / joint layout must equal the stock G1
 2. cross-check vs the standalone model built on Windows (robot/g1_revo2_epee_menagerie): epee tip and wrist-subtree COM
    on random joint configurations (<= 0.1 mm)
 3. static gravity torque of the right arm on sampled frames of data/fencing vs the actuator limits (stock G1 vs epee)
usage (inside the SuperMimic uv env):  python tools/check_robot.py [--g1-xml PATH] [--frames 5000]"""
import argparse, sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fencingrobot.data import G1_JOINT_NAMES, load_segment, segments
from fencingrobot.robot_spec import apply_grasp

ap = argparse.ArgumentParser(); ap.add_argument("--g1-xml", type=Path, default=None); ap.add_argument("--frames", type=int, default=5000)
ap.add_argument("--seed", type=int, default=20261110); A = ap.parse_args()
if A.g1_xml is None:
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
    A.g1_xml = Path(G1_XML)
print("G1 xml:", A.g1_xml)
stock = mujoco.MjSpec.from_file(str(A.g1_xml)).compile()
spec = mujoco.MjSpec.from_file(str(A.g1_xml)); apply_grasp(spec); m = spec.compile(); d = mujoco.MjData(m)
names = [m.joint(i).name for i in range(1, m.njnt)]
ok = (m.nbody == stock.nbody and m.nq == stock.nq == 36 and tuple(names) == G1_JOINT_NAMES)
wy = m.body("right_wrist_yaw_link").id
print(f"1. bodies {stock.nbody} -> {m.nbody}, nq {m.nq}, joint order ok {tuple(names) == G1_JOINT_NAMES}; total mass "
      f"{stock.body_subtreemass[0]:.3f} -> {m.body_subtreemass[0]:.3f} kg (wrist subtree {m.body_subtreemass[wy]:.4f}) -> {'PASS' if ok else 'FAIL'}")

ref_xml = ROOT / "robot" / "g1_revo2_epee_menagerie" / "g1_revo2_epee.xml"
if ref_xml.exists():
    r = mujoco.MjModel.from_xml_path(str(ref_xml)); dr = mujoco.MjData(r); rng = np.random.default_rng(0); e1 = e2 = 0.0
    for _ in range(200):
        q = np.zeros(36); q[2] = 0.8; q[3] = 1; q[7:] = rng.uniform(m.jnt_range[1:, 0], m.jnt_range[1:, 1])
        d.qpos[:] = q; dr.qpos[:] = q; mujoco.mj_forward(m, d); mujoco.mj_forward(r, dr)
        e1 = max(e1, np.linalg.norm(d.site_xpos[m.site("epee_tip").id] - dr.site_xpos[r.site("epee_tip").id]))
        e2 = max(e2, np.linalg.norm(d.subtree_com[wy] - dr.subtree_com[r.body("right_wrist_yaw_link").id]))
    print(f"2. vs standalone model: epee tip {e1 * 1000:.3f} mm, wrist-subtree COM {e2 * 1000:.3f} mm -> {'PASS' if max(e1, e2) < 1e-4 else 'CHECK'}")

arm = [j for j in G1_JOINT_NAMES if j.startswith("right_") and ("shoulder" in j or "elbow" in j or "wrist" in j)]
try:
    from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as C
    lim = {}
    for a in C.get_g1_robot_cfg().articulation.actuators:
        import re
        for j in arm:
            if any(re.fullmatch(p, j) for p in a.target_names_expr):
                lim[j] = a.effort_limit
except Exception as ex:  # noqa: BLE001
    print("   (mjlab not importable:", repr(ex)[:60], ") -> Unitree motor table: 5020 = 25 N m, 4010 (wrist pitch / yaw) = 5 N m")
    lim = {j: 5.0 if ("wrist_pitch" in j or "wrist_yaw" in j) else 25.0 for j in arm}
rng = np.random.default_rng(A.seed); segs = segments(); tau = {k: {j: [] for j in arm} for k in ("stock", "epee")}
ds = mujoco.MjData(stock)
# Pure gravity torque: disable contacts, otherwise reference frames that penetrate the floor / self-collide (~0.6 %)
# add penalty contact forces to qfrc_inverse (up to ~1e3 N m) and show up as spurious over-limit frames.
for M in (stock, m):
    M.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
for p in rng.choice(segs, min(len(segs), A.frames // 20 + 1), replace=False):
    Q = load_segment(p)["qpos"].astype(float)
    for t in rng.choice(len(Q), min(len(Q), 20), replace=False):
        for k, M, D in (("stock", stock, ds), ("epee", m, d)):
            D.qpos[:] = Q[t]; D.qvel[:] = 0; D.qacc[:] = 0; mujoco.mj_inverse(M, D)
            for j in arm:
                tau[k][j].append(abs(D.qfrc_inverse[M.jnt_dofadr[M.joint(j).id]]))
print("3. static gravity torque on sampled frames (|tau| p50 / p99, N m) vs effort limit")
for j in arm:
    s, e = np.array(tau["stock"][j]), np.array(tau["epee"][j])
    print(f"   {j:28s} limit {lim.get(j, float('nan')):6.1f}  stock {np.median(s):5.2f} / {np.percentile(s, 99):5.2f}"
          f"  epee {np.median(e):5.2f} / {np.percentile(e, 99):5.2f}  over-limit {np.mean(e > lim.get(j, np.inf)):.4f}")
sys.exit(0 if ok else 1)
