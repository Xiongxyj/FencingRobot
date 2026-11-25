"""Check our mjlab_research additions (mjlab v1.5.0 + setup/mjlab_research.patch).

 1. ForwardKinematics vs MuJoCo on the G1 at random states: body origin position,
    orientation, linear and angular velocity (MuJoCo: xpos, xquat, mj_objectVelocity).
 2. Gradients of ForwardKinematics: float64 gradcheck on a small batch.
 3. MotionLoader on real clips (optional --npz-dir): clip ends, and FK bodies vs the
    bodies stored in the npz (the npz were written by another converter / mujoco
    version, so this checks the data, not the FK).
usage (official supermimic env):  python tools/check_mjlab_research.py [--npz-dir DIR]
"""

import argparse
import glob

import mujoco
import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.forward_kinematics import ForwardKinematics

ap = argparse.ArgumentParser()
ap.add_argument("--npz-dir", default=None)
ap.add_argument("--samples", type=int, default=200)
A = ap.parse_args()

m = mujoco.MjModel.from_xml_path(str(G1_XML))
d = mujoco.MjData(m)
root = next(b for b in range(m.nbody) if m.body_jntnum[b] and m.jnt_type[m.body_jntadr[b]] == 0)
body_ids = [b for b in range(m.nbody) if b >= root]  # G1 xml: one robot tree after world
hinges = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
q_adr = [int(m.jnt_qposadr[j]) for j in hinges]
v_adr = [int(m.jnt_dofadr[j]) for j in hinges]
names = unitree_g1_flat_tracking_env_cfg().commands["motion"].body_names
rows = [body_ids.index(m.body(n).id) for n in names]
print(f"G1 {G1_XML}\n  bodies {len(body_ids)} (root {m.body(root).name}), hinges {len(hinges)}, tracked {len(rows)}")

fk64 = ForwardKinematics(m, body_ids, q_adr, rows, dtype=torch.float64)
rng = np.random.default_rng(0)
err = np.zeros(4)
for _ in range(A.samples):
  d.qpos[:] = 0
  d.qpos[0:3] = rng.normal(0, 1, 3)
  qr = rng.normal(size=4); d.qpos[3:7] = qr / np.linalg.norm(qr)
  lo, hi = m.jnt_range[hinges].T
  d.qpos[q_adr] = rng.uniform(lo, hi)
  d.qvel[:] = rng.normal(0, 2, m.nv)
  mujoco.mj_forward(m, d)
  # MuJoCo free joint: qvel[0:3] world linear vel of the root origin, qvel[3:6] local ang vel.
  rlin = d.qvel[0:3].copy()
  rang = np.zeros(3); mujoco.mju_rotVecQuat(rang, d.qvel[3:6], d.qpos[3:7])
  T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float64)[None]
  pos, quat, lin, ang = (t[0].numpy() for t in fk64(T(d.qpos[0:3]), T(d.qpos[3:7]), T(d.qpos[q_adr]), T(rlin), T(rang), T(d.qvel[v_adr])))
  for k, r in enumerate(rows):
    b = body_ids[r]
    # mjOBJ_XBODY: twist at the body frame origin (mjOBJ_BODY would give it at the COM).
    vel = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, vel, 0)
    qd = d.xquat[b] * (1 if np.dot(d.xquat[b], quat[k]) >= 0 else -1)
    err = np.maximum(err, [np.abs(pos[k] - d.xpos[b]).max(), np.abs(quat[k] - qd).max(),
                           np.abs(lin[k] - vel[3:]).max(), np.abs(ang[k] - vel[:3]).max()])
ok1 = err.max() < 1e-9
print(f"1. FK vs MuJoCo ({A.samples} random states, float64) max |err|: pos {err[0]:.1e} quat {err[1]:.1e} "
      f"lin_vel {err[2]:.1e} ang_vel {err[3]:.1e} -> {'PASS' if ok1 else 'FAIL'}")

gen = torch.Generator().manual_seed(0)
args = [torch.randn(3, n, dtype=torch.float64, generator=gen, requires_grad=True) for n in (3, 4, len(hinges), 3, 3, len(hinges))]
ok2 = torch.autograd.gradcheck(lambda *a: tuple(fk64(a[0], a[1] / a[1].norm(dim=-1, keepdim=True), *a[2:])), args, eps=1e-6, atol=1e-6)
print(f"2. gradcheck -> {'PASS' if ok2 else 'FAIL'}")

ok3 = True
if A.npz_dir:
  files = sorted(glob.glob(f"{A.npz_dir}/*.npz"))
  fk = ForwardKinematics(m, body_ids, q_adr, rows)
  ml = MotionLoader(files, torch.tensor(rows), fk=fk)
  lens = [np.load(f)["joint_pos"].shape[0] for f in files]
  ok3 = ml.motion_ends.tolist() == np.cumsum(lens).tolist() and ml.time_step_total == sum(lens)
  last = ml.clip_last_frame(torch.tensor([0, lens[0] - 1, lens[0]]))
  ok3 &= last.tolist() == [lens[0] - 1, lens[0] - 1, lens[0] + lens[1] - 1]
  pe, re, ve = [], [], []
  for f, s in zip(files, np.r_[0, np.cumsum(lens)[:-1]]):
    z = np.load(f); n = z["joint_pos"].shape[0]
    stored_names = list(z["body_names"]) if "body_names" in z.files else None
    idx = [stored_names.index(x) for x in names] if stored_names else rows
    pe.append(np.abs(ml.body_pos_w[s:s + n].numpy() - z["body_pos_w"][:, idx]).max())
    qs = z["body_quat_w"][:, idx]; qf = ml.body_quat_w[s:s + n].numpy()
    re.append((1 - np.abs((qs * qf).sum(-1))).max())
    ve.append(np.abs(ml.body_lin_vel_w[s:s + n].numpy() - z["body_lin_vel_w"][:, idx]).max())
  print(f"3. MotionLoader: {len(files)} clips, {ml.time_step_total} frames, motion_ends ok {ok3}; "
        f"FK vs stored bodies max: pos {max(pe) * 1000:.2f} mm, 1-|<q,q'>| {max(re):.1e}, lin_vel {max(ve):.3f} m/s")
raise SystemExit(0 if (ok1 and ok2 and ok3) else 1)
