"""Soft blade contact (fencingrobot/robot_spec.soften_blade): parameters in the compiled task model and the contact
stiffness they realise, on CPU MuJoCo and on mujoco_warp.

Test: everything except one blade segment and the floor is made non-colliding, gravity and actuators off; the robot is placed so that
the tip-side end of that segment is r below the floor (blade 30 degrees below horizontal); an external force F pushes
that point down. The F at which contact force = F (static balance) gives the realised stiffness F / r, compared with the
bending stiffness at that point, 3 EI / x^3, and with the segment's own target (its centre).
usage (official env, from ~/fencing/supermimic):
  uv run --frozen python ../FencingRobot/tools/check_blade_contact.py --motion-file ../FencingRobot/build/fencing/npz
"""
import argparse
import sys
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mjlab.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

from fencingrobot import robot_spec as RS  # noqa: E402
from fencingrobot import sm_task  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--motion-file", required=True)
ap.add_argument("--debug", action="store_true")
A = ap.parse_args()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
cfg = load_env_cfg(sm_task.TASK_ID)
cfg.commands["motion"].motion_file = A.motion_file
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device=device)
m = env.sim.mj_model
nm = lambda i: m.geom(i).name.split("/")[-1]  # noqa: E731
blade = [i for i in range(m.ngeom) if nm(i).startswith("epee_blade")]
bid = next(i for i in range(m.nbody) if m.body(i).name.endswith("right_wrist_yaw_link"))
ok = True
print(f"body_invweight0 of the wrist link: {float(m.body_invweight0[bid, 0]):.4f} 1/kg; robot_spec.BLADE_INVWEIGHT {RS.BLADE_INVWEIGHT}")
for i in blade:
  print(f"    {nm(i)}: priority {int(m.geom_priority[i])} solref {m.geom_solref[i].round(1).tolist()} solimp d {m.geom_solimp[i][0]:.2f} condim {int(m.geom_condim[i])}")
env.close()

segs = RS.blade_segments(RS.load_grasp_spec())
floor = next(i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE)
m.opt.gravity[:] = 0
m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION   # no PD torques: the only forces are F and the contact
ct, ca = m.geom_contype.copy(), m.geom_conaffinity.copy()
d = mujoco.MjData(m)
rng = np.random.default_rng(0)
axis = segs[0]["p1"] - segs[0]["p0"]
axis /= np.linalg.norm(axis)


def contact_force(F, pc, warp):
  d.qfrc_applied[:] = 0
  mujoco.mj_applyFT(m, d, np.array([0, 0, -F]), np.zeros(3), pc, bid, d.qfrc_applied)
  if not warp:
    mujoco.mj_forward(m, d)
    return float(sum(d.efc_force[c.efc_address] for c in d.contact if c.efc_address >= 0))
  wm, wd = mjw.put_model(m), mjw.put_data(m, d, nworld=1)
  mjw.forward(wm, wd)
  n = int(wd.nefc.numpy()[0])
  return float(wd.efc.force.numpy()[0, :n].sum())  # the only constraints are this contact (limits inactive, see below)


print("realised static stiffness F / r (N/m) at the tip-side end of each segment, r = 1 / 3 / 10 cm:")
print("  seg  x(m)   3EI/x^3   segment target   CPU MuJoCo              mujoco_warp")
worst = 0.0
for si, s in enumerate(segs):
  m.geom_contype[:] = 0
  m.geom_conaffinity[:] = 0
  for g in (blade[si], floor):
    m.geom_contype[g], m.geom_conaffinity[g] = ct[g], ca[g]
  res = {False: [], True: []}
  for r in (0.01, 0.03, 0.10):
    mujoco.mj_resetData(m, d)
    d.qpos[7:] = 0.5 * (m.jnt_range[1:, 0] + m.jnt_range[1:, 1])     # mid range: no joint limit active
    for _ in range(5000):
      q = rng.normal(size=4)
      d.qpos[3:7] = q / np.linalg.norm(q)
      mujoco.mj_kinematics(m, d)
      if abs((d.xmat[bid].reshape(3, 3) @ axis)[2] + 0.5) < 0.02:
        break
    end = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ s["p1"]
    d.qpos[2] += -(end[2] - s["radius"]) - r
    mujoco.mj_kinematics(m, d)
    pc = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ s["p1"] - np.array([0, 0, s["radius"]])
    for warp in (False, True):
      f0, f1 = contact_force(0.0, pc, warp), contact_force(1.0, pc, warp)   # f is affine in F: solve f(F) = F
      F = f0 / (1.0 - (f1 - f0))
      res[warp].append(F / r)
      if not warp and A.debug: print('dbg', si, r, 'ncon', d.ncon, [(nm(c.geom1), nm(c.geom2), round(c.dist, 4)) for c in d.contact], 'f0', f0, 'f1', f1, 'nefc', d.nefc)
  x = float(np.linalg.norm(s["p1"] - segs[0]["p0"]))
  worst = max(worst, max(abs(k / s["k"] - 1) for v in res.values() for k in v))
  fmt = lambda v: " / ".join(f"{k:8.1f}" for k in v)  # noqa: E731
  print(f"   {si}  {x:.3f}  {3 * RS.BLADE_EI / x ** 3:8.1f}   {s['k']:9.1f}     {fmt(res[False])}   {fmt(res[True])}")
ok2 = worst < 0.05
print(f"[{'ok' if ok2 else 'FAIL'}] realised stiffness vs the segment target: worst relative difference {worst * 100:.1f} %")
sys.exit(0 if ok and ok2 else 1)
