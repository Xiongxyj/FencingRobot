"""Checks for the fencing task on the official SuperMimic code (fencingrobot/sm_task.py).

 1. weighted world-model reward == official tracking_reward at weight 1 (random inputs).
 2. Task env builds with the converted fencing npz: bodies / joints identical to the stock G1, epee sites present,
    merged wrist-link mass, wrist effort limits (Unitree: pitch / yaw 5 N m), weight-table statistics.
 3. Simulator reward terms == weighted world-model reward evaluated on the same simulator state (the two rewards the
    policy is trained / scored with must agree), on reset states with the usual reset noise.
 4. Reference motions on the epee robot (CPU MuJoCo, every --stride-th frame): contacts of the hand / guard / blade
    geoms with the body or the floor (depth), i.e. what a reset into that frame starts with.
usage (official env, from ~/fencing/supermimic):
  uv run --frozen python ../FencingRobot/tools/check_fencing_task.py --motion-file ../FencingRobot/build/fencing/npz
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mjlab.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.utils.lab_api.math import matrix_from_quat  # noqa: E402

from fencingrobot import sm_task  # noqa: E402
from fencingrobot.robot_spec import get_g1_epee_spec  # noqa: E402
from supermimic.utils.tracking_reward import tracking_reward  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--motion-file", required=True)
ap.add_argument("--num-envs", type=int, default=256)
ap.add_argument("--stride", type=int, default=10)
ap.add_argument("--skip-contacts", action="store_true")
A = ap.parse_args()
ok = True


def report(name, passed, detail):
  global ok
  ok &= bool(passed)
  print(f"[{'ok' if passed else 'FAIL'}] {name}: {detail}")


# 1 ---------------------------------------------------------------------------------------------------------------
g = torch.Generator().manual_seed(0)
B, NB, J = 64, 14, 29
r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)  # noqa: E731
quat = lambda *s: torch.nn.functional.normalize(r(*s, 4), dim=-1)  # noqa: E731
args = (torch.cat([r(B, 3), quat(B)], -1), r(B, NB, 3), matrix_from_quat(quat(B, NB)), r(B, NB, 3), r(B, NB, 3),
        r(B, 2 * J), r(B, J), r(B, J), r(B, NB, 3), quat(B, NB), r(B, NB, 3), r(B, NB, 3),
        torch.stack([-torch.ones(J, dtype=torch.float64), torch.ones(J, dtype=torch.float64)], -1), 0.02)
d = (tracking_reward(*args) - sm_task.weighted_tracking_reward(*args, torch.ones(B, NB, dtype=torch.float64))).abs().max()
report("weighted WM reward == official at w = 1", d < 1e-12, f"max |diff| {d:.2e}")
w = torch.rand(B, NB, generator=g, dtype=torch.float64)
d2 = (tracking_reward(*args) - sm_task.weighted_tracking_reward(*args, w)).abs().max()
report("weights change the reward", d2 > 1e-6, f"max |diff| {d2:.2e} with random weights")

# 2 ---------------------------------------------------------------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
cfg = load_env_cfg(sm_task.TASK_ID)
cfg.commands["motion"].motion_file = A.motion_file
cfg.commands["motion"].sampling_mode = "uniform"
cfg.scene.num_envs = A.num_envs
env = ManagerBasedRlEnv(cfg=cfg, device=device)
robot = env.scene["robot"]
cmd = env.command_manager.get_term("motion")
stock = load_env_cfg("SuperMimic-GenCoord-G1").scene.entities["robot"].spec_fn().compile()
m = env.sim.mj_model
names = lambda mm, t, n: [mujoco.mj_id2name(mm, t, i) for i in range(n)]  # noqa: E731
strip = lambda xs: [x.split("/")[-1] for x in xs if x]  # noqa: E731
report("bodies identical to stock G1", strip(names(stock, mujoco.mjtObj.mjOBJ_BODY, stock.nbody))[1:] == list(robot.body_names),
       f"{len(robot.body_names)} bodies")
report("joints identical to stock G1", strip(names(stock, mujoco.mjtObj.mjOBJ_JOINT, stock.njnt))[1:] == list(robot.joint_names),
       f"{len(robot.joint_names)} joints, nq {m.nq}")
report("epee sites", {"epee_tip", "epee_guard_apex"} <= set(robot.site_names), str([s for s in robot.site_names if "epee" in s]))
bid = next(i for i in range(m.nbody) if (m.body(i).name or "").endswith("right_wrist_yaw_link"))
report("merged wrist link mass", abs(m.body_mass[bid] - 1.2446) < 1e-3, f"{m.body_mass[bid]:.4f} kg (stock "
       f"{stock.body_mass[stock.body('right_wrist_yaw_link').id]:.4f})")
lim = {}
for a in range(m.nu):
  jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a, 0]).split("/")[-1]
  if jn.startswith("right_") and any(k in jn for k in ("shoulder", "elbow", "wrist")):
    lim[jn] = float(m.actuator_forcerange[a, 1])
report("weapon-arm effort limits (N m)", lim.get("right_wrist_pitch_joint") == 5.0 and lim.get("right_wrist_yaw_joint") == 5.0,
       json.dumps(lim))
tab = cmd.motion.fr_body_w
bn = cfg.commands["motion"].body_names
print("    clips", cmd.motion.motion_ends.numel(), "frames", cmd.motion.time_step_total, f"({cmd.motion.time_step_total / 50 / 3600:.3f} h)")
print("    mean weight per body:", {n: round(float(tab[:, i].mean()), 4) for i, n in enumerate(bn) if float(tab[:, i].mean()) < 1})
print(f"    nconmax {env.sim.cfg.nconmax} njmax {env.sim.cfg.njmax}")

# 3 ---------------------------------------------------------------------------------------------------------------
env.reset()
env.sim.forward()
cmd.update_relative_body_poses()
rm = env.reward_manager
sim_r = torch.zeros(env.num_envs, device=device)
track = ("motion_global_root_pos", "motion_global_root_ori") + sm_task.BODY_REWARD_TERMS
for n in track:
  t = cfg.rewards[n]
  sim_r += t.weight * t.func(env, **t.params)
org = env.scene.env_origins
root = torch.cat([robot.data.root_link_pos_w - org, robot.data.root_link_quat_w], -1)
big = torch.tensor([-1e9, 1e9], device=device).expand(29, 2)
zero = torch.zeros(env.num_envs, 29, device=device)
wm_r = sm_task.weighted_tracking_reward(
  root, cmd.robot_body_pos_w - org[:, None], matrix_from_quat(cmd.robot_body_quat_w), cmd.robot_body_lin_vel_w,
  cmd.robot_body_ang_vel_w, torch.cat([zero, zero], -1), zero, zero, *cmd.motion.body_state(cmd.time_steps), big, 1.0,
  tab[cmd.time_steps])
d = (sim_r - wm_r).abs().max()
frac = float((tab[cmd.time_steps].min(1).values < 1).float().mean())
report("simulator reward == world-model reward on the same state", d < 1e-4,
       f"max |diff| {d:.2e} over {env.num_envs} reset states (reward mean {sim_r.mean():.3f} of 5; {frac * 100:.0f} % of them masked)")
env.close()

# 4 ---------------------------------------------------------------------------------------------------------------
if not A.skip_contacts:
  mm = get_g1_epee_spec().compile()
  # collision setup as in mjlab's FULL_COLLISION: every *_collision geom collides
  dd = mujoco.MjData(mm)
  weapon = {i for i in range(mm.ngeom) if (mm.geom(i).name or "") in ("right_hand_collision", "epee_guard_collision")
            or (mm.geom(i).name or "").startswith("epee_blade")}
  files = sorted(glob.glob(str(Path(A.motion_file) / "*.npz")))
  n = 0
  stat = {"blade_floor": [], "weapon_body": []}
  per_geom: dict[str, int] = {}
  for f in files:
    with np.load(f) as z:
      pos, quat_, jp = z["body_pos_w"][::A.stride, 0], z["body_quat_w"][::A.stride, 0], z["joint_pos"][::A.stride]
    for t in range(len(jp)):
      dd.qpos[:3], dd.qpos[3:7], dd.qpos[7:] = pos[t], quat_[t], jp[t]
      mujoco.mj_kinematics(mm, dd)
      n += 1
      # floor: lowest point of the blade capsule / guard / fist (z of geom centre minus extent bound)
      tip = dd.site("epee_tip").xpos[2]
      if tip < 0.004:
        stat["blade_floor"].append(0.004 - tip)
      mujoco.mj_collision(mm, dd)
      worst = 0.0
      for c in dd.contact[: dd.ncon]:
        if (c.geom1 in weapon) != (c.geom2 in weapon) and c.dist < 0:
          other = c.geom2 if c.geom1 in weapon else c.geom1
          if mm.geom_bodyid[other] == mm.geom_bodyid[next(iter(weapon))]:
            continue
          worst = max(worst, -c.dist)
          k = f"{mm.geom(c.geom1).name} x {mm.geom(c.geom2).name}"
          per_geom[k] = per_geom.get(k, 0) + 1
      if worst > 0:
        stat["weapon_body"].append(worst)
  for k, v in stat.items():
    v = np.array(v)
    print(f"    {k}: {len(v)} / {n} sampled frames ({len(v) / n * 100:.2f} %)"
          + (f", depth median {np.median(v) * 1000:.1f} mm, max {v.max() * 1000:.1f} mm, > 2 cm: {(v > 0.02).sum()}" if len(v) else ""))
  for k, v in sorted(per_geom.items(), key=lambda kv: -kv[1])[:8]:
    print(f"      {k}: {v}")

print("ALL OK" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
