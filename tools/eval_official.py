"""Whole-clip benchmark for an official SuperMimic checkpoint (paper Appendix F protocol).

Port of SuperMimic-Reproduced's eval/benchmark.py to the official code (which only ships a
speed benchmark). Official code is used unmodified: env cfg from the task registry, world
model + actor rebuilt with build_world_model / build_actor, actor input built exactly like
the official viewer's sim step (scripts/viewers.py step_and_predict, "Actor step on
ground-truth obs").

Protocol (Appendix F): every clip is played once from its first frame with the
deterministic actor (no action noise); no observation noise; no domain randomization
(foot friction fixed, default 1.0; CoM offset 0; encoder bias 0; no pushes; no reset
noise). A clip FAILS if a non-time-out training termination (anchor height / tilt) fires
before the clip's last frame (motion_end).
Normalized return (declared before any result):
  norm_return       = sum of step rewards / (max_step_reward * (frames - 1))  [paper: max
                      possible sum over the full episode, so early termination scores low]
  norm_return_alive = sum of step rewards / (max_step_reward * steps survived)
                      [what SuperMimic-Reproduced's script reports; for comparison only]
max_step_reward = step_dt * sum of positive reward weights (= the official WorldModelEnv's
max_step_reward; 0.02 * 5 = 0.1 for the G1 tracking task).

usage (official env, from ~/fencing/supermimic):
  WANDB_MODE=offline uv run --frozen python ../FencingRobot/tools/eval_official.py \
    --run-dir logs/SuperMimic-GenCoord-G1/<time> --motion-file <npz | dir | glob> [--num-envs 40]
The runner config is read from the matching offline W&B run (wandb/offline-run-<time>-*/
files/config.yaml) unless --config is given.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mjlab.tasks  # noqa: F401
import torch
import tyro
import yaml
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fencingrobot.sm_task  # noqa: E402,F401  (registers Fencing-SuperMimic-GenCoord-G1)
from mjlab.utils.lab_api.math import (  # noqa: E402
  quat_apply, quat_error_magnitude, quat_inv, quat_mul, yaw_quat,
)
from supermimic.representations import extract_actor_goal
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import (
  SuperMimicRunnerCfg,
  build_actor,
  build_world_model,
)
from supermimic.tasks.task_spec import build_actor_obs_td


@dataclass
class EvalConfig:
  run_dir: Path
  """Training log dir with actor.pt and world_model.pt (logs/<task>/<time>)."""
  motion_file: str
  """npz file, directory of npz, glob, or comma-separated list (clips in sorted order)."""
  task_id: str = "SuperMimic-GenCoord-G1"
  config: Path | None = None
  """W&B config.yaml of the run; default: wandb/offline-run-<time>-*/files/config.yaml.
  Offline runs only write it when they finish; without it, --runner.* is used."""
  runner: SuperMimicRunnerCfg = field(default_factory=SuperMimicRunnerCfg)
  """Used only when no config.yaml is found: pass the same --runner.* architecture flags
  as in training (a mismatch fails loudly in load_state_dict)."""
  num_envs: int = 64
  foot_friction: float = 1.0
  device: str | None = None
  seed: int = 0
  out: Path | None = None


def _motion_files(spec: str) -> list[str]:
  files: list[str] = []
  for item in [s.strip() for s in spec.split(",") if s.strip()]:
    item = os.path.expanduser(item)
    if os.path.isdir(item):
      files += sorted(glob.glob(os.path.join(item, "*.npz")))
    elif glob.has_magic(item):
      files += sorted(glob.glob(item))
    else:
      files.append(item)
  if not files:
    raise FileNotFoundError(spec)
  return files


def _runner_cfg(cfg: EvalConfig) -> SuperMimicRunnerCfg:
  path = cfg.config
  if path is None:
    run = cfg.run_dir.parent.parent if cfg.run_dir.parent.name == "snapshots" else cfg.run_dir
    stamp = run.name.replace("-", "")  # log dir name like 2026-11-11_13-44-48
    hits = sorted(glob.glob(f"wandb/*run-{stamp}-*/files/config.yaml"))
    if not hits:
      print(f"[WARN] no W&B config.yaml for {stamp}; using --runner.* from the command line")
      return cfg.runner
    path = Path(hits[0])
  raw = yaml.safe_load(Path(path).read_text())
  print(f"[INFO] runner config from {path}")
  return load_runner_cfg(
    {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in raw.items()}
  )


class Bench:
  """Appendix F evaluation env + actor for an official checkpoint (shared by
  eval_official.py and record_official.py so both run exactly the same rollout).

  ``files`` are concatenated in order (clip i = files[i]); ``num_envs`` parallel envs.
  ``render`` builds the env with mjlab's offscreen renderer (``render_mode="rgb_array"``:
  robot + translucent reference ghost of ``viewer.env_idx``)."""

  def __init__(
    self, cfg: EvalConfig, files: list[str], num_envs: int,
    render: tuple[int, int] | None = None,
  ) -> None:
    configure_torch_backends()
    torch.manual_seed(cfg.seed)
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    runner_cfg = _runner_cfg(cfg)
    check_representation_matches(cfg.task_id, runner_cfg.representation)
    self.files, self.names = files, [Path(f).stem for f in files]

    # play=True: infinite episode (the clip end decides), no actor-obs corruption, no pushes.
    env_cfg = load_env_cfg(cfg.task_id, play=True)
    motion = env_cfg.commands["motion"]
    assert isinstance(motion, MotionCommandCfg)
    motion.motion_file = files
    motion.sampling_mode = "start"
    motion.pose_range, motion.velocity_range = {}, {}
    motion.joint_position_range = (0.0, 0.0)
    # Keep the DR events (the sim context the world model was trained with reads them)
    # but make them constant; the trained sim-context normalization comes with the WM.
    ev = env_cfg.events
    if "foot_friction" in ev:
      ev["foot_friction"].params["ranges"] = (cfg.foot_friction, cfg.foot_friction)
    if "base_com" in ev:
      ev["base_com"].params["ranges"] = {i: (0.0, 0.0) for i in ev["base_com"].params["ranges"]}
    if "encoder_bias" in ev:
      ev["encoder_bias"].params["bias_range"] = (0.0, 0.0)
    for name in [n for n in ev if n not in ("foot_friction", "base_com", "encoder_bias")]:
      print(f"[INFO] dropping event {name!r}"); ev.pop(name)
    env_cfg.scene.num_envs = num_envs
    render_mode = None
    if render is not None:
      env_cfg.viewer.height, env_cfg.viewer.width = render
      env_cfg.viewer.max_extra_envs = 0
      render_mode = "rgb_array"
    self.env = RslRlVecEnvWrapper(
      ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    )
    self.base = self.env.unwrapped
    self.n_env = self.base.num_envs

    wm = build_world_model(runner_cfg, self.env, device)
    wm.load_bundle(torch.load(cfg.run_dir / "world_model.pt", map_location=device))
    self.rep = wm.representation
    self.actor = build_actor(runner_cfg, wm, device)
    self.actor.load_state_dict(torch.load(cfg.run_dir / "actor.pt", map_location=device))
    self.actor.eval()
    self.actor.set_slot_mix({"motion": 1.0})
    self.proprio = self.rep.actor_obs_mode == "proprio"
    self.max_step = self.base.step_dt * sum(
      t.weight for t in self.base.cfg.rewards.values() if t.weight > 0.0
    )
    self.cmd = self.base.command_manager.get_term("motion")
    self.ends = self.cmd.motion.motion_ends.tolist()
    self.starts = [0] + self.ends[:-1]
    assert len(self.ends) == len(files)

    # Fencing diagnostics (only when the robot carries the epee): weapon wrist link, tip site, wrist actuators.
    robot = self.base.scene["robot"]
    self.has_epee = "epee_tip" in robot.site_names
    if self.has_epee:
      mj = self.base.sim.mj_model
      self.wrist_col = list(self.cmd.cfg.body_names).index("right_wrist_yaw_link")
      sid = next(i for i in range(mj.nsite) if (mj.site(i).name or "").endswith("epee_tip"))
      assert (mj.body(int(mj.site_bodyid[sid])).name or "").endswith("right_wrist_yaw_link")
      self.tip_local = torch.tensor(mj.site_pos[sid], dtype=torch.float32, device=device)
      acts = list(robot.actuator_names)
      self.wrist_act = [next(i for i, n in enumerate(acts) if j in n)
                        for j in ("right_wrist_pitch", "right_wrist_yaw")]
      ctrl = robot.indexing.ctrl_ids[self.wrist_act].tolist()
      self.wrist_limit = torch.tensor(mj.actuator_forcerange[ctrl, 1], dtype=torch.float32, device=device)

    self.env.reset()
    self.sim_ctx = wm.sim_context.read(self.base)
    self.history: torch.Tensor | None = None
    self.fresh = torch.ones(self.n_env, dtype=torch.bool, device=self.base.device)

  def start_clip(self, env_ids: list[int], clips: list[int]) -> None:
    """Put each env at its clip's first frame like a training reset (state = frame s,
    reference already advanced to s + 1), without touching the other envs."""
    last = len(self.files) - 1
    for e, c in zip(env_ids, clips, strict=True):
      self.cmd.reset_to_frame(torch.tensor([e], device=self.base.device), self.starts[min(c, last)])
    ids = torch.tensor(env_ids, device=self.base.device)
    self.cmd.time_steps[ids] += 1
    self.base.scene.write_data_to_sim()
    self.base.sim.forward()
    self.cmd.update_relative_body_poses()
    self.base.episode_length_buf[ids] = 0
    self.fresh[ids] = True

  @torch.no_grad()
  def act(self) -> torch.Tensor:
    """Deterministic actor on ground-truth obs, as the official viewer's sim step."""
    obs = self.env.get_observations()
    goal = extract_actor_goal(obs["goal"])
    sensor = self.rep.compute_actor_sensor(
      self.rep.represent(obs["full_state"]), self.sim_ctx,
      sim_actor_obs=obs["actor"] if self.proprio else None,
    )
    H = self.actor.history_length
    if self.history is None:
      self.history = sensor.unsqueeze(1).repeat(1, H, 1)
    else:
      self.history = torch.cat([self.history[:, 1:], sensor.unsqueeze(1)], dim=1)
      self.history[self.fresh] = sensor[self.fresh].unsqueeze(1).repeat(1, H, 1)
    self.fresh[:] = False
    return self.actor(build_actor_obs_td({"motion": goal}, self.history))

  def drift(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Robot root vs reference root after the last step (reference frame time_steps - 1,
    the one the step was scored against): horizontal distance (m) and |yaw diff| (deg).
    Meaningless for envs that were just reset."""
    m, ts = self.cmd.motion, (self.cmd.time_steps - 1).clamp(min=0)
    a = self.cmd.motion_anchor_body_index
    ref_pos = m.body_pos_w[ts, a] + self.base.scene.env_origins
    xy = (self.cmd.robot_anchor_pos_w[:, :2] - ref_pos[:, :2]).norm(dim=-1)

    def yaw(q: torch.Tensor) -> torch.Tensor:
      w, x, y, z = q.unbind(-1)
      return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    d = yaw(self.cmd.robot_anchor_quat_w) - yaw(m.body_quat_w[ts, a])
    d = torch.atan2(torch.sin(d), torch.cos(d)).abs() * 57.29577951308232
    return xy, d

  def fencing(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """After the last step, against the reference frame it was scored with (time_steps - 1), the reference
    re-anchored to the robot root xy + yaw exactly as the tracking reward does (so root drift is not counted):
    weapon-wrist orientation error (deg), epee-tip position error (m), wrist pitch / yaw actuator at >= 99 % of
    its effort limit (bool, (E, 2); sampled at the last physics substep of the control step), and the
    reference's weapon_arm_weight at that frame (0 where no_track masks the wrist)."""
    m, ts = self.cmd.motion, (self.cmd.time_steps - 1).clamp(min=0)
    a, k = self.cmd.motion_anchor_body_index, self.wrist_col
    org = self.base.scene.env_origins
    ref_pos, ref_q = m.body_pos_w[ts] + org[:, None], m.body_quat_w[ts]
    rob_q, rob_p = self.cmd.robot_body_quat_w[:, k], self.cmd.robot_body_pos_w[:, k]
    d = yaw_quat(quat_mul(self.cmd.robot_anchor_quat_w, quat_inv(ref_q[:, a])))
    anchor = self.cmd.robot_anchor_pos_w.clone()
    anchor[:, 2] = ref_pos[:, a, 2]
    tip = self.tip_local.expand(self.n_env, 3)
    ref_tip = anchor + quat_apply(d, ref_pos[:, k] + quat_apply(ref_q[:, k], tip) - ref_pos[:, a])
    ori = quat_error_magnitude(quat_mul(d, ref_q[:, k]), rob_q) * 57.29577951308232
    tip_err = (rob_p + quat_apply(rob_q, tip) - ref_tip).norm(dim=-1)
    tau = self.base.scene["robot"].data.actuator_force[:, self.wrist_act]
    sat = tau.abs() >= 0.99 * self.wrist_limit
    table = getattr(m, "fr_body_w", None)
    w = table[ts, k] if table is not None else torch.ones(self.n_env, device=ori.device)
    return ori, tip_err, sat, w

  def step(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One env step; returns (reward, failed, done). As the official runner, a failure
    on a step that also times out counts as the time-out."""
    with torch.no_grad():
      _, rew, _, _ = self.env.step(self.act())
    failed = self.base.reset_terminated & ~self.base.reset_time_outs
    return rew, failed, failed | self.base.reset_time_outs


def evaluate(cfg: EvalConfig) -> dict:
  files = _motion_files(cfg.motion_file)
  n_clips = len(files)
  b = Bench(cfg, files, min(cfg.num_envs, n_clips))
  n_env, starts, ends, names = b.n_env, b.starts, b.ends, b.names
  current = list(range(n_env))
  b.start_clip(list(range(n_env)), current)
  ret = torch.zeros(n_env, device=b.base.device)
  steps = torch.zeros(n_env, dtype=torch.long, device=b.base.device)
  results: dict[int, dict] = {}
  active = [c < n_clips for c in current]

  # Drift diagnostics (reported, not part of the score): root xy / yaw vs reference.
  dev = b.base.device
  xy_sum = torch.zeros(n_env, device=dev); xy_max = torch.zeros(n_env, device=dev)
  yaw_sum = torch.zeros(n_env, device=dev); xy_last = torch.zeros(n_env, device=dev)
  yaw_last = torch.zeros(n_env, device=dev)
  t_xy03 = torch.full((n_env,), -1, dtype=torch.long, device=dev)
  t_yaw15 = torch.full((n_env,), -1, dtype=torch.long, device=dev)
  f_w = torch.zeros(n_env, device=dev); f_ori = torch.zeros(n_env, device=dev)
  f_tip = torch.zeros(n_env, device=dev); f_sat = torch.zeros(n_env, 2, device=dev)
  f_n = torch.zeros(n_env, device=dev)
  while len(results) < n_clips:
    rew, failed, done_mask = b.step()
    ret += rew
    steps += 1
    xy, yaw = b.drift()
    live = ~done_mask
    xy_sum += torch.where(live, xy, 0.0); yaw_sum += torch.where(live, yaw, 0.0)
    xy_max = torch.where(live, torch.maximum(xy_max, xy), xy_max)
    xy_last = torch.where(live, xy, xy_last); yaw_last = torch.where(live, yaw, yaw_last)
    t_xy03 = torch.where(live & (t_xy03 < 0) & (xy > 0.3), steps, t_xy03)
    t_yaw15 = torch.where(live & (t_yaw15 < 0) & (yaw > 15.0), steps, t_yaw15)
    if b.has_epee:
      ori, tip_err, sat, w = b.fencing()
      w = torch.where(live, w, 0.0)
      f_w += w; f_ori += w * ori; f_tip += w * tip_err
      f_sat += (sat & live[:, None]).float(); f_n += live.float()
    restart_ids, restart_clips = [], []
    for e in done_mask.nonzero().flatten().tolist():
      c = current[e]
      if active[e] and c not in results:
        frames = ends[c] - starts[c]
        r, s = float(ret[e]), int(steps[e])
        results[c] = dict(
          name=names[c], frames=frames, steps=s, failed=bool(failed[e]),
          fail_step=s if bool(failed[e]) else -1,
          norm_return=r / (b.max_step * max(frames - 1, 1)),
          norm_return_alive=r / (b.max_step * max(s, 1)),
          drift_xy_mean_m=float(xy_sum[e]) / max(s - 1, 1), drift_xy_max_m=float(xy_max[e]),
          drift_xy_end_m=float(xy_last[e]), drift_yaw_mean_deg=float(yaw_sum[e]) / max(s - 1, 1),
          drift_yaw_end_deg=float(yaw_last[e]),
          t_xy_over_0p3m_s=round(int(t_xy03[e]) * b.base.step_dt, 2) if int(t_xy03[e]) >= 0 else None,
          t_yaw_over_15deg_s=round(int(t_yaw15[e]) * b.base.step_dt, 2) if int(t_yaw15[e]) >= 0 else None,
        )
        if b.has_epee:
          fw, fn = max(float(f_w[e]), 1e-9), max(float(f_n[e]), 1.0)
          results[c].update(
            wrist_ori_err_deg=float(f_ori[e]) / fw if float(f_w[e]) > 0 else None,
            tip_err_m=float(f_tip[e]) / fw if float(f_w[e]) > 0 else None,
            weapon_arm_weight_sum=float(f_w[e]),
            wrist_pitch_sat_frac=float(f_sat[e, 0]) / fn, wrist_yaw_sat_frac=float(f_sat[e, 1]) / fn,
          )
        if len(results) % 10 == 0 or len(results) == n_clips:
          print(f"  evaluated {len(results)}/{n_clips}")
      current[e] += n_env
      active[e] = current[e] < n_clips
      restart_ids.append(e)
      restart_clips.append(current[e])
    if restart_ids:
      b.start_clip(restart_ids, restart_clips)
      idx = torch.tensor(restart_ids, device=b.base.device)
      ret[idx], steps[idx] = 0.0, 0
      xy_sum[idx], xy_max[idx], yaw_sum[idx], xy_last[idx], yaw_last[idx] = 0.0, 0.0, 0.0, 0.0, 0.0
      t_xy03[idx], t_yaw15[idx] = -1, -1
      f_w[idx], f_ori[idx], f_tip[idx], f_n[idx] = 0.0, 0.0, 0.0, 0.0
      f_sat[idx] = 0.0
  b.env.close()

  clips = [results[i] for i in range(n_clips)]
  tot = sum(c["steps"] for c in clips)
  summary = dict(
    run_dir=str(cfg.run_dir), motion_file=cfg.motion_file, clips=n_clips,
    failed=sum(c["failed"] for c in clips),
    mean_norm_return=sum(c["norm_return"] for c in clips) / n_clips,
    mean_norm_return_alive=sum(c["norm_return_alive"] for c in clips) / n_clips,
    step_weighted_norm_return_alive=sum(c["norm_return_alive"] * c["steps"] for c in clips) / max(tot, 1),
    max_step_reward=b.max_step, foot_friction=cfg.foot_friction,
    median_drift_xy_end_m=sorted(c["drift_xy_end_m"] for c in clips)[n_clips // 2],
    median_drift_xy_mean_m=sorted(c["drift_xy_mean_m"] for c in clips)[n_clips // 2],
    median_drift_yaw_mean_deg=sorted(c["drift_yaw_mean_deg"] for c in clips)[n_clips // 2],
  )
  if b.has_epee:
    # Weighted by weapon_arm_weight (frames whose reference wrist is unreliable count less / not at all);
    # tracking error in simulation against the retargeted reference, not real-robot accuracy.
    W = sum(c["weapon_arm_weight_sum"] for c in clips)
    med = lambda k: (lambda v: v[len(v) // 2] if v else None)(sorted(c[k] for c in clips if c[k] is not None))  # noqa: E731
    summary.update(
      wrist_ori_err_deg=sum(c["wrist_ori_err_deg"] * c["weapon_arm_weight_sum"] for c in clips if c["wrist_ori_err_deg"] is not None) / max(W, 1e-9),
      tip_err_m=sum(c["tip_err_m"] * c["weapon_arm_weight_sum"] for c in clips if c["tip_err_m"] is not None) / max(W, 1e-9),
      median_clip_wrist_ori_err_deg=med("wrist_ori_err_deg"), median_clip_tip_err_m=med("tip_err_m"),
      wrist_pitch_sat_frac=sum(c["wrist_pitch_sat_frac"] * c["steps"] for c in clips) / max(tot, 1),
      wrist_yaw_sat_frac=sum(c["wrist_yaw_sat_frac"] * c["steps"] for c in clips) / max(tot, 1),
    )
  out = cfg.out or cfg.run_dir / "benchmark_official.json"
  out.write_text(json.dumps({"summary": summary, "clips": clips}, indent=1))
  print(json.dumps(summary, indent=1))
  for c in sorted(clips, key=lambda c: c["norm_return"])[:10]:
    tag = f"FAIL@{c['fail_step']}" if c["failed"] else ""
    print(f"  {c['name']:28s} ret {c['norm_return']:.3f} ({c['norm_return_alive']:.3f} alive) "
          f"steps {c['steps']}/{c['frames'] - 1} {tag}")
  print("saved", out)
  return summary


if __name__ == "__main__":
  evaluate(tyro.cli(EvalConfig))
