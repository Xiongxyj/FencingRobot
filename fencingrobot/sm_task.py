"""Fencing task for the OFFICIAL SuperMimic code (no official file is edited).

Importing this module (tools/train_fencing.py, tools/eval_official.py do it):
  1. registers the task ``Fencing-SuperMimic-GenCoord-G1`` = official ``gencoord_env_cfg`` with
       - the robot replaced by G1 + Revo2 + epee (``robot_spec.get_g1_epee_robot_cfg``; bodies / joints / actuators and
         their Unitree effort limits unchanged),
       - the four body-tracking reward terms replaced by per-body weighted versions (below),
       - the motion command replaced by a subclass cfg that builds the per-frame body-weight table from the ``fr_*``
         mask arrays inside the motion npz (``data.attach_masks``);
  2. patches ``WorldModelEnv._wm_step`` so the imagined (world-model) reward uses the SAME weights. The patch is inert
     unless the motion loader carries a weight table (other tasks, e.g. LAFAN1, run the official code path).

Per-body weight w[frame, body] over the 14 tracked bodies (official order):
    1, times 0 where ``no_track_mask`` marks the body's SMPL-X joint (elbow -> *_elbow_link, wrist -> *_wrist_yaw_link),
    times ``weapon_arm_weight`` on right_shoulder_roll_link / right_elbow_link / right_wrist_yaw_link,
    times ``legs_unreliable_weight`` (default 1 = no handling, TBD) on the six leg bodies where ``legs_unreliable``.
Weighted reward: the official terms are ``exp(-mean_b(err_b) / std^2)`` for body position, orientation, linear and
angular velocity; here ``mean_b(w_b * err_b)`` (the weight multiplies that link's error, the divisor stays 14). With all
weights 1 this is the official reward. The two root (pelvis) terms, action rate and joint limit are untouched.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from dataclasses import dataclass, fields
from typing import cast

import numpy as np
import torch
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import load_rl_cfg, register_mjlab_task
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_error_magnitude

import supermimic.tasks  # noqa: F401  (registers the official tasks; BASE_TASK_ID)
from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.tasks import BASE_TASK_ID
from supermimic.tasks.env_cfgs import gencoord_env_cfg
from supermimic.utils import tracking_reward as _official_reward
from supermimic.utils.helpers import rotation_error_magnitude

from .data import NO_TRACK_SMPLX_TO_G1_BODIES, WEAPON_ARM_G1_BODIES
from .robot_spec import get_g1_epee_robot_cfg

TASK_ID = "Fencing-SuperMimic-GenCoord-G1"
LEG_BODIES = ("left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
              "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link")
BODY_REWARD_TERMS = ("motion_body_pos", "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel")


# ----------------------------------------------------------------------------------------------- weight table
def body_weight_table(files: list[str], body_names: tuple[str, ...], legs_unreliable_weight: float = 1.0) -> np.ndarray:
    """(total frames, len(body_names)) float32, clips concatenated in ``files`` order (= MotionLoader order).
    A file without the fr_* arrays (e.g. LAFAN1) gets weight 1 everywhere."""
    col = {n: i for i, n in enumerate(body_names)}
    arm = [col[n] for n in WEAPON_ARM_G1_BODIES if n in col]
    legs = [col[n] for n in LEG_BODIES if n in col]
    out = []
    for path in files:
        with np.load(path) as d:
            T = d["joint_pos"].shape[0]
            w = np.ones((T, len(body_names)), np.float32)
            if "fr_weapon_arm_weight" in d.files:
                assert d["fr_weapon_arm_weight"].shape[0] == T, path
                w[:, arm] *= d["fr_weapon_arm_weight"][:, None]
                nt = d["fr_no_track_mask"]
                for j, bodies in NO_TRACK_SMPLX_TO_G1_BODIES.items():
                    cols = [col[n] for n in bodies if n in col]
                    w[np.ix_(nt[:, j], cols)] = 0.0
                unmapped = [j for j in range(nt.shape[1]) if j not in NO_TRACK_SMPLX_TO_G1_BODIES and nt[:, j].any()]
                assert not unmapped, f"{path}: no_track_mask marks SMPL-X joints {unmapped} with no G1 mapping"
                w[np.ix_(d["fr_legs_unreliable"], legs)] *= legs_unreliable_weight
        out.append(w)
    return np.concatenate(out)


@dataclass(kw_only=True)
class FencingMotionCommandCfg(MotionCommandCfg):
    legs_unreliable_weight: float = 1.0
    """Weight on the six leg bodies at frames flagged legs_unreliable (0.14 % of frames). 1 = flag ignored (TBD)."""

    def build(self, env) -> MotionCommand:
        cmd = MotionCommand(self, env)
        table = body_weight_table(cmd.motion.motion_files, self.body_names, self.legs_unreliable_weight)
        assert table.shape[0] == cmd.motion.time_step_total, (table.shape, cmd.motion.time_step_total)
        cmd.motion.fr_body_w = torch.tensor(table, device=cmd.device)
        print(f"[fencing] body-weight table {tuple(table.shape)}: mean {table.mean():.4f}, "
              f"frames with any weight < 1: {(table.min(1) < 1).mean() * 100:.1f} %, "
              f"legs_unreliable_weight {self.legs_unreliable_weight}")
        return cmd


# ----------------------------------------------------------------------------------------------- simulator reward
def _cmd_w(env, command_name: str) -> tuple[MotionCommand, torch.Tensor]:
    cmd = cast(MotionCommand, env.command_manager.get_term(command_name))
    return cmd, cmd.motion.fr_body_w[cmd.time_steps]


def body_pos_exp(env, command_name: str, std: float) -> torch.Tensor:
    c, w = _cmd_w(env, command_name)
    err = torch.sum(torch.square(c.body_pos_relative_w - c.robot_body_pos_w), dim=-1)
    return torch.exp(-(w * err).mean(-1) / std**2)


def body_ori_exp(env, command_name: str, std: float) -> torch.Tensor:
    c, w = _cmd_w(env, command_name)
    err = quat_error_magnitude(c.body_quat_relative_w, c.robot_body_quat_w) ** 2
    return torch.exp(-(w * err).mean(-1) / std**2)


def body_lin_vel_exp(env, command_name: str, std: float) -> torch.Tensor:
    c, w = _cmd_w(env, command_name)
    err = torch.sum(torch.square(c.body_lin_vel_w - c.robot_body_lin_vel_w), dim=-1)
    return torch.exp(-(w * err).mean(-1) / std**2)


def body_ang_vel_exp(env, command_name: str, std: float) -> torch.Tensor:
    c, w = _cmd_w(env, command_name)
    err = torch.sum(torch.square(c.body_ang_vel_w - c.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-(w * err).mean(-1) / std**2)


_SIM_TERMS = dict(zip(BODY_REWARD_TERMS, (body_pos_exp, body_ori_exp, body_lin_vel_exp, body_ang_vel_exp), strict=True))


# ----------------------------------------------------------------------------------------------- world-model reward
def weighted_tracking_reward(aux_state, pred_pos, pred_rot_mat, pred_lin_vel, pred_ang_vel, joint_state, action,
                             last_action, target_pos, target_quat, target_lin_vel, target_ang_vel, soft_joint_pos_limits,
                             dt: float, body_w: torch.Tensor) -> torch.Tensor:
    """supermimic.utils.tracking_reward.tracking_reward with ``body_w`` (B, bodies) inside the four body means.
    Constants are the official ones (tools/check_fencing_task.py asserts equality at body_w = 1)."""
    tgt_pos, tgt_mat = _official_reward._reanchor_targets(target_pos, target_quat, aux_state[..., :3], aux_state[..., 3:7])
    terms = [
        (torch.sum((pred_pos[..., 0, :] - target_pos[..., 0, :]) ** 2, dim=-1), 0.5, 0.3),
        (rotation_error_magnitude(pred_rot_mat[..., 0, :, :], matrix_from_quat(target_quat[..., 0, :])) ** 2, 0.5, 0.4),
        ((body_w * torch.sum((pred_pos - tgt_pos) ** 2, dim=-1)).mean(-1), 1.0, 0.3),
        ((body_w * rotation_error_magnitude(pred_rot_mat, tgt_mat) ** 2).mean(-1), 1.0, 0.4),
        ((body_w * torch.sum((pred_lin_vel - target_lin_vel) ** 2, dim=-1)).mean(-1), 1.0, 1.0),
        ((body_w * torch.sum((pred_ang_vel - target_ang_vel) ** 2, dim=-1)).mean(-1), 1.0, 3.14),
    ]
    reward = sum(w * torch.exp(-err / std**2) for err, w, std in terms)
    action_rate = torch.sum((action - last_action) ** 2, dim=-1)
    joint_pos, _ = joint_state.chunk(2, dim=-1)
    lo, hi = soft_joint_pos_limits.unbind(-1)
    joint_limit = ((lo - joint_pos).clamp(min=0) + (joint_pos - hi).clamp(min=0)).sum(-1)
    return dt * (reward - 0.1 * action_rate - 2.0 * joint_limit)


def _code_hash(fn) -> str:
    """Hash of a function's AST (comments / formatting do not change it)."""
    return hashlib.sha256(ast.dump(ast.parse(textwrap.dedent(inspect.getsource(fn)))).encode()).hexdigest()[:16]


# The two official functions re-implemented above / below, at supermimic 5efbf11. If the official code changes, re-derive
# the weighted versions and update these hashes.
_EXPECTED = {"_wm_step": "8dde4b4510edf7e2", "tracking_reward": "f58e279ad7adee9e"}
_official_wm_step = WorldModelEnv._wm_step


def _wm_step(self, rep_state, aux_state, action, last_action, sim_ctx):
    """Official WorldModelEnv._wm_step with the weighted reward (same dynamics, termination, argument order)."""
    table = getattr(self.motion, "fr_body_w", None)
    if table is None:
        return _official_wm_step(self, rep_state, aux_state, action, last_action, sim_ctx)
    rep = self.representation
    next_aux_state, next_rep_state = self.dynamics(rep_state, action, aux_state, sim_ctx)
    next_joint_state = rep.joint_state(next_rep_state)
    ref_body_pos_w, ref_body_quat_w, ref_body_lin_vel_w, ref_body_ang_vel_w = self._ref_body_state
    term_prob = self._compute_term_prob(next_aux_state, ref_body_pos_w, ref_body_quat_w)
    reward = weighted_tracking_reward(
        next_aux_state, *rep.world_frame_bodies(next_aux_state, next_rep_state), next_joint_state, action, last_action,
        ref_body_pos_w, ref_body_quat_w, ref_body_lin_vel_w, ref_body_ang_vel_w, rep._soft_joint_pos_limits, rep.dt,
        table[self.motion_times],  # the frame self._ref_body_state was computed for
    )
    return next_aux_state, next_rep_state, next_joint_state, term_prob, reward


def official_code_hashes() -> dict[str, str]:
    return {"_wm_step": _code_hash(_official_wm_step), "tracking_reward": _code_hash(_official_reward.tracking_reward)}


def _install() -> None:
    got = official_code_hashes()
    if got != _EXPECTED:
        raise RuntimeError(f"official supermimic code changed ({got} != {_EXPECTED}): the weighted world-model reward in "
                           "fencingrobot/sm_task.py must be re-derived from it before training")
    WorldModelEnv._wm_step = _wm_step


# ----------------------------------------------------------------------------------------------- task
SOFT_BLADE = True  # user 2026-11-12: blade contact stiffness of a bending blade (robot_spec.soften_blade)
UNSTABLE_JOINT_VEL = 100.0  # rad / s; G1 motors top out below 40 rad / s, so this is far outside anything trackable


def sim_unstable(env, max_joint_vel: float = UNSTABLE_JOINT_VEL) -> torch.Tensor:
    """Failure termination for a diverged simulation: non-finite physics state, or a joint speed no motor can reach.

    Needed because the official anchor terminations compare with ``>`` and therefore never fire on NaN: a single env
    whose physics diverged kept running until its time-out, its NaN states went into the replay buffer, the world-model
    loss became NaN and with it every network (run fencing_v1, 2026-11-12). The runner stores the PRE-step state, and a
    terminated env is reset before the next observation, so a state flagged here never reaches the buffer.
    """
    d = env.sim.data
    bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for t in (d.qpos, d.qvel, d.qacc):
        bad |= ~torch.isfinite(t).all(dim=-1)
    qv = env.scene["robot"].data.joint_vel
    return bad | ~(qv.abs().amax(dim=-1) <= max_joint_vel)


def fencing_gencoord_env_cfg(play: bool = False):
    cfg = gencoord_env_cfg(play=play)
    cfg.scene.entities = {"robot": get_g1_epee_robot_cfg(soft_blade=SOFT_BLADE)}
    old = cfg.commands["motion"]
    assert type(old) is MotionCommandCfg
    cfg.commands["motion"] = FencingMotionCommandCfg(**{f.name: getattr(old, f.name) for f in fields(old)})
    for name, fn in _SIM_TERMS.items():
        term = cfg.rewards[name]
        assert set(term.params) == {"command_name", "std"}, (name, term.params)
        term.func = fn
    cfg.terminations["sim_unstable"] = TerminationTermCfg(func=sim_unstable)
    return cfg


_install()
try:
    register_mjlab_task(task_id=TASK_ID, env_cfg=fencing_gencoord_env_cfg(),
                        play_env_cfg=fencing_gencoord_env_cfg(play=True), rl_cfg=load_rl_cfg(BASE_TASK_ID))
except ValueError:
    pass  # already registered
