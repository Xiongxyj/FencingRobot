"""Record videos of selected clips with an official SuperMimic checkpoint, under exactly the
benchmark rollout of eval_official.py (Appendix F protocol, shared Bench class).

Each selected clip gets its own env and plays once from its first frame until the clip ends
or a training termination fires (failure). Frames come from mjlab's offscreen renderer
(MuJoCo, EGL): the robot plus the reference motion as a translucent green ghost, camera
following torso_link. Writes <out-dir>/<clip>_{ok|FAIL}.mp4, for failures also
<clip>_FAIL_last<N>s.mp4 (the N seconds before the failure), and record.json with this
rollout's per-clip result next to the benchmark's (GPU nondeterminism can flip a
borderline clip between runs).

--clips auto: from <run-dir>/benchmark_official.json take every failed clip plus the
best, median and worst successful clips. Or give names: --clips jumps1_subject2,walk4_subject1

usage (official env, from ~/fencing/supermimic):
  MUJOCO_GL=egl WANDB_MODE=offline uv run --frozen python ../FencingRobot/tools/record_official.py \
    --run-dir logs/SuperMimic-GenCoord-G1/<time>/snapshots/000 \
    --motion-file ../SuperMimic-Reproduced/data/npz/lafan1 --clips auto
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import imageio_ffmpeg
import mediapy
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_official import Bench, EvalConfig, _motion_files  # noqa: E402
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg  # noqa: E402


@dataclass
class RecordConfig:
  run_dir: Path
  """Checkpoint dir (actor.pt, world_model.pt), e.g. a snapshot dir."""
  motion_file: str
  """The benchmark clips (dir / glob / list); --clips selects among them by file stem."""
  clips: str = "auto"
  out_dir: Path | None = None
  """Default: <run-dir>/videos."""
  width: int = 960
  height: int = 540
  fps: int = 25
  """Video fps; the control rate is 50 Hz, so every (50 / fps)-th step is rendered."""
  max_seconds: float | None = None
  """Stop each clip after this much clip time (default: the whole clip)."""
  tail_seconds: float = 15.0
  webm: bool = True
  """Also write VP9 .webm copies: Ubuntu's default player (GStreamer) plays VP9 out of
  the box but needs gstreamer1.0-libav for the H.264 .mp4."""
  task_id: str = "SuperMimic-GenCoord-G1"
  config: Path | None = None
  runner: SuperMimicRunnerCfg = field(default_factory=SuperMimicRunnerCfg)
  foot_friction: float = 1.0
  device: str | None = None
  seed: int = 0


def pick_auto(bench_json: Path) -> list[str]:
  clips = json.loads(bench_json.read_text())["clips"]
  failed = [c["name"] for c in clips if c["failed"]]
  ok = sorted((c for c in clips if not c["failed"]), key=lambda c: c["norm_return"])
  chosen = failed + ([ok[-1]["name"], ok[len(ok) // 2]["name"], ok[0]["name"]] if ok else [])
  return list(dict.fromkeys(chosen))


def main(a: RecordConfig) -> None:
  all_files = _motion_files(a.motion_file)
  by_name = {Path(f).stem: f for f in all_files}
  bench_json = a.run_dir / "benchmark_official.json"
  bench = {c["name"]: c for c in json.loads(bench_json.read_text())["clips"]} if bench_json.exists() else {}
  if a.clips == "auto":
    if not bench:
      raise SystemExit(f"--clips auto needs {bench_json} (run eval_official / eval_snapshots first)")
    names = pick_auto(bench_json)
  else:
    names = [s.strip() for s in a.clips.split(",") if s.strip()]
  missing = [n for n in names if n not in by_name]
  if missing:
    raise SystemExit(f"unknown clips {missing}; available: {sorted(by_name)}")
  print(f"[record] clips: {names}")

  cfg = EvalConfig(run_dir=a.run_dir, motion_file=a.motion_file, task_id=a.task_id, config=a.config,
                   runner=a.runner, foot_friction=a.foot_friction, device=a.device, seed=a.seed)
  b = Bench(cfg, [by_name[n] for n in names], num_envs=len(names), render=(a.height, a.width))
  n = b.n_env
  b.start_clip(list(range(n)), list(range(n)))
  renderer = b.base._offline_renderer
  assert renderer is not None
  stride = max(1, round(1.0 / (b.base.step_dt * a.fps)))
  fps = 1.0 / (b.base.step_dt * stride)
  max_steps = int(a.max_seconds / b.base.step_dt) if a.max_seconds else None
  out_dir = a.out_dir or a.run_dir / "videos"
  out_dir.mkdir(parents=True, exist_ok=True)
  mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

  tmp = [out_dir / f".{nm}.partial.mp4" for nm in names]
  writers = [mediapy.VideoWriter(p, shape=(a.height, a.width), fps=fps) for p in tmp]
  for w in writers:
    w.__enter__()
  ret = torch.zeros(n, device=b.base.device)
  steps = [0] * n
  active = [True] * n
  results: dict[str, dict] = {}

  def render_env(e: int) -> None:
    renderer._cfg.env_idx = e
    writers[e].add_image(b.base.render())

  for e in range(n):
    render_env(e)
  while any(active):
    rew, failed, done = b.step()
    ret += rew
    for e in range(n):
      if not active[e]:
        continue
      steps[e] += 1
      cut = max_steps is not None and steps[e] >= max_steps
      if bool(done[e]) or cut:
        frames = b.ends[e] - b.starts[e]
        results[names[e]] = dict(
          steps=steps[e], frames=frames, failed=bool(failed[e]), stopped_early=cut and not bool(done[e]),
          fail_time_s=round(steps[e] * b.base.step_dt, 2) if bool(failed[e]) else None,
          norm_return=float(ret[e]) / (b.max_step * max(frames - 1, 1)),
          benchmark=bench.get(names[e]),
        )
        active[e] = False
        writers[e].__exit__(None, None, None)
        print(f"[record] {names[e]}: {'FAIL' if failed[e] else 'ok'} after {steps[e]}/{frames - 1} steps "
              f"(benchmark: {bench.get(names[e], {}).get('steps', '?')} steps, "
              f"failed {bench.get(names[e], {}).get('failed', '?')})", flush=True)
      elif steps[e] % stride == 0:
        render_env(e)
  b.env.close()

  ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
  for e, nm in enumerate(names):
    r = results[nm]
    final = out_dir / f"{nm}_{'FAIL' if r['failed'] else 'ok'}.mp4"
    tmp[e].replace(final)
    r["video"] = final.name
    if r["failed"] and a.tail_seconds > 0:
      tail = out_dir / f"{nm}_FAIL_last{int(a.tail_seconds)}s.mp4"
      subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-sseof", f"-{a.tail_seconds}", "-i", str(final),
                      "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tail)], check=True)
      r["video_tail"] = tail.name
    if a.webm:
      for key in ("video", "video_tail"):
        if key in r:
          src = out_dir / r[key]
          subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libvpx-vp9",
                          "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1", "-b:v", "0",
                          "-crf", "34", str(src.with_suffix(".webm"))], check=True)
  (out_dir / "record.json").write_text(json.dumps(
    dict(run_dir=str(a.run_dir), fps=fps, size=[a.width, a.height], clips=results), indent=1))
  print(f"[record] saved {len(names)} videos to {out_dir}")


if __name__ == "__main__":
  main(tyro.cli(RecordConfig))
