"""Run tools/eval_official.py on every snapshot of a run (skips ones already evaluated)
and write <run-dir>/snapshots/summary.csv (one row per snapshot: wall time, log
iteration, failures, normalized returns).

usage (official env, from ~/fencing/supermimic):
  WANDB_MODE=offline uv run --frozen python ../FencingRobot/tools/eval_snapshots.py \
    --run-dir logs/SuperMimic-GenCoord-G1/<time> --motion-file ../SuperMimic-Reproduced/data/npz/lafan1 \
    [--every 1] [--num-envs 40] [--runner.* same architecture flags as training, if changed]
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_official import EvalConfig, evaluate  # noqa: E402
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg  # noqa: E402


@dataclass
class Args:
  run_dir: Path
  motion_file: str
  every: int = 1
  """Evaluate every N-th snapshot (the newest one is always included)."""
  num_envs: int = 40
  task_id: str = "SuperMimic-GenCoord-G1"
  config: Path | None = None
  runner: SuperMimicRunnerCfg = field(default_factory=SuperMimicRunnerCfg)


def main(a: Args) -> None:
  snaps = sorted((a.run_dir / "snapshots").glob("[0-9][0-9][0-9]"))
  if not snaps:
    raise SystemExit(f"no snapshots in {a.run_dir / 'snapshots'} (run tools/snapshot_ckpts.py during training)")
  pick = [s for i, s in enumerate(snaps) if i % a.every == 0 or s == snaps[-1]]
  base = EvalConfig(run_dir=a.run_dir, motion_file=a.motion_file, num_envs=a.num_envs, task_id=a.task_id,
                    config=a.config, runner=a.runner)
  rows = []
  for s in snaps:
    res = s / "benchmark_official.json"
    if s in pick and not res.exists():
      print(f"=== snapshot {s.name}")
      evaluate(replace(base, run_dir=s, out=res))
    if res.exists():
      meta = json.loads((s / "meta.json").read_text())
      summ = json.loads(res.read_text())["summary"]
      rows.append(dict(snapshot=s.name, wall_h=round(meta["wall_s"] / 3600, 4),
                       log_iteration=meta.get("log_iteration", ""), total_steps=meta.get("total_steps", ""),
                       failed=summ["failed"], clips=summ["clips"],
                       mean_norm_return=round(summ["mean_norm_return"], 4),
                       mean_norm_return_alive=round(summ["mean_norm_return_alive"], 4)))
  out = a.run_dir / "snapshots" / "summary.csv"
  with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
  for r in rows:
    print(r)
  print("saved", out)


if __name__ == "__main__":
  main(tyro.cli(Args))
