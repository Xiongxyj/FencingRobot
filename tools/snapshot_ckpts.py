"""Keep every checkpoint of an official SuperMimic run (the runner overwrites actor.pt /
critic.pt / world_model.pt in its log dir every --runner.save-interval steps).

Polls the run dir; when actor.pt changes and has stopped changing, copies the checkpoint
files to <run-dir>/snapshots/<NNN>/ with meta.json (wall time since the run started, and
the latest "Learning iteration" / "Total steps" / "Time elapsed" from the console log).
CPU only, no torch import. Start it next to training, stop it with Ctrl-C.

usage (from ~/fencing/supermimic):
  python3 ../FencingRobot/tools/snapshot_ckpts.py --run-dir logs/SuperMimic-GenCoord-G1/<time> \
    --log ../runs/<name>.log
  (--run-dir latest: the newest dir under logs/SuperMimic-GenCoord-G1)
"""

import argparse
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

FILES = ("actor.pt", "critic.pt", "world_model.pt")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def last_log_state(log: Path | None) -> dict:
  if log is None or not log.exists():
    return {}
  with open(log, "rb") as f:
    f.seek(0, 2)
    f.seek(max(0, f.tell() - 200_000))
    tail = ANSI.sub("", f.read().decode("utf-8", "replace"))
  out = {}
  for key, pat in (("log_iteration", r"Learning iteration (\d+)/"), ("total_steps", r"Total steps:\s*(\d+)"),
                   ("time_elapsed", r"Time elapsed:\s*([\d:]+)")):
    hits = re.findall(pat, tail)
    if hits:
      out[key] = int(hits[-1]) if key != "time_elapsed" else hits[-1]
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--log", type=Path, default=None, help="console log of the training run (tee)")
  ap.add_argument("--task-dir", default="logs/SuperMimic-GenCoord-G1")
  ap.add_argument("--poll", type=float, default=10.0)
  ap.add_argument("--min-interval-min", type=float, default=30.0,
                  help="keep a checkpoint only if at least this many minutes after the last kept one "
                       "(the runner saves every ~1-2 min at 512 envs; each snapshot is ~75 MB)")
  A = ap.parse_args()
  if A.run_dir == "latest":
    # The newest run dir whose name (its start time) is at most 10 min before this tool
    # started: start it right before or after the training command, not hours later.
    t0 = time.time() - 600
    while True:
      runs = [p for p in Path(A.task_dir).glob("20*") if p.is_dir()
              and datetime.strptime(p.name, "%Y-%m-%d_%H-%M-%S").timestamp() >= t0]
      if runs:
        break
      time.sleep(A.poll)
    run = max(runs, key=lambda p: p.name)
  else:
    run = Path(A.run_dir)
  start = datetime.strptime(run.name, "%Y-%m-%d_%H-%M-%S").timestamp()
  snaps = run / "snapshots"
  snaps.mkdir(exist_ok=True)
  done = sorted(snaps.glob("[0-9][0-9][0-9]"))
  last_mtime = json.loads((done[-1] / "meta.json").read_text())["actor_mtime"] if done else 0.0
  print(f"[snapshot] watching {run} ({len(done)} snapshots so far)", flush=True)
  while True:
    actor = run / "actor.pt"
    if actor.exists():
      m = actor.stat().st_mtime
      due = m - last_mtime >= A.min_interval_min * 60 if last_mtime else True
      if m > last_mtime and due and time.time() - max((run / f).stat().st_mtime for f in FILES if (run / f).exists()) > 5:
        k = len(list(snaps.glob("[0-9][0-9][0-9]")))
        dst = snaps / f"{k:03d}"
        dst.mkdir()
        for f in FILES:
          if (run / f).exists():
            shutil.copy2(run / f, dst / f)
        meta = dict(index=k, actor_mtime=m, wall_s=m - start, **last_log_state(A.log))
        (dst / "meta.json").write_text(json.dumps(meta, indent=1))
        last_mtime = m
        print(f"[snapshot] {dst.name}: wall {meta['wall_s'] / 3600:.2f} h {meta}", flush=True)
    time.sleep(A.poll)


if __name__ == "__main__":
  main()
