"""Live TensorBoard for an official SuperMimic run (the official logger only writes W&B and
the console): follows the training console log (`| tee`) and writes every finished
"Learning iteration" block as TensorBoard scalars (step = "Total steps"; the same keys
as the console, e.g. Train/Mean reward, Episode_Reward/*, Episode_Termination/*,
Metrics/motion/*, losses, Perf/*). Also writes the snapshot benchmark
(tools/eval_snapshots.py: snapshots/summary.csv) when --run-dir is given.
CPU only; safe to start before, during or after training; Ctrl-C to stop.

usage (official env, from ~/fencing/supermimic):
  uv run --frozen python ../FencingRobot/tools/log_to_tensorboard.py --log ../runs/<name>.log \
    --tb-dir ../runs/tb/<name> [--run-dir logs/SuperMimic-GenCoord-G1/<time>]
  uvx --python 3.12 --from "tensorboard==2.20.0" --with "setuptools<81" tensorboard \
    --logdir ../runs/tb --port 6006      # another terminal; the official .venv's tensorboard
                                         # needs pkg_resources (no setuptools in the lock)
"""

import argparse
import csv
import re
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SKIP = {"Run name", "ETA", "Time elapsed"}
PERF = {"Total steps", "Steps per second", "Collection time", "WM Learning time",
        "Policy Learning time", "Iteration time"}


def tag(key: str) -> str:
  if "/" in key:
    return key
  if key in PERF:
    return f"Perf/{key}"
  if key in ("Mean reward", "Mean episode length"):
    return f"Train/{key}"
  return f"Loss/{key}"


def parse_block(lines: list[str]) -> dict:
  out = {}
  for line in lines:
    key, sep, val = line.strip().partition(": ")
    key = key.rstrip(":").strip()
    if not sep or key in SKIP:
      continue
    try:
      out[key] = float(val.strip().rstrip("s"))
    except ValueError:
      pass
  m = re.search(r"Time elapsed:\s*(\d+):(\d+):(\d+)", "\n".join(lines))
  if m:
    out["time_h"] = int(m.group(1)) + int(m.group(2)) / 60 + int(m.group(3)) / 3600
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--log", type=Path, required=True)
  ap.add_argument("--tb-dir", type=Path, required=True)
  ap.add_argument("--run-dir", type=Path, default=None)
  ap.add_argument("--poll", type=float, default=15.0)
  ap.add_argument("--once", action="store_true", help="convert what is there and exit")
  A = ap.parse_args()
  w = SummaryWriter(str(A.tb_dir))
  pos, buf, n_blocks, n_bench = 0, "", 0, 0
  print(f"[tb] {A.log} -> {A.tb_dir}", flush=True)
  while True:
    if A.log.exists():
      with open(A.log, "rb") as f:
        f.seek(pos)
        chunk = f.read()
        pos = f.tell()
      buf += ANSI.sub("", chunk.decode("utf-8", "replace"))
      # A block is complete once its "ETA:" line has arrived. Walk with an offset and cut the buffer once at the
      # end: cutting it per block copies the whole remaining log each time (a 60 MB log took hours to convert,
      # so TensorBoard only showed the start of the run).
      cur = 0
      while True:
        start = buf.find("Learning iteration", cur)
        end = buf.find("ETA:", start) if start >= 0 else -1
        nl = buf.find("\n", end) if end >= 0 else -1
        if nl < 0:
          break
        vals = parse_block(buf[start:nl].splitlines())
        cur = nl + 1
        step = int(vals.get("Total steps", n_blocks))
        for k, v in vals.items():
          if k == "time_h":
            w.add_scalar("Perf/Time elapsed (h)", v, step)
          else:
            w.add_scalar(tag(k), v, step)
        n_blocks += 1
      buf = buf[cur:]
    summary = A.run_dir / "snapshots" / "summary.csv" if A.run_dir else None
    if summary and summary.exists():
      rows = list(csv.DictReader(open(summary)))
      for r in rows[n_bench:]:
        step = int(r["total_steps"]) if r.get("total_steps") else int(r["snapshot"])
        w.add_scalar("Benchmark/failed", float(r["failed"]), step)
        w.add_scalar("Benchmark/mean_norm_return", float(r["mean_norm_return"]), step)
        w.add_scalar("Benchmark/mean_norm_return_alive", float(r["mean_norm_return_alive"]), step)
        w.add_scalar("Benchmark/wall_h", float(r["wall_h"]), step)
      n_bench = len(rows)
    w.flush()
    if A.once:
      print(f"[tb] {n_blocks} log blocks, {n_bench} benchmark rows", flush=True)
      break
    time.sleep(A.poll)
  w.close()


if __name__ == "__main__":
  main()
