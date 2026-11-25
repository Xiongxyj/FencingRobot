"""Training curves of an official SuperMimic run from its console log (W&B offline needs no
login), plus the snapshot benchmark (tools/eval_snapshots.py) if present.

Parses every "Learning iteration i/N" block of the log ("key: value" lines) into
<out>.csv and plots, against training time ("Time elapsed"): mean reward, mean episode
length, every Episode*/Metrics* term, every loss, and (second figure) failures / 40 and the
normalized returns of the snapshots against their wall time.

usage: python3 ../FencingRobot/tools/plot_training.py --log ../runs/<name>.log \
         [--run-dir logs/SuperMimic-GenCoord-G1/<time>] [--out ../runs/<name>]
       (any Python with matplotlib, e.g. the official env: uv run --frozen python ...)
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SKIP = {"Run name", "ETA"}


def parse(log: Path) -> list[dict]:
  rows, cur = [], None
  for line in ANSI.sub("", log.read_text(errors="replace")).splitlines():
    m = re.search(r"Learning iteration (\d+)/(\d+)", line)
    if m:
      cur = {"log_iteration": int(m.group(1))}
      rows.append(cur)
      continue
    if cur is None or ":" not in line:
      continue
    key, _, val = line.strip().partition(": ")
    key = key.rstrip(":").strip()
    if key in SKIP or not val:
      continue
    val = val.strip().rstrip("s")
    if key == "Time elapsed":
      h, mi, s = (int(x) for x in val.split(":"))
      cur["time_h"] = h + mi / 60 + s / 3600
      continue
    try:
      cur[key] = float(val)
    except ValueError:
      pass
  return [r for r in rows if "time_h" in r]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--log", type=Path, required=True)
  ap.add_argument("--run-dir", type=Path, default=None)
  ap.add_argument("--out", type=Path, default=None)
  A = ap.parse_args()
  out = A.out or A.log.with_suffix("")
  rows = parse(A.log)
  if not rows:
    raise SystemExit(f"no 'Learning iteration' blocks in {A.log}")
  keys = sorted({k for r in rows for k in r} - {"log_iteration", "time_h"})
  with open(f"{out}_curves.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["log_iteration", "time_h", *keys])
    w.writeheader(); w.writerows(rows)

  first = ["Mean reward", "Mean episode length"]
  plot_keys = [k for k in first if k in keys] + [k for k in keys if k not in first and k not in {
    "Total steps", "Steps per second", "Collection time", "WM Learning time", "Policy Learning time",
    "Iteration time"}]
  n = len(plot_keys)
  cols = 4
  fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(4.2 * cols, 2.6 * ((n + cols - 1) // cols)),
                           squeeze=False)
  for ax, k in zip(axes.flat, plot_keys):
    pts = [(r["time_h"], r[k]) for r in rows if k in r]
    ax.plot(*zip(*pts), lw=1, marker="." if len(pts) < 50 else None, ms=3)
    ax.set_title(k, fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
  for ax in axes.flat[n:]:
    ax.axis("off")
  fig.supxlabel("training time (h, 'Time elapsed')", fontsize=9)
  fig.tight_layout(); fig.savefig(f"{out}_curves.png", dpi=110); plt.close(fig)
  print(f"{len(rows)} log rows, {n} series -> {out}_curves.csv / .png")

  summary = A.run_dir / "snapshots" / "summary.csv" if A.run_dir else None
  if summary and summary.exists():
    s = list(csv.DictReader(open(summary)))
    t = [float(r["wall_h"]) for r in s]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(t, [int(r["failed"]) for r in s], "o-")
    ax[0].set_title(f"benchmark failures / {s[0]['clips']}"); ax[0].set_xlabel("wall time (h)"); ax[0].grid(alpha=0.3)
    ax[1].plot(t, [float(r["mean_norm_return"]) for r in s], "o-", label="norm_return (paper)")
    ax[1].plot(t, [float(r["mean_norm_return_alive"]) for r in s], "s--", label="norm_return_alive")
    ax[1].set_title("benchmark normalized return"); ax[1].set_xlabel("wall time (h)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out}_benchmark.png", dpi=110); plt.close(fig)
    print(f"{len(s)} snapshots -> {out}_benchmark.png")


if __name__ == "__main__":
  main()
