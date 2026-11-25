"""Train the fencing task with the official SuperMimic trainer (src/supermimic/scripts/train.py, unmodified): this
launcher only imports fencingrobot.sm_task first (registers Fencing-SuperMimic-GenCoord-G1 and installs the weighted
world-model reward), then hands over to the official main(). All official flags work unchanged.

One default differs from the official trainer: the actor observation is ``proprio`` (deployable; user 2026-11-12) instead
of ``rep_state`` (contains root height and base linear velocity). Pass --runner.actor-obs rep_state to get the official
default back.

usage (official env, from ~/fencing/supermimic):
  WANDB_MODE=offline uv run --frozen python ../FencingRobot/tools/train_fencing.py Fencing-SuperMimic-GenCoord-G1 \
    --motion-file ../FencingRobot/build/fencing/npz --num-envs N --runner.policy.batch-size B --runner.run-name NAME
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fencingrobot.sm_task  # noqa: E402,F401
from supermimic.scripts.train import main  # noqa: E402

if __name__ == "__main__":
  if not any(a.startswith("--runner.actor-obs") for a in sys.argv[1:]):
    sys.argv += ["--runner.actor-obs", "proprio"]
    print("[fencing] actor observation: proprio (launcher default)")
  main()
