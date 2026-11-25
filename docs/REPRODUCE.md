# Baseline reproduction

This release uses the first successful fencing baseline, before subsequent reward and sampling experiments. The official SuperMimic training implementation remains unchanged; the fencing task is registered externally. The demonstration was recorded during initial baseline development using an earlier internal export; the released dataset contains refined motion trajectories.

## Environment

Linux, NVIDIA CUDA-capable GPU and compatible drivers, Git, curl and uv. Clone the repository as `FencingRobot` so relative paths in the commands match.

```bash
mkdir -p fencing && cd fencing
git clone https://github.com/Xiongxyj/FencingRobot.git
cd FencingRobot
bash setup/setup_official.sh
bash setup/setup_converter.sh
```

The first setup pins official SuperMimic to `5efbf114e11f725c31d83e217be987c13c0e4ce2`. Its private mjlab dependency is substituted by public mjlab v1.5.0 plus `setup/mjlab_research.patch`, in a sibling directory `mjlab_research`. The official uv lockfile is used. The second setup installs the separate, pinned converter environment (not a training dependency).

## Prepare motion

```bash
bash setup/prepare_fencing_data.sh
```

Conversion produces 50 Hz NPZ trajectories in `build/fencing/npz` with per-frame masks. This recipe trains the baseline on the released dataset, not the exact historical demonstration input.

## Train the baseline

From the checkout, change to the official SuperMimic sibling:

```bash
cd ../supermimic
WANDB_MODE=offline uv run --frozen python ../FencingRobot/tools/train_fencing.py \
  Fencing-SuperMimic-GenCoord-G1 \
  --motion-file ../FencingRobot/build/fencing/npz \
  --num-envs 512 --runner.policy.batch-size 1024 --runner.run-name fencing_baseline
```

The launcher defaults to proprioceptive actor observations. Adjust environment and batch sizes for available GPU memory. The original successful policy ran with 512 environments and batch size 1024 on an 8 GB GPU. This is a reproduction recipe, not a newly verified clean-machine training run; no checkpoints are included.

The evaluation and recording scripts are `tools/eval_official.py` and `tools/record_official.py`; use `--help` in the official environment for arguments. Match the actor observation configuration to the trained checkpoint (`--runner.actor-obs proprio`).
