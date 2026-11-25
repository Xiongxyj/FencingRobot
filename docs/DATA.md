# Fencing motion dataset

## Overview

`data/fencing` contains 829 clips, 348,995 frames at 30 Hz (3.231 hours). The dataset retains valid poses across gaps in piste translation observations; only the missing translation correction is interpolated. Clips without reliable piste anchors, heavily fragmented bouts and clips shorter than two seconds were rejected during construction.

## Format

Each `segments/<segment_id>.npz` contains:

| Key | Shape | Description |
|---|---|---|
| `qpos` | (T,36) | Root xyz in metres, root quaternion wxyz, then 29 joint angles in radians |
| `joint_names` | (29,) | G1 joint order |
| `fps` | scalar | 30 Hz |
| `mirrored` | scalar | Whether original left-handed motion was mirrored |
| `weapon_side_original`, `grip_type` | scalar strings | Original weapon side and grip category; robot uses a fixed pistol-style grip |
| `weapon_arm_weight` | (T,) | Suggested confidence weight for right shoulder, elbow and wrist links only |
| `weapon_arm_flags` | (T,) | Observation flags; see `weapon_arm_flags_legend` in file |
| `weapon_arm_observed`, `weapon_arm_interpolated` | (T,) | Direct observation or interpolation flags |
| `legs_unreliable` | (T,) | Unreliable lower-body observations |
| `no_track_mask` | (T,22) | SMPL-X body-joint mask; mapping to G1 in `fencingrobot/data.py` |
| `x_filled` | (T,) | Inferred rather than measured piste-direction position |
| `source_frame_index`, `container_slot` | (T,) | Source frame indices |
| `e_x`, `smpl_scale`, `segment_id`, `provenance` | varies | Coordinate, retargeting and lineage metadata |

Coordinates are z-up. Left-handed fencers are mirrored to right-handed robot motion. The original data files and checksums are retained unchanged. Package `manifest.csv`, `summary.json` and `files.sha256` document and verify the release. Internal lineage identifiers inside the immutable NPZ files are preserved for provenance.

```python
import numpy as np
from pathlib import Path
p = next(Path('data/fencing/segments').glob('*.npz'))
with np.load(p, allow_pickle=False) as clip:
    qpos = clip['qpos']
    fps = float(clip['fps'])
    names = clip['joint_names']
    confidence = clip['weapon_arm_weight']
```

Run `python tools/verify_data.py --pkg data/fencing` to check hashes, joint order, frame counts, finite values and quaternion normalization. Use `bash setup/prepare_fencing_data.sh` after setting up dependencies to convert to 50 Hz by time interpolation and attach the tracking masks. Original motion files are not overwritten.

## Limitations and evaluation

Monocular pose and depth estimates are imperfect, especially during occlusion and fast actions. The dataset contains inferred piste positions (approximately 4.89% of frames); these are not direct measurements. Retargeting introduces body-scale and forearm-direction errors and occasional self-intersections. Fixed hand geometry does not model grip changes. Some reference clips end during dynamic motions. Data are single-fencer robot trajectories, not a synchronized two-agent interaction benchmark. There is no predefined independent train/test benchmark; split by source bout rather than randomly splitting frames or near-duplicate clips to avoid leakage.
