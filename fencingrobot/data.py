"""Reading the G1 motion package (data/fencing) and exporting it for SuperMimic-Reproduced / mjlab.

Package segment npz (30 fps, already de-duplicated and resampled to real time):
  qpos (T,36) = root xyz (m) + root quat **wxyz** + 29 joints (G1_JOINT_NAMES order), joint_names, fps
  weapon_arm_weight (T,), weapon_arm_flags (T,), weapon_arm_observed / weapon_arm_interpolated (T,), legs_unreliable (T,),
  no_track_mask (T,22) over SMPL-X body joints, mirrored, grip_type, weapon_side_original, source_frame_index, segment_id ...
SuperMimic CSV (no header, one row per frame): root xyz, root quat **xyzw**, 29 joints; converted by its
scripts/data/csv_to_npz_batch.py (--input-fps 30 --output-fps 50, time-based interpolation).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "data" / "fencing"

G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
# no_track_mask is over SMPL-X body joints; suggested G1 bodies to drop from the tracking reward (package README).
NO_TRACK_SMPLX_TO_G1_BODIES = {
    18: ("left_elbow_link",), 19: ("right_elbow_link",),
    20: ("left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link"),
    21: ("right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link"),
}
WEAPON_ARM_G1_BODIES = ("right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link",
                        "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link")
META_KEYS = ("weapon_arm_weight", "weapon_arm_flags", "weapon_arm_observed", "weapon_arm_interpolated", "legs_unreliable",
             "no_track_mask", "source_frame_index")


def segments(pkg: Path = PKG) -> list[Path]:
    return sorted((pkg / "segments").glob("*.npz"))


def load_segment(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def to_csv_rows(qpos: np.ndarray) -> np.ndarray:
    """(T,36) wxyz -> (T,36) xyzw (SuperMimic / LAFAN1-retarget CSV layout)."""
    out = qpos.astype(np.float64).copy()
    out[:, 3:7] = qpos[:, [4, 5, 6, 3]]
    return out


def export(pkg: Path, out_dir: Path, limit: int | None = None) -> dict:
    """Write <out>/csv/<seg>.csv, <out>/meta/<seg>.npz (per-frame masks at 30 fps + t_s), <out>/fps_map.csv, <out>/export.json."""
    (out_dir / "csv").mkdir(parents=True, exist_ok=True); (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    rows, n_frames = [], 0
    for p in segments(pkg)[:limit]:
        d = load_segment(p)
        assert tuple(str(j) for j in d["joint_names"]) == G1_JOINT_NAMES, p
        fps = float(d["fps"])
        np.savetxt(out_dir / "csv" / f"{p.stem}.csv", to_csv_rows(d["qpos"]), delimiter=",", fmt="%.7f")
        T = len(d["qpos"])
        meta = {k: d[k] for k in META_KEYS if k in d}
        meta.update(t_s=np.arange(T) / fps, fps=np.array(fps), mirrored=d["mirrored"], grip_type=d["grip_type"],
                    segment_id=d["segment_id"])
        np.savez_compressed(out_dir / "meta" / f"{p.stem}.npz", **meta)
        rows.append({"name": p.stem, "fps": fps}); n_frames += T
    with open(out_dir / "fps_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "fps"]); w.writeheader(); w.writerows(rows)
    info = dict(source=str(pkg), segments=len(rows), frames=n_frames, csv_layout="root xyz, root quat xyzw, 29 joints (G1_JOINT_NAMES)",
                meta="per-frame masks at the ORIGINAL 30 fps with t_s; resample to the control rate by time (nearest / min for weights)")
    json.dump(info, open(out_dir / "export.json", "w"), indent=1)
    return info


def resample_mask(t_src: np.ndarray, values: np.ndarray, fps_out: float, reduce: str = "nearest") -> np.ndarray:
    """Masks to the 50 Hz motion clock by time. reduce='min' for weights (conservative), 'any' for flags, 'nearest' otherwise."""
    t_out = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / fps_out)
    i0 = np.clip(np.searchsorted(t_src, t_out, side="right") - 1, 0, len(t_src) - 1); i1 = np.clip(i0 + 1, 0, len(t_src) - 1)
    a, b = values[i0], values[i1]
    if reduce == "min":
        return np.minimum(a, b)
    if reduce == "any":
        return np.logical_or(a, b) if a.dtype == bool else np.bitwise_or(a, b)
    near = np.where((t_out - t_src[i0]) <= (t_src[i1] - t_out), i0, i1)
    return values[near]


# ---------------------------------------------------------------------------------------------------------------------
# Masks on the 50 Hz motion clock, written INTO the converted motion npz (keys `fr_*`; the official MotionLoader only reads
# joint_pos / joint_vel / body_*_w, so extra keys are ignored by it and read by fencingrobot.sm_task).
MASK_KEYS = ("fr_weapon_arm_weight", "fr_no_track_mask", "fr_legs_unreliable")


def masks_on_converter_grid(n_in: int, fps_in: float, fps_out: float, weight: np.ndarray, no_track: np.ndarray,
                            legs: np.ndarray) -> dict:
    """Same output grid as the converter (SuperMimic-Reproduced motion_convert.resample_motion): output frame k at
    t = k / fps_out is interpolated from source frames i0 = floor(t * fps_in), i1 = i0 + 1. A mask applies to an output
    frame if it applies to EITHER source frame it was blended from (weight: minimum); on-grid frames use i0 only."""
    duration = (n_in - 1) / fps_in
    n_out = int(np.floor(duration * fps_out + 1e-6)) + 1
    phase = (np.arange(n_out) / fps_out) / duration if duration > 0 else np.zeros(n_out)
    x = phase * (n_in - 1)
    i0 = np.clip(np.floor(x).astype(np.int64), 0, n_in - 1); i1 = np.minimum(i0 + 1, n_in - 1)
    i1 = np.where(x - i0 < 1e-9, i0, i1)
    return {"fr_weapon_arm_weight": np.minimum(weight[i0], weight[i1]).astype(np.float32),
            "fr_no_track_mask": np.logical_or(no_track[i0], no_track[i1]),
            "fr_legs_unreliable": np.logical_or(legs[i0], legs[i1])}


def attach_masks(npz_dir: Path, pkg: Path = PKG) -> dict:
    """Add the fr_* mask arrays to every <npz_dir>/<seg>.npz (rewrites the file; motion arrays untouched)."""
    n_files = n_frames = 0
    stats = dict(arm_w_lt1=0, arm_w_zero=0, legs_unreliable=0, no_track_any=0)
    for seg in segments(pkg):
        dst = npz_dir / f"{seg.stem}.npz"
        if not dst.exists():
            raise FileNotFoundError(dst)
        d = load_segment(seg)
        with np.load(dst) as z:
            motion = {k: z[k] for k in z.files if k not in MASK_KEYS}
        T = motion["joint_pos"].shape[0]
        assert abs(float(motion["fps"][0]) - 50.0) < 1e-6, (dst, motion["fps"])
        m = masks_on_converter_grid(len(d["qpos"]), float(d["fps"]), 50.0, d["weapon_arm_weight"].astype(np.float64),
                                    d["no_track_mask"].astype(bool), d["legs_unreliable"].astype(bool))
        assert len(m["fr_weapon_arm_weight"]) == T, (dst, T, len(m["fr_weapon_arm_weight"]))
        np.savez(dst, **motion, **m)
        n_files += 1; n_frames += T
        stats["arm_w_lt1"] += int((m["fr_weapon_arm_weight"] < 1).sum()); stats["arm_w_zero"] += int((m["fr_weapon_arm_weight"] == 0).sum())
        stats["legs_unreliable"] += int(m["fr_legs_unreliable"].sum()); stats["no_track_any"] += int(m["fr_no_track_mask"].any(1).sum())
    extra = sorted(p.stem for p in npz_dir.glob("*.npz") if not (pkg / "segments" / p.name).exists())
    return dict(npz_dir=str(npz_dir), files=n_files, frames=n_frames, hours=n_frames / 50 / 3600, extra_npz_not_in_package=extra,
                **{f"frames_{k}": v for k, v in stats.items()})
