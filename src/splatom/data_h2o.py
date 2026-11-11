"""H2O dataset parsing and CPU-side hand-angle baseline utilities.

H2O hand_pose records contain two 64-value blocks:
    [valid_flag, 21 * (x, y, z)] for the left hand, then the same for right.
The 21-point topology follows the H2OPlayer skeleton edges:
    wrist(0) -> chains (1..4), (5..8), (9..12), (13..16), (17..20).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from splatom.geometry.kinematics import joint_angle_degrees


FINGER_CHAINS: dict[str, tuple[int, int, int, int]] = {
    "chain_1": (1, 2, 3, 4),
    "chain_2": (5, 6, 7, 8),
    "chain_3": (9, 10, 11, 12),
    "chain_4": (13, 14, 15, 16),
    "chain_5": (17, 18, 19, 20),
}
ANGLE_FEATURES = tuple(
    f"{chain}_{location}_angle_deg"
    for chain in FINGER_CHAINS
    for location in ("base", "joint1", "joint2")
)


def parse_h2o_hand_pose_values(values: np.ndarray | list[float]) -> dict[str, np.ndarray | None]:
    """Parse one H2O hand_pose record into left/right arrays shaped (21, 3).

    The validity flags are at indices 0 and 64. Coordinates are read from
    [1:64] and [65:128], respectively. Missing hands are returned as None.
    """
    row = np.asarray(values, dtype=np.float64).reshape(-1)
    if row.size != 128:
        raise ValueError(f"Expected 128 values in H2O hand_pose record, got {row.size}")

    left_valid = np.isfinite(row[0]) and int(round(float(row[0]))) == 1
    right_valid = np.isfinite(row[64]) and int(round(float(row[64]))) == 1

    left = row[1:64].reshape(21, 3).copy() if left_valid else None
    right = row[65:128].reshape(21, 3).copy() if right_valid else None

    # H2O marks an entire hand as invalid with a flag. Non-finite individual
    # points remain in place so angle calculations can mark only affected joints NaN.
    return {"left": left, "right": right}


def load_h2o_hand_pose_file(path: str | Path) -> dict[str, np.ndarray | None]:
    """Load and parse a one-line H2O hand_pose text file."""
    path = Path(path)
    try:
        values = np.fromstring(path.read_text(encoding="utf-8").replace(",", " "), sep=" ")
    except OSError as exc:
        raise OSError(f"Could not read H2O pose file: {path}") from exc
    return parse_h2o_hand_pose_values(values)


def hand_angle_features(joints: np.ndarray) -> dict[str, float]:
    """Compute geometric internal angles along each H2O finger chain.

    For each chain, `base` uses wrist -> base -> next landmark. `joint1` and
    `joint2` use the next three-point windows within the chain. These are
    geometric landmark-chain angles, not validated clinical goniometer angles.
    """
    points = np.asarray(joints, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"Expected joints with shape (21, 3), got {points.shape}")

    result: dict[str, float] = {}
    wrist = points[0]
    for chain_name, (base, joint1, joint2, tip) in FINGER_CHAINS.items():
        triplets = (
            (wrist, points[base], points[joint1]),
            (points[base], points[joint1], points[joint2]),
            (points[joint1], points[joint2], points[tip]),
        )
        labels = ("base", "joint1", "joint2")
        for label, (parent, joint, child) in zip(labels, triplets):
            value = float(joint_angle_degrees(parent, joint, child))
            result[f"{chain_name}_{label}_angle_deg"] = value
    return result


def _metadata_from_pose_path(path: Path, subject_root: Path) -> dict[str, Any]:
    rel = path.relative_to(subject_root)
    # Expected: <sequence>/<take>/cam4/hand_pose/<frame>.txt
    if len(rel.parts) < 5:
        raise ValueError(f"Unexpected H2O pose path layout: {rel}")
    return {
        "sequence": rel.parts[0],
        "take": rel.parts[1],
        "camera": rel.parts[2],
        "frame": int(path.stem) if path.stem.isdigit() else path.stem,
    }


def run_h2o_pose_baseline(subject_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Build per-frame left/right hand angle CSV and a summary JSON for cam4."""
    subject_root = Path(subject_root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_files = sorted(
        p for p in subject_root.rglob("*.txt")
        if p.parent.name == "hand_pose" and p.parent.parent.name == "cam4"
    )
    if not pose_files:
        raise FileNotFoundError(
            f"No cam4/hand_pose/*.txt files found below {subject_root}. "
            "Check that the H2O subject archive was extracted correctly."
        )

    csv_path = output_dir / "subject1_cam4_angles.csv"
    fieldnames = [
        "subject", "sequence", "take", "camera", "frame", "hand", "hand_present",
        *ANGLE_FEATURES,
    ]

    hand_counts = {"left_present": 0, "right_present": 0, "left_missing": 0, "right_missing": 0}
    angle_values: dict[str, list[float]] = {name: [] for name in ANGLE_FEATURES}
    sequence_keys: set[tuple[str, str]] = set()
    malformed_files: list[dict[str, str]] = []
    rows_written = 0

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for pose_path in pose_files:
            try:
                hands = load_h2o_hand_pose_file(pose_path)
                metadata = _metadata_from_pose_path(pose_path, subject_root)
            except (OSError, ValueError) as exc:
                malformed_files.append({"file": str(pose_path), "error": str(exc)})
                continue

            sequence_keys.add((metadata["sequence"], str(metadata["take"])))
            for side in ("left", "right"):
                joints = hands[side]
                row: dict[str, Any] = {
                    "subject": subject_root.name,
                    **metadata,
                    "hand": side,
                    "hand_present": int(joints is not None),
                }
                if joints is None:
                    hand_counts[f"{side}_missing"] += 1
                    row.update({name: "" for name in ANGLE_FEATURES})
                else:
                    hand_counts[f"{side}_present"] += 1
                    features = hand_angle_features(joints)
                    row.update(features)
                    for name, value in features.items():
                        if np.isfinite(value):
                            angle_values[name].append(value)
                writer.writerow(row)
                rows_written += 1

    angle_summary: dict[str, dict[str, float | int | None]] = {}
    for name, values in angle_values.items():
        arr = np.asarray(values, dtype=np.float64)
        angle_summary[name] = {
            "count": int(arr.size),
            "mean_deg": float(np.mean(arr)) if arr.size else None,
            "std_deg": float(np.std(arr)) if arr.size else None,
        }

    summary: dict[str, Any] = {
        "dataset": "H2O",
        "subject": subject_root.name,
        "camera": "cam4",
        "pose_files_found": len(pose_files),
        "rows_written": rows_written,
        "sequences_and_takes": len(sequence_keys),
        "hand_annotations": hand_counts,
        "angle_summary": angle_summary,
        "malformed_file_count": len(malformed_files),
        "malformed_files": malformed_files[:20],
        "outputs": {"angles_csv": str(csv_path.resolve())},
        "interpretation_note": (
            "Angles are geometric 3D internal angles from H2O's 21-point hand skeleton. "
            "The base angle uses wrist landmark 0 as the preceding point; these values "
            "are not yet validated clinical MCP/PIP/DIP goniometry measurements."
        ),
        "topology_source": "https://github.com/taeinkwon/h2oplayer",
        "dataset_source": "https://github.com/taeinkwon/h2odataset",
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
