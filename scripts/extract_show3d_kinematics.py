#!/usr/bin/env python3
"""Export per-frame 3D landmarks and approximate finger flexion angles from SHOW3D UmeTrack JSON.

The script uses only Python stdlib + NumPy. The official 21-landmark topology is:
0..4 tips; 5 wrist; 6..7 thumb intermediate/distal;
8..10 index proximal/intermediate/distal; 11..13 middle;
14..16 ring; 17..19 pinky; 20 palm center.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

FINGERS = {
    "index":  {"tip": 1, "prox": 8, "mid": 9, "dist": 10},
    "middle": {"tip": 2, "prox": 11, "mid": 12, "dist": 13},
    "ring":   {"tip": 3, "prox": 14, "mid": 15, "dist": 16},
    "pinky":  {"tip": 4, "prox": 17, "mid": 18, "dist": 19},
}

def flexion_deg(a, b, c):
    """0 degrees means straight; larger means more bent."""
    a, b, c = (np.asarray(x, dtype=np.float64) for x in (a, b, c))
    u, v = a - b, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-8 or nv < 1e-8:
        return float("nan")
    cosine = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    internal = math.degrees(math.acos(float(cosine)))
    return 180.0 - internal

def compute_angles(points):
    out = {}
    palm = points[20]
    for finger, ids in FINGERS.items():
        # The palm centre is used as the proximal anchor. This is a proxy
        # for MCP flexion, not a clinical-grade anatomical MCP angle.
        out[f"{finger}_mcp_proxy_deg"] = flexion_deg(palm, points[ids["prox"]], points[ids["mid"]])
        out[f"{finger}_pip_deg"] = flexion_deg(points[ids["prox"]], points[ids["mid"]], points[ids["dist"]])
        out[f"{finger}_dip_deg"] = flexion_deg(points[ids["mid"]], points[ids["dist"]], points[ids["tip"]])

    # The released 21-point topology gives only two thumb intermediate/distal
    # landmarks plus its tip. Report an explicit proxy and IP angle.
    # Landmark order: 5=wrist, 6=thumb intermediate, 7=thumb distal, 0=thumb tip.
    # These remain geometric proxies, not clinical anatomical joint ground truth.
    out["thumb_mcp_proxy_deg"] = flexion_deg(points[5], points[6], points[7])
    out["thumb_ip_proxy_deg"] = flexion_deg(points[6], points[7], points[0])
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--out", default="data/processed/show3d/sample")
    parser.add_argument("--min-confidence", type=float, default=0.1)
    args = parser.parse_args()

    annotation_path = Path(args.annotation)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(annotation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Current v3 files use a JSON object keyed by frame index.
    frames = payload.get("frames", payload) if isinstance(payload, dict) else payload
    if not isinstance(frames, dict):
        raise ValueError("Unexpected SHOW3D annotation format: expected a frame-indexed JSON object.")

    angle_rows = []
    landmark_rows = []
    valid_counts = {"left": 0, "right": 0}
    confidence_values = {"left": [], "right": []}
    frame_keys = sorted(frames.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    for key in frame_keys:
        frame = frames[key]
        if not isinstance(frame, dict):
            continue
        frame_index = int(frame.get("index", key)) if str(frame.get("index", key)).isdigit() else key
        timestamp = frame.get("timestamp")
        hands = frame.get("hand_poses", {}) or {}

        for hand_key, hand_name in (("0", "left"), ("1", "right")):
            hand = hands.get(hand_key)
            if not isinstance(hand, dict):
                continue
            confidence = hand.get("confidence")
            if confidence is not None:
                try:
                    confidence = float(confidence)
                    confidence_values[hand_name].append(confidence)
                except (TypeError, ValueError):
                    confidence = float("nan")

            raw_points = hand.get("landmarks_3d_mm")
            if raw_points is None:
                continue
            points = np.asarray(raw_points, dtype=np.float64)
            if points.shape != (21, 3) or not np.isfinite(points).all():
                continue
            if confidence is not None and np.isfinite(confidence) and confidence <= args.min_confidence:
                continue

            valid_counts[hand_name] += 1
            angles = compute_angles(points)
            row = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "hand": hand_name,
                "confidence": confidence,
            }
            row.update(angles)
            angle_rows.append(row)

            for landmark_idx, xyz in enumerate(points):
                landmark_rows.append({
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "hand": hand_name,
                    "landmark_index": landmark_idx,
                    "x_mm": float(xyz[0]),
                    "y_mm": float(xyz[1]),
                    "z_mm": float(xyz[2]),
                    "confidence": confidence,
                })

    if not angle_rows:
        raise RuntimeError(
            "No valid 3D hand frames found. Check whether the annotation file is "
            "hand_pose_umetrack.json and whether this scene has accepted world landmarks."
        )

    angle_fields = list(angle_rows[0].keys())
    landmark_fields = list(landmark_rows[0].keys())

    angle_csv = out_dir / "hand_kinematics.csv"
    landmark_csv = out_dir / "landmarks_3d.csv"
    with open(angle_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=angle_fields)
        writer.writeheader()
        writer.writerows(angle_rows)
    with open(landmark_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=landmark_fields)
        writer.writeheader()
        writer.writerows(landmark_rows)

    summary = {
        "annotation_file": str(annotation_path),
        "frames_in_annotation": len(frames),
        "valid_hand_frames": valid_counts,
        "min_confidence_threshold": args.min_confidence,
        "mean_reported_confidence": {
            hand: (float(np.mean(vals)) if vals else None)
            for hand, vals in confidence_values.items()
        },
        "outputs": {
            "kinematics_csv": str(angle_csv),
            "landmarks_csv": str(landmark_csv),
        },
        "angle_convention": "flexion_deg = 180 - internal_angle; 0 is straight",
        "limitations": [
            "MCP angles are palm-centre-anchored proxies, not clinical-grade anatomical joint angles.",
            "The dataset provides 21 landmarks; thumb base/IP values are approximate proxies.",
            "Use the source joint_angles/MANO parameters and validate topology before making clinical claims.",
            "This is a dataset inspection baseline, not yet a 4D Gaussian Splatting reconstruction."
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\n[OK] Wrote: {angle_csv}")
    print(f"[OK] Wrote: {landmark_csv}")

if __name__ == "__main__":
    main()
