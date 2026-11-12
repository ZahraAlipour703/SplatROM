#!/usr/bin/env python3
"""Audit one downloaded SHOW3D sample before research experiments.

Outputs data-quality, confidence, joint-parameter, timestamp, and wrist/world
coordinate-consistency reports. Does NOT claim clinical joint-angle accuracy.
Requires only Python stdlib and NumPy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
import numpy as np

def pct(x, q):
    return float(np.percentile(np.asarray(x, dtype=float), q)) if x else None

def summary(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": pct(vals, 95),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }

def angle_deg(a, b, c):
    a, b, c = (np.asarray(v, dtype=np.float64) for v in (a, b, c))
    u, v = a-b, c-b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-8 or nv < 1e-8:
        return float("nan")
    cosine = float(np.clip(np.dot(u, v)/(nu*nv), -1.0, 1.0))
    return 180.0 - math.degrees(math.acos(cosine))

FINGERS = {
    "index":  (8, 9, 10, 1),
    "middle": (11, 12, 13, 2),
    "ring":   (14, 15, 16, 3),
    "pinky":  (17, 18, 19, 4),
}

def landmark_angle_proxies(points):
    out = {}
    for name, (prox, mid, dist, tip) in FINGERS.items():
        out[f"{name}_pip_proxy_deg"] = angle_deg(points[prox], points[mid], points[dist])
        out[f"{name}_dip_proxy_deg"] = angle_deg(points[mid], points[dist], points[tip])
        # This is explicitly a palm-centre-anchored proxy, not a true MCP angle.
        out[f"{name}_mcp_proxy_deg"] = angle_deg(points[20], points[prox], points[mid])
    out["thumb_ip_proxy_deg"] = angle_deg(points[7], points[6], points[0])
    return out

def valid_xyz(x, shape):
    try:
        arr = np.asarray(x, dtype=np.float64)
        return arr.shape == shape and np.isfinite(arr).all(), arr
    except Exception:
        return False, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/show3d")
    ap.add_argument("--subject", default="ISH822")
    ap.add_argument("--scene", default="aria_inspecting_3ab0")
    ap.add_argument("--out", default="data/processed/show3d/ISH822__aria_inspecting_3ab0/audit")
    ap.add_argument("--threshold", type=float, default=0.1)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_path = root / "hand_pose" / "v3" / "scenes" / args.subject / args.scene / "hand_pose_umetrack.json"
    if not pose_path.exists():
        raise FileNotFoundError(f"Missing annotation: {pose_path}")

    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", payload) if isinstance(payload, dict) else payload
    if not isinstance(frames, dict):
        raise ValueError("Expected flat JSON object keyed by frame index.")

    frame_keys = sorted(frames.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    rows = []
    per_hand = {
        "left": {"conf": [], "world_errors_mm": [], "native_angle_lengths": [], "world_frames": 0, "accepted": 0, "high_conf": 0, "local_world_pairs": 0, "missing_world": 0, "malformed": 0},
        "right": {"conf": [], "world_errors_mm": [], "native_angle_lengths": [], "world_frames": 0, "accepted": 0, "high_conf": 0, "local_world_pairs": 0, "missing_world": 0, "malformed": 0},
    }
    timestamps = []
    frame_missing_counts = []
    native_angle_rows = []
    proxy_angle_rows = []
    local_world_errors = []

    for key in frame_keys:
        frame = frames[key]
        if not isinstance(frame, dict):
            continue
        idx = frame.get("index", int(key) if str(key).isdigit() else key)
        ts = frame.get("timestamp")
        try:
            if ts is not None:
                timestamps.append((int(idx), float(ts)))
        except (ValueError, TypeError):
            pass
        missing = frame.get("missing_cameras") or []
        frame_missing_counts.append(len(missing))
        hands = frame.get("hand_poses") or {}
        row = {"frame_index": idx, "timestamp": ts, "missing_camera_count": len(missing)}
        for hand_key, name in (("0", "left"), ("1", "right")):
            hand = hands.get(hand_key)
            p = per_hand[name]
            row[f"{name}_present"] = int(isinstance(hand, dict))
            if not isinstance(hand, dict):
                row[f"{name}_confidence"] = None
                row[f"{name}_accepted"] = 0
                continue

            conf = hand.get("confidence")
            try:
                conf = float(conf) if conf is not None else float("nan")
            except (ValueError, TypeError):
                conf = float("nan")
            if math.isfinite(conf):
                p["conf"].append(conf)
            row[f"{name}_confidence"] = conf if math.isfinite(conf) else None

            world_ok, world = valid_xyz(hand.get("landmarks_3d_mm"), (21, 3))
            local_ok, local = valid_xyz(hand.get("landmarks_3d_mm_local"), (21, 3))
            rot_ok, rot = valid_xyz(hand.get("wrist_rotation"), (3, 3))
            tran_ok, tran = valid_xyz(hand.get("wrist_translation"), (3,))
            angles = hand.get("joint_angles")
            try:
                angles = np.asarray(angles, dtype=np.float64).reshape(-1)
                angles_ok = angles.size == 22 and np.isfinite(angles).all()
            except Exception:
                angles = np.asarray([], dtype=float)
                angles_ok = False

            accepted = bool(conf > args.threshold and world_ok and rot_ok and tran_ok)
            high_conf = bool(conf >= 0.5 and world_ok and rot_ok and tran_ok)
            p["accepted"] += int(accepted)
            p["high_conf"] += int(high_conf)
            p["world_frames"] += int(world_ok)
            p["missing_world"] += int(not world_ok)
            p["malformed"] += int(not world_ok and hand.get("landmarks_3d_mm") is not None)
            row[f"{name}_accepted"] = int(accepted)
            row[f"{name}_high_conf"] = int(high_conf)
            row[f"{name}_world_geometry"] = int(world_ok)
            row[f"{name}_local_geometry"] = int(local_ok)
            row[f"{name}_joint_angles_22"] = int(angles_ok)

            if angles_ok:
                p["native_angle_lengths"].append(int(angles.size))
                nr = {"frame_index": idx, "timestamp": ts, "hand": name}
                for j, val in enumerate(angles):
                    nr[f"joint_{j:02d}_rad"] = float(val)
                    nr[f"joint_{j:02d}_deg"] = float(np.degrees(val))
                native_angle_rows.append(nr)

            if accepted:
                proxies = landmark_angle_proxies(world)
                pr = {"frame_index": idx, "timestamp": ts, "hand": name, "confidence": conf}
                pr.update(proxies)
                proxy_angle_rows.append(pr)

            if world_ok and local_ok and rot_ok and tran_ok:
                # Dataset contract: wrist_rotation is world-from-wrist and
                # wrist_translation is mm; local landmarks are in wrist coordinates.
                reconstructed = (rot @ local.T).T + tran.reshape(1, 3)
                err = np.linalg.norm(reconstructed - world, axis=1)
                mean_err = float(np.mean(err))
                p["world_errors_mm"].append(mean_err)
                p["local_world_pairs"] += 1
                local_world_errors.extend(err.tolist())
                row[f"{name}_local_to_world_mean_error_mm"] = mean_err
            else:
                row[f"{name}_local_to_world_mean_error_mm"] = None

        rows.append(row)

    # Timestamp cadence and monotonicity
    ordered_ts = sorted(timestamps, key=lambda x: x[0])
    diffs = np.diff([x[1] for x in ordered_ts]) if len(ordered_ts) > 1 else np.asarray([])
    positive_diffs = diffs[diffs > 0]
    median_dt = float(np.median(positive_diffs)) if len(positive_diffs) else None
    inferred_fps = (1.0 / median_dt) if median_dt and median_dt > 0 else None

    report = {
        "dataset": "SHOW3D",
        "subject": args.subject,
        "scene": args.scene,
        "frames_in_json": len(frames),
        "timestamps": {
            "count": len(ordered_ts),
            "strictly_increasing": bool(np.all(diffs > 0)) if len(diffs) else None,
            "median_delta_seconds": median_dt,
            "inferred_fps": inferred_fps,
            "duration_seconds": (ordered_ts[-1][1] - ordered_ts[0][1]) if len(ordered_ts) > 1 else None,
        },
        "missing_cameras_per_frame": summary(frame_missing_counts),
        "hands": {},
        "local_to_world_consistency": {
            "paired_landmark_error_mm": summary(local_world_errors),
            "definition": "R_world_from_wrist @ landmarks_3d_mm_local + t_world_from_wrist compared with landmarks_3d_mm",
        },
        "outputs": {
            "frame_quality_csv": str(out_dir / "frame_quality.csv"),
            "native_joint_parameters_csv": str(out_dir / "umetrack_joint_parameters.csv"),
            "landmark_angle_proxies_csv": str(out_dir / "landmark_angle_proxies.csv"),
        },
        "interpretation_notes": [
            "UmeTrack joint_angles contains 22 radians-valued parameters; columns are kept index-based until the calibrated subject profile defines the joint mapping.",
            "Landmark angle outputs are geometric proxies. MCP values use the palm centre as an anchor and are not clinical-grade joint angles.",
            "SHOW3D world coordinates are in the back-mounted rig frame, which moves with the participant.",
            "Confidence is the automatic solver's score, not ground-truth measurement uncertainty.",
        ],
    }
    for name, p in per_hand.items():
        report["hands"][name] = {
            "confidence": summary(p["conf"]),
            "accepted_frames_conf_gt_threshold_and_world_pose": p["accepted"],
            "high_confidence_frames_conf_ge_0_5_and_world_pose": p["high_conf"],
            "frames_with_world_landmarks": p["world_frames"],
            "frames_missing_world_landmarks": p["missing_world"],
            "malformed_non_null_world_landmarks": p["malformed"],
            "local_world_transform_pairs": p["local_world_pairs"],
            "mean_per_frame_local_world_error_mm": summary(p["world_errors_mm"]),
            "joint_angle_vector_lengths": sorted(set(p["native_angle_lengths"])),
        }

    def write_csv(path, data):
        if not data:
            path.write_text("no_rows\n", encoding="utf-8")
            return
        fields = []
        for item in data:
            for k in item:
                if k not in fields:
                    fields.append(k)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    write_csv(out_dir / "frame_quality.csv", rows)
    write_csv(out_dir / "umetrack_joint_parameters.csv", native_angle_rows)
    write_csv(out_dir / "landmark_angle_proxies.csv", proxy_angle_rows)
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\n[OK] Audit reports written to:", out_dir)

if __name__ == "__main__":
    main()
