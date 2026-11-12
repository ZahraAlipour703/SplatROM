#!/usr/bin/env python3
"""Audit SHOW3D camera geometry and 3D-to-2D hand reprojection.

Uses the official facebookresearch/show3d API. This is a dataset consistency
check, not independent motion-capture ground truth or clinical validation.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from show3d.camera import project_to_image
from show3d.dataset import Show3DFrameRef, Show3DPaths, load_camera_calibration, load_missing_cameras

VIEWS = ("rig0", "rig1", "rig2", "rig3", "headset0")

def as_3d(value: Any):
    try: a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError): return None
    return a if a.shape == (21, 3) and np.isfinite(a).all() else None

def as_2d(value: Any):
    out = np.full((21, 2), np.nan, dtype=np.float64)
    if not isinstance(value, list) or len(value) != 21: return out
    for i, p in enumerate(value):
        if p is None: continue
        try:
            a = np.asarray(p, dtype=np.float64).reshape(-1)
            if a.size == 2 and np.isfinite(a).all(): out[i] = a
        except (TypeError, ValueError): pass
    return out

def describe(values):
    a = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if not len(a): return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {"count": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95))}

def video_info(path: Path):
    if not path.exists(): return {"exists": False}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened(): return {"exists": True, "opened": False}
        return {"exists": True, "opened": True, "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally: cap.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/show3d")
    ap.add_argument("--subject", default="ISH822")
    ap.add_argument("--scene", default="aria_inspecting_3ab0")
    ap.add_argument("--hand-version", default="v3")
    ap.add_argument("--object-version", default="v2")
    ap.add_argument("--out", default="results/show3d_geometry_audit")
    ap.add_argument("--min-confidence", type=float, default=0.1)
    args = ap.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref0 = Show3DFrameRef(args.subject, args.scene, 0)
    paths = Show3DPaths(root, hand_pose_version=args.hand_version, object_pose_version=args.object_version)
    hand_file, object_file = paths.hand_pose_path(ref0), paths.object_pose_path(ref0)
    info_file = paths.frame_info_path(ref0)
    field_file = root / "interaction_field/v1/scenes" / args.subject / args.scene / "interaction_field.jsonl"
    for p in (hand_file, object_file, info_file):
        if not p.exists(): raise FileNotFoundError(p)
    hand_data = json.loads(hand_file.read_text(encoding="utf-8"))
    frame_ids = sorted(int(k) for k in hand_data)
    if not frame_ids: raise RuntimeError("No frames in hand annotation")

    cal_cache, missing_cache = {}, {}
    rig_cal, rig_centers = {}, {}
    missing_counts = {v: 0 for v in VIEWS}
    projection = {"left": [], "right": []}
    per_frame_rows = []
    headset_valid = skipped_headset = 0
    confidence = {"left": [], "right": []}
    world_counts = {"left": 0, "right": 0}

    for frame_index in frame_ids:
        ref = Show3DFrameRef(args.subject, args.scene, frame_index)
        cals = {}
        missing = load_missing_cameras(info_file, frame_index, cache=missing_cache)
        for view in VIEWS:
            if view in missing: missing_counts[view] += 1
            cal = load_camera_calibration(paths.camera_calibration_path(ref, view), frame_index, cache=cal_cache)
            cals[view] = cal
            if cal is None or cal.t_world_from_camera is None: continue
            T = np.asarray(cal.t_world_from_camera, dtype=np.float64)
            R = T[:3, :3]
            ortho = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
            det = float(np.linalg.det(R))
            if view.startswith("rig") and frame_index == frame_ids[0]:
                rig_centers[view] = T[:3, 3].copy()
                rig_cal[view] = {"fx": cal.fx, "fy": cal.fy, "cx": cal.cx, "cy": cal.cy,
                    "width": cal.image_width, "height": cal.image_height,
                    "rotation_orthogonality_fro": ortho, "rotation_determinant": det,
                    "camera_center_world_mm": T[:3, 3].tolist()}
        headset = cals.get("headset0")
        if headset is None or headset.t_world_from_camera is None:
            skipped_headset += 1
            continue
        headset_valid += 1
        frame = hand_data[str(frame_index)]
        hands = frame.get("hand_poses", {}) if isinstance(frame, dict) else {}
        for hand_id, side in (("0", "left"), ("1", "right")):
            h = hands.get(hand_id) if isinstance(hands, dict) else None
            if not isinstance(h, dict): continue
            try: conf = float(h.get("confidence", float("nan")))
            except (TypeError, ValueError): conf = float("nan")
            if math.isfinite(conf): confidence[side].append(conf)
            if not math.isfinite(conf) or conf <= args.min_confidence: continue
            xyz = as_3d(h.get("landmarks_3d_mm"))
            if xyz is None: continue
            world_counts[side] += 1
            lm2d = h.get("landmarks_2d")
            ref2d = as_2d(lm2d.get("headset0") if isinstance(lm2d, dict) else None)
            try: uv, in_image = project_to_image(xyz, headset)
            except Exception: continue
            mask = np.isfinite(ref2d).all(axis=1) & in_image & np.isfinite(uv).all(axis=1)
            if not mask.any(): continue
            err = np.linalg.norm(uv[mask] - ref2d[mask], axis=1)
            projection[side].extend(err.tolist())
            per_frame_rows.append({"frame_index": frame_index, "hand": side, "confidence": conf,
                "matched_landmarks": int(mask.sum()), "mean_error_px": float(err.mean()),
                "median_error_px": float(np.median(err)), "p90_error_px": float(np.percentile(err, 90))})

    baselines = {}
    rig_names = sorted(rig_centers)
    for i, a in enumerate(rig_names):
        for b in rig_names[i+1:]: baselines[f"{a}-{b}_mm"] = float(np.linalg.norm(rig_centers[a]-rig_centers[b]))
    videos = {v: video_info(paths.scene_dir(ref0) / f"{v}.mp4") for v in VIEWS}

    object_data = json.loads(object_file.read_text(encoding="utf-8"))
    object_conf, object_pose_frames = [], set()
    for k, val in object_data.items():
        if isinstance(val, dict):
            try: c = float(val.get("confidence", 0.0))
            except (TypeError, ValueError): continue
            object_conf.append(c)
            if c > 0.5: object_pose_frames.add(int(k))

    field_summary = {"exists": field_file.exists(), "rows": 0, "unique_frames": 0,
        "frames_with_object_pose_conf_gt_0_5": 0, "malformed_arrays": 0, "left": [], "right": []}
    field_frames = set()
    if field_file.exists():
        with field_file.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line); field_summary["rows"] += 1
                sid = str(row.get("sample_id", ""))
                if ":" in sid:
                    try: field_frames.add(int(sid.rsplit(":", 1)[1]))
                    except ValueError: pass
                for side in ("left", "right"):
                    raw = row.get(f"{side}_to_object")
                    if raw is None: continue
                    try: arr = np.asarray(raw, dtype=np.float64)
                    except (TypeError, ValueError): field_summary["malformed_arrays"] += 1; continue
                    if arr.shape != (21, 3): field_summary["malformed_arrays"] += 1; continue
                    field_summary[side].extend(np.linalg.norm(arr, axis=1)[np.isfinite(arr).all(axis=1)].tolist())
    field_summary["unique_frames"] = len(field_frames)
    field_summary["frames_with_object_pose_conf_gt_0_5"] = len(field_frames & object_pose_frames)
    for side in ("left", "right"):
        vals = field_summary[side]
        s = describe(vals)
        a = np.asarray(vals, dtype=float)
        s["below_5mm_fraction"] = float(np.mean(a < 5)) if len(a) else None
        s["below_10mm_fraction"] = float(np.mean(a < 10)) if len(a) else None
        field_summary[side] = s
    report = {
        "dataset": "SHOW3D", "subject": args.subject, "scene": args.scene,
        "annotation_versions": {"hand_pose": args.hand_version, "object_pose": args.object_version, "interaction_field": "v1"},
        "hand_frame_count": len(frame_ids), "video_metadata": videos,
        "missing_counts_by_selected_view": missing_counts,
        "fixed_rig_camera_calibration": rig_cal, "pairwise_rig_baselines_mm": baselines,
        "headset0_pose_valid_frames": headset_valid, "frames_skipped_invalid_headset0_pose": skipped_headset,
        "world_hand_frames_above_confidence_threshold": world_counts,
        "mean_hand_confidence": {s: (float(np.mean(v)) if v else None) for s, v in confidence.items()},
        "headset0_reprojection_error_px": {s: describe(v) for s, v in projection.items()},
        "object_pose_v2": {"entries": len(object_data), "confidence_gt_0_5": len(object_pose_frames), "confidence": describe(object_conf)},
        "interaction_field_v1": {k: v for k, v in field_summary.items() if k not in ("left", "right")} | {
            "left_distance_to_nearest_object_vertex_mm": field_summary["left"],
            "right_distance_to_nearest_object_vertex_mm": field_summary["right"],
            "note": "Vector norms are unsigned distances to nearest mesh vertices, not binary contact labels."},
        "interpretation": [
            "The prior local-to-world residual near 1e-14 mm is an algebraic identity check, not independent camera validation.",
            "Headset0 reprojection compares projected 3D landmarks with provided 2D headset0 annotations; it is dataset-derived consistency evidence, not external mocap ground truth.",
            "Rig transforms are parsed and checked for rigidity by the official SHOW3D API.",
            "Interaction-field distances are exploratory proximity measurements, not contact ground truth.",
            "ISH822 is a development/train subject and must not be used as the held-out test subject in the paper."],
        "outputs": {"reprojection_csv": str(out / "headset0_reprojection.csv"), "report_json": str(out / "geometry_audit.json")}}
    fields = ["frame_index", "hand", "confidence", "matched_landmarks", "mean_error_px", "median_error_px", "p90_error_px"]
    with (out / "headset0_reprojection.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(per_frame_rows)
    (out / "geometry_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\n[OK] Geometry audit written to", out)

if __name__ == "__main__": main()
