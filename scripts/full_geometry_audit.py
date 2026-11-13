from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# SplatROM full-scene geometry audit
# Scene: ISH822 / aria_inspecting_3ab0
# Uses the official SHOW3D loader and camera projection implementation.
# All spatial quantities remain in millimeters.
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SHOW3D_API = ROOT / "show3d"
DATA_ROOT = ROOT / "data" / "raw" / "show3d"
FRAME_DIR = (
    ROOT / "data" / "processed" / "show3d"
    / "ISH822__aria_inspecting_3ab0" / "frames_10fps"
)
INDEX_PATH = FRAME_DIR / "index.jsonl"
INTERACTION_PATH = (
    DATA_ROOT / "interaction_field" / "v1" / "scenes"
    / "ISH822" / "aria_inspecting_3ab0" / "interaction_field.jsonl"
)
OUTPUT_DIR = ROOT / "data" / "processed" / "splatrom_diagnostics"
REPORT_PATH = OUTPUT_DIR / "full_geometry_report.json"
CSV_PATH = OUTPUT_DIR / "full_geometry_per_camera.csv"

SUBJECT = "ISH822"
SCENE = "aria_inspecting_3ab0"
OBJECT_ALIAS = "aria"
VIEWS = ("rig0", "rig1", "rig2", "rig3")
CONFIDENCE_THRESHOLD = 0.5
CAMERA_TRANSFORM_TOLERANCE = 1e-4
CAMERA_ROTATION_TOLERANCE = 1e-4
OBJECT_ROTATION_TOLERANCE = 1e-3
EXPECTED_TIMESTAMPS = 297
EXPECTED_IMAGE_ROWS = EXPECTED_TIMESTAMPS * len(VIEWS)


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"Required file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"Invalid JSONL at {path}:{line_number}: {exc}")
            if not isinstance(item, dict):
                fail(f"Expected JSON object at {path}:{line_number}")
            rows.append(item)
    return rows


def finite_array(value: Any, shape: tuple[int, ...] | None = None) -> bool:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if shape is not None and arr.shape != shape:
        return False
    return bool(np.isfinite(arr).all())


def rotation_metrics(rotation: np.ndarray) -> tuple[float, float]:
    orth_error = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
    )
    determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
    return orth_error, determinant_error


def validate_interaction_vectors(
    value: Any,
) -> tuple[bool, int, str | None]:
    if not isinstance(value, list):
        return False, 0, "not_list"
    if len(value) != 21:
        return False, 0, f"length_{len(value)}_not_21"
    valid = 0
    for joint_index, vector in enumerate(value):
        if vector is None:
            continue
        if not finite_array(vector, (3,)):
            return False, valid, f"joint_{joint_index}_not_finite_3vector"
        valid += 1
    if valid == 0:
        return False, 0, "no_non_null_vectors"
    return True, valid, None


def main() -> int:
    if not (SHOW3D_API / "show3d" / "dataset.py").is_file():
        fail(
            "SHOW3D API checkout not found. Expected: "
            f"{SHOW3D_API / 'show3d' / 'dataset.py'}"
        )
    if not INDEX_PATH.is_file():
        fail(f"Extracted image index not found: {INDEX_PATH}")

    # Import from the user's local SHOW3D checkout without installing it.
    sys.path.insert(0, str(SHOW3D_API))
    from show3d import camera
    from show3d.dataset import (
        Show3DDataset,
        Show3DFrameRef,
        load_object_mesh_with_faces,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading extracted-image index...")
    image_rows = load_jsonl(INDEX_PATH)
    interaction_rows = load_jsonl(INTERACTION_PATH)

    errors: list[str] = []
    warnings: list[str] = []

    # Group image records by sample ID and confirm exactly one record per view.
    rows_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        required = ("sample_id", "frame_index", "subject_id", "scene_id", "view", "image")
        missing = [key for key in required if key not in row]
        if missing:
            errors.append(f"Index row missing fields {missing}: {row}")
            continue
        rows_by_sample[str(row["sample_id"])].append(row)

    unique_records: list[dict[str, Any]] = []
    for sample_id, group in rows_by_sample.items():
        first = group[0]
        expected_id = (
            f"{first['subject_id']}/{first['scene_id']}:{int(first['frame_index']):06d}"
        )
        if sample_id != expected_id:
            errors.append(f"Sample ID mismatch: {sample_id} != {expected_id}")
        found_views = [str(row["view"]) for row in group]
        if len(found_views) != len(set(found_views)):
            errors.append(f"Duplicate camera row for {sample_id}: {found_views}")
        if set(found_views) != set(VIEWS):
            errors.append(
                f"View-set mismatch for {sample_id}: found={sorted(set(found_views))}, "
                f"expected={list(VIEWS)}"
            )
        if any(
            str(row["subject_id"]) != SUBJECT or str(row["scene_id"]) != SCENE
            for row in group
        ):
            errors.append(f"Unexpected subject/scene in group {sample_id}")
        unique_records.append(first)

    unique_records.sort(key=lambda row: int(row["frame_index"]))
    sample_ids = [str(row["sample_id"]) for row in unique_records]
    if len(image_rows) != EXPECTED_IMAGE_ROWS:
        errors.append(
            f"Expected {EXPECTED_IMAGE_ROWS} image rows, found {len(image_rows)}"
        )
    if len(unique_records) != EXPECTED_TIMESTAMPS:
        errors.append(
            f"Expected {EXPECTED_TIMESTAMPS} unique timestamps, found {len(unique_records)}"
        )
    if len(set(sample_ids)) != len(sample_ids):
        errors.append("Duplicate sample IDs in unique timestamp list")

    # Interaction Field is not loaded by Show3DDataset; validate its keyed JSONL
    # separately, then join by the exact SHOW3D sample_id.
    interactions: dict[str, dict[str, Any]] = {}
    for row in interaction_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            errors.append("Interaction Field row has no sample_id")
            continue
        if sample_id in interactions:
            errors.append(f"Duplicate Interaction Field sample_id: {sample_id}")
        interactions[sample_id] = row

    interaction_counts = {
        "frames_present": 0,
        "left_frames_valid": 0,
        "right_frames_valid": 0,
        "left_vectors_non_null": 0,
        "right_vectors_non_null": 0,
    }
    interaction_valid_by_sample: dict[str, dict[str, int]] = {}

    for sample_id in sample_ids:
        row = interactions.get(sample_id)
        if row is None:
            errors.append(f"Missing Interaction Field row: {sample_id}")
            continue
        interaction_counts["frames_present"] += 1
        frame_status: dict[str, int] = {}
        for side, key in (("left", "left_to_object"), ("right", "right_to_object")):
            valid, count, reason = validate_interaction_vectors(row.get(key))
            if not valid:
                errors.append(f"{sample_id} {key}: {reason}")
            else:
                interaction_counts[f"{side}_frames_valid"] += 1
                interaction_counts[f"{side}_vectors_non_null"] += count
            frame_status[side] = count
        interaction_valid_by_sample[sample_id] = frame_status

    # Load the official SHOW3D mesh and dataset frames.
    print("Loading official SHOW3D annotations and camera calibrations...")
    mesh = load_object_mesh_with_faces(OBJECT_ALIAS)
    if mesh is None:
        fail(f"SHOW3D object mesh not found for alias '{OBJECT_ALIAS}'")

    vertices_object = np.asarray(mesh.vertices_mm, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices_object.ndim != 2 or vertices_object.shape[1] != 3:
        fail(f"Unexpected object mesh vertex shape: {vertices_object.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        fail(f"Unexpected object mesh face shape: {faces.shape}")
    if not np.isfinite(vertices_object).all():
        fail("Object mesh contains non-finite vertices")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices_object)):
        fail("Object mesh contains out-of-range face indices")

    frame_refs = [
        Show3DFrameRef(
            subject_id=str(row["subject_id"]),
            scene_id=str(row["scene_id"]),
            frame_index=int(row["frame_index"]),
            object_alias=OBJECT_ALIAS,
        )
        for row in unique_records
    ]

    dataset = Show3DDataset(
        DATA_ROOT,
        frame_refs,
        hand_pose_version="v3",
        object_pose_version="v2",
        views=VIEWS,
        decode_images=False,
    )

    if len(dataset) != len(unique_records):
        errors.append(
            f"SHOW3D loader length {len(dataset)} != index timestamps {len(unique_records)}"
        )

    # First-seen calibration per camera is used to measure rig constancy.
    camera_reference: dict[str, dict[str, Any]] = {}
    camera_aggregate: dict[str, dict[str, Any]] = {
        view: {
            "frames_checked": 0,
            "image_files_ok": 0,
            "hands_checked": 0,
            "hand_landmarks_total": 0,
            "hand_landmarks_in_front": 0,
            "hand_landmarks_inside_image": 0,
            "object_frames_checked": 0,
            "object_vertices_total": 0,
            "object_vertices_in_front": 0,
            "object_vertices_inside_image": 0,
            "max_transform_deviation": 0.0,
            "max_intrinsics_deviation": 0.0,
            "max_rotation_orthogonality_error": 0.0,
            "max_rotation_determinant_error": 0.0,
            "resolution": None,
            "pose_sources": Counter(),
        }
        for view in VIEWS
    }

    frame_count_checked = 0
    object_pose_confidences: list[float] = []
    hand_confidences: dict[str, list[float]] = {"left": [], "right": []}
    object_rotation_max_orth_error = 0.0
    object_rotation_max_det_error = 0.0
    per_camera_rows: list[dict[str, Any]] = []

    print(f"Auditing {len(dataset)} timestamps x {len(VIEWS)} cameras...")

    for frame_idx in range(len(dataset)):
        frame = dataset[frame_idx]
        frame_ref = frame.frame
        sample_id = frame_ref.sample_id
        frame_count_checked += 1

        expected_record = unique_records[frame_idx]
        if sample_id != str(expected_record["sample_id"]):
            errors.append(
                f"Loader/index order mismatch at {frame_idx}: "
                f"{sample_id} != {expected_record['sample_id']}"
            )

        # Validate both hand poses once per timestamp.
        hands: dict[str, Any] = {
            "left": frame.left_hand,
            "right": frame.right_hand,
        }
        hand_points: dict[str, np.ndarray] = {}
        for side, hand in hands.items():
            if hand is None:
                errors.append(f"{sample_id}: missing {side} hand pose")
                continue
            confidence = float(hand.confidence)
            hand_confidences[side].append(confidence)
            if not math.isfinite(confidence) or confidence <= CONFIDENCE_THRESHOLD:
                errors.append(
                    f"{sample_id}: {side} hand confidence {confidence} "
                    f"<= {CONFIDENCE_THRESHOLD}"
                )
            if hand.landmarks_world_mm is None:
                errors.append(f"{sample_id}: {side} hand landmarks are None")
                continue
            points = np.asarray(hand.landmarks_world_mm, dtype=np.float64)
            if points.shape != (21, 3):
                errors.append(f"{sample_id}: {side} hand shape is {points.shape}, expected (21, 3)")
                continue
            if not np.isfinite(points).all():
                errors.append(f"{sample_id}: {side} hand has non-finite coordinates")
                continue
            hand_points[side] = points

        # Validate object 6DoF pose and transform the mesh into the rig/world frame.
        pose = frame.object_pose
        object_world_vertices: np.ndarray | None = None
        if pose is None:
            errors.append(f"{sample_id}: missing object pose")
        else:
            pose_confidence = float(pose.confidence)
            object_pose_confidences.append(pose_confidence)
            if not math.isfinite(pose_confidence) or pose_confidence <= CONFIDENCE_THRESHOLD:
                errors.append(
                    f"{sample_id}: object confidence {pose_confidence} <= {CONFIDENCE_THRESHOLD}"
                )
            rotation = np.asarray(pose.rotation, dtype=np.float64)
            translation = np.asarray(pose.translation_mm, dtype=np.float64)
            if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
                errors.append(f"{sample_id}: invalid object rotation matrix")
            elif translation.shape != (3,) or not np.isfinite(translation).all():
                errors.append(f"{sample_id}: invalid object translation vector")
            else:
                orth_error, det_error = rotation_metrics(rotation)
                object_rotation_max_orth_error = max(
                    object_rotation_max_orth_error, orth_error
                )
                object_rotation_max_det_error = max(
                    object_rotation_max_det_error, det_error
                )
                if orth_error > OBJECT_ROTATION_TOLERANCE or det_error > OBJECT_ROTATION_TOLERANCE:
                    warnings.append(
                        f"{sample_id}: object rotation is slightly non-rigid "
                        f"(orth={orth_error:.3g}, det_error={det_error:.3g})"
                    )
                object_world_vertices = np.asarray(
                    pose.pose_vertices(vertices_object), dtype=np.float64
                )
                if object_world_vertices.shape != vertices_object.shape:
                    errors.append(f"{sample_id}: posed object mesh shape changed unexpectedly")
                    object_world_vertices = None
                elif not np.isfinite(object_world_vertices).all():
                    errors.append(f"{sample_id}: posed object mesh has non-finite coordinates")
                    object_world_vertices = None

        # Per-camera checks.
        for view_name in VIEWS:
            stats = camera_aggregate[view_name]
            stats["frames_checked"] += 1
            view = frame.views.get(view_name)
            if view is None:
                errors.append(f"{sample_id}/{view_name}: view absent from loader")
                continue
            if view.missing:
                errors.append(f"{sample_id}/{view_name}: camera marked missing")
            calibration = view.calibration
            if calibration is None:
                errors.append(f"{sample_id}/{view_name}: calibration missing")
                continue
            T = calibration.t_world_from_camera
            if T is None:
                errors.append(f"{sample_id}/{view_name}: T_world_from_camera is None")
                continue
            T = np.asarray(T, dtype=np.float64)
            if T.shape != (4, 4) or not np.isfinite(T).all():
                errors.append(f"{sample_id}/{view_name}: invalid 4x4 camera transform")
                continue

            intrinsics = np.asarray(
                [calibration.fx, calibration.fy, calibration.cx, calibration.cy],
                dtype=np.float64,
            )
            if not np.isfinite(intrinsics).all() or calibration.fx <= 0 or calibration.fy <= 0:
                errors.append(f"{sample_id}/{view_name}: invalid camera intrinsics")

            if calibration.image_width <= 0 or calibration.image_height <= 0:
                errors.append(f"{sample_id}/{view_name}: invalid calibrated image dimensions")

            camera_rotation = T[:3, :3]
            orth_error, det_error = rotation_metrics(camera_rotation)
            stats["max_rotation_orthogonality_error"] = max(
                stats["max_rotation_orthogonality_error"], orth_error
            )
            stats["max_rotation_determinant_error"] = max(
                stats["max_rotation_determinant_error"], det_error
            )
            if orth_error > CAMERA_ROTATION_TOLERANCE or det_error > CAMERA_ROTATION_TOLERANCE:
                errors.append(
                    f"{sample_id}/{view_name}: camera rotation invalid "
                    f"(orth={orth_error:.3g}, det_error={det_error:.3g})"
                )
            if not np.allclose(T[3], [0, 0, 0, 1], atol=CAMERA_TRANSFORM_TOLERANCE):
                errors.append(f"{sample_id}/{view_name}: invalid homogeneous transform bottom row")

            stats["pose_sources"][str(getattr(calibration, "pose_source", "unknown"))] += 1
            resolution = [int(calibration.image_width), int(calibration.image_height)]
            reference = camera_reference.get(view_name)
            if reference is None:
                camera_reference[view_name] = {
                    "T": T.copy(),
                    "intrinsics": intrinsics.copy(),
                    "resolution": resolution,
                }
                stats["resolution"] = f"{resolution[0]}x{resolution[1]}"
            else:
                transform_deviation = float(np.max(np.abs(T - reference["T"])))
                intrinsics_deviation = float(np.max(np.abs(intrinsics - reference["intrinsics"])))
                stats["max_transform_deviation"] = max(
                    stats["max_transform_deviation"], transform_deviation
                )
                stats["max_intrinsics_deviation"] = max(
                    stats["max_intrinsics_deviation"], intrinsics_deviation
                )
                if transform_deviation > CAMERA_TRANSFORM_TOLERANCE:
                    errors.append(
                        f"{sample_id}/{view_name}: rig camera transform drift "
                        f"{transform_deviation:.6g}"
                    )
                if intrinsics_deviation > CAMERA_TRANSFORM_TOLERANCE:
                    errors.append(
                        f"{sample_id}/{view_name}: camera intrinsics drift "
                        f"{intrinsics_deviation:.6g}"
                    )
                if resolution != reference["resolution"]:
                    errors.append(f"{sample_id}/{view_name}: calibrated resolution changed")

            # Verify every extracted JPG against the calibrated size.
            image_row = next(
                (r for r in rows_by_sample.get(sample_id, []) if r.get("view") == view_name),
                None,
            )
            if image_row is None:
                errors.append(f"{sample_id}/{view_name}: image record absent")
            else:
                image_path = FRAME_DIR / str(image_row["image"])
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    errors.append(f"{sample_id}/{view_name}: image missing/unreadable: {image_path}")
                else:
                    image_height, image_width = image_bgr.shape[:2]
                    if image_width != calibration.image_width or image_height != calibration.image_height:
                        errors.append(
                            f"{sample_id}/{view_name}: image size {image_width}x{image_height} "
                            f"!= calibration {calibration.image_width}x{calibration.image_height}"
                        )
                    else:
                        stats["image_files_ok"] += 1

            # Project both 3D hand skeletons with SHOW3D's camera implementation.
            hand_stats: dict[str, dict[str, int]] = {}
            for side, points in hand_points.items():
                camera_points = camera.world_to_camera(points, T)
                in_front_mask = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 0)
                uv, inside_mask = camera.project_to_image(points, calibration)
                finite_uv = np.isfinite(uv).all(axis=1)
                # SHOW3D's validity includes both positive depth and in-image bounds.
                inside_count = int(np.count_nonzero(inside_mask))
                front_count = int(np.count_nonzero(in_front_mask))
                stats["hands_checked"] += 1
                stats["hand_landmarks_total"] += int(len(points))
                stats["hand_landmarks_in_front"] += front_count
                stats["hand_landmarks_inside_image"] += inside_count
                if front_count == 0:
                    warnings.append(f"{sample_id}/{view_name}: all {side} hand joints behind camera")
                if inside_count == 0:
                    warnings.append(f"{sample_id}/{view_name}: no {side} hand joints inside image")
                if np.any(in_front_mask & ~finite_uv):
                    errors.append(f"{sample_id}/{view_name}: non-finite projected {side} hand pixels")
                hand_stats[side] = {
                    "landmarks": int(len(points)),
                    "in_front": front_count,
                    "inside_image": inside_count,
                }

            # Project object mesh vertices after applying the frame's 6DoF pose.
            object_front = 0
            object_inside = 0
            if object_world_vertices is not None:
                object_camera = camera.world_to_camera(object_world_vertices, T)
                object_front_mask = (
                    np.isfinite(object_camera).all(axis=1)
                    & (object_camera[:, 2] > 0)
                )
                object_uv, object_inside_mask = camera.project_to_image(
                    object_world_vertices, calibration
                )
                object_front = int(np.count_nonzero(object_front_mask))
                object_inside = int(np.count_nonzero(object_inside_mask))
                stats["object_frames_checked"] += 1
                stats["object_vertices_total"] += int(len(object_world_vertices))
                stats["object_vertices_in_front"] += object_front
                stats["object_vertices_inside_image"] += object_inside
                if object_front == 0:
                    warnings.append(f"{sample_id}/{view_name}: object mesh entirely behind camera")
                if np.any(object_front_mask & ~np.isfinite(object_uv).all(axis=1)):
                    errors.append(f"{sample_id}/{view_name}: non-finite projected object pixels")

            # One row per timestamp-camera for reproducibility and later filtering.
            per_camera_rows.append({
                "sample_id": sample_id,
                "frame_index": int(frame_ref.frame_index),
                "view": view_name,
                "width": int(calibration.image_width),
                "height": int(calibration.image_height),
                "fx": float(calibration.fx),
                "fy": float(calibration.fy),
                "cx": float(calibration.cx),
                "cy": float(calibration.cy),
                "camera_pose_source": str(getattr(calibration, "pose_source", "unknown")),
                "camera_pose_valid": bool(getattr(calibration, "is_pose_valid", False)),
                "left_in_front": hand_stats.get("left", {}).get("in_front", 0),
                "left_inside_image": hand_stats.get("left", {}).get("inside_image", 0),
                "right_in_front": hand_stats.get("right", {}).get("in_front", 0),
                "right_inside_image": hand_stats.get("right", {}).get("inside_image", 0),
                "object_vertices_in_front": object_front,
                "object_vertices_inside_image": object_inside,
                "object_confidence": float(pose.confidence) if pose is not None else None,
                "left_hand_confidence": float(frame.left_hand.confidence) if frame.left_hand is not None else None,
                "right_hand_confidence": float(frame.right_hand.confidence) if frame.right_hand is not None else None,
            })

        if (frame_idx + 1) % 25 == 0 or frame_idx + 1 == len(dataset):
            print(f"  completed {frame_idx + 1}/{len(dataset)} timestamps")

    # Compute cross-camera baselines from the static camera centers.
    camera_centers: dict[str, list[float]] = {}
    for view_name, reference in camera_reference.items():
        camera_centers[view_name] = reference["T"][:3, 3].astype(float).tolist()
    baselines: dict[str, float] = {}
    for i, view_a in enumerate(VIEWS):
        for view_b in VIEWS[i + 1:]:
            if view_a in camera_centers and view_b in camera_centers:
                distance = float(
                    np.linalg.norm(
                        np.asarray(camera_centers[view_a]) - np.asarray(camera_centers[view_b])
                    )
                )
                baselines[f"{view_a}-{view_b}"] = distance

    # Write detailed per-camera CSV.
    if per_camera_rows:
        with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_camera_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_camera_rows)

    def confidence_summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "min": None, "max": None}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    camera_summary: dict[str, Any] = {}
    for view_name, stats in camera_aggregate.items():
        camera_summary[view_name] = dict(stats)
        camera_summary[view_name]["pose_sources"] = dict(stats["pose_sources"])
        total = int(stats["hand_landmarks_total"])
        camera_summary[view_name]["hand_in_front_fraction"] = (
            float(stats["hand_landmarks_in_front"] / total) if total else None
        )
        camera_summary[view_name]["hand_inside_image_fraction"] = (
            float(stats["hand_landmarks_inside_image"] / total) if total else None
        )
        object_total = int(stats["object_vertices_total"])
        camera_summary[view_name]["object_inside_image_fraction"] = (
            float(stats["object_vertices_inside_image"] / object_total)
            if object_total else None
        )

    status = "PASS" if not errors else "FAIL"
    report = {
        "project": "SplatROM",
        "scene": f"{SUBJECT}/{SCENE}",
        "object_alias": OBJECT_ALIAS,
        "units": "millimeters",
        "hand_pose_version": "v3",
        "object_pose_version": "v2",
        "views": list(VIEWS),
        "expected_timestamps": EXPECTED_TIMESTAMPS,
        "timestamps_loaded": len(dataset),
        "timestamps_checked": frame_count_checked,
        "image_records": len(image_rows),
        "unique_sample_ids": len(unique_records),
        "expected_image_records": EXPECTED_IMAGE_ROWS,
        "object_mesh": {
            "vertices": int(len(vertices_object)),
            "faces": int(len(faces)),
        },
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "object_pose_confidence": confidence_summary(object_pose_confidences),
        "left_hand_confidence": confidence_summary(hand_confidences["left"]),
        "right_hand_confidence": confidence_summary(hand_confidences["right"]),
        "interaction_field": interaction_counts,
        "camera_centers_world_mm": camera_centers,
        "camera_pairwise_baselines_mm": baselines,
        "object_rotation_max_orthogonality_error": object_rotation_max_orth_error,
        "object_rotation_max_determinant_error": object_rotation_max_det_error,
        "camera_statistics": camera_summary,
        "per_camera_csv": str(CSV_PATH),
        "errors_count": len(errors),
        "warnings_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "status": status,
        "interpretation": (
            "PASS means the selected files, annotation schemas, rigid transforms, "
            "and projection calculations passed these automated checks. It does "
            "not establish pixel-accurate alignment or independent 3D ground-truth accuracy."
        ),
    }

    with REPORT_PATH.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print("\n=== FULL GEOMETRY AUDIT ===")
    print("Scene:", report["scene"])
    print("Timestamps:", frame_count_checked)
    print("Image records:", len(image_rows))
    print("Camera-frame combinations:", len(per_camera_rows))
    print("Object mesh:", len(vertices_object), "vertices /", len(faces), "faces")
    print("Object-pose confidence:", report["object_pose_confidence"])
    print("Left-hand confidence:", report["left_hand_confidence"])
    print("Right-hand confidence:", report["right_hand_confidence"])
    print("Interaction Field:", interaction_counts)

    print("\n=== CAMERA SUMMARY ===")
    for view_name in VIEWS:
        info = camera_summary[view_name]
        print(f"\n{view_name} ({info['resolution']})")
        print("  Frames checked:", info["frames_checked"])
        print("  Images verified:", info["image_files_ok"])
        print(
            "  Hand landmarks in front:",
            info["hand_landmarks_in_front"], "/", info["hand_landmarks_total"],
        )
        print(
            "  Hand landmarks inside image:",
            info["hand_landmarks_inside_image"], "/", info["hand_landmarks_total"],
        )
        print(
            "  Object vertices inside image:",
            info["object_vertices_inside_image"], "/", info["object_vertices_total"],
        )
        print("  Max camera-transform deviation:", info["max_transform_deviation"])
        print("  Max intrinsics deviation:", info["max_intrinsics_deviation"])
        print("  Pose sources:", info["pose_sources"])

    print("\n=== CAMERA BASELINES (mm) ===")
    for pair, distance in baselines.items():
        print(f"  {pair}: {distance:.3f}")

    print("\n=== ERRORS ===")
    if errors:
        for item in errors[:50]:
            print("  ERROR:", item)
        if len(errors) > 50:
            print(f"  ... {len(errors) - 50} additional errors in JSON report")
    else:
        print("  None")

    print("\n=== WARNINGS ===")
    if warnings:
        for item in warnings[:30]:
            print("  WARNING:", item)
        if len(warnings) > 30:
            print(f"  ... {len(warnings) - 30} additional warnings in JSON report")
    else:
        print("  None")

    print("\n=== OUTPUT FILES ===")
    print("Report:", REPORT_PATH)
    print("Per-camera CSV:", CSV_PATH)
    print("\nFINAL STATUS:", status)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise
