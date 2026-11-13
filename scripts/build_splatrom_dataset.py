from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SUBJECT = "ISH822"
SCENE = "aria_inspecting_3ab0"
OBJECT_ALIAS = "aria"
VIEWS = ("rig0", "rig1", "rig2", "rig3")
HAND_POSE_VERSION = "v3"
OBJECT_POSE_VERSION = "v2"
INTERACTION_FIELD_VERSION = "v1"
CONFIDENCE_THRESHOLD = 0.5
EXPECTED_TIMESTAMPS = 297
EXPECTED_IMAGE_ROWS = EXPECTED_TIMESTAMPS * len(VIEWS)


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        fail(f"Expected a JSON object in {path}")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"Required JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"Invalid JSONL at {path}:{line_number}: {exc}")
            if not isinstance(value, dict):
                fail(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(json_safe(value), stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def finite_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        fail(f"Expected array shape {shape}, found {array.shape}")
    if not np.isfinite(array).all():
        fail("Array contains NaN or infinity")
    return array


def git_revision(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def safe_relative_image(frame_dir: Path, relative: str) -> Path:
    candidate = (frame_dir / relative).resolve()
    try:
        candidate.relative_to(frame_dir.resolve())
    except ValueError:
        fail(f"Image path escapes frame directory: {relative}")
    return candidate


def validate_interaction_field(value: Any, sample_id: str, side: str) -> tuple[list[Any], int]:
    if not isinstance(value, list) or len(value) != 21:
        fail(f"{sample_id}: {side} Interaction Field must contain exactly 21 vectors")
    output: list[Any] = []
    non_null = 0
    for joint_index, vector in enumerate(value):
        if vector is None:
            output.append(None)
            continue
        arr = finite_array(vector, (3,))
        output.append(arr.tolist())
        non_null += 1
    if non_null == 0:
        fail(f"{sample_id}: {side} Interaction Field has no valid vectors")
    return output, non_null


def export_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"# SHOW3D object mesh: {OBJECT_ALIAS}\n")
        stream.write("# Vertices are in object-local coordinates, millimeters.\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.9f} {y:.9f} {z:.9f}\n")
        for a, b, c in faces:
            stream.write(f"f {int(a)+1} {int(b)+1} {int(c)+1}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a portable, versioned SplatROM v1 dataset package from the validated SHOW3D sample."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="SplatROM repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: data/processed/splatrom_v1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory. Use only if you intend to rebuild it.",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    show3d_root = root / "show3d"
    data_root = root / "data" / "raw" / "show3d"
    frame_dir = (
        root / "data" / "processed" / "show3d"
        / f"{SUBJECT}__{SCENE}" / "frames_10fps"
    )
    output_dir = (args.out or (root / "data" / "processed" / "splatrom_v1")).resolve()
    audit_dir = root / "data" / "processed" / "splatrom_diagnostics"
    audit_report_path = audit_dir / "full_geometry_report.json"
    audit_csv_path = audit_dir / "full_geometry_per_camera.csv"
    index_path = frame_dir / "index.jsonl"
    extract_info_path = frame_dir / "extract_info.json"
    interaction_path = (
        data_root / "interaction_field" / INTERACTION_FIELD_VERSION / "scenes"
        / SUBJECT / SCENE / "interaction_field.jsonl"
    )

    if not (show3d_root / "show3d" / "dataset.py").is_file():
        fail(f"SHOW3D API checkout not found: {show3d_root / 'show3d' / 'dataset.py'}")
    if not audit_report_path.is_file():
        fail(f"Full geometry audit report is missing: {audit_report_path}")
    audit_report = load_json(audit_report_path)
    if audit_report.get("status") != "PASS":
        fail("The full geometry audit report is not PASS; refusing to package the sample")
    if audit_report.get("timestamps_checked") != EXPECTED_TIMESTAMPS:
        fail("Audit report timestamp count differs from the expected 297")
    if audit_report.get("image_records") != EXPECTED_IMAGE_ROWS:
        fail("Audit report image count differs from the expected 1,188")

    if output_dir.exists():
        if not args.force:
            fail(
                f"Output already exists: {output_dir}. Choose another --out path "
                "or pass --force to replace it."
            )
        shutil.rmtree(output_dir)

    staging_dir = output_dir.with_name(output_dir.name + ".building")
    if staging_dir.exists():
        fail(
            f"Temporary build directory already exists: {staging_dir}. "
            "Inspect it and remove it manually before retrying."
        )

    # Import the exact local version of the official SHOW3D API.
    sys.path.insert(0, str(show3d_root))
    from show3d.dataset import (  # type: ignore[import-not-found]
        Show3DDataset,
        Show3DFrameRef,
        load_object_mesh_with_faces,
    )

    print("Reading extracted-image manifest and Interaction Field...")
    image_rows = load_jsonl(index_path)
    interaction_rows = load_jsonl(interaction_path)
    extract_info = load_json(extract_info_path) if extract_info_path.is_file() else {}

    rows_by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    sample_meta: dict[str, dict[str, Any]] = {}
    for row in image_rows:
        for key in ("sample_id", "frame_index", "subject_id", "scene_id", "view", "image"):
            if key not in row:
                fail(f"Image index row is missing {key}: {row}")
        sid = str(row["sample_id"])
        view = str(row["view"])
        if view not in VIEWS:
            fail(f"Unexpected view {view} in extracted-image index")
        if view in rows_by_sample[sid]:
            fail(f"Duplicate image row for {sid}/{view}")
        if str(row["subject_id"]) != SUBJECT or str(row["scene_id"]) != SCENE:
            fail(f"Unexpected subject/scene in index row {sid}")
        rows_by_sample[sid][view] = row
        sample_meta[sid] = {
            "sample_id": sid,
            "subject_id": SUBJECT,
            "scene_id": SCENE,
            "frame_index": int(row["frame_index"]),
        }

    for sid, view_rows in rows_by_sample.items():
        if set(view_rows) != set(VIEWS):
            fail(f"{sid}: expected views {VIEWS}, got {tuple(sorted(view_rows))}")
        frames = {int(row["frame_index"]) for row in view_rows.values()}
        if len(frames) != 1:
            fail(f"{sid}: per-view frame indexes differ: {frames}")

    unique_samples = sorted(sample_meta.values(), key=lambda row: row["frame_index"])
    if len(unique_samples) != EXPECTED_TIMESTAMPS:
        fail(f"Expected {EXPECTED_TIMESTAMPS} timestamps; found {len(unique_samples)}")
    if len(image_rows) != EXPECTED_IMAGE_ROWS:
        fail(f"Expected {EXPECTED_IMAGE_ROWS} image records; found {len(image_rows)}")

    interaction_map: dict[str, dict[str, Any]] = {}
    for row in interaction_rows:
        sid = str(row.get("sample_id", ""))
        if not sid:
            fail("Interaction Field row missing sample_id")
        if sid in interaction_map:
            fail(f"Duplicate Interaction Field row: {sid}")
        interaction_map[sid] = row

    expected_sample_ids = {row["sample_id"] for row in unique_samples}
    missing_interactions = expected_sample_ids - set(interaction_map)
    if missing_interactions:
        fail(f"Missing Interaction Field rows, first examples: {sorted(missing_interactions)[:5]}")

    print("Loading official SHOW3D mesh and annotations...")
    mesh = load_object_mesh_with_faces(OBJECT_ALIAS)
    if mesh is None:
        fail(f"Could not load object mesh alias '{OBJECT_ALIAS}'")
    object_vertices = finite_array(mesh.vertices_mm)
    object_faces = np.asarray(mesh.faces, dtype=np.int64)
    if object_vertices.ndim != 2 or object_vertices.shape[1] != 3:
        fail(f"Unexpected object mesh vertex shape: {object_vertices.shape}")
    if object_faces.ndim != 2 or object_faces.shape[1] != 3:
        fail(f"Unexpected object mesh face shape: {object_faces.shape}")
    if object_faces.size and (object_faces.min() < 0 or object_faces.max() >= len(object_vertices)):
        fail("Object mesh contains invalid face indices")

    refs = [
        Show3DFrameRef(
            subject_id=SUBJECT,
            scene_id=SCENE,
            frame_index=int(row["frame_index"]),
            object_alias=OBJECT_ALIAS,
        )
        for row in unique_samples
    ]
    dataset = Show3DDataset(
        data_root,
        refs,
        hand_pose_version=HAND_POSE_VERSION,
        object_pose_version=OBJECT_POSE_VERSION,
        views=VIEWS,
        decode_images=False,
    )
    if len(dataset) != EXPECTED_TIMESTAMPS:
        fail(f"SHOW3D loader returned {len(dataset)} frames, expected {EXPECTED_TIMESTAMPS}")

    # Stage the new package. Final output appears only after all writes succeed.
    staging_dir.mkdir(parents=True, exist_ok=False)
    (staging_dir / "images").mkdir()
    (staging_dir / "cameras").mkdir()
    (staging_dir / "annotations").mkdir()
    (staging_dir / "geometry").mkdir()
    (staging_dir / "audit").mkdir()
    (staging_dir / "licenses").mkdir()

    camera_reference: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    hand_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    interaction_output_rows: list[dict[str, Any]] = []
    copied_images = 0
    left_non_null_total = 0
    right_non_null_total = 0
    max_object_pose_reconstruction_error = 0.0
    max_camera_transform_orthogonality_error = 0.0

    print(f"Building package: {EXPECTED_TIMESTAMPS} timestamps x {len(VIEWS)} views...")
    for frame_position, frame in enumerate(dataset):
        frame_ref = frame.frame
        sid = frame_ref.sample_id
        expected = unique_samples[frame_position]
        if sid != expected["sample_id"]:
            fail(f"SHOW3D frame order mismatch: {sid} != {expected['sample_id']}")

        # Hand landmarks are in the SHOW3D back-mounted-rig/world frame, in mm.
        hand_payload: dict[str, Any] = {
            "sample_id": sid,
            "frame_index": int(frame_ref.frame_index),
            "coordinate_frame": "show3d_rig_world",
            "units": "millimeters",
            "left": None,
            "right": None,
        }
        for side, hand in (("left", frame.left_hand), ("right", frame.right_hand)):
            if hand is None:
                fail(f"{sid}: missing {side} hand annotation")
            confidence = float(hand.confidence)
            if not math.isfinite(confidence) or confidence <= CONFIDENCE_THRESHOLD:
                fail(f"{sid}: {side} hand confidence is not above {CONFIDENCE_THRESHOLD}")
            landmarks = finite_array(hand.landmarks_world_mm, (21, 3))
            hand_payload[side] = {
                "confidence": confidence,
                "landmarks_world_mm": landmarks.tolist(),
            }

        # Object pose convention is world-from-object; retain the source R and t.
        pose = frame.object_pose
        if pose is None:
            fail(f"{sid}: missing object pose")
        object_confidence = float(pose.confidence)
        if not math.isfinite(object_confidence) or object_confidence <= CONFIDENCE_THRESHOLD:
            fail(f"{sid}: object confidence is not above {CONFIDENCE_THRESHOLD}")
        rotation = finite_array(pose.rotation, (3, 3))
        translation = finite_array(pose.translation_mm, (3,))
        T_world_from_object = np.eye(4, dtype=np.float64)
        T_world_from_object[:3, :3] = rotation
        T_world_from_object[:3, 3] = translation

        # Verify the exported transform reproduces SHOW3D's own posed vertices.
        posed_by_api = finite_array(pose.pose_vertices(object_vertices))
        posed_by_matrix = object_vertices @ rotation.T + translation.reshape(1, 3)
        if posed_by_api.shape != posed_by_matrix.shape:
            fail(f"{sid}: object pose vertex output shape mismatch")
        pose_error = float(np.max(np.abs(posed_by_api - posed_by_matrix)))
        max_object_pose_reconstruction_error = max(
            max_object_pose_reconstruction_error, pose_error
        )
        if pose_error > 1e-3:
            fail(
                f"{sid}: exported T_world_from_object does not reproduce SHOW3D posed mesh "
                f"(max abs error {pose_error:.6g} mm)"
            )

        # Interaction Field vectors are vectors to nearest object-mesh vertices,
        # not absolute target points and not binary contact labels.
        interaction = interaction_map[sid]
        left_vectors, left_count = validate_interaction_field(
            interaction.get("left_to_object"), sid, "left"
        )
        right_vectors, right_count = validate_interaction_field(
            interaction.get("right_to_object"), sid, "right"
        )
        left_non_null_total += left_count
        right_non_null_total += right_count

        frame_images: dict[str, str] = {}
        for view_name in VIEWS:
            view = frame.views.get(view_name)
            if view is None or view.missing or view.calibration is None:
                fail(f"{sid}/{view_name}: missing view or calibration")
            calibration = view.calibration
            if calibration.t_world_from_camera is None:
                fail(f"{sid}/{view_name}: missing T_world_from_camera")
            T_world_from_camera = finite_array(
                calibration.t_world_from_camera, (4, 4)
            )
            if not np.allclose(T_world_from_camera[3], [0, 0, 0, 1], atol=1e-6):
                fail(f"{sid}/{view_name}: invalid homogeneous transform bottom row")
            R_camera = T_world_from_camera[:3, :3]
            ortho_error = float(np.linalg.norm(R_camera.T @ R_camera - np.eye(3), ord="fro"))
            max_camera_transform_orthogonality_error = max(
                max_camera_transform_orthogonality_error, ortho_error
            )
            if ortho_error > 1e-4 or abs(float(np.linalg.det(R_camera)) - 1.0) > 1e-4:
                fail(f"{sid}/{view_name}: camera rotation is not a valid SO(3) rotation")

            intrinsics = {
                "fx": float(calibration.fx),
                "fy": float(calibration.fy),
                "cx": float(calibration.cx),
                "cy": float(calibration.cy),
            }
            if not all(math.isfinite(v) for v in intrinsics.values()):
                fail(f"{sid}/{view_name}: non-finite camera intrinsics")
            if intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0:
                fail(f"{sid}/{view_name}: non-positive focal length")
            width = int(calibration.image_width)
            height = int(calibration.image_height)
            if width <= 0 or height <= 0:
                fail(f"{sid}/{view_name}: invalid image resolution")

            current_camera = {
                "camera_name": view_name,
                "model": "pinhole",
                "width": width,
                "height": height,
                "intrinsics": intrinsics,
                "K": [
                    [intrinsics["fx"], 0.0, intrinsics["cx"]],
                    [0.0, intrinsics["fy"], intrinsics["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                "T_world_from_camera": T_world_from_camera.tolist(),
                "T_camera_from_world": np.linalg.inv(T_world_from_camera).tolist(),
                "camera_center_world_mm": T_world_from_camera[:3, 3].tolist(),
                "pose_source": str(getattr(calibration, "pose_source", "unknown")),
                "is_pose_valid": bool(getattr(calibration, "is_pose_valid", False)),
                "is_synthesized": bool(getattr(calibration, "is_synthesized", False)),
                "pose_contract_version": getattr(calibration, "pose_contract_version", None),
            }

            reference = camera_reference.get(view_name)
            if reference is None:
                camera_reference[view_name] = current_camera
            else:
                if not np.allclose(
                    reference["T_world_from_camera"],
                    current_camera["T_world_from_camera"],
                    atol=1e-4,
                    rtol=0.0,
                ):
                    fail(f"{sid}/{view_name}: camera transform changed during sequence")
                if not np.allclose(
                    np.asarray(list(reference["intrinsics"].values())),
                    np.asarray(list(intrinsics.values())),
                    atol=1e-4,
                    rtol=0.0,
                ):
                    fail(f"{sid}/{view_name}: camera intrinsics changed during sequence")
                if reference["width"] != width or reference["height"] != height:
                    fail(f"{sid}/{view_name}: image resolution changed during sequence")

            source_row = rows_by_sample[sid][view_name]
            source_image = safe_relative_image(frame_dir, str(source_row["image"]))
            image_bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if image_bgr is None:
                fail(f"{sid}/{view_name}: missing or unreadable image {source_image}")
            image_h, image_w = image_bgr.shape[:2]
            if (image_w, image_h) != (width, height):
                fail(
                    f"{sid}/{view_name}: image {image_w}x{image_h} does not match "
                    f"calibration {width}x{height}"
                )

            target_rel = Path("images") / view_name / f"{int(frame_ref.frame_index):06d}.jpg"
            target_path = staging_dir / target_rel
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, target_path)
            copied_images += 1
            frame_images[view_name] = target_rel.as_posix()

        hand_rows.append(hand_payload)
        object_rows.append({
            "sample_id": sid,
            "frame_index": int(frame_ref.frame_index),
            "confidence": object_confidence,
            "object_alias": OBJECT_ALIAS,
            "units": "millimeters",
            "coordinate_frame": "show3d_rig_world",
            "pose_convention": "T_world_from_object; column-vector homogeneous transform",
            "rotation_object_to_world": rotation.tolist(),
            "translation_world_mm": translation.tolist(),
            "T_world_from_object": T_world_from_object.tolist(),
            "mesh": "geometry/object_mesh.npz",
        })
        interaction_output_rows.append({
            "sample_id": sid,
            "frame_index": int(frame_ref.frame_index),
            "units": "millimeters",
            "coordinate_frame": "show3d_rig_world_vector_components",
            "semantics": "vector from each hand landmark to its nearest object-mesh vertex",
            "left_to_object": left_vectors,
            "right_to_object": right_vectors,
        })
        manifest_rows.append({
            "sample_id": sid,
            "subject_id": SUBJECT,
            "scene_id": SCENE,
            "source_frame_index": int(frame_ref.frame_index),
            "images": frame_images,
            "annotations": {
                "hands": "annotations/hands.jsonl",
                "object": "annotations/objects.jsonl",
                "interaction_field": "annotations/interactions.jsonl",
            },
        })

        if (frame_position + 1) % 25 == 0 or frame_position + 1 == len(dataset):
            print(f"  packaged {frame_position + 1}/{len(dataset)} timestamps")

    # Static calibrated cameras: all rig transforms/intrinsics are fixed in the
    # moving back-mounted-rig coordinate frame for this recording.
    write_json(staging_dir / "cameras" / "cameras.json", {
        "coordinate_frame": "show3d_rig_world",
        "coordinate_frame_description": (
            "SHOW3D back-mounted camera-rig frame. It moves with the wearer; "
            "it is not a room-fixed global frame."
        ),
        "units": "millimeters",
        "camera_extrinsic_convention": "T_world_from_camera maps camera-frame column points into SHOW3D rig-world coordinates",
        "camera_intrinsic_convention": "pinhole K in pixel units; no distortion coefficients are exposed by the loaded CameraCalibration object",
        "cameras": camera_reference,
    })
    write_json(staging_dir / "transforms.json", {
        "units": "millimeters",
        "world_frame": "show3d_rig_world",
        "world_frame_is_rig_attached_and_moves_with_wearer": True,
        "camera_transform_convention": "T_world_from_camera; 4x4 homogeneous matrix; column vectors",
        "object_transform_convention": "T_world_from_object; 4x4 homogeneous matrix; column vectors",
        "cameras": {
            name: {
                "T_world_from_camera": data["T_world_from_camera"],
                "T_camera_from_world": data["T_camera_from_world"],
            }
            for name, data in camera_reference.items()
        },
        "object_transforms_file": "annotations/objects.jsonl",
    })

    mesh_npz = staging_dir / "geometry" / "object_mesh.npz"
    np.savez_compressed(
        mesh_npz,
        vertices_object_frame_mm=object_vertices.astype(np.float32),
        faces=object_faces.astype(np.int32),
    )
    export_obj(staging_dir / "geometry" / "object_mesh.obj", object_vertices, object_faces)
    write_json(staging_dir / "geometry" / "object_mesh_metadata.json", {
        "object_alias": OBJECT_ALIAS,
        "vertex_count": int(len(object_vertices)),
        "face_count": int(len(object_faces)),
        "coordinate_frame": "object_local",
        "units": "millimeters",
        "pose_applied_per_frame": "annotations/objects.jsonl:T_world_from_object",
        "note": "Mesh vertices are local/object-frame vertices, not pre-transformed world coordinates.",
    })

    write_jsonl(staging_dir / "manifest.jsonl", manifest_rows)
    write_jsonl(staging_dir / "annotations" / "hands.jsonl", hand_rows)
    write_jsonl(staging_dir / "annotations" / "objects.jsonl", object_rows)
    write_jsonl(staging_dir / "annotations" / "interactions.jsonl", interaction_output_rows)

    # Preserve the exact extracted-image index and extraction parameters as provenance.
    shutil.copy2(index_path, staging_dir / "source_index.jsonl")
    if extract_info_path.is_file():
        shutil.copy2(extract_info_path, staging_dir / "source_extract_info.json")
    shutil.copy2(audit_report_path, staging_dir / "audit" / "full_geometry_report.json")
    if audit_csv_path.is_file():
        shutil.copy2(audit_csv_path, staging_dir / "audit" / "full_geometry_per_camera.csv")

    # Copy upstream mesh attribution and API license when present. Keep them distinct.
    show3d_license = show3d_root / "LICENSE"
    attribution = show3d_root / "show3d" / "assets" / "objects" / "ATTRIBUTION.md"
    if show3d_license.is_file():
        shutil.copy2(show3d_license, staging_dir / "licenses" / "SHOW3D_API_LICENSE.txt")
    if attribution.is_file():
        shutil.copy2(attribution, staging_dir / "licenses" / "HOT3D_OBJECT_MESH_ATTRIBUTION.md")

    api_revision = git_revision(show3d_root)
    splatrom_revision = git_revision(root)
    extract_fps = extract_info.get("fps", 10)
    source_fps = 60
    metadata = {
        "package_name": "SplatROM Dataset Package",
        "package_version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": "SplatROM",
        "dataset_role": "single-scene development sample; not a held-out evaluation set",
        "source_dataset": "SHOW3D",
        "source_scene": f"{SUBJECT}/{SCENE}",
        "object_alias": OBJECT_ALIAS,
        "views": list(VIEWS),
        "timestamp_count": len(manifest_rows),
        "image_count": copied_images,
        "hand_pose_version": HAND_POSE_VERSION,
        "object_pose_version": OBJECT_POSE_VERSION,
        "interaction_field_version": INTERACTION_FIELD_VERSION,
        "confidence_threshold_used_for_build": CONFIDENCE_THRESHOLD,
        "frame_sampling": {
            "source_video_fps": source_fps,
            "extracted_fps_from_source_metadata": extract_fps,
            "sampling_note": "Source frame indexes and sample IDs are retained. No wall-clock timestamps are inferred.",
        },
        "spatial_conventions": {
            "units": "millimeters",
            "world_frame": "SHOW3D back-mounted-rig frame, moving with the wearer",
            "hand_points": "21 xyz landmarks per hand in SHOW3D rig-world coordinates",
            "object_mesh": "object-local coordinates; apply per-frame T_world_from_object",
            "interaction_field": "per-joint vector to nearest object mesh vertex; not absolute points and not binary contact labels",
            "camera_extrinsics": "T_world_from_camera; camera-to-world homogeneous transform",
        },
        "counts": {
            "image_records": len(image_rows),
            "unique_timestamps": len(unique_samples),
            "views_per_timestamp": len(VIEWS),
            "left_interaction_vectors_non_null": left_non_null_total,
            "right_interaction_vectors_non_null": right_non_null_total,
            "object_mesh_vertices": int(len(object_vertices)),
            "object_mesh_faces": int(len(object_faces)),
        },
        "consistency_checks": {
            "full_geometry_audit_status": audit_report.get("status"),
            "max_camera_rotation_orthogonality_error": max_camera_transform_orthogonality_error,
            "max_object_pose_matrix_vs_official_api_vertex_error_mm": max_object_pose_reconstruction_error,
            "images_copied_and_resolution_checked": copied_images,
        },
        "source_revisions": {
            "show3d_api_git_commit": api_revision,
            "splatrom_git_commit_at_build": splatrom_revision,
        },
        "license_and_attribution": {
            "source_dataset": "Retain and follow the SHOW3D dataset's original terms; this derivative package does not relicense source data.",
            "show3d_api": "See licenses/SHOW3D_API_LICENSE.txt when included.",
            "object_mesh": "See licenses/HOT3D_OBJECT_MESH_ATTRIBUTION.md when included and the applicable HOT3D terms. Do not assume mesh and dataset assets share identical licenses.",
            "paper_citation": "Rim et al., SHOW3D: Capturing Scenes of 3D Hands and Objects in the Wild, CVPR 2026.",
        },
        "files": {
            "manifest": "manifest.jsonl",
            "cameras": "cameras/cameras.json",
            "transforms": "transforms.json",
            "hands": "annotations/hands.jsonl",
            "objects": "annotations/objects.jsonl",
            "interactions": "annotations/interactions.jsonl",
            "object_mesh_npz": "geometry/object_mesh.npz",
            "object_mesh_obj": "geometry/object_mesh.obj",
            "audit_report": "audit/full_geometry_report.json",
        },
    }
    write_json(staging_dir / "metadata.json", metadata)

    readme = f"""# SplatROM Dataset Package v1.0.0

## Contents

- `images/rig0..rig3/`: copied synchronized JPEG frames.
- `manifest.jsonl`: one record per synchronized timestamp, keyed by the original SHOW3D `sample_id`.
- `cameras/cameras.json`: camera intrinsics and static camera transforms.
- `transforms.json`: explicit camera and object transform conventions.
- `annotations/hands.jsonl`: left/right 21-joint 3D hand landmarks and confidence.
- `annotations/objects.jsonl`: per-frame object pose and confidence.
- `annotations/interactions.jsonl`: per-joint interaction vectors.
- `geometry/object_mesh.npz` and `.obj`: object-local mesh vertices and triangular faces.
- `audit/`: the full-sequence audit report and per-camera CSV.
- `source_index.jsonl`, `source_extract_info.json`: extraction provenance.

## Coordinate conventions

All spatial coordinates and translations are in **millimeters**. Hand landmarks are expressed in the SHOW3D back-mounted-rig/world frame. This frame moves with the wearer; it is not a room-fixed global coordinate frame. The four rig cameras are static relative to this rig frame.

Camera extrinsics use `T_world_from_camera`, a 4x4 camera-to-rig-world transform for column vectors. The inverse `T_camera_from_world` is also included. Intrinsics are represented by the pinhole matrix `K` in pixel units.

The object mesh is stored in object-local coordinates. Apply the per-frame `T_world_from_object` in `annotations/objects.jsonl` to obtain its posed vertices in rig-world coordinates.

## Interaction Field semantics

Each left/right list contains 21 vectors in millimeters, one per hand landmark. Each vector points from that landmark to the nearest object-mesh vertex. These are **vectors**, not absolute 3D target positions and not binary contact labels.

## Dataset scope

This is a single-scene development package for `ISH822/aria_inspecting_3ab0`, containing {len(manifest_rows)} timestamps at the extracted sampling rate reported in `metadata.json`. It is not a subject-independent benchmark or a held-out test set. Do not make generalization claims from this scene alone.

## Validation

The package was built only after the full geometry audit reported PASS. That automated audit checks file/annotation coverage, finite coordinates, calibration consistency, camera transforms, and projection domains. PASS does not establish pixel-perfect alignment or independent 3D ground-truth accuracy.

## Moving-camera-frame caution

The SHOW3D world frame is attached to the body-worn rig. The camera rig moves with the participant. Consequently, the room background is not static in this coordinate frame. A reconstruction baseline should explicitly handle this or restrict the first experiments to the hand-object interaction region; do not assume a conventional static-background scene without checking it.

## Attribution and licensing

Source dataset: SHOW3D (`facebook/show3d-dataset`). Cite Rim et al., *SHOW3D: Capturing Scenes of 3D Hands and Objects in the Wild*, CVPR 2026. The source dataset, API code, and HOT3D object mesh assets may have different license terms. Review the original dataset terms, `licenses/SHOW3D_API_LICENSE.txt`, and `licenses/HOT3D_OBJECT_MESH_ATTRIBUTION.md` before redistribution or any commercial use. This package does not grant additional rights.
"""
    (staging_dir / "README.md").write_text(readme, encoding="utf-8", newline="\n")

    # Lightweight package self-check before publishing the final directory.
    if copied_images != EXPECTED_IMAGE_ROWS:
        fail(f"Copied {copied_images} images; expected {EXPECTED_IMAGE_ROWS}")
    for filename, rows in (
        ("hands.jsonl", hand_rows),
        ("objects.jsonl", object_rows),
        ("interactions.jsonl", interaction_output_rows),
    ):
        if len(rows) != EXPECTED_TIMESTAMPS:
            fail(f"{filename}: expected {EXPECTED_TIMESTAMPS} rows, found {len(rows)}")
    if len(camera_reference) != len(VIEWS):
        fail(f"Expected {len(VIEWS)} cameras in output, found {len(camera_reference)}")

    staging_dir.rename(output_dir)
    print("\n=== SPLATROM DATASET PACKAGE BUILT ===")
    print("Output:", output_dir)
    print("Timestamps:", len(manifest_rows))
    print("Copied images:", copied_images)
    print("Cameras:", ", ".join(camera_reference))
    print("Object mesh:", len(object_vertices), "vertices /", len(object_faces), "faces")
    print("Interaction vectors (non-null):", left_non_null_total, "left /", right_non_null_total, "right")
    print("Max API-vs-matrix object vertex discrepancy (mm):", max_object_pose_reconstruction_error)
    print("Package status: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise
