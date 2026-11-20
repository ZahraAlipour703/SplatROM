from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

EXPECTED_TIMESTAMPS = 297
EXPECTED_VIEWS = ("rig0", "rig1", "rig2", "rig3")
EXPECTED_IMAGES = EXPECTED_TIMESTAMPS * len(EXPECTED_VIEWS)
CONFIDENCE_THRESHOLD = 0.5


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_json(path: Path) -> Any:
    if not path.is_file():
        fail(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"Missing required file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"Invalid JSON at {path}:{line_no}: {exc}")
            if not isinstance(row, dict):
                fail(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception as exc:
        fail(f"{label}: cannot convert to numeric array ({exc})")
    if arr.shape != shape:
        fail(f"{label}: expected shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        fail(f"{label}: NaN or infinity found")
    return arr


def finite_scalar(value: Any, label: str) -> float:
    try:
        result = float(value)
    except Exception:
        fail(f"{label}: not numeric")
    if not math.isfinite(result):
        fail(f"{label}: not finite")
    return result


def resolve_inside(root: Path, relative: str, label: str) -> Path:
    p = (root / relative).resolve()
    try:
        p.relative_to(root.resolve())
    except ValueError:
        fail(f"{label}: path escapes package root: {relative}")
    if not p.is_file():
        fail(f"{label}: referenced file missing: {relative}")
    return p


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser_default = project_root / "data" / "processed" / "splatrom_v1"

    import argparse
    parser = argparse.ArgumentParser(
        description="Verify the generated SplatROM v1 package without modifying it."
    )
    parser.add_argument("--package", type=Path, default=parser_default)
    args = parser.parse_args()
    package = args.package.resolve()

    if not package.is_dir():
        fail(f"Package directory not found: {package}")

    print("Loading package metadata and manifests...")
    metadata = read_json(package / "metadata.json")
    cameras_doc = read_json(package / "cameras" / "cameras.json")
    transforms_doc = read_json(package / "transforms.json")
    audit = read_json(package / "audit" / "full_geometry_report.json")

    manifest = read_jsonl(package / "manifest.jsonl")
    hands = read_jsonl(package / "annotations" / "hands.jsonl")
    objects = read_jsonl(package / "annotations" / "objects.jsonl")
    interactions = read_jsonl(package / "annotations" / "interactions.jsonl")

    expected_paths = [
        "README.md",
        "source_index.jsonl",
        "source_extract_info.json",
        "geometry/object_mesh.npz",
        "geometry/object_mesh.obj",
        "geometry/object_mesh_metadata.json",
        "audit/full_geometry_report.json",
        "audit/full_geometry_per_camera.csv",
    ]
    missing = [p for p in expected_paths if not (package / p).is_file()]
    if missing:
        fail(f"Required package files missing: {missing}")

    if metadata.get("package_name") != "SplatROM Dataset Package":
        fail("Unexpected package_name in metadata.json")
    if metadata.get("package_version") != "1.0.0":
        fail("Unexpected package_version; expected 1.0.0")
    if metadata.get("source_scene") != "ISH822/aria_inspecting_3ab0":
        fail("Unexpected source scene in metadata")
    if metadata.get("hand_pose_version") != "v3":
        fail("Unexpected hand pose version")
    if metadata.get("object_pose_version") != "v2":
        fail("Unexpected object pose version")
    if metadata.get("interaction_field_version") != "v1":
        fail("Unexpected Interaction Field version")
    if audit.get("status") != "PASS":
        fail("Packaged full geometry audit is not PASS")

    if not (
        len(manifest) == len(hands) == len(objects) == len(interactions)
        == EXPECTED_TIMESTAMPS
    ):
        fail(
            "Expected 297 rows in manifest/hands/objects/interactions; got "
            f"{len(manifest)}/{len(hands)}/{len(objects)}/{len(interactions)}"
        )

    # All annotation records must join one-to-one by the original sample_id.
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    for name, rows in (
        ("manifest", manifest),
        ("hands", hands),
        ("objects", objects),
        ("interactions", interactions),
    ):
        mapping: dict[str, dict[str, Any]] = {}
        for row in rows:
            sid = str(row.get("sample_id", ""))
            if not sid:
                fail(f"{name}: record without sample_id")
            if sid in mapping:
                fail(f"{name}: duplicate sample_id {sid}")
            mapping[sid] = row
        maps[name] = mapping

    canonical_ids = set(maps["manifest"])
    for name in ("hands", "objects", "interactions"):
        if set(maps[name]) != canonical_ids:
            fail(f"sample_id set mismatch between manifest and {name}")

    # Camera metadata and transform conventions.
    cameras = cameras_doc.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != set(EXPECTED_VIEWS):
        fail(f"Expected camera entries {EXPECTED_VIEWS}; got {list(cameras or {})}")
    if cameras_doc.get("coordinate_frame") != "show3d_rig_world":
        fail("Camera coordinate frame is not show3d_rig_world")
    if cameras_doc.get("units") != "millimeters":
        fail("Camera units are not millimeters")

    transform_cameras = transforms_doc.get("cameras")
    if not isinstance(transform_cameras, dict) or set(transform_cameras) != set(EXPECTED_VIEWS):
        fail("transforms.json does not contain all four cameras")

    camera_matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for view in EXPECTED_VIEWS:
        camera = cameras[view]
        width = int(camera["width"])
        height = int(camera["height"])
        if (width, height) != ((1024, 1280) if view == "rig3" else (1280, 1024)):
            fail(f"Unexpected resolution for {view}: {width}x{height}")
        intr = camera["intrinsics"]
        fx = finite_scalar(intr["fx"], f"{view}.fx")
        fy = finite_scalar(intr["fy"], f"{view}.fy")
        cx = finite_scalar(intr["cx"], f"{view}.cx")
        cy = finite_scalar(intr["cy"], f"{view}.cy")
        if fx <= 0 or fy <= 0:
            fail(f"{view}: focal length must be positive")
        K = array(camera["K"], (3, 3), f"{view}.K")
        expected_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        if not np.allclose(K, expected_K, atol=1e-6, rtol=0):
            fail(f"{view}: K does not match intrinsics fields")

        T_wc = array(camera["T_world_from_camera"], (4, 4), f"{view}.T_world_from_camera")
        T_cw = array(camera["T_camera_from_world"], (4, 4), f"{view}.T_camera_from_world")
        if not np.allclose(T_wc[3], [0, 0, 0, 1], atol=1e-6):
            fail(f"{view}: invalid homogeneous bottom row")
        R = T_wc[:3, :3]
        ortho = np.linalg.norm(R.T @ R - np.eye(3), ord="fro")
        det = np.linalg.det(R)
        if ortho > 1e-4 or abs(det - 1.0) > 1e-4:
            fail(f"{view}: camera rotation is not SO(3): ortho={ortho}, det={det}")
        if not np.allclose(T_wc @ T_cw, np.eye(4), atol=1e-5, rtol=0):
            fail(f"{view}: camera transforms are not mutual inverses")
        tdata = transform_cameras[view]
        if not np.allclose(array(tdata["T_world_from_camera"], (4, 4), f"transforms {view} T_wc"), T_wc, atol=1e-8):
            fail(f"{view}: transforms.json T_world_from_camera disagrees with cameras.json")
        if not np.allclose(array(tdata["T_camera_from_world"], (4, 4), f"transforms {view} T_cw"), T_cw, atol=1e-8):
            fail(f"{view}: transforms.json T_camera_from_world disagrees with cameras.json")
        center = array(camera["camera_center_world_mm"], (3,), f"{view}.camera_center_world_mm")
        if not np.allclose(center, T_wc[:3, 3], atol=1e-6):
            fail(f"{view}: camera center does not match T_world_from_camera")
        camera_matrices[view] = (T_wc, K)

    # Object mesh array and face index integrity.
    mesh = np.load(package / "geometry" / "object_mesh.npz")
    vertices = np.asarray(mesh["vertices_object_frame_mm"], dtype=np.float64)
    faces = np.asarray(mesh["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        fail(f"Invalid object mesh vertex array: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        fail(f"Invalid object mesh face array: {faces.shape}")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        fail("Object mesh has out-of-range face indices")
    mesh_meta = read_json(package / "geometry" / "object_mesh_metadata.json")
    if int(mesh_meta.get("vertex_count", -1)) != len(vertices):
        fail("Mesh vertex count differs from object_mesh_metadata.json")
    if int(mesh_meta.get("face_count", -1)) != len(faces):
        fail("Mesh face count differs from object_mesh_metadata.json")

    print("Checking annotation shapes, confidence values, paths, and images...")
    image_count = 0
    left_vectors = 0
    right_vectors = 0
    max_object_R_orthogonality = 0.0
    max_pose_det_error = 0.0
    max_object_world_reconstruction_error = 0.0
    max_hand_projection_domain_failures = 0
    object_vertices_projected = 0
    object_vertices_in_frame = 0

    expected_image_refs = 0
    for sid in sorted(canonical_ids, key=lambda x: int(maps["manifest"][x]["source_frame_index"])):
        m = maps["manifest"][sid]
        source_index = int(m["source_frame_index"])
        frame_index = int(maps["hands"][sid]["frame_index"])
        if source_index != frame_index:
            fail(f"{sid}: manifest and annotation frame indexes differ")
        image_refs = m.get("images")
        if not isinstance(image_refs, dict) or set(image_refs) != set(EXPECTED_VIEWS):
            fail(f"{sid}: expected image references for all four views")

        hand = maps["hands"][sid]
        if hand.get("coordinate_frame") != "show3d_rig_world" or hand.get("units") != "millimeters":
            fail(f"{sid}: hand coordinate metadata mismatch")
        for side in ("left", "right"):
            record = hand.get(side)
            if not isinstance(record, dict):
                fail(f"{sid}: missing {side} hand record")
            conf = finite_scalar(record.get("confidence"), f"{sid}/{side} confidence")
            if conf <= CONFIDENCE_THRESHOLD:
                fail(f"{sid}: {side} confidence is not above {CONFIDENCE_THRESHOLD}")
            array(record.get("landmarks_world_mm"), (21, 3), f"{sid}/{side} landmarks")

        obj = maps["objects"][sid]
        conf = finite_scalar(obj.get("confidence"), f"{sid} object confidence")
        if conf <= CONFIDENCE_THRESHOLD:
            fail(f"{sid}: object confidence is not above {CONFIDENCE_THRESHOLD}")
        R = array(obj.get("rotation_object_to_world"), (3, 3), f"{sid}.object R")
        t = array(obj.get("translation_world_mm"), (3,), f"{sid}.object t")
        T = array(obj.get("T_world_from_object"), (4, 4), f"{sid}.object T")
        if not np.allclose(T[:3, :3], R, atol=1e-7) or not np.allclose(T[:3, 3], t, atol=1e-7):
            fail(f"{sid}: object T matrix disagrees with R/t fields")
        if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7):
            fail(f"{sid}: invalid object transform bottom row")
        rerr = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
        derr = abs(float(np.linalg.det(R)) - 1.0)
        max_object_R_orthogonality = max(max_object_R_orthogonality, rerr)
        max_pose_det_error = max(max_pose_det_error, derr)
        if rerr > 1e-4 or derr > 1e-4:
            fail(f"{sid}: object rotation is not SO(3)")
        posed_vertices = vertices @ R.T + t.reshape(1, 3)
        if not np.isfinite(posed_vertices).all():
            fail(f"{sid}: posed object mesh has non-finite coordinates")
        # Check a deterministic subset of mesh vertices against the camera projection domain.
        vertices_sample = posed_vertices[:: max(1, len(posed_vertices) // 500)]

        interaction = maps["interactions"][sid]
        if interaction.get("units") != "millimeters":
            fail(f"{sid}: interaction units mismatch")
        for side in ("left", "right"):
            vectors = interaction.get(f"{side}_to_object")
            if not isinstance(vectors, list) or len(vectors) != 21:
                fail(f"{sid}: {side} interaction field must contain 21 entries")
            count = 0
            for j, vector in enumerate(vectors):
                if vector is None:
                    continue
                array(vector, (3,), f"{sid}/{side} interaction vector {j}")
                count += 1
            if count == 0:
                fail(f"{sid}: no non-null {side} interaction vectors")
            if side == "left":
                left_vectors += count
            else:
                right_vectors += count

        for view in EXPECTED_VIEWS:
            rel = image_refs[view]
            image_path = resolve_inside(package, rel, f"{sid}/{view} image")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                fail(f"{sid}/{view}: image unreadable")
            h, w = image.shape[:2]
            T_wc, K = camera_matrices[view]
            camera_info = cameras[view]
            if (w, h) != (int(camera_info["width"]), int(camera_info["height"])):
                fail(f"{sid}/{view}: image dimensions differ from camera metadata")

            # Verify every hand landmark is in front of the camera. Also count its
            # projection within the image using the exported pinhole K and T_cw.
            T_cw = np.linalg.inv(T_wc)
            for side in ("left", "right"):
                pts = np.asarray(hand[side]["landmarks_world_mm"], dtype=np.float64)
                xyz1 = np.concatenate([pts, np.ones((21, 1))], axis=1)
                cam = (T_cw @ xyz1.T).T[:, :3]
                if not np.isfinite(cam).all() or np.any(cam[:, 2] <= 0):
                    fail(f"{sid}/{view}/{side}: hand landmark behind camera or non-finite")
                uvw = (K @ cam.T).T
                uv = uvw[:, :2] / uvw[:, 2:3]
                inside = (
                    np.isfinite(uv).all(axis=1)
                    & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                    & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                )
                if not inside.all():
                    max_hand_projection_domain_failures += int((~inside).sum())
                    fail(f"{sid}/{view}/{side}: one or more hand landmarks outside image")

            # Project a deterministic object-mesh subset into each view.
            xyz1 = np.concatenate([vertices_sample, np.ones((len(vertices_sample), 1))], axis=1)
            cam_obj = (T_cw @ xyz1.T).T[:, :3]
            valid_depth = np.isfinite(cam_obj).all(axis=1) & (cam_obj[:, 2] > 0)
            if not valid_depth.all():
                fail(f"{sid}/{view}: sampled object mesh vertices behind camera")
            uvw_obj = (K @ cam_obj.T).T
            uv_obj = uvw_obj[:, :2] / uvw_obj[:, 2:3]
            inside_obj = (
                np.isfinite(uv_obj).all(axis=1)
                & (uv_obj[:, 0] >= 0) & (uv_obj[:, 0] < w)
                & (uv_obj[:, 1] >= 0) & (uv_obj[:, 1] < h)
            )
            object_vertices_projected += len(vertices_sample)
            object_vertices_in_frame += int(inside_obj.sum())
            image_count += 1
            expected_image_refs += 1

    # Counts and metadata cross-check.
    counts = metadata.get("counts", {})
    checks = {
        "image_records": EXPECTED_IMAGES,
        "unique_timestamps": EXPECTED_TIMESTAMPS,
        "views_per_timestamp": len(EXPECTED_VIEWS),
        "left_interaction_vectors_non_null": left_vectors,
        "right_interaction_vectors_non_null": right_vectors,
        "object_mesh_vertices": len(vertices),
        "object_mesh_faces": len(faces),
    }
    for key, expected in checks.items():
        if int(counts.get(key, -1)) != expected:
            fail(f"metadata counts.{key}: expected {expected}, found {counts.get(key)}")

    if len(list((package / "images" / "rig0").glob("*.jpg"))) != EXPECTED_TIMESTAMPS:
        fail("rig0 image directory does not contain 297 JPGs")
    per_view_counts = {
        view: len(list((package / "images" / view).glob("*.jpg")))
        for view in EXPECTED_VIEWS
    }
    if any(value != EXPECTED_TIMESTAMPS for value in per_view_counts.values()):
        fail(f"Per-camera image counts are incorrect: {per_view_counts}")
    if image_count != EXPECTED_IMAGES:
        fail(f"Expected {EXPECTED_IMAGES} image references verified, got {image_count}")

    declared_max_pose_error = metadata.get("consistency_checks", {}).get(
        "max_object_pose_matrix_vs_official_api_vertex_error_mm"
    )
    if declared_max_pose_error is None:
        fail("metadata is missing object pose reconstruction error")
    declared_max_pose_error = finite_scalar(declared_max_pose_error, "declared object pose error")
    if declared_max_pose_error > 1e-3:
        fail(f"Object matrix/API vertex discrepancy too high: {declared_max_pose_error} mm")

    print("\n=== SPLATROM PACKAGE VERIFICATION ===")
    print("Package:", package)
    print("Package version:", metadata["package_version"])
    print("Timestamps:", len(manifest))
    print("Images verified:", image_count)
    print("Images per camera:", per_view_counts)
    print("Hand annotation records:", len(hands))
    print("Object annotation records:", len(objects))
    print("Interaction records:", len(interactions))
    print("Non-null interaction vectors:", left_vectors, "left /", right_vectors, "right")
    print("Mesh:", len(vertices), "vertices /", len(faces), "faces")
    print("Camera extrinsics/intrinsics:", "4 cameras; inverse and metadata checks passed")
    print("Maximum object rotation orthogonality error:", max_object_R_orthogonality)
    print("Maximum object rotation determinant error:", max_pose_det_error)
    print("Object mesh sample projections:", object_vertices_in_frame, "/", object_vertices_projected, "inside image domains")
    print("Declared official API object-pose matrix discrepancy (mm):", declared_max_pose_error)
    print("Audit report:", audit.get("status"))
    print("\nFINAL STATUS: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        raise
