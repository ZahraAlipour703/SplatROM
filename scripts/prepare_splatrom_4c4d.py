from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any

import cv2
import numpy as np

VIEWS = ("rig0", "rig1", "rig2", "rig3")
CAMERA_MODEL_ID_PINHOLE = 1  # COLMAP PINHOLE
METERS_PER_MM = 1e-3
EXPECTED_TIMESTAMPS = 297


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        fail(f"Expected JSON object: {path}")
    return obj


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"Required file not found: {path}")
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


def finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        fail(f"{label}: expected shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        fail(f"{label}: contains NaN or infinity")
    return arr


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to COLMAP [qw, qx, qy, qz]."""
    R = finite_array(R, (3, 3), "rotation")
    if np.linalg.norm(R.T @ R - np.eye(3), ord="fro") > 1e-5:
        fail("COLMAP export encountered a non-orthogonal rotation")
    if abs(np.linalg.det(R) - 1.0) > 1e-5:
        fail("COLMAP export encountered a rotation with determinant != +1")

    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 2.0
        if s == 0:
            fail("Quaternion conversion encountered a zero divisor")
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0)) * 2.0
        if s == 0:
            fail("Quaternion conversion encountered a zero divisor")
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0)) * 2.0
        if s == 0:
            fail("Quaternion conversion encountered a zero divisor")
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q *= -1
    return q


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(q, dtype=np.float64)
    return np.asarray([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def write_colmap_cameras_binary(path: Path, cameras: list[dict[str, Any]]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras:
            f.write(struct.pack("<i", int(cam["camera_id"])))
            f.write(struct.pack("<i", CAMERA_MODEL_ID_PINHOLE))
            f.write(struct.pack("<Q", int(cam["width"])))
            f.write(struct.pack("<Q", int(cam["height"])))
            for value in (cam["fx"], cam["fy"], cam["cx"], cam["cy"]):
                f.write(struct.pack("<d", float(value)))


def write_colmap_images_binary(path: Path, images: list[dict[str, Any]]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for image in images:
            f.write(struct.pack("<I", int(image["image_id"])))
            for value in image["qvec"]:
                f.write(struct.pack("<d", float(value)))
            for value in image["tvec"]:
                f.write(struct.pack("<d", float(value)))
            f.write(struct.pack("<I", int(image["camera_id"])))
            f.write(image["name"].encode("utf-8") + b"\x00")
            # The 4C4D reader uses image records for camera metadata/poses;
            # it does not require 2D feature observations here.
            f.write(struct.pack("<Q", 0))


def write_colmap_text(sparse_dir: Path, cameras: list[dict[str, Any]], images: list[dict[str, Any]]) -> None:
    with (sparse_dir / "cameras.txt").open("w", encoding="utf-8", newline="\n") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam in cameras:
            f.write(
                f'{cam["camera_id"]} PINHOLE {cam["width"]} {cam["height"]} '
                f'{cam["fx"]:.12g} {cam["fy"]:.12g} {cam["cx"]:.12g} {cam["cy"]:.12g}\n'
            )

    with (sparse_dir / "images.txt").open("w", encoding="utf-8", newline="\n") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] (empty for camera anchors)\n")
        for image in images:
            q = image["qvec"]
            t = image["tvec"]
            f.write(
                f'{image["image_id"]} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} '
                f'{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} '
                f'{image["camera_id"]} {image["name"]}\n\n'
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare SplatROM images and calibrated COLMAP camera files for "
            "the 4C4D COLMAP scene reader. Point-cloud initialization is separate."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--package", type=Path, default=None)
    parser.add_argument("--fourc4d-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    package = (args.package or project_root / "data" / "processed" / "splatrom_v1").resolve()
    fourc4d_root = (args.fourc4d_root or project_root.parent / "4C4D").resolve()
    output = (args.out or fourc4d_root / "data" / "SplatROM" / "ISH822_aria_inspecting_3ab0").resolve()

    if not package.is_dir():
        fail(f"SplatROM package directory not found: {package}")
    if not (fourc4d_root / "scene" / "dataset_readers.py").is_file():
        fail(f"4C4D checkout not found or incomplete: {fourc4d_root}")

    metadata = load_json(package / "metadata.json")
    if metadata.get("package_version") != "1.0.0":
        fail(f"Expected SplatROM package v1.0.0, got {metadata.get('package_version')}")
    if metadata.get("timestamp_count") != EXPECTED_TIMESTAMPS:
        fail(f"Expected {EXPECTED_TIMESTAMPS} timestamps in package metadata")

    camera_document = load_json(package / "cameras" / "cameras.json")
    camera_map = camera_document.get("cameras")
    if not isinstance(camera_map, dict) or set(camera_map) != set(VIEWS):
        fail(f"Expected camera records for {VIEWS}")

    rows = load_jsonl(package / "manifest.jsonl")
    rows.sort(key=lambda r: int(r["source_frame_index"]))
    if len(rows) != EXPECTED_TIMESTAMPS:
        fail(f"Expected {EXPECTED_TIMESTAMPS} manifest records, got {len(rows)}")
    source_indices = [int(r["source_frame_index"]) for r in rows]
    if len(set(source_indices)) != EXPECTED_TIMESTAMPS:
        fail("Duplicate source_frame_index values in manifest")

    if output.exists():
        if not args.force:
            fail(f"Output already exists: {output}. Use --force to replace this generated scene.")
        allowed_root = (fourc4d_root / "data" / "SplatROM").resolve()
        try:
            output.relative_to(allowed_root)
        except ValueError:
            fail(f"--force is only allowed inside {allowed_root}")
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=False)
    images_dir = output / "images"
    sparse_dir = output / "sparse" / "0"
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)

    cameras: list[dict[str, Any]] = []
    camera_transforms: dict[str, dict[str, Any]] = {}
    for camera_id, view in enumerate(VIEWS, start=1):
        record = camera_map[view]
        intr = record["intrinsics"]
        width, height = int(record["width"]), int(record["height"])
        fx, fy = float(intr["fx"]), float(intr["fy"])
        cx, cy = float(intr["cx"]), float(intr["cy"])
        if min(width, height) <= 0 or min(fx, fy) <= 0:
            fail(f"{view}: invalid image dimensions/focal length")
        if not (0 <= cx < width and 0 <= cy < height):
            fail(f"{view}: principal point outside image domain")

        T_world_from_camera_mm = finite_array(
            record["T_world_from_camera"], (4, 4), f"{view} T_world_from_camera"
        )
        if not np.allclose(T_world_from_camera_mm[3], [0, 0, 0, 1], atol=1e-8):
            fail(f"{view}: invalid homogeneous transform bottom row")
        R_c2w = T_world_from_camera_mm[:3, :3]
        if np.linalg.norm(R_c2w.T @ R_c2w - np.eye(3), ord="fro") > 1e-4:
            fail(f"{view}: camera rotation is not orthogonal")
        if abs(np.linalg.det(R_c2w) - 1.0) > 1e-4:
            fail(f"{view}: camera rotation determinant is not +1")

        # Convert the SHOW3D rig-world coordinate translations from millimeters
        # to meters. Intrinsics remain in pixel units; rotation is unchanged.
        T_world_from_camera_m = T_world_from_camera_mm.copy()
        T_world_from_camera_m[:3, 3] *= METERS_PER_MM
        T_camera_from_world_m = np.linalg.inv(T_world_from_camera_m)
        R_w2c = T_camera_from_world_m[:3, :3]
        t_w2c = T_camera_from_world_m[:3, 3]
        qvec = rotmat_to_qvec(R_w2c)
        if np.max(np.abs(qvec_to_rotmat(qvec) - R_w2c)) > 1e-6:
            fail(f"{view}: quaternion round-trip failed")

        cameras.append({
            "camera_id": camera_id, "view": view,
            "width": width, "height": height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        })
        camera_transforms[view] = {
            "camera_id": camera_id,
            "T_world_from_camera_mm": T_world_from_camera_mm.tolist(),
            "T_world_from_camera_m": T_world_from_camera_m.tolist(),
            "T_camera_from_world_m": T_camera_from_world_m.tolist(),
            "colmap_qvec_wxyz": qvec.tolist(),
            "colmap_tvec_m": t_w2c.tolist(),
        }

    mapping_rows: list[dict[str, Any]] = []
    colmap_images: list[dict[str, Any]] = []
    png_count = 0
    time_denominator = EXPECTED_TIMESTAMPS / 10.0

    print(f"Preparing 4C4D scene from {len(rows)} timestamps...")
    for dense_idx, row in enumerate(rows):
        sid = str(row["sample_id"])
        source_frame = int(row["source_frame_index"])
        image_refs = row.get("images")
        if not isinstance(image_refs, dict) or set(image_refs) != set(VIEWS):
            fail(f"{sid}: manifest must contain image references for {VIEWS}")

        for camera_id, view in enumerate(VIEWS, start=1):
            rel = Path(str(image_refs[view]))
            src = (package / rel).resolve()
            try:
                src.relative_to(package.resolve())
            except ValueError:
                fail(f"{sid}/{view}: image path escapes package directory")
            if not src.is_file():
                fail(f"{sid}/{view}: image missing: {src}")

            image_bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if image_bgr is None:
                fail(f"{sid}/{view}: could not decode {src}")
            height, width = image_bgr.shape[:2]
            cam = cameras[camera_id - 1]
            if (width, height) != (cam["width"], cam["height"]):
                fail(
                    f"{sid}/{view}: source image {width}x{height} does not match "
                    f'calibration {cam["width"]}x{cam["height"]}'
                )

            image_name = f"cam{camera_id - 1:02d}_{dense_idx:04d}.png"
            destination = images_dir / image_name
            ok = cv2.imwrite(str(destination), image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                fail(f"Failed to write PNG: {destination}")
            png_count += 1

            mapping_rows.append({
                "adapter_frame_index": dense_idx,
                "source_frame_index": source_frame,
                "sample_id": sid,
                "camera": view,
                "camera_index": camera_id - 1,
                "image_name": image_name,
                # Matches 4C4D process_camera_info() exactly when max suffix is 0296.
                "time_4c4d": dense_idx / time_denominator,
            })

            if dense_idx == 0:
                transform = camera_transforms[view]
                colmap_images.append({
                    "image_id": camera_id,
                    "camera_id": camera_id,
                    "name": image_name,
                    "qvec": transform["colmap_qvec_wxyz"],
                    "tvec": transform["colmap_tvec_m"],
                })

        if (dense_idx + 1) % 25 == 0 or dense_idx + 1 == len(rows):
            print(f"  converted {dense_idx + 1}/{len(rows)} timestamps")

    if png_count != EXPECTED_TIMESTAMPS * len(VIEWS):
        fail(f"Expected 1,188 PNG images, wrote {png_count}")
    if len(colmap_images) != len(VIEWS):
        fail(f"Expected four time-zero COLMAP camera anchors, got {len(colmap_images)}")

    write_colmap_cameras_binary(sparse_dir / "cameras.bin", cameras)
    write_colmap_images_binary(sparse_dir / "images.bin", colmap_images)
    write_colmap_text(sparse_dir, cameras, colmap_images)

    with (output / "frame_mapping.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mapping_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mapping_rows)

    metadata_out = {
        "adapter": "SplatROM-to-4C4D",
        "adapter_version": "0.1.0",
        "source_package": str(package),
        "source_package_version": metadata.get("package_version"),
        "source_dataset": metadata.get("source_dataset"),
        "source_scene": metadata.get("source_scene"),
        "timestamp_count": EXPECTED_TIMESTAMPS,
        "image_count": png_count,
        "views": list(VIEWS),
        "image_pattern": "cam{camera_index:02d}_{dense_frame_index:04d}.png",
        "source_frame_index_mapping": "frame_mapping.csv",
        "time_mapping": {
            "formula": "dense_frame_index / (297 / 10)",
            "min": 0.0,
            "max": (EXPECTED_TIMESTAMPS - 1) / time_denominator,
            "4c4d_default_time_duration": [0.0, 10.0],
            "note": "4C4D model time, not source wall-clock seconds.",
        },
        "world_coordinate_frame": "SHOW3D back-mounted-rig frame",
        "colmap_world_units": "meters",
        "source_world_units": "millimeters",
        "translation_scale_from_mm": METERS_PER_MM,
        "camera_pose_convention": "COLMAP qvec/tvec encode T_camera_from_world; derived by inverting SHOW3D T_world_from_camera.",
        "camera_intrinsics": "Four separate PINHOLE records; per-camera width, height, fx, fy, cx, cy retained.",
        "colmap_camera_records": cameras,
        "colmap_anchor_images": colmap_images,
        "points3d_status": "PENDING: generate valid points3D.bin or points3D.txt from training views only.",
        "training_warning": "Do not train until point-cloud initialization exists and the principal-point patch is applied.",
    }
    with (output / "adapter_metadata.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(metadata_out, f, indent=2)
        f.write("\n")

    (sparse_dir / "POINTS3D_PENDING.txt").write_text(
        "points3D.bin / points3D.txt intentionally not generated by this adapter.\n"
        "Generate valid COLMAP-format sparse point-cloud data from training views only.\n"
        "Do not start 4C4D training until this placeholder is replaced.\n",
        encoding="utf-8",
    )

    readme = f"""# SplatROM -> 4C4D adapter output

- Frames: {EXPECTED_TIMESTAMPS}
- Images: {png_count}
- Cameras: {', '.join(VIEWS)}
- Image names: `camXX_YYYY.png`
- 4C4D time range: [0, 10), last time = {(EXPECTED_TIMESTAMPS - 1) / time_denominator:.8f}
- COLMAP world units: meters (SHOW3D translations converted from millimeters by 0.001).
- Camera intrinsics: four PINHOLE models, with each camera's own dimensions and principal point.
- `images.bin` / `images.txt`: four frame-zero camera anchors.
- `points3D.bin` is NOT generated and is required before training.

## Notes

1. `n3v2colmap.py` is N3V/LLFF-specific and expects `poses_bounds.npy`; do not use it for SHOW3D.
2. 4C4D's COLMAP reader scans PNG files, reads the last four filename digits as frame index, and propagates each `camXX` anchor pose across that camera's sequence.
3. `frame_mapping.csv` preserves original SHOW3D source frame IDs and sample IDs.
4. Apply the principal-point patch described in the SplatROM instructions before training; the unmodified COLMAP reader drops `cx/cy` and falls back to centered principal point.
5. SHOW3D's rig-world frame moves with the wearer. The camera rig is static in this coordinate frame, but the room background is not.
"""
    (output / "README_4C4D.md").write_text(readme, encoding="utf-8", newline="\n")

    print("\n=== SPLATROM -> 4C4D PREPARATION COMPLETE ===")
    print("Output:", output)
    print("Timestamps:", EXPECTED_TIMESTAMPS)
    print("PNG images:", png_count)
    print("COLMAP cameras:", len(cameras))
    print("COLMAP time-zero anchors:", len(colmap_images))
    print("Camera models: 4 x PINHOLE")
    print("COLMAP world units: meters")
    print("Time range: 0.00000000 to", f"{(EXPECTED_TIMESTAMPS - 1) / time_denominator:.8f}")
    print("points3D: PENDING (not fabricated)")
    print("Do not start 4C4D training until a valid point cloud is generated.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise
