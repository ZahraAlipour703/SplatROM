
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT.parent / "show3d"
DATA_ROOT = ROOT / "data" / "raw" / "show3d"
FRAME_DIR = (
    ROOT / "data" / "processed" / "show3d"
    / "ISH822__aria_inspecting_3ab0" / "frames_10fps"
)

OUT_DIR = ROOT / "data" / "processed" / "splatrom_diagnostics"
OVERLAY_DIR = OUT_DIR / "overlays"

SUBJECT = "ISH822"
SCENE = "aria_inspecting_3ab0"
VIEWS = ("rig0", "rig1", "rig2", "rig3")
CONFIDENCE_THRESHOLD = 0.5

if not (API_DIR / "show3d" / "dataset.py").exists():
    raise FileNotFoundError(
        f"SHOW3D API not found at {API_DIR}. "
        "Clone facebookresearch/show3d beside SplatROM first."
    )

sys.path.insert(0, str(API_DIR))

from show3d.dataset import (
    Show3DDataset,
    Show3DFrameRef,
    load_object_mesh_with_faces,
)
from show3d.camera import project_to_image, world_to_camera
from show3d import viz


def load_index():
    path = FRAME_DIR / "index.jsonl"
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # One row per timestamp, not one row per camera.
    unique = {}

    for row in rows:
        unique.setdefault(row["sample_id"], row)

    records = sorted(
        unique.values(),
        key=lambda r: int(r["frame_index"])
    )

    return rows, records


def make_report():
    all_rows, records = load_index()

    if len(records) != 297:
        raise RuntimeError(
            f"Expected 297 timestamps, found {len(records)}"
        )

    refs = [
        Show3DFrameRef(
            subject_id=row["subject_id"],
            scene_id=row["scene_id"],
            frame_index=int(row["frame_index"]),
        )
        for row in records
    ]

    dataset = Show3DDataset(
        DATA_ROOT,
        refs,
        hand_pose_version="v3",
        object_pose_version="v2",
        views=VIEWS,
        decode_images=False,
    )

    image_map = {
        (row["sample_id"], row["view"]): row
        for row in all_rows
    }

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_indices = sorted({
        0,
        len(records) // 2,
        len(records) - 1,
    })

    selected_frames = {}
    reference_transforms = {}

    errors = []
    warnings = []

    stats = {
        view: {
            "width": None,
            "height": None,
            "frames_checked": 0,
            "hand_landmarks_total": 0,
            "hand_landmarks_in_front": 0,
            "hand_landmarks_inside_image": 0,
            "max_transform_deviation": 0.0,
            "max_rotation_orthogonality_error": 0.0,
            "max_rotation_determinant_error": 0.0,
        }
        for view in VIEWS
    }

    object_pose_count = 0
    hand_pose_count = {"left": 0, "right": 0}

    object_mesh = load_object_mesh_with_faces("aria")

    if object_mesh is None:
        warnings.append("ARIA object mesh unavailable; overlays will show hands only.")

    for index in range(len(dataset)):
        frame = dataset[index]
        sample_id = frame.frame.sample_id

        if frame.object_pose is None:
            errors.append(f"{sample_id}: missing object pose")
        else:
            object_pose_count += 1

        for side, hand in (
            ("left", frame.left_hand),
            ("right", frame.right_hand),
        ):
            if (
                hand is not None
                and hand.confidence > CONFIDENCE_THRESHOLD
                and hand.landmarks_world_mm is not None
            ):
                hand_pose_count[side] += 1

        for view_name in VIEWS:
            view = frame.views[view_name]
            cal = view.calibration

            if view.missing:
                errors.append(f"{sample_id}/{view_name}: camera marked missing")
                continue

            if cal is None or cal.t_world_from_camera is None:
                errors.append(f"{sample_id}/{view_name}: invalid calibration")
                continue

            T = np.asarray(cal.t_world_from_camera, dtype=np.float64)

            if T.shape != (4, 4) or not np.isfinite(T).all():
                errors.append(f"{sample_id}/{view_name}: invalid transform matrix")
                continue

            R = T[:3, :3]

            orth_error = float(
                np.linalg.norm(R.T @ R - np.eye(3), ord="fro")
            )
            det_error = abs(float(np.linalg.det(R)) - 1.0)

            if orth_error > 1e-4:
                errors.append(
                    f"{sample_id}/{view_name}: rotation orthogonality error "
                    f"{orth_error:.6g}"
                )

            if det_error > 1e-4:
                errors.append(
                    f"{sample_id}/{view_name}: rotation determinant error "
                    f"{det_error:.6g}"
                )

            reference = reference_transforms.get(view_name)

            if reference is None:
                reference_transforms[view_name] = T.copy()
            else:
                drift = float(np.max(np.abs(T - reference)))
                stats[view_name]["max_transform_deviation"] = max(
                    stats[view_name]["max_transform_deviation"],
                    drift,
                )

                if drift > 1e-4:
                    errors.append(
                        f"{sample_id}/{view_name}: rig transform is not static "
                        f"(max deviation {drift:.6g})"
                    )

            camera_stats = stats[view_name]
            camera_stats["frames_checked"] += 1
            camera_stats["width"] = int(cal.image_width)
            camera_stats["height"] = int(cal.image_height)

            camera_stats["max_rotation_orthogonality_error"] = max(
                camera_stats["max_rotation_orthogonality_error"],
                orth_error,
            )
            camera_stats["max_rotation_determinant_error"] = max(
                camera_stats["max_rotation_determinant_error"],
                det_error,
            )

            for hand in (frame.left_hand, frame.right_hand):
                if (
                    hand is None
                    or hand.confidence <= CONFIDENCE_THRESHOLD
                    or hand.landmarks_world_mm is None
                ):
                    continue

                points = np.asarray(
                    hand.landmarks_world_mm,
                    dtype=np.float64,
                )

                if points.shape != (21, 3):
                    errors.append(
                        f"{sample_id}/{view_name}: unexpected hand shape "
                        f"{points.shape}"
                    )
                    continue

                camera_points = world_to_camera(points, T)
                in_front = np.isfinite(camera_points[:, 2]) & (
                    camera_points[:, 2] > 0
                )

                _, inside = project_to_image(points, cal)

                camera_stats["hand_landmarks_total"] += len(points)
                camera_stats["hand_landmarks_in_front"] += int(
                    in_front.sum()
                )
                camera_stats["hand_landmarks_inside_image"] += int(
                    inside.sum()
                )

        if index in selected_indices:
            selected_frames[sample_id] = frame

    # Compare actual extracted image sizes with camera calibration.
    for sample_id in selected_frames:
        for view_name in VIEWS:
            row = image_map.get((sample_id, view_name))

            if row is None:
                errors.append(f"{sample_id}/{view_name}: no extracted image record")
                continue

            image_path = FRAME_DIR / row["image"]
            image = cv2.imread(str(image_path))

            if image is None:
                errors.append(f"{sample_id}/{view_name}: cannot read extracted image")
                continue

            frame = selected_frames[sample_id]
            cal = frame.views[view_name].calibration

            if cal is None:
                continue

            height, width = image.shape[:2]

            if width != cal.image_width or height != cal.image_height:
                errors.append(
                    f"{sample_id}/{view_name}: image is {width}x{height}, "
                    f"calibration is {cal.image_width}x{cal.image_height}"
                )

    # Render the exact extracted JPGs with official SHOW3D geometry.
    for sample_id, frame in selected_frames.items():
        fig, axes = plt.subplots(2, 2, figsize=(16, 13))
        axes = axes.ravel()

        for ax, view_name in zip(axes, VIEWS):
            row = image_map[(sample_id, view_name)]
            image_path = FRAME_DIR / row["image"]

            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                ax.set_title(f"{view_name}: image unavailable")
                ax.axis("off")
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            cal = frame.views[view_name].calibration

            if cal is None or cal.t_world_from_camera is None:
                ax.imshow(image_rgb)
                ax.set_title(f"{view_name}: invalid calibration")
                ax.axis("off")
                continue

            # Render the posed object mesh first.
            if object_mesh is not None and frame.object_pose is not None:
                image_rgb = viz.draw_object_mesh(
                    image_rgb,
                    object_mesh,
                    frame.object_pose,
                    cal,
                )

            # Project and draw both 3D hand skeletons.
            for hand, color in (
                (frame.left_hand, viz.MESH_COLORS[0]),
                (frame.right_hand, viz.MESH_COLORS[1]),
            ):
                if (
                    hand is None
                    or hand.confidence <= CONFIDENCE_THRESHOLD
                    or hand.landmarks_world_mm is None
                ):
                    continue

                points = np.asarray(
                    hand.landmarks_world_mm,
                    dtype=np.float64,
                )
                uv, valid = project_to_image(points, cal)

                for a, b in viz.HAND_EDGES:
                    if valid[a] and valid[b]:
                        p1 = tuple(np.round(uv[a]).astype(int))
                        p2 = tuple(np.round(uv[b]).astype(int))

                        cv2.line(
                            image_rgb, p1, p2, color, 2, cv2.LINE_AA
                        )

                for joint_index in np.flatnonzero(valid):
                    x, y = np.round(uv[joint_index]).astype(int)
                    cv2.circle(
                        image_rgb,
                        (x, y),
                        4,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )

            ax.imshow(image_rgb)
            ax.set_title(
                f"{view_name} | {cal.image_width}x{cal.image_height}"
            )
            ax.axis("off")

        fig.suptitle(f"SplatROM geometry check: {sample_id}")
        fig.tight_layout()

        output = OVERLAY_DIR / f"overlay_{frame.frame.frame_index:06d}.png"
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)

    report = {
        "scene": f"{SUBJECT}/{SCENE}",
        "hand_pose_version": "v3",
        "object_pose_version": "v2",
        "timestamps_checked": len(dataset),
        "object_poses_loaded": object_pose_count,
        "hand_poses_loaded": hand_pose_count,
        "camera_statistics": stats,
        "overlay_files": [
            str(OVERLAY_DIR / f"overlay_{selected_frames[sid].frame.frame_index:06d}.png")
            for sid in selected_frames
        ],
        "errors": errors,
        "warnings": warnings,
        "status": "PASS" if not errors else "FAIL",
    }

    report_path = OUT_DIR / "projection_report.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== PROJECTION VALIDATION ===")
    print("Timestamps:", len(dataset))
    print("Object poses loaded:", object_pose_count)
    print("Left hand poses loaded:", hand_pose_count["left"])
    print("Right hand poses loaded:", hand_pose_count["right"])

    print("\n=== CAMERA STATISTICS ===")

    for view_name, values in stats.items():
        print(f"\n{view_name}")
        print(
            f"  Resolution: {values['width']}x{values['height']}"
        )
        print(
            f"  Frames checked: {values['frames_checked']}"
        )
        print(
            "  Hand landmarks inside image:",
            values["hand_landmarks_inside_image"],
            "/",
            values["hand_landmarks_total"],
        )
        print(
            "  Maximum transform deviation:",
            values["max_transform_deviation"],
        )
        print(
            "  Maximum rotation orthogonality error:",
            values["max_rotation_orthogonality_error"],
        )

    print("\n=== ERRORS ===")
    if errors:
        for error in errors[:30]:
            print(error)
        if len(errors) > 30:
            print(f"... and {len(errors) - 30} more")
    else:
        print("None")

    print("\n=== WARNINGS ===")
    print("\n".join(warnings) if warnings else "None")

    print("\n=== OUTPUT ===")
    print("Report:", report_path)
    print("Overlays:", OVERLAY_DIR)

    print("\nFINAL STATUS:", report["status"])


if __name__ == "__main__":
    make_report()
