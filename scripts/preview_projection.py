
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "show3d"
DATA_ROOT = ROOT / "data" / "raw" / "show3d"

FRAME_DIR = (
    ROOT / "data" / "processed" / "show3d"
    / "ISH822__aria_inspecting_3ab0" / "frames_10fps"
)

OUT_DIR = (
    ROOT / "data" / "processed" / "splatrom_diagnostics"
    / "preview_overlays"
)

VIEWS = ("rig0", "rig1", "rig2", "rig3")
SUBJECT = "ISH822"
SCENE = "aria_inspecting_3ab0"

sys.path.insert(0, str(API_DIR))

from show3d.dataset import (
    Show3DDataset,
    Show3DFrameRef,
    load_object_mesh_with_faces,
)
from show3d import camera, viz


def load_records():
    index_path = FRAME_DIR / "index.jsonl"

    rows = []

    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    unique = {}

    for row in rows:
        unique.setdefault(row["sample_id"], row)

    return sorted(
        unique.values(),
        key=lambda row: int(row["frame_index"])
    ), rows


def sample_id_from_frame(frame_ref):
    return (
        f"{frame_ref.subject_id}/"
        f"{frame_ref.scene_id}:"
        f"{frame_ref.frame_index:06d}"
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records, all_rows = load_records()

    if len(records) != 297:
        raise RuntimeError(
            f"Expected 297 timestamps, found {len(records)}"
        )

    selected_indices = sorted({
        0,
        len(records) // 2,
        len(records) - 1,
    })

    selected_records = [
        records[i] for i in selected_indices
    ]

    refs = [
        Show3DFrameRef(
            subject_id=SUBJECT,
            scene_id=SCENE,
            frame_index=int(row["frame_index"]),
        )
        for row in selected_records
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
        (row["sample_id"], row["view"]): row["image"]
        for row in all_rows
    }

    mesh = load_object_mesh_with_faces("aria")

    if mesh is None:
        raise RuntimeError("Could not load ARIA object mesh")

    print("Mesh vertices:", mesh.vertices_mm.shape)
    print("Mesh faces:", mesh.faces.shape)

    report = {
        "scene": f"{SUBJECT}/{SCENE}",
        "selected_frames": [],
        "camera_results": [],
    }

    for frame_index in range(len(dataset)):
        frame = dataset[frame_index]
        frame_ref = frame.frame

        sid = sample_id_from_frame(frame_ref)

        print(f"\nProcessing {sid}")

        fig, axes = plt.subplots(
            2, 2,
            figsize=(16, 13)
        )
        axes = axes.ravel()

        for ax, view_name in zip(axes, VIEWS):
            relative_path = image_map.get(
                (sid, view_name)
            )

            if relative_path is None:
                raise RuntimeError(
                    f"Missing image index: {sid}/{view_name}"
                )

            image_path = FRAME_DIR / relative_path
            image_bgr = cv2.imread(str(image_path))

            if image_bgr is None:
                raise RuntimeError(
                    f"Could not read image: {image_path}"
                )

            image_rgb = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2RGB
            )

            view = frame.views[view_name]
            cal = view.calibration

            if view.missing or cal is None:
                raise RuntimeError(
                    f"Missing view/calibration: {sid}/{view_name}"
                )

            height, width = image_rgb.shape[:2]

            if (
                width != cal.image_width
                or height != cal.image_height
            ):
                raise RuntimeError(
                    f"Resolution mismatch {view_name}: "
                    f"image={width}x{height}, "
                    f"calibration="
                    f"{cal.image_width}x{cal.image_height}"
                )

            if not cal.is_pose_valid:
                raise RuntimeError(
                    f"Invalid camera pose: {view_name}"
                )

            # Draw the transformed object mesh.
            if frame.object_pose is not None:
                image_rgb = viz.draw_object_mesh(
                    image_rgb,
                    mesh,
                    frame.object_pose,
                    cal,
                )

            camera_result = {
                "sample_id": sid,
                "view": view_name,
                "width": width,
                "height": height,
                "pose_valid": bool(cal.is_pose_valid),
                "hands": {},
            }

            # Left and right hand projections.
            hands = (
                ("left", frame.left_hand, (255, 70, 70)),
                ("right", frame.right_hand, (70, 170, 255)),
            )

            for side, hand, color in hands:

                if (
                    hand is None
                    or hand.landmarks_world_mm is None
                    or hand.confidence <= 0.5
                ):
                    camera_result["hands"][side] = {
                        "available": False
                    }
                    continue

                points = np.asarray(
                    hand.landmarks_world_mm,
                    dtype=np.float64,
                )

                if points.shape != (21, 3):
                    raise RuntimeError(
                        f"Unexpected {side} hand shape: "
                        f"{points.shape}"
                    )

                # Use official SHOW3D projection.
                uv, api_valid = camera.project_to_image(
                    points,
                    cal,
                )

                camera_points = camera.world_to_camera(
                    points,
                    cal.t_world_from_camera,
                )

                in_front = (
                    np.isfinite(camera_points).all(axis=1)
                    & (camera_points[:, 2] > 0)
                )

                inside_image = (
                    np.isfinite(uv).all(axis=1)
                    & (uv[:, 0] >= 0)
                    & (uv[:, 0] < width)
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < height)
                )

                drawable = in_front & inside_image

                # Draw skeleton edges.
                for a, b in viz.HAND_EDGES:
                    if drawable[a] and drawable[b]:
                        p1 = tuple(
                            np.round(uv[a]).astype(int)
                        )
                        p2 = tuple(
                            np.round(uv[b]).astype(int)
                        )

                        cv2.line(
                            image_rgb,
                            p1,
                            p2,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

                # Draw joints.
                for joint_id in np.flatnonzero(drawable):
                    x, y = np.round(
                        uv[joint_id]
                    ).astype(int)

                    cv2.circle(
                        image_rgb,
                        (x, y),
                        4,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )

                camera_result["hands"][side] = {
                    "available": True,
                    "confidence": float(hand.confidence),
                    "landmarks": 21,
                    "in_front": int(in_front.sum()),
                    "inside_image": int(drawable.sum()),
                    "api_valid": int(np.asarray(api_valid).sum()),
                }

            report["camera_results"].append(
                camera_result
            )

            ax.imshow(image_rgb)
            ax.set_title(
                f"{view_name} | {width}x{height}"
            )
            ax.axis("off")

            print(
                f"  {view_name}: "
                f"{width}x{height}; "
                f"left={camera_result['hands'].get('left')}; "
                f"right={camera_result['hands'].get('right')}"
            )

        frame_number = frame_ref.frame_index

        fig.suptitle(
            f"SplatROM projection preview: {sid}"
        )
        fig.tight_layout()

        output_path = (
            OUT_DIR /
            f"projection_{frame_number:06d}.png"
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        report["selected_frames"].append({
            "sample_id": sid,
            "frame_index": frame_number,
            "overlay": str(output_path),
        })

    report_path = OUT_DIR / "preview_report.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== PREVIEW COMPLETE ===")
    print("Report:", report_path)
    print("Overlays:", OUT_DIR)


if __name__ == "__main__":
    main()
