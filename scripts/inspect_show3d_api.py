
import sys
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "show3d"
DATA_ROOT = ROOT / "data" / "raw" / "show3d"

sys.path.insert(0, str(API_DIR))

from show3d.dataset import (
    Show3DDataset,
    Show3DFrameRef,
    load_object_mesh_with_faces,
)
from show3d import camera, viz


def public_attributes(obj):
    return [
        name for name in dir(obj)
        if not name.startswith("_")
    ]


print("=== PYTHON AND API ===")
print("API directory:", API_DIR)
print("Dataset signature:", inspect.signature(Show3DDataset))
print("Frame reference:", inspect.signature(Show3DFrameRef))
print("Projection:", inspect.signature(camera.project_to_image))
print("World to camera:", inspect.signature(camera.world_to_camera))
print("Object renderer:", inspect.signature(viz.draw_object_mesh))

print("\n=== DATASET LOAD TEST ===")

frame_ref = Show3DFrameRef(
    subject_id="ISH822",
    scene_id="aria_inspecting_3ab0",
    frame_index=7,
)

dataset = Show3DDataset(
    DATA_ROOT,
    [frame_ref],
    hand_pose_version="v3",
    object_pose_version="v2",
    views=("rig0", "rig1", "rig2", "rig3"),
    decode_images=False,
)

frame = dataset[0]

print("Frame loaded:", type(frame).__name__)
print("Frame attributes:", public_attributes(frame))

print("\n=== HANDS ===")

for side in ("left_hand", "right_hand"):
    hand = getattr(frame, side, None)

    print(f"\n{side}:")

    if hand is None:
        print("Not available")
        continue

    print("Type:", type(hand).__name__)
    print("Attributes:", public_attributes(hand))

    if hasattr(hand, "confidence"):
        print("Confidence:", hand.confidence)

    if hasattr(hand, "landmarks_world_mm"):
        points = hand.landmarks_world_mm
        print("Landmarks shape:", getattr(points, "shape", None))

print("\n=== OBJECT POSE ===")

pose = frame.object_pose

if pose is None:
    print("Object pose unavailable")
else:
    print("Type:", type(pose).__name__)
    print("Attributes:", public_attributes(pose))

    for name in ("confidence", "rotation", "translation_mm"):
        if hasattr(pose, name):
            print(name, ":", getattr(pose, name))

print("\n=== CAMERA CALIBRATIONS ===")

for view_name in ("rig0", "rig1", "rig2", "rig3"):
    view = frame.views[view_name]
    calibration = view.calibration

    print(f"\n{view_name}:")
    print("View attributes:", public_attributes(view))

    if calibration is None:
        print("Calibration missing")
        continue

    print("Calibration type:", type(calibration).__name__)
    print("Calibration attributes:", public_attributes(calibration))

    for name in (
        "image_width",
        "image_height",
        "width",
        "height",
        "fx",
        "fy",
        "cx",
        "cy",
        "t_world_from_camera",
    ):
        if hasattr(calibration, name):
            value = getattr(calibration, name)
            print(name, ":", getattr(value, "shape", value))

print("\n=== OBJECT MESH TEST ===")

mesh = load_object_mesh_with_faces("aria")

if mesh is None:
    print("No standalone mesh returned for alias: aria")
else:
    print("Mesh type:", type(mesh).__name__)
    print("Mesh attributes:", public_attributes(mesh))

print("\n=== INSPECTION COMPLETE ===")
