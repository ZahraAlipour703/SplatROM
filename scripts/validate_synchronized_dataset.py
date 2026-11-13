
import json
import math
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

SCENE = "ISH822/aria_inspecting_3ab0"
FRAME_DIR = ROOT / "data/processed/show3d/ISH822__aria_inspecting_3ab0/frames_10fps"

OBJECT_FILE = (
    ROOT / "data/raw/show3d/object_pose/v2/scenes/"
    / SCENE / "object_pose.json"
)

INTERACTION_FILE = (
    ROOT / "data/raw/show3d/interaction_field/v1/scenes/"
    / SCENE / "interaction_field.jsonl"
)

INDEX_FILE = FRAME_DIR / "index.jsonl"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def finite_number(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def flatten_numeric(value):
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(flatten_numeric(item))
        return result
    if finite_number(value):
        return [float(value)]
    return []


def validate_pose(pose):
    if not isinstance(pose, dict):
        return False, "missing_pose"

    R = pose.get("R")
    t = pose.get("t")

    r_values = flatten_numeric(R)
    t_values = flatten_numeric(t)

    if len(r_values) != 9:
        return False, "invalid_rotation_shape"

    if len(t_values) != 3:
        return False, "invalid_translation_shape"

    if not all(finite_number(v) for v in r_values + t_values):
        return False, "non_finite_pose"

    confidence = pose.get("confidence")
    if not finite_number(confidence) or confidence <= 0.5:
        return False, "low_confidence"

    return True, "valid"


def validate_vectors(hand_vectors):
    if not isinstance(hand_vectors, list):
        return False, "not_a_list"

    if len(hand_vectors) != 21:
        return False, "not_21_vectors"

    valid_count = 0

    for vector in hand_vectors:
        if vector is None:
            continue

        values = flatten_numeric(vector)

        if len(values) != 3:
            return False, "invalid_vector_shape"

        if not all(math.isfinite(v) for v in values):
            return False, "non_finite_vector"

        valid_count += 1

    return valid_count > 0, f"{valid_count}_valid_vectors"


def main():
    print("Loading annotations...")

    images = load_jsonl(INDEX_FILE)
    interactions = load_jsonl(INTERACTION_FILE)

    with OBJECT_FILE.open("r", encoding="utf-8") as f:
        object_data = json.load(f)

    interaction_map = {
        row["sample_id"]: row for row in interactions
    }

    object_map = {
        f"{SCENE}:{int(key):06d}": value
        for key, value in object_data.items()
    }

    print("\n=== DATASET COUNTS ===")
    print("Image records:", len(images))
    print("Interaction records:", len(interactions))
    print("Object pose records:", len(object_data))

    errors = Counter()
    pose_confidences = []
    left_valid = 0
    right_valid = 0
    checked_timestamps = set()

    for image in images:
        sample_id = image["sample_id"]
        frame_index = int(image["frame_index"])

        if sample_id in checked_timestamps:
            continue

        checked_timestamps.add(sample_id)

        # Check all four image files.
        for view in ("rig0", "rig1", "rig2", "rig3"):
            relative = (
                f"{SCENE}/{view}/{frame_index:06d}.jpg"
            )
            image_path = FRAME_DIR / relative

            if not image_path.is_file():
                errors["missing_image"] += 1

        # Object pose.
        pose = object_map.get(sample_id)

        valid_pose, reason = validate_pose(pose)

        if not valid_pose:
            errors[f"pose_{reason}"] += 1
        else:
            pose_confidences.append(float(pose["confidence"]))

        # Interaction field.
        interaction = interaction_map.get(sample_id)

        if interaction is None:
            errors["missing_interaction"] += 1
            continue

        for hand in ("left_to_object", "right_to_object"):
            valid, reason = validate_vectors(interaction.get(hand))

            if valid:
                if hand == "left_to_object":
                    left_valid += 1
                else:
                    right_valid += 1
            else:
                errors[f"{hand}_{reason}"] += 1

    print("\n=== VALIDATION RESULTS ===")
    print("Unique timestamps checked:", len(checked_timestamps))
    print("Valid left interaction fields:", left_valid)
    print("Valid right interaction fields:", right_valid)

    if pose_confidences:
        print("Valid object poses:", len(pose_confidences))
        print("Mean object confidence:",
              sum(pose_confidences) / len(pose_confidences))
        print("Minimum object confidence:", min(pose_confidences))
    else:
        print("Valid object poses: 0")

    print("\n=== ERRORS ===")
    if errors:
        for name, count in sorted(errors.items()):
            print(f"{name}: {count}")
    else:
        print("No validation errors.")

    passed = (
        len(checked_timestamps) == 297
        and left_valid == 297
        and right_valid == 297
        and len(pose_confidences) == 297
        and not errors
    )

    print("\n=== FINAL STATUS ===")
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
