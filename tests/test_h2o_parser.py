import numpy as np
import pytest

from splatom.data_h2o import (
    ANGLE_FEATURES,
    hand_angle_features,
    parse_h2o_hand_pose_values,
)


def _synthetic_pose_record(left_valid=1, right_valid=1):
    record = np.zeros(128, dtype=np.float64)
    record[0] = left_valid
    record[64] = right_valid
    left = np.zeros((21, 3), dtype=np.float64)
    right = np.zeros((21, 3), dtype=np.float64)
    left[0] = [0.0, -1.0, 0.0]
    right[0] = [0.0, -1.0, 0.0]
    for base in (1, 5, 9, 13, 17):
        for j in range(4):
            left[base + j] = [float(j + 1), float(base), 0.0]
            right[base + j] = [float(j + 1), float(base), 0.0]
    record[1:64] = left.reshape(-1)
    record[65:128] = right.reshape(-1)
    return record


def test_parse_both_hands_and_flags():
    parsed = parse_h2o_hand_pose_values(_synthetic_pose_record())
    assert parsed["left"].shape == (21, 3)
    assert parsed["right"].shape == (21, 3)
    assert np.allclose(parsed["left"][1], [1.0, 1.0, 0.0])


def test_missing_hand_flag_returns_none():
    parsed = parse_h2o_hand_pose_values(_synthetic_pose_record(left_valid=0, right_valid=1))
    assert parsed["left"] is None
    assert parsed["right"] is not None


def test_h2o_record_size_is_validated():
    with pytest.raises(ValueError, match="128 values"):
        parse_h2o_hand_pose_values(np.zeros(127))


def test_angle_features_have_expected_names_and_straight_chain_angles():
    parsed = parse_h2o_hand_pose_values(_synthetic_pose_record())
    result = hand_angle_features(parsed["left"])
    assert set(result) == set(ANGLE_FEATURES)
    assert len(result) == 15
    for chain_name in ("chain_1", "chain_2", "chain_3", "chain_4", "chain_5"):
        assert np.isclose(result[f"{chain_name}_joint1_angle_deg"], 180.0)
        assert np.isclose(result[f"{chain_name}_joint2_angle_deg"], 180.0)


def test_degenerate_chain_returns_nan_not_crash():
    joints = np.zeros((21, 3), dtype=np.float64)
    result = hand_angle_features(joints)
    assert all(np.isnan(value) for value in result.values())


def test_end_to_end_baseline_writes_csv_and_summary(tmp_path):
    from splatom.data_h2o import run_h2o_pose_baseline

    subject = tmp_path / "subject1"
    pose_dir = subject / "h1" / "0" / "cam4" / "hand_pose"
    pose_dir.mkdir(parents=True)
    record = _synthetic_pose_record(left_valid=1, right_valid=0)
    (pose_dir / "000000.txt").write_text(" ".join(map(str, record.tolist())), encoding="utf-8")

    summary = run_h2o_pose_baseline(subject, tmp_path / "processed")
    assert summary["pose_files_found"] == 1
    assert summary["rows_written"] == 2
    assert summary["hand_annotations"]["left_present"] == 1
    assert summary["hand_annotations"]["right_missing"] == 1
    assert (tmp_path / "processed" / "subject1_cam4_angles.csv").exists()
    assert (tmp_path / "processed" / "summary.json").exists()
