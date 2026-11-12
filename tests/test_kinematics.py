import numpy as np
import pytest

from splatom.geometry.kinematics import joint_angle_degrees, joint_angles_from_indices


def test_right_angle():
    angle = joint_angle_degrees(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    assert float(angle) == pytest.approx(90.0)


def test_batch_and_degenerate():
    parent = np.array([[1, 0, 0], [0, 0, 0]], dtype=float)
    joint = np.zeros((2, 3), dtype=float)
    child = np.array([[0, 1, 0], [1, 0, 0]], dtype=float)
    angles = joint_angle_degrees(parent, joint, child)
    assert angles[0] == pytest.approx(90.0)
    assert np.isnan(angles[1])


def test_named_angles():
    points = np.array([[[1, 0, 0], [0, 0, 0], [0, 1, 0]]], dtype=float)
    result = joint_angles_from_indices(points, {"demo": (0, 1, 2)})
    assert result["demo"][0] == pytest.approx(90.0)
