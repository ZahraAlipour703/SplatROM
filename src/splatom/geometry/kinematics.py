"""Small, dependency-light hand kinematics helpers."""
from __future__ import annotations

import numpy as np


def joint_angle_degrees(parent: np.ndarray, joint: np.ndarray, child: np.ndarray) -> np.ndarray:
    """Return the 0..180 degree angle at ``joint`` for arrays shaped (..., 3).

    Invalid or degenerate triplets return NaN. Inputs must use the same length
    unit (e.g. mm or m); the angle itself is unit-independent.
    """
    parent = np.asarray(parent, dtype=np.float64)
    joint = np.asarray(joint, dtype=np.float64)
    child = np.asarray(child, dtype=np.float64)
    if parent.shape != joint.shape or joint.shape != child.shape:
        raise ValueError("parent, joint and child must have identical shapes")
    if parent.ndim < 1 or parent.shape[-1] != 3:
        raise ValueError("each point must have shape (..., 3)")

    a = parent - joint
    b = child - joint
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    denom = norm_a * norm_b
    valid = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1) & (denom > 1e-12)

    cosine = np.full(denom.shape, np.nan, dtype=np.float64)
    cosine[valid] = np.sum(a[valid] * b[valid], axis=-1) / denom[valid]
    cosine[valid] = np.clip(cosine[valid], -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def joint_angles_from_indices(joints: np.ndarray, triplets: dict[str, tuple[int, int, int]]) -> dict[str, np.ndarray]:
    """Compute named joint angles from (..., J, 3) joint arrays.

    ``triplets`` maps a name to (parent_index, joint_index, child_index).
    """
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim < 2 or joints.shape[-1] != 3:
        raise ValueError("joints must have shape (..., J, 3)")
    result: dict[str, np.ndarray] = {}
    for name, (parent_idx, joint_idx, child_idx) in triplets.items():
        indices = (parent_idx, joint_idx, child_idx)
        if min(indices) < 0 or max(indices) >= joints.shape[-2]:
            raise IndexError(f"joint index out of bounds for {name}: {indices}")
        result[name] = joint_angle_degrees(
            joints[..., parent_idx, :],
            joints[..., joint_idx, :],
            joints[..., child_idx, :],
        )
    return result
