import numpy as np
import pytest

from splatom.evaluation.metrics import angle_error_metrics


def test_metrics_ignore_nan_pairs():
    result = angle_error_metrics(np.array([10.0, 20.0, np.nan]), np.array([12.0, 18.0, 3.0]))
    assert result["count"] == 2
    assert result["mae_deg"] == pytest.approx(2.0)
    assert result["rmse_deg"] == pytest.approx(2.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        angle_error_metrics(np.zeros(2), np.zeros(3))
