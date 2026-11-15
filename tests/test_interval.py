
import pandas as pd
import pytest

from src.calibration.interval import IntervalCalibrator


def make_calibration_data():
    """Create a small deterministic calibration dataset."""

    return pd.DataFrame({
        "module": ["execution"] * 6,
        "correct": [1, 1, 1, 1, 0, 0],
        "top_confidence": [
            0.80, 0.85, 0.90, 0.88, 0.40, 0.50
        ],
        "runner_up_confidence": [
            0.10, 0.15, 0.12, 0.18, 0.30, 0.35
        ],
    })


def test_interval_calibrator_fits():
    calibrator = IntervalCalibrator()
    calibrator.fit(make_calibration_data())

    assert calibrator.is_fitted
    assert "execution" in calibrator.module_intervals


def test_interval_prediction():
    calibrator = IntervalCalibrator()
    calibrator.fit(make_calibration_data())

    # High top confidence should fall in the top interval.
    prediction = calibrator.predict_one("execution", 0.85)

    assert prediction == 1


def test_interval_explanation():
    calibrator = IntervalCalibrator()
    calibrator.fit(make_calibration_data())

    explanation = calibrator.explain_one("execution", 0.85)

    assert explanation["calibrated"] == 1
    assert explanation["inside_top_interval"] is True
    assert explanation["inside_runner_up_interval"] is False


def test_predict_requires_fit():
    calibrator = IntervalCalibrator()

    with pytest.raises(RuntimeError):
        calibrator.predict_one("execution", 0.85)


def test_unknown_module():
    calibrator = IntervalCalibrator()
    calibrator.fit(make_calibration_data())

    with pytest.raises(ValueError):
        calibrator.predict_one("unknown", 0.85)
