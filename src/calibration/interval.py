
from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Confidence interval representation
# --------------------------------------------------

@dataclass
class ConfidenceInterval:
    """Stores the bounds of a confidence interval."""

    mean: float
    std: float
    lower: float
    upper: float

    def contains(self, value):
        """Return True if value falls inside the interval."""
        return self.lower <= value <= self.upper


# --------------------------------------------------
# 2. Interval calibration model
# --------------------------------------------------

class IntervalCalibrator:
    """
    Paper-inspired interval calibration.

    Fits separate confidence intervals for each module
    using correctly predicted calibration examples.

    A prediction is classified as calibrated (1) when:
      - Top confidence is inside the top interval.
      - Top confidence is outside the runner-up interval.

    Otherwise, it is classified as uncalibrated (0).
    """

    def __init__(self, n_std=2.0):
        if n_std <= 0:
            raise ValueError("n_std must be positive.")

        self.n_std = n_std
        self.module_intervals = {}
        self.is_fitted = False

    # --------------------------------------------------
    # 3. Calculate an interval
    # --------------------------------------------------

    def _calculate_interval(self, values):
        """
        Calculate mean, standard deviation and bounds.

        We use population standard deviation (ddof=0)
        because we are estimating the distribution of
        the calibration sample.
        """

        values = np.asarray(values, dtype=float)

        if values.size == 0:
            raise ValueError("Cannot fit an interval on empty data.")

        if not np.all(np.isfinite(values)):
            raise ValueError("Confidence values must be finite.")

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))

        lower = max(0.0, mean - self.n_std * std)
        upper = min(1.0, mean + self.n_std * std)

        return ConfidenceInterval(
            mean=mean,
            std=std,
            lower=lower,
            upper=upper,
        )

    # --------------------------------------------------
    # 4. Fit intervals for each module
    # --------------------------------------------------

    def fit(self, calibration_data):
        """
        Fit module-specific intervals.

        Parameters
        ----------
        calibration_data : pandas.DataFrame

            Required columns:
                module
                correct
                top_confidence
                runner_up_confidence
        """

        required_columns = {
            "module",
            "correct",
            "top_confidence",
            "runner_up_confidence",
        }

        missing = required_columns - set(calibration_data.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        if calibration_data.empty:
            raise ValueError("Calibration dataset is empty.")

        if calibration_data[list(required_columns)].isnull().any().any():
            raise ValueError("Calibration data contains missing values.")

        if not calibration_data["correct"].isin([0, 1]).all():
            raise ValueError("'correct' must contain only 0 and 1.")

        confidence_columns = [
            "top_confidence",
            "runner_up_confidence",
        ]

        for column in confidence_columns:
            values = calibration_data[column].to_numpy(dtype=float)

            if not np.all(np.isfinite(values)):
                raise ValueError(f"{column} contains non-finite values.")

            if np.any(values < 0) or np.any(values > 1):
                raise ValueError(f"{column} must be between 0 and 1.")

        self.module_intervals = {}

        # Fit each module independently.
        for module, module_data in calibration_data.groupby("module"):

            # Use only correctly predicted examples.
            correct_data = module_data[
                module_data["correct"] == 1
            ]

            if len(correct_data) < 2:
                raise ValueError(
                    f"Module '{module}' needs at least two "
                    "correct calibration examples."
                )

            top_interval = self._calculate_interval(
                correct_data["top_confidence"]
            )

            runner_up_interval = self._calculate_interval(
                correct_data["runner_up_confidence"]
            )

            self.module_intervals[module] = {
                "top": top_interval,
                "runner_up": runner_up_interval,
                "correct_samples": len(correct_data),
                "total_samples": len(module_data),
            }

        self.is_fitted = True

        return self

    # --------------------------------------------------
    # 5. Predict one observation
    # --------------------------------------------------

    def predict_one(self, module, top_confidence):
        """
        Classify one prediction.

        Returns:
            1 = calibrated
            0 = uncalibrated
        """

        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_one().")

        if module not in self.module_intervals:
            raise ValueError(
                f"Unknown module: '{module}'. "
                "Fit the calibrator with this module first."
            )

        if not np.isfinite(top_confidence):
            raise ValueError("top_confidence must be finite.")

        if not 0 <= top_confidence <= 1:
            raise ValueError("top_confidence must be between 0 and 1.")

        intervals = self.module_intervals[module]

        inside_top = intervals["top"].contains(top_confidence)
        inside_runner_up = intervals["runner_up"].contains(
            top_confidence
        )

        return int(inside_top and not inside_runner_up)

    # --------------------------------------------------
    # 6. Predict multiple observations
    # --------------------------------------------------

    def predict(self, test_data):
        """
        Classify every row in a DataFrame.

        Returns a NumPy array containing binary
        calibrated / uncalibrated classifications.
        """

        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        required_columns = {"module", "top_confidence"}

        missing = required_columns - set(test_data.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        predictions = []

        for _, row in test_data.iterrows():
            predictions.append(
                self.predict_one(
                    module=row["module"],
                    top_confidence=float(row["top_confidence"]),
                )
            )

        return np.asarray(predictions, dtype=int)

    # --------------------------------------------------
    # 7. Explain one prediction
    # --------------------------------------------------

    def explain_one(self, module, top_confidence):
        """
        Return the intervals and decision behind a prediction.
        """

        if not self.is_fitted:
            raise RuntimeError("Call fit() before explain_one().")

        if module not in self.module_intervals:
            raise ValueError(f"Unknown module: '{module}'.")

        intervals = self.module_intervals[module]

        top = intervals["top"]
        runner = intervals["runner_up"]

        inside_top = top.contains(top_confidence)
        inside_runner = runner.contains(top_confidence)

        calibrated = int(inside_top and not inside_runner)

        if not inside_top:
            reason = "Top confidence is outside the top interval."
        elif inside_runner:
            reason = "Top confidence overlaps the runner-up interval."
        else:
            reason = "Top confidence is inside the top interval only."

        return {
            "module": module,
            "top_confidence": float(top_confidence),
            "top_interval": (top.lower, top.upper),
            "runner_up_interval": (runner.lower, runner.upper),
            "inside_top_interval": inside_top,
            "inside_runner_up_interval": inside_runner,
            "calibrated": calibrated,
            "reason": reason,
        }

    # --------------------------------------------------
    # 8. Inspect fitted intervals
    # --------------------------------------------------

    def get_interval_summary(self):
        """
        Return fitted interval parameters as a DataFrame.
        """

        if not self.is_fitted:
            raise RuntimeError("Call fit() before inspecting intervals.")

        rows = []

        for module, intervals in self.module_intervals.items():

            for interval_name in ["top", "runner_up"]:

                interval = intervals[interval_name]

                rows.append({
                    "module": module,
                    "interval_type": interval_name,
                    "mean": interval.mean,
                    "std": interval.std,
                    "lower": interval.lower,
                    "upper": interval.upper,
                    "correct_samples": intervals["correct_samples"],
                    "total_samples": intervals["total_samples"],
                })

        return pd.DataFrame(rows)
