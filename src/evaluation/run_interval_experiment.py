
from pathlib import Path

import pandas as pd

from src.calibration.interval import IntervalCalibrator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)


def main():

    calibration_data = pd.read_csv(
        DATA_DIR / "calibration.csv"
    )

    id_test = pd.read_csv(
        DATA_DIR / "test_in_distribution.csv"
    )

    shifted_test = pd.read_csv(
        DATA_DIR / "test_shifted.csv"
    )

    # Fit exclusively on calibration data.
    calibrator = IntervalCalibrator(n_std=2.0)
    calibrator.fit(calibration_data)

    # Inspect learned intervals.
    interval_summary = calibrator.get_interval_summary()

    print("\nFITTED CONFIDENCE INTERVALS")
    print(interval_summary.round(4).to_string(index=False))

    interval_summary.to_csv(
        TABLES_DIR / "interval_parameters.csv",
        index=False,
    )

    # Evaluate both test environments.
    results = []

    for dataset_name, test_data in [
        ("id_test", id_test),
        ("shifted_test", shifted_test),
    ]:

        test_data = test_data.copy()

        test_data["interval_prediction"] = calibrator.predict(
            test_data
        )

        for module, module_data in test_data.groupby("module"):

            results.append({
                "dataset": dataset_name,
                "module": module,
                "trials": len(module_data),
                "actual_accuracy": module_data["correct"].mean(),
                "calibrated_rate": (
                    module_data["interval_prediction"].mean()
                ),
                "uncalibrated_rate": (
                    1 - module_data["interval_prediction"].mean()
                ),
                "correctly_flagged_error_rate": (
                    (
                        (module_data["correct"] == 0)
                        & (module_data["interval_prediction"] == 0)
                    ).sum()
                    / max((module_data["correct"] == 0).sum(), 1)
                ),
            })

    results_df = pd.DataFrame(results)

    print("\nINTERVAL CALIBRATION RESULTS")
    print(results_df.round(4).to_string(index=False))

    results_df.to_csv(
        TABLES_DIR / "interval_calibration_results.csv",
        index=False,
    )

    print("\nSaved interval experiment results.")


if __name__ == "__main__":
    main()
