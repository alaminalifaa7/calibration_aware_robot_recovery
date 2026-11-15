
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

ID_TEST_FILE = DATA_DIR / "test_in_distribution.csv"
SHIFTED_TEST_FILE = DATA_DIR / "test_shifted.csv"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 10


# --------------------------------------------------
# 2. Input validation
# --------------------------------------------------

def validate_inputs(y_true, y_prob):
    """
    Validate ground-truth labels and predicted probabilities.

    y_true: actual correctness labels (0 or 1)
    y_prob: predicted probability that the prediction is correct
    """

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have equal lengths.")

    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics on empty data.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1.")

    if not np.all(np.isfinite(y_prob)):
        raise ValueError("Predicted probabilities must be finite.")

    if np.any(y_prob < 0) or np.any(y_prob > 1):
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    return y_true, y_prob


# --------------------------------------------------
# 3. Brier score
# --------------------------------------------------

def brier_score(y_true, y_prob):
    """
    Calculate the Brier score.

    Lower is better.

    Formula:
        BS = mean((predicted_probability - actual_outcome)^2)
    """

    y_true, y_prob = validate_inputs(y_true, y_prob)

    return float(np.mean((y_prob - y_true) ** 2))


# --------------------------------------------------
# 4. Expected Calibration Error
# --------------------------------------------------

def expected_calibration_error(y_true, y_prob, n_bins=N_BINS):
    """
    Calculate Expected Calibration Error (ECE).

    Divide predictions into equal-width confidence bins.

    For each bin:
        confidence = average predicted probability
        accuracy = average actual correctness

    ECE is the weighted average absolute difference
    between confidence and accuracy.

    Lower is better.
    """

    y_true, y_prob = validate_inputs(y_true, y_prob)

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Assign each prediction to a bin.
    # The final bin includes confidence = 1.0.
    bin_indices = np.digitize(
        y_prob,
        bin_edges[1:-1],
        right=False,
    )

    total_predictions = len(y_true)
    ece = 0.0

    for bin_index in range(n_bins):
        mask = bin_indices == bin_index

        if not np.any(mask):
            continue

        bin_accuracy = np.mean(y_true[mask])
        bin_confidence = np.mean(y_prob[mask])
        bin_weight = np.sum(mask) / total_predictions

        ece += bin_weight * abs(
            bin_accuracy - bin_confidence
        )

    return float(ece)


# --------------------------------------------------
# 5. Accuracy
# --------------------------------------------------

def accuracy_score(y_true):
    """
    Calculate the proportion of correct predictions.
    """

    y_true = np.asarray(y_true, dtype=float)

    if y_true.size == 0:
        raise ValueError("Cannot calculate accuracy on empty data.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1.")

    return float(np.mean(y_true))


# --------------------------------------------------
# 6. Mean confidence
# --------------------------------------------------

def mean_confidence(y_prob):
    """
    Calculate average predicted confidence.
    """

    y_prob = np.asarray(y_prob, dtype=float)

    if y_prob.size == 0:
        raise ValueError("Cannot calculate confidence on empty data.")

    if not np.all(np.isfinite(y_prob)):
        raise ValueError("Predicted probabilities must be finite.")

    if np.any(y_prob < 0) or np.any(y_prob > 1):
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    return float(np.mean(y_prob))


# --------------------------------------------------
# 7. Overconfidence gap
# --------------------------------------------------

def overconfidence_gap(y_true, y_prob):
    """
    Calculate mean confidence minus accuracy.

    Positive value: average overconfidence.
    Negative value: average underconfidence.
    Zero: equal average confidence and accuracy.

    Note:
        This is an aggregate gap, not a complete
        measure of calibration.
    """

    y_true, y_prob = validate_inputs(y_true, y_prob)

    return float(np.mean(y_prob) - np.mean(y_true))


# --------------------------------------------------
# 8. Calculate all metrics together
# --------------------------------------------------

def calculate_metrics(y_true, y_prob, n_bins=N_BINS):
    """
    Calculate all evaluation metrics for one group
    of predictions.
    """

    y_true, y_prob = validate_inputs(y_true, y_prob)

    return {
        "trials": len(y_true),
        "accuracy": accuracy_score(y_true),
        "mean_confidence": mean_confidence(y_prob),
        "overconfidence_gap": overconfidence_gap(y_true, y_prob),
        "brier_score": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(
            y_true,
            y_prob,
            n_bins=n_bins,
        ),
    }


# --------------------------------------------------
# 9. Evaluate a complete dataset
# --------------------------------------------------

def evaluate_dataset(df, dataset_name):
    """
    Evaluate each module separately.

    The predicted probability is top_confidence:
    the raw confidence score produced by our simulator.
    """

    results = []

    for module, module_data in df.groupby("module"):

        metrics = calculate_metrics(
            y_true=module_data["correct"],
            y_prob=module_data["top_confidence"],
        )

        metrics["dataset"] = dataset_name
        metrics["module"] = module

        results.append(metrics)

    return results


# --------------------------------------------------
# 10. Main execution
# --------------------------------------------------

def main():

    print("\n" + "=" * 75)
    print("RAW CONFIDENCE BASELINE EVALUATION")
    print("=" * 75)

    id_test = pd.read_csv(ID_TEST_FILE)
    shifted_test = pd.read_csv(SHIFTED_TEST_FILE)

    all_results = []

    all_results.extend(
        evaluate_dataset(id_test, "id_test")
    )

    all_results.extend(
        evaluate_dataset(shifted_test, "shifted_test")
    )

    results = pd.DataFrame(all_results)

    # Organise columns for readability
    results = results[
        [
            "dataset",
            "module",
            "trials",
            "accuracy",
            "mean_confidence",
            "overconfidence_gap",
            "brier_score",
            "ece",
        ]
    ]

    # Round metric values for display
    metric_columns = [
        "accuracy",
        "mean_confidence",
        "overconfidence_gap",
        "brier_score",
        "ece",
    ]

    results[metric_columns] = results[metric_columns].round(4)

    print("\nResults:")
    print(results.to_string(index=False))

    # Save results
    output_file = TABLES_DIR / "raw_confidence_baseline.csv"

    results.to_csv(output_file, index=False)

    print("\n" + "=" * 75)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 75)

    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()
