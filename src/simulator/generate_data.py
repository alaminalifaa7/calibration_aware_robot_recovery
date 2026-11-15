from pathlib import Path
import numpy as np
import pandas as pd

MODULES = {
    "perception": {
        "id_reliability": 0.92,
        "shift_reliability": 0.75,
    },
    "planning": {
        "id_reliability": 0.88,
        "shift_reliability": 0.88,
    },
    "execution": {
        "id_reliability": 0.90,
        "shift_reliability": 0.65,
    },
}

def generate_dataset(
    n_trials: int = 2000,
    environment: str = "in_distribution",
    seed: int = 42,
) -> pd.DataFrame:
    if environment not in {"in_distribution", "shifted"}:
        raise ValueError("environment must be 'in_distribution' or 'shifted'")

    rng = np.random.default_rng(seed)
    records = []

    for trial_id in range(n_trials):
        for module_name, settings in MODULES.items():
            if environment == "in_distribution":
                reliability = settings["id_reliability"]
            else:
                reliability = settings["shift_reliability"]

            correct = rng.random() < reliability

            if correct:
                top_confidence = rng.beta(8, 2)
                runner_up_confidence = rng.beta(2, 8)
            elif environment == "in_distribution":
                top_confidence = rng.beta(5, 5)
                runner_up_confidence = rng.beta(4, 6)
            else:
                top_confidence = rng.beta(8, 2)
                runner_up_confidence = rng.beta(2, 8)

            runner_up_confidence = min(
                runner_up_confidence,
                top_confidence - 0.001,
            )

            top_confidence = np.clip(top_confidence, 0.001, 0.999)
            runner_up_confidence = np.clip(runner_up_confidence, 0.001, 0.999)

            top_logit = np.log(top_confidence / (1 - top_confidence))
            runner_up_logit = np.log(runner_up_confidence / (1 - runner_up_confidence))
            margin = top_logit - runner_up_logit

            records.append({
                "trial_id": trial_id,
                "environment": environment,
                "module": module_name,
                "correct": int(correct),
                "top_confidence": float(top_confidence),
                "runner_up_confidence": float(runner_up_confidence),
                "logit_margin": float(margin),
            })

    return pd.DataFrame(records)

def main():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_data = generate_dataset(
        n_trials=2000,
        environment="in_distribution",
        seed=42,
    )

    test_id_data = generate_dataset(
        n_trials=3000,
        environment="in_distribution",
        seed=123,
    )

    test_shifted_data = generate_dataset(
        n_trials=3000,
        environment="shifted",
        seed=456,
    )

    calibration_data.to_csv(output_dir / "calibration.csv", index=False)
    test_id_data.to_csv(output_dir / "test_in_distribution.csv", index=False)
    test_shifted_data.to_csv(output_dir / "test_shifted.csv", index=False)

    print("Dataset generation complete.")
    print(f"Calibration rows (2000 trials * 3 modules): {len(calibration_data)}")
    print(f"ID test rows (3000 trials * 3 modules): {len(test_id_data)}")
    print(f"Shifted test rows (3000 trials * 3 modules): {len(test_shifted_data)}")

    print("\nModule accuracy by environment:")
    combined = pd.concat([test_id_data, test_shifted_data], ignore_index=True)
    summary = combined.groupby(["environment", "module"])["correct"].mean().round(3)
    print(summary)

if __name__ == "__main__":
    main()
