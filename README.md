# Calibration-Aware Robot Recovery

This repository is a simulation study of confidence calibration for a modular robot-recovery pipeline. It explores how a system could identify predictions that may need human review when operating conditions shift, and compares module-specific confidence intervals with raw confidence scores.

The current implementation simulates three modules (perception, planning, and execution), generates calibration and test data, and evaluates confidence behavior in in-distribution and shifted environments. It is an offline simulation: it does not connect to a robot, implement a recovery controller, or issue real human queries. The interval approach is inspired by concepts from HRI '26 (Banerjee et al., 2026).

## How It Works

1. The simulator generates labeled predictions for each module. In the shifted environment, the simulated reliability of perception and execution decreases while planning remains unchanged.
2. `IntervalCalibrator` fits separate top-confidence and runner-up-confidence intervals for each module, using only correctly predicted examples in the calibration dataset. By default, each interval spans the mean plus or minus two population standard deviations, clipped to `[0, 1]`.
3. A test prediction is marked calibrated when its top confidence is inside the module's top-confidence interval and outside its runner-up interval. A prediction marked uncalibrated can be treated as a candidate for review; the current code does not perform that review.
4. Evaluation reports module-level accuracy, calibration rates, and how often incorrect predictions are flagged. Separate scripts evaluate raw-confidence metrics and generate diagnostic summaries and plots.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Study

Run these commands from the repository root. The first command generates deterministic synthetic data in `data/raw/`; it overwrites the three CSV files there.

```bash
python -m src.simulator.generate_data
python -m src.evaluation.metrics
python -m src.evaluation.inspect_data
python -m src.evaluation.run_interval_experiment
```

The metrics command evaluates the raw top-confidence baseline. The inspection command writes confidence summaries, reliability-bin tables, and plots to `results/`. The interval experiment fits on `calibration.csv`, evaluates on both test sets, and writes interval parameters and results. Run the tests with:

```bash
pytest
```

## Data

Each generated CSV has one row per trial and module. The main columns are:

| Column | Meaning |
| --- | --- |
| `trial_id` | Trial index within the generated dataset |
| `environment` | `in_distribution` or `shifted` |
| `module` | Simulated pipeline module |
| `correct` | Whether the module prediction is correct (`0` or `1`) |
| `top_confidence` | Confidence assigned to the top prediction |
| `runner_up_confidence` | Confidence assigned to the runner-up prediction |
| `logit_margin` | Difference between the top and runner-up confidence logits |

The generator creates `calibration.csv` (2,000 trials per module), `test_in_distribution.csv` (3,000 trials per module), and `test_shifted.csv` (3,000 trials per module). The checked-in data can be used directly; regenerate it when you want to reproduce the configured simulation from its fixed seeds.

## Outputs

- `results/tables/raw_confidence_baseline.csv`: accuracy, mean confidence, overconfidence gap, Brier score, and expected calibration error by module and test environment.
- `results/tables/confidence_summary.csv` and reliability-bin CSV files: descriptive statistics and confidence-bin accuracy data.
- `results/tables/interval_parameters.csv`: fitted intervals and calibration sample counts.
- `results/tables/interval_calibration_results.csv`: interval decisions and error-flagging rates by module and test environment.
- `results/figures/`: confidence distribution, accuracy-versus-confidence, and reliability-diagram plots.

## Project Structure

```text
src/
	calibration/   Module-specific confidence interval calibrator
	evaluation/    Baseline metrics, data inspection, and interval experiment
	simulator/     Synthetic prediction and dataset generator
data/raw/        Calibration and test CSV files
results/         Generated tables and figures
tests/           Unit tests for interval calibration
```
