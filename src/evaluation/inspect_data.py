
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

CALIBRATION_FILE = DATA_DIR / "calibration.csv"
ID_TEST_FILE = DATA_DIR / "test_in_distribution.csv"
SHIFTED_TEST_FILE = DATA_DIR / "test_shifted.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

def load_datasets():
    """Load calibration, in-distribution and shifted datasets."""

    calibration = pd.read_csv(CALIBRATION_FILE)
    id_test = pd.read_csv(ID_TEST_FILE)
    shifted_test = pd.read_csv(SHIFTED_TEST_FILE)

    return calibration, id_test, shifted_test


# --------------------------------------------------
# 3. Basic dataset inspection
# --------------------------------------------------

def inspect_dataset_sizes(datasets):
    """Print dataset dimensions and basic information."""

    print("\n" + "=" * 65)
    print("DATASET OVERVIEW")
    print("=" * 65)

    for name, df in datasets.items():
        print(f"\n{name}")
        print("-" * 40)
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print(f"Modules: {sorted(df['module'].unique())}")
        print(f"Environments: {sorted(df['environment'].unique())}")

    print("\nColumns:")
    print(list(next(iter(datasets.values())).columns))


# --------------------------------------------------
# 4. Accuracy and confidence summary
# --------------------------------------------------

def calculate_summary(datasets):
    """
    Calculate accuracy and average confidence for each
    module in each environment.
    """

    combined = pd.concat(
        [
            df.assign(dataset=name)
            for name, df in datasets.items()
        ],
        ignore_index=True,
    )

    summary = (
        combined.groupby(["dataset", "module"])
        .agg(
            trials=("correct", "count"),
            accuracy=("correct", "mean"),
            mean_confidence=("top_confidence", "mean"),
            mean_runner_up=("runner_up_confidence", "mean"),
            mean_logit_margin=("logit_margin", "mean"),
        )
        .reset_index()
    )

    summary["accuracy"] = summary["accuracy"].round(3)
    summary["mean_confidence"] = summary["mean_confidence"].round(3)
    summary["mean_runner_up"] = summary["mean_runner_up"].round(3)
    summary["mean_logit_margin"] = summary["mean_logit_margin"].round(3)

    print("\n" + "=" * 65)
    print("ACCURACY AND CONFIDENCE SUMMARY")
    print("=" * 65)

    print(summary.to_string(index=False))

    summary.to_csv(TABLES_DIR / "confidence_summary.csv", index=False)

    return combined, summary


# --------------------------------------------------
# 5. Confidence by prediction correctness
# --------------------------------------------------

def plot_confidence_distributions(combined):
    """
    Compare top confidence distributions for correct
    and incorrect predictions.
    """

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 5),
        sharey=True,
    )

    modules = sorted(combined["module"].unique())

    for ax, module in zip(axes, modules):
        module_data = combined[combined["module"] == module]

        sns.histplot(
            data=module_data,
            x="top_confidence",
            hue="correct",
            stat="density",
            common_norm=False,
            bins=25,
            element="step",
            fill=False,
            ax=ax,
        )

        ax.set_title(module.capitalize())
        ax.set_xlabel("Top prediction confidence")
        ax.set_ylabel("Density")

    fig.suptitle(
        "Confidence distributions: correct vs incorrect predictions",
        fontsize=14,
    )

    plt.tight_layout()

    output = FIGURES_DIR / "confidence_distributions.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {output}")


# --------------------------------------------------
# 6. Compare accuracy and confidence under shift
# --------------------------------------------------

def plot_accuracy_vs_confidence(summary):
    """
    Compare module accuracy and average confidence
    across the two test environments.
    """

    test_summary = summary[
        summary["dataset"].isin(
            ["id_test", "shifted_test"]
        )
    ].copy()

    melted = test_summary.melt(
        id_vars=["dataset", "module"],
        value_vars=["accuracy", "mean_confidence"],
        var_name="metric",
        value_name="value",
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5),
    )

    for ax, metric in zip(
        axes,
        ["accuracy", "mean_confidence"],
    ):
        metric_data = melted[melted["metric"] == metric]

        sns.barplot(
            data=metric_data,
            x="module",
            y="value",
            hue="dataset",
            ax=ax,
        )

        ax.set_title(metric.replace("_", " ").capitalize())
        ax.set_xlabel("Module")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)

    fig.suptitle(
        "Module accuracy vs average confidence under distribution shift",
        fontsize=14,
    )

    plt.tight_layout()

    output = FIGURES_DIR / "accuracy_vs_confidence.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {output}")


# --------------------------------------------------
# 7. Reliability diagram
# --------------------------------------------------


def plot_reliability_diagram(combined):
    """
    Compare predicted confidence with observed accuracy
    separately for each module and test environment.

    Each point represents a confidence bin.
    The annotation shows the number of predictions
    contributing to that point.
    """

    test_data = combined[
        combined["dataset"].isin(["id_test", "shifted_test"])
    ].copy()

    test_data["confidence_bin"] = pd.cut(
        test_data["top_confidence"],
        bins=10,
        labels=False,
        include_lowest=True,
    )

    reliability = (
        test_data.groupby(
            ["dataset", "module", "confidence_bin"],
            observed=True,
        )
        .agg(
            mean_confidence=("top_confidence", "mean"),
            accuracy=("correct", "mean"),
            count=("correct", "size"),
        )
        .reset_index()
    )

    modules = sorted(test_data["module"].unique())

    fig, axes = plt.subplots(
        len(modules),
        2,
        figsize=(14, 5 * len(modules)),
        sharex=True,
        sharey=True,
    )

    environments = ["id_test", "shifted_test"]

    for row, module in enumerate(modules):
        for col, dataset in enumerate(environments):

            ax = axes[row, col]

            data = reliability[
                (reliability["module"] == module)
                & (reliability["dataset"] == dataset)
            ]

            # Perfect calibration reference line
            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                label="Perfect calibration",
            )

            # Observed calibration
            ax.plot(
                data["mean_confidence"],
                data["accuracy"],
                marker="o",
                label="Observed accuracy",
            )

            # Annotate each point with its sample count
            for _, point in data.iterrows():
                ax.annotate(
                    f"n={int(point['count'])}",
                    (
                        point["mean_confidence"],
                        point["accuracy"],
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

            ax.set_title(
                f"{module.capitalize()} - "
                f"{dataset.replace('_', ' ').capitalize()}"
            )

            ax.set_xlabel("Mean predicted confidence")
            ax.set_ylabel("Observed accuracy")

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            ax.legend()

    fig.suptitle(
        "Module-level reliability diagrams under distribution shift",
        fontsize=15,
    )

    plt.tight_layout()

    output = FIGURES_DIR / "reliability_diagrams_by_module.png"

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    reliability.to_csv(
        TABLES_DIR / "reliability_bins_by_module.csv",
        index=False,
    )

    print(f"\nSaved: {output}")
    print(
        f"Saved: "
        f"{TABLES_DIR / 'reliability_bins_by_module.csv'}"
    )



# --------------------------------------------------
# 8. Main execution
# --------------------------------------------------

def main():
    datasets = {
        "calibration": pd.read_csv(CALIBRATION_FILE),
        "id_test": pd.read_csv(ID_TEST_FILE),
        "shifted_test": pd.read_csv(SHIFTED_TEST_FILE),
    }

    inspect_dataset_sizes(datasets)

    combined, summary = calculate_summary(datasets)

    plot_confidence_distributions(combined)

    plot_accuracy_vs_confidence(summary)

    plot_reliability_diagram(combined)

    print("\n" + "=" * 65)
    print("DATA INSPECTION COMPLETE")
    print("=" * 65)

    print(f"\nFigures directory: {FIGURES_DIR}")
    print(f"Tables directory: {TABLES_DIR}")


if __name__ == "__main__":
    main()
