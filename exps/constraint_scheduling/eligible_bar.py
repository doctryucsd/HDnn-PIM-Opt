"""Plot grouped bar charts of feasible rates for EHVI and NEHVI experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "constraint_scheduling"
OUTPUT_FILENAMES = {
    "ehvi": "feasible_rate_bar_ehvi.pdf",
    "nehvi": "feasible_rate_bar_nehvi.pdf",
}

EXPERIMENT_VALUES: Dict[str, Dict[str, List[float]]] = {
    "ehvi": {
        "MNIST": [0.012, 0.048, 0.144, 0.112],
        "Fashion-MNIST": [0.024, 0.064, 0.100, 0.072],
        "CIFAR-10": [0.012, 0.156, 0.108, 0.136],
    },
    "nehvi": {
        "MNIST": [0.012, 0.020, 0.108, 0.088],
        "Fashion-MNIST": [0.024, 0.084, 0.072, 0.104],
        "CIFAR-10": [0.012, 0.072, 0.096, 0.100],
    },
}

# Dataset and plotting configuration.
DATASETS: List[str] = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
BASE_METHOD_LABELS: List[str] = ["Random", "NC", "SC", "CS(Ours)"]
EXPERIMENT_METHOD_LABELS: Dict[str, List[str]] = {
    "ehvi": BASE_METHOD_LABELS,
    "nehvi": ["Random", "NC [4]", "SC", "CS(Ours)"],
}
BAR_COLORS: List[str] = ["#8dd3c7", "#80b1d3", "#fb8072", "#b15928"]
BAR_WIDTH: float = 0.23
GROUP_SPACING: float = 1  # Reduce this to bring dataset groups closer together.
YMAX: float = 0.20  # Adjust this limit to change the visible range of the y-axis.
ANNOTATION_FONT_SIZE: int = 9  # Increase or decrease to change bar-label text size.
AXIS_LABEL_FONT_SIZE: int = 12  # Controls the y-axis label size.
TICK_LABEL_FONT_SIZE: int = 10  # Controls both x and y tick label sizes.
LEGEND_FONT_SIZE: int = 10  # Controls the legend text size.

def plot_feasible_rates(
    experiment: str,
    dataset_values: Dict[str, List[float]],
    method_labels: List[str],
) -> Path:
    # Calculate bar positions centered around each dataset tick.
    dataset_indices = [index * GROUP_SPACING for index in range(len(DATASETS))]
    offsets = [(-1.5 + i) * BAR_WIDTH for i in range(len(method_labels))]

    fig, ax = plt.subplots(figsize=(6.2, 2.2))

    for method_idx, (label, color, offset) in enumerate(zip(method_labels, BAR_COLORS, offsets)):
        heights = [dataset_values[dataset][method_idx] for dataset in DATASETS]
        x_positions = [index + offset for index in dataset_indices]
        bars = ax.bar(x_positions, heights, BAR_WIDTH, label=label, color=color)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.002,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_FONT_SIZE,
            )

    ax.set_ylabel("Feasibility rate", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylim(0.0, YMAX)
    ax.set_xticks(dataset_indices)
    ax.set_xticklabels(DATASETS, fontsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=len(method_labels),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / OUTPUT_FILENAMES[experiment]
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for experiment, dataset_values in EXPERIMENT_VALUES.items():
        method_labels = EXPERIMENT_METHOD_LABELS.get(experiment, BASE_METHOD_LABELS)
        output_path = plot_feasible_rates(experiment, dataset_values, method_labels)
        print(f"Saved feasibility rate bar chart for {experiment.upper()} to {output_path}")


if __name__ == "__main__":
    main()
