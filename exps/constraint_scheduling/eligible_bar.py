"""Plot a grouped bar chart of feasible rates for EHVI experiments."""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "constraint_scheduling"
OUTPUT_FILENAME = "feasible_rate_bar_nehvi.pdf"

# Paste feasible rate values here before running the script.
MNIST_RANDOM: float = 0.012
MNIST_NO_CONSTRAINT: float = 0.020
MNIST_STATIC: float = 0.108
MNIST_SCHED: float = 0.088

FASHION_RANDOM: float = 0.024
FASHION_NO_CONSTRAINT: float = 0.084
FASHION_STATIC: float = 0.072
FASHION_SCHED: float = 0.104

CIFAR_RANDOM: float = 0.012
CIFAR_NO_CONSTRAINT: float = 0.072
CIFAR_STATIC: float = 0.096
CIFAR_SCHED: float = 0.100

# Dataset and plotting configuration.
DATASETS: List[str] = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
METHOD_LABELS: List[str] = ["Random", "No-constraint", "Static", "Scheduling"]
BAR_COLORS: List[str] = ["#8dd3c7", "#80b1d3", "#fb8072", "#b15928"]
BAR_WIDTH: float = 0.18
YMAX: float = 0.20  # Adjust this limit to change the visible range of the y-axis.

# Bundle the dataset values so they are easy to update without touching plotting code.
MNIST_VALUES: List[float] = [
    MNIST_RANDOM,
    MNIST_NO_CONSTRAINT,
    MNIST_STATIC,
    MNIST_SCHED,
]
FASHION_VALUES: List[float] = [
    FASHION_RANDOM,
    FASHION_NO_CONSTRAINT,
    FASHION_STATIC,
    FASHION_SCHED,
]
CIFAR_VALUES: List[float] = [
    CIFAR_RANDOM,
    CIFAR_NO_CONSTRAINT,
    CIFAR_STATIC,
    CIFAR_SCHED,
]
DATASET_VALUES: List[List[float]] = [MNIST_VALUES, FASHION_VALUES, CIFAR_VALUES]


def main() -> None:
    # Calculate bar positions centered around each dataset tick.
    dataset_indices = list(range(len(DATASETS)))
    offsets = [(-1.5 + i) * BAR_WIDTH for i in range(len(METHOD_LABELS))]

    fig, ax = plt.subplots(figsize=(6.2, 2.2))

    for method_idx, (label, color, offset) in enumerate(zip(METHOD_LABELS, BAR_COLORS, offsets)):
        heights = [values[method_idx] for values in DATASET_VALUES]
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
                fontsize=7,
            )

    ax.set_ylabel("Feasible rate")
    ax.set_ylim(0.0, YMAX)
    ax.set_xticks(dataset_indices)
    ax.set_xticklabels(DATASETS)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=len(METHOD_LABELS), frameon=False)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    fig.savefig(output_path)

    print(f"Saved feasible rate bar chart to {output_path}")


if __name__ == "__main__":
    main()
