from __future__ import annotations

from argparse import ArgumentParser
from typing import List, Tuple

from botorch.utils.multi_objective import is_non_dominated
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch import Tensor
from database import Point, PointSet
from utils import read_metric_file


def filter_constraints(
    points: PointSet, constraints: List[float]
) -> Tuple[PointSet, PointSet]:
    assert len(constraints) == 4, "Constraints should have 4 values"

    indices: List[int] = [
        i for i, point in enumerate(points) if point.is_eligible(constraints)
    ]

    return points.subset(indices), points.exclude(indices)


def get_pareto_set(points: PointSet) -> Tuple[PointSet, PointSet]:
    y_list = points.to_list_compare()

    y: Tensor = Tensor(y_list)  # type: ignore
    output = is_non_dominated(y)
    indices = [i for i, x in enumerate(output) if x]
    assert len(output.shape) == 1, "output should be 1D"

    return points.subset(indices), points.exclude(indices)


def get_label(metric: str) -> str:
    if metric == "accuracy":
        return "Accuracy (%)"
    elif metric == "energy":
        return "Energy (uJ)"
    elif metric == "performance":
        return "Timing (us)"
    elif metric == "area":
        return "Area (mm2)"
    else:
        raise ValueError(f"Unknown metric: {metric}")


def plot_3d_scatter(
    num_iters: int,
    metrics: List[str],
    pareto_set: PointSet,
    dominated_set: PointSet | None,
    uneligible_set: PointSet | None,
    baseline: Point | None,
    notes: List[int],
    labels: List[str],
    point_label: str,
    point_color: str,
):
    """
    Plots a 3D scatter plot with an additional point plotted in a distinct color and thresholds.
    """

    plt.rcParams["font.size"] = 15

    # Creating a figure
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    # plot pareto set
    processed_pareto_set = pareto_set.plot_process(metrics)
    pareto_x, pareto_y, pareto_z = processed_pareto_set
    pareto_indices = pareto_set.get_indices()
    pareto_scatter = ax.scatter(pareto_x, pareto_y, pareto_z, c=pareto_indices, cmap="viridis", s=80)  # type: ignore

    # plot colorbar for pareto set
    num_ticks = 5  # Number of ticks to display on the colorbar
    tick_indices = [
        int(i) for i in [num_iters * j / (num_ticks - 1) for j in range(num_ticks)]
    ]
    # tick_indices.append(len(indices) - 1)  # Add the final index
    sorted_tick_indices = sorted(set(tick_indices))  # Remove duplicates and sort
    colorbar = plt.colorbar(pareto_scatter, ax=ax, ticks=tick_indices, pad=0.1)
    colorbar.ax.set_yticklabels(sorted_tick_indices)  # type: ignore
    colorbar.set_label("Iteration", rotation=270, labelpad=20)

    # plot dominated set
    if dominated_set is not None:
        processed_dominated_set = dominated_set.plot_process(metrics)
        dominated_x, dominated_y, dominated_z = processed_dominated_set
        ax.scatter(dominated_x, dominated_y, dominated_z, color="black", s=80)

    # plot ineligible set
    if uneligible_set is not None:
        processed_ineligible_set = uneligible_set.plot_process(metrics)
        ineligible_x, ineligible_y, ineligible_z = processed_ineligible_set
        ax.scatter(ineligible_x, ineligible_y, ineligible_z, color="gray", s=80)

    # plot baseline
    if baseline is not None:
        processed_baseline = baseline.process_plot(metrics)
        baseline_x, baseline_y, baseline_z = processed_baseline
        ax.scatter(baseline_x, baseline_y, baseline_z, color=point_color, s=100, label=point_label)  # type: ignore
        ax.legend()

    # plot notes
    for pareto_point in pareto_set:
        if pareto_point.idx in notes:
            processed_pareto_point = pareto_point.process_plot(metrics)
            note_x, note_y, note_z = processed_pareto_point
            ax.text(
                note_x,
                note_y,
                note_z,
                str(pareto_point.idx),
                fontweight="bold",
                zorder=5,
                fontsize=22,
            )

    # set label
    ax.set_xlabel(labels[0], labelpad=10)
    ax.set_ylabel(labels[1], labelpad=10)
    ax.set_zlabel(labels[2], labelpad=10)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.savefig("pareto_set_constraint.png")


def main(
    metric_file: str,
    metrics: List[str],
    plot_baseline: bool,
    plot_ineligible: bool,
    plot_dominated: bool,
    constraint: List[float],
    baseline: List[float],
    baseline_label: str,
    baseline_color: str,
    notes: List[int],
) -> None:
    data = read_metric_file(metric_file)
    point_set = PointSet.from_dict(data)
    num_iters = len(point_set)

    eligible_set, ineligible_set = filter_constraints(point_set, constraint)
    pareto_set, dominated_set = get_pareto_set(eligible_set)

    # prepare plot arguments
    plot_ineligible_set = ineligible_set if plot_ineligible else None
    plot_dominated_set = dominated_set if plot_dominated else None
    plot_baseline_point = Point.from_list(baseline, -1) if plot_baseline else None
    labels = [get_label(metric) for metric in metrics]

    plot_3d_scatter(
        num_iters,
        metrics,
        pareto_set,
        plot_dominated_set,
        plot_ineligible_set,
        plot_baseline_point,
        notes,
        labels,
        baseline_label,
        baseline_color,
    )


if __name__ == "__main__":
    # Create the parser
    parser = ArgumentParser()

    # Add arguments
    parser.add_argument("--metric_file", type=str, help="The metrics file.")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs=3,
        choices=["accuracy", "energy", "performance", "area"],
        help="The metrics to plot. Should have 3 metrics among 'accuracy', 'energy', 'performance', 'area'.",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="If True, plot the baseline."
    )
    parser.add_argument(
        "--ineligible", action="store_true", help="If True, plot the ineligible points."
    )
    parser.add_argument(
        "--dominated",
        action="store_true",
        help="If True, plot the eligible dominated points.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Numerical settings
    # accuracy, energy, performance, area
    constraint: List[float] = [0.9, 0.5, 0.2, 0.5]
    baseline: List[float] = [0.93, 0.2, 0.2, 0.2]
    baseline_label: str = "Baseline"
    baseline_color: str = "red"
    notes: List[int] = [1, 10, 39]

    # get parameters
    metric_file: str = args.metric_file
    metrics: List[str] = args.metrics
    plot_baseline: bool = args.baseline
    plot_ineligible: bool = args.ineligible
    plot_dominated: bool = args.dominated

    main(
        metric_file,
        metrics,
        plot_baseline,
        plot_ineligible,
        plot_dominated,
        constraint,
        baseline,
        baseline_label,
        baseline_color,
        notes,
    )
