from __future__ import annotations

import json
from sys import argv
from typing import Any, Dict, List, Tuple

from botorch.utils.multi_objective import is_non_dominated
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from torch import Tensor

NUM_ITERS: int = 40

# METRIC_NAMES = ["hd_dim", "reram_size", "kron", "frequency"]
METRIC_NAMES = ["hd_dim", "frequency", "kron"]


def read_metric_file(file: str) -> Any:
    with open(file, "r") as f:
        obj = json.load(f)
        return obj


def get_pareto_set(
    metrics: Dict[str, List[float]], metrics_type: Dict[str, str]
) -> Tuple[Dict[str, List[float]], List[int]]:
    y_list: List[List[float]] = [[] for _ in metrics["area"]]
    for metric_name, values in metrics.items():
        metric_type = metrics_type[metric_name]
        for iter, value in enumerate(values):
            if metric_type == "min":
                y_list[iter].append(-value)
            elif metric_type == "max":
                y_list[iter].append(value)
            else:
                raise ValueError(f"Unknown metric_type: {metric_type}")

    for point in y_list:
        assert len(point) == len(
            metrics
        ), f"All metrics should have the same number of values: {len(metrics)}"

    y: Tensor = Tensor(y_list)  # type: ignore
    output = is_non_dominated(y)
    indices = [i for i, x in enumerate(output) if x]
    assert len(output.shape) == 1, "output should be 1D"

    ret: Dict[str, List[float]] = {}
    for metric_name, values in metrics.items():
        for idx, is_in_pareto in enumerate(output):
            if is_in_pareto:
                if metric_name not in ret:
                    ret[metric_name] = []
                ret[metric_name].append(metrics[metric_name][idx])
    return ret, indices


def plot_3d_scatter(
    metrics: Dict[str, List[float]],
    indices: List[int],
    labels: List[str],
    extra_point: List[float],
    notes: List[int],
    point_label="HyperMetric",
    point_color="red",
):
    """
    Plots a 2x2 grid of 3D scatter plots for four combinations of three lists,
    with an additional point plotted in a distinct color.

    Args:
    metrics (dict): A dictionary containing four lists of data points.
    labels (list of str): A list containing labels for each list.
    extra_point (list): A list of four values, one for each dimension.
    point_label (str): Label for the extra point.
    point_color (str): Color for the extra point.
    """
    # Ensure there are exactly four lists and four labels
    # if len(metrics) != 4 or len(labels) != 4 or len(extra_point) != 4:
    if len(metrics) != 3 or len(labels) != 3 or len(extra_point) != 3:
        # raise ValueError("There must be exactly four lists, four labels, and four coordinates for the extra point.")
        raise ValueError(
            "There must be exactly three lists, three labels, and three coordinates for the extra point."
        )

    # Enable LaTeX rendering and set font size
    # plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 26

    # Creating a figure
    fig = plt.figure(figsize=(14, 10))

    # Titles and combinations setup
    # combinations = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    combinations = [(0, 1, 2)]

    # Loop through each subplot configuration
    for i, (idx1, idx2, idx3) in enumerate(combinations):
        # metric names
        metric1 = METRIC_NAMES[idx1]
        metric2 = METRIC_NAMES[idx2]
        metric3 = METRIC_NAMES[idx3]

        # ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax = fig.add_subplot(1, 1, i + 1, projection="3d")
        pareto_scatter = ax.scatter(metrics[metric1], metrics[metric2], metrics[metric3], c=indices, cmap="viridis", s=160)  # type: ignore
        for j, pareto_index in enumerate(indices):
            if pareto_index in notes:
                ax.text(metrics[metric1][j], metrics[metric2][j], metrics[metric3][j], str(pareto_index), fontweight="bold", zorder=5)  # type: ignore
        ax.scatter(extra_point[idx1], extra_point[idx2], extra_point[idx3], color=point_color, s=200, label=point_label)  # type: ignore
        ax.set_xlabel(labels[idx1], labelpad=20)
        ax.set_ylabel(labels[idx2], labelpad=20)
        ax.set_zlabel(labels[idx3], labelpad=20)  # type: ignore
        ax.set_zlim(-0.01, 1.01)  # type: ignore
        ax.legend()
        ax.xaxis.set_major_locator(MultipleLocator(2048))
        ax.zaxis.set_major_locator(MultipleLocator(1))  # type: ignore
        ax.zaxis.set_major_formatter(FuncFormatter(lambda x, _: "RP" if x == 0 else "KP"))  # type: ignore

        # Add a colorbar with selected indices
        num_ticks = 5  # Number of ticks to display on the colorbar
        tick_indices = [
            int(i) for i in [NUM_ITERS * j / (num_ticks - 1) for j in range(num_ticks)]
        ]
        # tick_indices.append(len(indices) - 1)  # Add the final index
        tick_indices_pareto = sorted(set(tick_indices))  # Remove duplicates and sort
        cbar = plt.colorbar(pareto_scatter, ax=ax, ticks=tick_indices, pad=0.1)
        cbar.ax.set_yticklabels(tick_indices_pareto)  # type: ignore
        cbar.set_label("Iteration", rotation=270, labelpad=30)

    # Adjust layout
    # plt.tight_layout()

    # Show the plot
    plt.savefig("pareto_set_param.png")


def get_pareto_indices() -> List[int]:
    metrics_type = {
        "accuracy": "max",
        "power": "min",
        "performance": "min",
        "area": "min",
    }
    metrics = read_metric_file(argv[1])
    metrics.pop("hv")
    metrics.pop("param")
    _, indices = get_pareto_set(metrics, metrics_type)
    return indices


def process_params(
    params: List[Dict[str, float]], indices: List[int]
) -> Dict[str, List[float]]:
    ret: Dict[str, List[float]] = {}
    for param_name in params[0].keys():
        ret[param_name] = []

    for idx in indices:
        param_iter = params[idx]
        for param_name, param_value in param_iter.items():
            ret[param_name].append(param_value)

    return ret


def main() -> None:
    assert len(argv) == 2, "Usage: python pareto_set.py <metrics.json>"

    # labels = ["hd_dim", "reram_size", "kron", "frequency(Hz)"]
    labels = ["hd_dim", "freq(Hz)", "kron"]

    indices = get_pareto_indices()
    params = read_metric_file(argv[1])["param"]
    processed_params = process_params(params, indices)
    input_params = {}
    for metric_name in METRIC_NAMES:
        input_params[metric_name] = processed_params[metric_name]
    # HyperMetric: List[float] = [2048, 64, True, 5e8]
    HyperMetric: List[float] = [2048, 5e8, True]

    note = [1, 10, 39]

    plot_3d_scatter(input_params, indices, labels, HyperMetric, note)


if __name__ == "__main__":
    main()
