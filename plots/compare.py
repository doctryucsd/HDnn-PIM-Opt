from __future__ import annotations

import json
from sys import argv
from typing import Dict, List, Tuple

from botorch.utils.multi_objective import is_non_dominated
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch import Tensor

NUM_ITERS: int = 40
METRIC_NAMES = ["accuracy", "power", "performance", "area"]


def read_metric_file(file: str) -> Dict[str, List[float]]:
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


def compare(
    pareto_set: Dict[str, List[float]], hypermetric: List[float], indices: List[int]
) -> List[Tuple[List[float], int]]:
    # accuracy
    accuracy = pareto_set["accuracy"]
    accuracy_comparison = [(acc - hypermetric[0]) for acc in accuracy]

    # power
    power = pareto_set["power"]
    power_comparison = [(1 - po / hypermetric[1]) for po in power]

    # performance
    performance = pareto_set["performance"]
    performance_comparison = [(1 - perf / hypermetric[2]) for perf in performance]

    # area
    area = pareto_set["area"]
    area_comparison = [(1 - ar / hypermetric[3]) for ar in area]

    ret: List[Tuple[List[float], int]] = []
    for i in range(len(accuracy)):
        ret.append(
            (
                [
                    accuracy_comparison[i] * 100,
                    power_comparison[i] * 100,
                    performance_comparison[i] * 100,
                    area_comparison[i] * 100,
                ],
                indices[i],
            )
        )

    return ret


def main() -> None:
    assert len(argv) == 2, "Usage: python pareto_set.py <metrics.json>"

    metrics_type = {
        "accuracy": "max",
        "power": "min",
        "performance": "min",
        "area": "min",
    }
    # labels = [r"Accuracy(\%)", r"Power(W)", r"Timing($\mu$s)", r"Area($mm^2$)"]
    labels = ["Accuracy(%)", "Power(W)", "Timing(us)", "Area(mm2)"]

    metrics = read_metric_file(argv[1])
    metrics.pop("hv")
    metrics.pop("param")
    pareto_set, indices = get_pareto_set(metrics, metrics_type)

    HyperMetric: List[float] = [0.88506, 0.06331275, 4.216, 0.058311]

    from rich import print

    print(compare(pareto_set, HyperMetric, indices))


if __name__ == "__main__":
    main()
