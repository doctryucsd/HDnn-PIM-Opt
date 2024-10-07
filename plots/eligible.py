from __future__ import annotations
from typing import List
from argparse import ArgumentParser
from database import PointSet
from utils import read_metric_file, get_file_name, get_folder_name
import numpy as np
import matplotlib.pyplot as plt
import os

def get_eligible_ratio(metric_file: str, start_iter: int, end_iter: int, constraints: List[float]) -> float:
    data = read_metric_file(metric_file)
    point_set = PointSet.from_dict(data)[start_iter: end_iter]
    assert isinstance(point_set, PointSet), f"point_set is not of type PointSet, but {type(point_set)}"

    eligible_points = [point for point in point_set if point.is_eligible(constraints)]
    return len(eligible_points) / len(point_set)

def plot_bar_chart(accuracy: list[list[float]], workloads: list[str], methods: list[str]):
    # Width of a single bar
    bar_width = 0.2

    # Set positions of the bars on the x-axis for each workload
    gap = 0.3
    indices = np.arange(len(workloads)) * (len(methods) * bar_width + gap)
    
    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot each workload's bars with a small offset from the others
    for i in range(len(methods)):
        method_accuracies = [accuracy[w][i] for w in range(len(workloads))]  # Get accuracies for method i across all workloads
        ax.bar(indices + i * bar_width, method_accuracies, bar_width, label=methods[i])

    # Add labels, title, and legend
    ax.set_ylabel('Eligible Rate', fontsize=14)
    ax.set_xticks(indices + bar_width * (len(methods) - 1) / 2)
    ax.set_xticklabels(workloads, fontsize=12)
    ax.legend(fontsize=11)

    plt.savefig("eligible_rates.png")


def main(metric_files: List[str], methods: List[str], workloads: List[str], constraints: List[List[float]], start_iter: int, end_iter: int) -> None:
    workload_eligible_rates: List[List[float]] = []

    metric_files_workloads = [metric_files[i:i + len(methods)] for i in range(0, len(metric_files), len(methods))]
    assert len(metric_files_workloads) == len(workloads), f"Number of workloads {len(workloads)} does not match the number of metric files {len(metric_files_workloads)}"

    for metric_files_workload, constraint in zip(metric_files_workloads, constraints):
        method_eligible_rates: List[float] = []
        for metric_file in metric_files_workload:
            eligible_ratio = get_eligible_ratio(metric_file, start_iter, end_iter, constraint)
            method_eligible_rates.append(eligible_ratio)
        workload_eligible_rates.append(method_eligible_rates)

    plot_bar_chart(workload_eligible_rates, workloads, methods)

    from rich import print
    print(workload_eligible_rates)
    

if __name__ == "__main__":
    # Create the parser
    parser = ArgumentParser()

    # Add arguments
    parser.add_argument("--metric_files", nargs="+", type=str, help="The metric files.")
    parser.add_argument("--start_iter", type=int, default=10, help="The start iteration.")
    parser.add_argument("--end_iter", type=int, default=-1, help="The end iteration.")

    # Parse arguments
    args = parser.parse_args()

    # Numerical settings
    # accuracy, energy, performance, area
    constraints: List[List[float]] = [
        [0.9, 0.2, 0.2, 0.2],
        [0.8, 0.2, 0.2, 0.2],
        [0.3, 0.2, 0.2, 0.2],
        ]
    
    methods: List[str] = ["random", "EHVI", "NEHVI", "EHVI_constraint", "NEHVI_constraint"]
    workloads: List[str] = ["MNIST", "Fashion-MNIST", "CIFAR-10"]

    # get parameters
    metric_files: List[str] = args.metric_files
    start_iter: int = args.start_iter
    end_iter: int = args.end_iter

    main(metric_files, methods, workloads, constraints, start_iter, end_iter)