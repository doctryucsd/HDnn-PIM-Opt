from __future__ import annotations

import json
from sys import argv
from typing import List, Tuple

from matplotlib import pyplot as plt
from argparse import ArgumentParser
from utils import get_file_name


def plot_hvs(hvs: List[Tuple[str, List[float]]], start_iter: int, end_iter: int, title: str):
    plt.rcParams.update({"font.size": 17})
    # plot all hvs in one figure
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, hv in hvs:
        ax.plot(hv[start_iter:end_iter], label=name, linewidth=3)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Hypervolume")
    ax.legend(loc="upper left")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(f"{title}.png")


def main(json_files: List[str], title: str, start_iter: int, end_iter: int, constraint: bool) -> None:
    hvs: List[Tuple[str, List[float]]] = []
    for file in json_files:
        name = get_file_name(file)
        with open(file, "r") as f:
            obj = json.load(f)
            if constraint:
                hv: List[float] = obj["hv_constrained"]
            else:
                hv: List[float] = obj["hv"]
            hvs.append((name, hv))

    plot_hvs(hvs, start_iter, end_iter, title)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Add arguments
    parser.add_argument("--json_files", type=str, nargs="+", help="The json files.")
    parser.add_argument("--title", type=str, default="hv", help="The json files.")
    parser.add_argument(
        "--constraint", action="store_true", help="If True, plot the constrained hv."
    )
    parser.add_argument("--start_iter", type=int, default=9, help="The starting iteration for plotting.")
    parser.add_argument("--end_iter", type=int, default=-1, help="The ending iteration for plotting.")

    args = parser.parse_args()

    # get the parameters
    json_files: List[str] = args.json_files
    title: str = args.title
    constraint: bool = args.constraint
    strart_iter: int = args.start_iter
    end_iter: int = args.end_iter

    main(json_files, title, strart_iter, end_iter, constraint)
