from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from pymoo.indicators.hv import HV
from utils import get_file_name

# Defaults to edit here if you want a baked-in setting
DEFAULT_REF_POINT: List[float] = [0.0, 1.0, 1.0, 1.0]
DEFAULT_CONSTRAINTS: List[float] = [0.3, 0.2, 0.2, 0.2]


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


def transform_points(acc: List[float], eng: List[float], tim: List[float], area: List[float]) -> np.ndarray:
    # Convert maximization of accuracy to minimization by negation
    return np.array([[-a, e, t, ar] for a, e, t, ar in zip(acc, eng, tim, area)], dtype=float)


def compute_hv_series(
    metrics: dict,
    ref_point: List[float],
    constrained: bool,
    constraints: List[float] | None,
) -> List[float]:
    acc = metrics["accuracy"]
    eng = metrics["energy"]
    tim = metrics["timing"]
    area = metrics["area"]

    # Prepare transformed points
    points_all = transform_points(acc, eng, tim, area)
    N = len(points_all)

    # Feasibility mask for constrained case
    if constrained:
        assert constraints is not None and len(constraints) == 4, "constraints must be 4 floats when --constraint is used with --compute"
        acc_thr, eng_thr, tim_thr, area_thr = constraints
        feasible = [
            (a >= acc_thr) and (e <= eng_thr) and (t <= tim_thr) and (ar <= area_thr)
            for a, e, t, ar in zip(acc, eng, tim, area)
        ]
    else:
        feasible = [True] * N

    hv_ind = HV(ref_point=np.array(ref_point, dtype=float))

    series: List[float] = []
    for i in range(N):
        # Select points up to i that are feasible (or all points if unconstrained)
        pts = points_all[: i + 1][[j for j in range(i + 1) if feasible[j]]]
        if pts.size == 0:
            series.append(0.0)
        else:
            val = float(hv_ind(pts))
            series.append(val)
    return series


def main(
    json_files: List[str],
    title: str,
    start_iter: int,
    end_iter: int,
    constraint: bool,
    compute: bool,
    ref_point: List[float],
    constraints: List[float] | None,
) -> None:
    hvs: List[Tuple[str, List[float]]] = []
    for file in json_files:
        name = get_file_name(file)
        with open(file, "r") as f:
            obj = json.load(f)
            if compute:
                hv: List[float] = compute_hv_series(obj, ref_point, constraint, constraints)
            else:
                if constraint:
                    hv = obj["hv_constrained"]
                else:
                    hv = obj["hv"]
            hvs.append((name, hv))

    plot_hvs(hvs, start_iter, end_iter, title)

    # Print final HV values for all files at the end
    print("\nFinal hypervolume values:")
    for name, hv in hvs:
        if not hv:
            print(f"- {name}: N/A (no data)")
            continue
        # Always print the absolute final HV, ignoring end_iter
        final_val = hv[-1]
        print(f"- {name}: {final_val:.6f}")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Add arguments
    parser.add_argument("--json_files", type=str, nargs="+", help="The json files.")
    parser.add_argument("--title", type=str, default="hv", help="Plot title and output filename prefix.")
    parser.add_argument("--constraint", action="store_true", help="If set, plot constrained HV.")
    parser.add_argument("--start_iter", type=int, default=0, help="Starting iteration (inclusive).")
    parser.add_argument("--end_iter", type=int, default=-1, help="Ending iteration (exclusive). -1 for end.")

    # New: compute HV from metrics instead of reading hv arrays
    parser.add_argument("--compute", action="store_true", help="Compute HV from metrics instead of reading precomputed values.")
    parser.add_argument(
        "--ref_point",
        type=float,
        nargs=4,
        default=DEFAULT_REF_POINT,
        help="Reference point for HV (minimization space). Default is defined in hv.py",
    )
    parser.add_argument(
        "--constraints",
        type=float,
        nargs=4,
        default=None,
        help="Constraints [acc_min, energy_max, timing_max, area_max]. If omitted with --compute --constraint, uses DEFAULT_CONSTRAINTS in hv.py.",
    )

    args = parser.parse_args()

    json_files: List[str] = args.json_files
    title: str = args.title
    constraint: bool = args.constraint
    start_iter: int = args.start_iter
    end_iter: int = args.end_iter
    compute: bool = args.compute
    ref_point: List[float] = args.ref_point
    constraints: List[float] | None = args.constraints

    if end_iter == -1:
        end_iter = None  # slice to end

    # Apply defaults if needed
    if compute and constraint and constraints is None:
        constraints = DEFAULT_CONSTRAINTS
        # Optional: print which defaults are used
        print(f"[hv.py] Using default constraints from hv.py: {constraints}")

    main(json_files, title, start_iter, end_iter, constraint, compute, ref_point, constraints)
