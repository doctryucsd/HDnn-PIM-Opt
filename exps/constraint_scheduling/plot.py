from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# Ensure repo root is importable to reach plots.hv
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOTS_DIR = os.path.join(REPO_ROOT, "plots")
for p in (REPO_ROOT, PLOTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from plots.hv import compute_hv_series  # type: ignore


# Defaults: single global threshold and ref point (minimization space)
# You can customize these for your analysis.
DEFAULT_REF_POINT: List[float] = [0.0, 1.0, 1.0, 1.0]
DEFAULT_CONSTRAINTS: List[float] = [0.3, 0.2, 0.2, 0.2]  # [acc_min, energy_max, timing_max, area_max]


def read_metrics(path: str) -> Dict[str, List[float]]:
    with open(path, "r") as f:
        return json.load(f)


def compute_eligibility_rate_from_metrics(metrics: Dict[str, List[float]], constraints: List[float]) -> float:
    acc = metrics.get("accuracy", [])
    eng = metrics.get("energy", [])
    tim = metrics.get("timing", [])
    area = metrics.get("area", [])
    N = min(len(acc), len(eng), len(tim), len(area))
    if N == 0:
        return 0.0
    acc_thr, eng_thr, tim_thr, area_thr = constraints
    feasible = [
        (a >= acc_thr) and (e <= eng_thr) and (t <= tim_thr) and (ar <= area_thr)
        for a, e, t, ar in zip(acc[:N], eng[:N], tim[:N], area[:N])
    ]
    return float(np.sum(feasible)) / float(N)


def compute_eligibility_rate(path: str, constraints: List[float]) -> float:
    metrics = read_metrics(path)
    return compute_eligibility_rate_from_metrics(metrics, constraints)


def compute_hv_at_iter(
    path: str, ref_point: List[float], constraints: List[float], iter_index: int
) -> float:
    metrics = read_metrics(path)
    hv_series = compute_hv_series(
        metrics=metrics,
        ref_point=ref_point,
        constrained=True,
        constraints=constraints,
    )
    if not hv_series:
        return 0.0
    if iter_index is None or iter_index < 0:
        return float(hv_series[-1])
    idx = min(iter_index, len(hv_series) - 1)
    return float(hv_series[idx])


def gather_method_hvs(
    dataset_dir: str,
    ref_point: List[float],
    constraints: List[float],
    iter_index: int,
) -> Tuple[Dict[str, List[float]], Dict[str, Tuple[float, str]], Dict[str, List[float]]]:
    method_to_hvs: Dict[str, List[float]] = {}
    method_to_best: Dict[str, Tuple[float, str]] = {}
    method_to_rates: Dict[str, List[float]] = {}

    for entry in sorted(os.listdir(dataset_dir)):
        method_path = os.path.join(dataset_dir, entry)
        if not os.path.isdir(method_path):
            continue

        # Collect all JSON files for this method (one per seed)
        json_files = [
            os.path.join(method_path, f)
            for f in os.listdir(method_path)
            if f.endswith(".json") and os.path.isfile(os.path.join(method_path, f))
        ]
        if not json_files:
            continue

        final_hvs: List[float] = []
        best_val: float = float("-inf")
        best_file: str = ""
        for jf in sorted(json_files):
            try:
                hv_val = compute_hv_at_iter(jf, ref_point, constraints, iter_index)
                final_hvs.append(hv_val)
                if hv_val > best_val:
                    best_val = hv_val
                    best_file = jf
                rate_val = compute_eligibility_rate(jf, constraints)
                method_to_rates.setdefault(entry, []).append(rate_val)
            except Exception as e:
                print(f"[warn] Skipping {jf}: {e}")

        if final_hvs:
            method_to_hvs[entry] = final_hvs
            method_to_best[entry] = (best_val, best_file)

    return method_to_hvs, method_to_best, method_to_rates


def plot_bar_with_error(
    method_stats: Dict[str, Tuple[float, float]],
    method_values: Dict[str, List[float]],
    title: str,
    out_path: str,
) -> None:
    plt.rcParams.update({"font.size": 14})
    methods = list(method_stats.keys())
    means = [method_stats[m][0] for m in methods]

    # Use range (min..max) as error bars instead of stddev
    yerr_lower: List[float] = []
    yerr_upper: List[float] = []
    for m, mean in zip(methods, means):
        vals = np.array(method_values[m], dtype=float)
        vmin = float(np.min(vals)) if vals.size else mean
        vmax = float(np.max(vals)) if vals.size else mean
        yerr_lower.append(max(mean - vmin, 0.0))
        yerr_upper.append(max(vmax - mean, 0.0))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=np.vstack([yerr_lower, yerr_upper]), capsize=6, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Hypervolume")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Shrink the bottom part: zoom y-axis around min..max with a small padding
    if means:
        lower_candidates = [float(np.min(method_values[m])) for m in methods]
        upper_candidates = [float(np.max(method_values[m])) for m in methods]
        y_low = float(np.min(lower_candidates))
        y_high = float(np.max(upper_candidates))
        rng = max(y_high - y_low, 1e-6)
        pad = 0.05 * rng
        ax.set_ylim(y_low - pad, y_high + pad)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_bar_with_std(
    method_stats: Dict[str, Tuple[float, float]], title: str, out_path: str, ylabel: str = "Eligibility Rate"
) -> None:
    plt.rcParams.update({"font.size": 14})
    methods = list(method_stats.keys())
    means = [method_stats[m][0] for m in methods]
    stds = [method_stats[m][1] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=6, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and plot constrained HV and eligibility rate across methods for a dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset results directory (e.g., exps/constraint_scheduling/cifar10)")
    parser.add_argument(
        "--ref_point",
        type=float,
        nargs=4,
        default=DEFAULT_REF_POINT,
        help="Reference point in minimization space [acc_neg_ref, energy_ref, timing_ref, area_ref]",
    )
    parser.add_argument(
        "--constraints",
        type=float,
        nargs=4,
        default=DEFAULT_CONSTRAINTS,
        help="Constraint thresholds [acc_min, energy_max, timing_max, area_max]",
    )
    parser.add_argument("--title", type=str, default=None, help="Plot title. Defaults to '<dataset> [Final|Iter k] HV (constrained)'")
    parser.add_argument(
        "--out_name",
        type=str,
        default="hv_bar_constrained.pdf",
        help="Filename for the saved PDF placed under the dataset directory.",
    )
    parser.add_argument(
        "--rate_out_name",
        type=str,
        default="eligibility_rate_bar.pdf",
        help="Filename for the eligibility rate PDF placed under the dataset directory.",
    )
    parser.add_argument(
        "--iter",
        dest="iter_index",
        type=int,
        default=-1,
        help="Iteration index to consider (0-based). Use -1 for final iteration.",
    )

    args = parser.parse_args()

    dataset_dir: str = args.dataset_dir
    ref_point: List[float] = args.ref_point
    constraints: List[float] = args.constraints
    out_name: str = args.out_name
    rate_out_name: str = args.rate_out_name
    iter_index: int = args.iter_index

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    # Build default title based on iteration selection
    if args.title:
        title = args.title
    else:
        if iter_index is None or iter_index < 0:
            title = f"{dataset_name} Final HV (constrained)"
        else:
            title = f"{dataset_name} HV (iter {iter_index}, constrained)"

    method_hvs, method_best, method_rates = gather_method_hvs(dataset_dir, ref_point, constraints, iter_index)
    if not method_hvs:
        print(f"No method results found in {dataset_dir}")
        return

    # Compute population statistics (ddof=0)
    method_stats: Dict[str, Tuple[float, float]] = {}
    for method, values in method_hvs.items():
        arr = np.array(values, dtype=float)
        mean = float(np.mean(arr)) if arr.size else 0.0
        std = float(np.std(arr, ddof=0)) if arr.size else 0.0  # population std
        method_stats[method] = (mean, std)

    # Eligibility rate stats (population std)
    method_rate_stats: Dict[str, Tuple[float, float]] = {}
    for method, values in method_rates.items():
        arr = np.array(values, dtype=float)
        mean = float(np.mean(arr)) if arr.size else 0.0
        std = float(np.std(arr, ddof=0)) if arr.size else 0.0
        method_rate_stats[method] = (mean, std)

    # Print stats
    print(f"Dataset: {dataset_name}")
    print(f"Ref point: {ref_point}")
    print(f"Constraints [acc_min, energy_max, timing_max, area_max]: {constraints}")
    print(
        "Iteration: final" if (iter_index is None or iter_index < 0) else f"Iteration: {iter_index}"
    )
    print("")
    for method in sorted(method_stats.keys()):
        mean, std = method_stats[method]
        n = len(method_hvs[method])
        print(f"{method}: mean={mean:.6f}, std={std:.6f} (N={n})")
        best_val, best_file = method_best.get(method, (float("nan"), ""))
        if best_file:
            print(f"  best={best_val:.6f}  file={os.path.basename(best_file)}")
        else:
            print(f"  best=N/A  file=")

    # Print eligibility rate stats
    print("\nEligibility Rate (feasible/total)")
    for method in sorted(method_rate_stats.keys()):
        mean, std = method_rate_stats[method]
        n = len(method_rates.get(method, []))
        print(f"{method}: mean={mean:.4f}, std={std:.4f} (N={n})")

    # Plot and save PDF
    out_path = os.path.join(dataset_dir, out_name)
    # Keep plotting order deterministic (sorted by method name)
    method_stats_sorted = {m: method_stats[m] for m in sorted(method_stats.keys())}
    method_values_sorted = {m: method_hvs[m] for m in sorted(method_stats.keys())}
    plot_bar_with_error(method_stats_sorted, method_values_sorted, title, out_path)
    print(f"\nSaved: {out_path}")

    # Plot eligibility rate bar with std error bars
    rate_title = f"{dataset_name} Eligibility Rate"
    rate_out_path = os.path.join(dataset_dir, rate_out_name)
    method_rate_stats_sorted = {m: method_rate_stats[m] for m in sorted(method_rate_stats.keys())}
    plot_bar_with_std(method_rate_stats_sorted, rate_title, rate_out_path, ylabel="Eligibility Rate")
    print(f"Saved: {rate_out_path}")


if __name__ == "__main__":
    main()
