from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, TypeVar
import re

import numpy as np
import matplotlib.pyplot as plt
import hashlib
import matplotlib.patheffects as pe


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
DEFAULT_CONSTRAINTS: List[float] = [0.85, 0.1, 0.1, 0.2]  # [acc_min, energy_max, timing_max, area_max]
# For HV-vs-iteration curves: start plotting from this iteration index (0-based)
CURVE_START_ITER: int = 10

# Color-blind friendly palette (Okabe–Ito)
COLORBLIND_PALETTE: List[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#000000",  # black
]

# Additional distinct styles to reduce ambiguity when lines overlap
LINESTYLES: List[str] = ["-", "--", "-.", ":"]
MARKERS: List[str] = ["o", "s", "^", "v", "D", "P", "X", "*", "h"]

NEHVI_METHOD_ORDER: List[str] = ["NEHVI_linear", "NEHVI_static", "NEHVI_no-constraint", "random"]
EHVI_METHOD_ORDER: List[str] = ["EHVI_linear", "EHVI_static", "EHVI_no-constraint", "random"]

T = TypeVar("T")


def _merge_seed_skips(target: Dict[str, set[int]], methods: List[str], seeds: set[int]) -> None:
    """Union the provided seeds into target[method] for each method in methods."""
    if not seeds:
        return
    for method in methods:
        target.setdefault(method, set()).update(seeds)


def _style_for_name(name: str) -> Tuple[str, str, str, int]:
    """Deterministically assign (color, linestyle, marker, markevery) from a name.

    Uses a stable hash so the same method keeps the same style across runs.
    markevery is chosen to stagger marker positions across methods to avoid
    perfect overlap of markers.
    """
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16)
    color = COLORBLIND_PALETTE[h % len(COLORBLIND_PALETTE)]
    linestyle = LINESTYLES[(h // len(COLORBLIND_PALETTE)) % len(LINESTYLES)]
    marker = MARKERS[(h // (len(COLORBLIND_PALETTE) * len(LINESTYLES))) % len(MARKERS)]
    markevery = 5 + (h % 5)  # place markers every 5..9 points with per-name offset
    return color, linestyle, marker, markevery


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
    skip_seeds: set[int] | None = None,
    skip_methods: set[str] | None = None,
    per_method_skip_seeds: Dict[str, set[int]] | None = None,
) -> Tuple[Dict[str, List[float]], Dict[str, Tuple[float, str]], Dict[str, List[float]]]:
    method_to_hvs: Dict[str, List[float]] = {}
    method_to_best: Dict[str, Tuple[float, str]] = {}
    method_to_rates: Dict[str, List[float]] = {}

    for entry in sorted(os.listdir(dataset_dir)):
        method_path = os.path.join(dataset_dir, entry)
        if not os.path.isdir(method_path):
            continue
        if skip_methods and entry in skip_methods:
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
        method_seeds_to_skip: set[int] = set(skip_seeds or [])
        if per_method_skip_seeds and entry in per_method_skip_seeds:
            method_seeds_to_skip.update(per_method_skip_seeds[entry])

        for jf in sorted(json_files):
            if method_seeds_to_skip:
                seed = _extract_seed_from_filename(jf)
                if seed is not None and seed in method_seeds_to_skip:
                    continue
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
    ylabel: str = "Hypervolume",
    show_error: bool = True,
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
    if show_error:
        ax.bar(x, means, yerr=np.vstack([yerr_lower, yerr_upper]), capsize=6, alpha=0.9)
    else:
        ax.bar(x, means, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Shrink the bottom part: zoom y-axis around min..max with a small padding
    if means:
        if show_error:
            lower_candidates = [float(np.min(method_values[m])) for m in methods]
            upper_candidates = [float(np.max(method_values[m])) for m in methods]
        else:
            lower_candidates = [float(mean) for mean in means]
            upper_candidates = [float(mean) for mean in means]
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


def compute_hv_series_for_file(path: str, ref_point: List[float], constraints: List[float]) -> List[float]:
    metrics = read_metrics(path)
    series = compute_hv_series(
        metrics=metrics,
        ref_point=ref_point,
        constrained=True,
        constraints=constraints,
    )
    return [float(x) for x in series]


def gather_method_hv_series(
    dataset_dir: str,
    ref_point: List[float],
    constraints: List[float],
    skip_seeds: set[int] | None = None,
    skip_methods: set[str] | None = None,
    per_method_skip_seeds: Dict[str, set[int]] | None = None,
) -> Dict[str, List[List[float]]]:
    method_to_series: Dict[str, List[List[float]]] = {}

    for entry in sorted(os.listdir(dataset_dir)):
        method_path = os.path.join(dataset_dir, entry)
        if not os.path.isdir(method_path):
            continue
        if skip_methods and entry in skip_methods:
            continue
        json_files = [
            os.path.join(method_path, f)
            for f in os.listdir(method_path)
            if f.endswith(".json") and os.path.isfile(os.path.join(method_path, f))
        ]
        if not json_files:
            continue
        series_list: List[List[float]] = []
        method_seeds_to_skip: set[int] = set(skip_seeds or [])
        if per_method_skip_seeds and entry in per_method_skip_seeds:
            method_seeds_to_skip.update(per_method_skip_seeds[entry])

        for jf in sorted(json_files):
            if method_seeds_to_skip:
                seed = _extract_seed_from_filename(jf)
                if seed is not None and seed in method_seeds_to_skip:
                    continue
            try:
                series_list.append(compute_hv_series_for_file(jf, ref_point, constraints))
            except Exception as e:
                print(f"[warn] HV series failed for {jf}: {e}")
        if series_list:
            method_to_series[entry] = series_list
    return method_to_series


def _extract_seed_from_filename(path: str) -> int | None:
    """Extract integer seed from filename patterns like '*_seed42.json'."""
    base = os.path.basename(path)
    m = re.search(r"seed(\d+)", base)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _filter_dict_by_order(source: Dict[str, T], order: List[str]) -> Dict[str, T]:
    """Return an ordered subset of a dict based on the provided method order."""
    return {method: source[method] for method in order if method in source}


def _filename_with_suffix(filename: str, suffix: str) -> str:
    """Append a suffix to a filename while preserving the extension."""
    base, ext = os.path.splitext(filename)
    return f"{base}_{suffix}{ext}" if ext else f"{filename}_{suffix}"


def gather_seed_hv_series(
    dataset_dir: str,
    ref_point: List[float],
    constraints: List[float],
    skip_seeds: set[int] | None = None,
    skip_methods: set[str] | None = None,
    per_method_skip_seeds: Dict[str, set[int]] | None = None,
) -> Tuple[Dict[int, Dict[str, List[float]]], List[str]]:
    """
    Build mapping: seed -> { method_name -> HV series (sliced from CURVE_START_ITER) }.
    Also return the sorted list of all method names to enable missing-method warnings.
    """
    seed_to_method_series: Dict[int, Dict[str, List[float]]] = {}
    all_methods: List[str] = []

    for entry in sorted(os.listdir(dataset_dir)):
        method_path = os.path.join(dataset_dir, entry)
        if not os.path.isdir(method_path):
            continue
        if skip_methods and entry in skip_methods:
            continue

        json_files = [
            os.path.join(method_path, f)
            for f in os.listdir(method_path)
            if f.endswith(".json") and os.path.isfile(os.path.join(method_path, f))
        ]
        if not json_files:
            continue
        all_methods.append(entry)
        method_seeds_to_skip: set[int] = set(skip_seeds or [])
        if per_method_skip_seeds and entry in per_method_skip_seeds:
            method_seeds_to_skip.update(per_method_skip_seeds[entry])

        for jf in sorted(json_files):
            seed = _extract_seed_from_filename(jf)
            if seed is None:
                print(f"[warn] Could not parse seed from filename: {os.path.basename(jf)}")
                continue
            if method_seeds_to_skip and seed in method_seeds_to_skip:
                continue
            try:
                series = compute_hv_series_for_file(jf, ref_point, constraints)
                if len(series) <= CURVE_START_ITER:
                    print(
                        f"[warn] Series shorter than CURVE_START_ITER={CURVE_START_ITER} for {os.path.basename(jf)}; skipping"
                    )
                    continue
                sliced = [float(x) for x in series[CURVE_START_ITER:]]
                seed_to_method_series.setdefault(seed, {})[entry] = sliced
            except Exception as e:
                print(f"[warn] HV series failed for {jf}: {e}")

    # Warn if any seed lacks some methods
    all_methods_sorted = sorted(all_methods)
    for seed, m2s in sorted(seed_to_method_series.items()):
        present = set(m2s.keys())
        missing = [m for m in all_methods_sorted if m not in present]
        if missing:
            print(f"[warn] Seed {seed} missing methods: {', '.join(missing)}")

    return seed_to_method_series, all_methods_sorted


def plot_per_seed_curves(
    seed_to_method_series: Dict[int, Dict[str, List[float]]],
    dataset_name: str,
    out_dir: str,
    method_order: List[str] | None = None,
    filename_suffix: str | None = None,
) -> None:
    """Plot HV vs iteration for each seed, overlaying all available methods."""
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 14})
    saved_any = False

    for seed in sorted(seed_to_method_series.keys()):
        m2s = seed_to_method_series[seed]
        if not m2s:
            continue

        if method_order:
            methods = [m for m in method_order if m in m2s]
        else:
            methods = sorted(m2s.keys())
        if not methods:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        for method in methods:
            y = np.array(m2s[method], dtype=float)
            x = np.arange(len(y))
            color, linestyle, marker, markevery = _style_for_name(method)
            line, = ax.plot(
                x,
                y,
                label=method,
                linewidth=2.5,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=markevery,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.9,
                alpha=0.95,
            )
            line.set_path_effects([
                pe.Stroke(linewidth=4, foreground="white", alpha=0.8),
                pe.Normal(),
            ])

        ax.set_xlabel("Iteration (offset)")
        ax.set_ylabel("Hypervolume")
        ax.set_title(
            f"{dataset_name} Seed {seed} HV vs Iteration (constrained, from iter {CURVE_START_ITER})"
        )
        ax.grid(axis="both", linestyle=":", alpha=0.4)
        ax.legend(loc="best")
        fig.tight_layout()

        suffix_str = f"_{filename_suffix}" if filename_suffix else ""
        out_path = os.path.join(out_dir, f"hv_curve_seed_{seed}_from{CURVE_START_ITER}{suffix_str}.pdf")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved: {out_path}")
        saved_any = True

    if not saved_any:
        label = filename_suffix or ""
        label_str = f" ({label})" if label else ""
        print(f"[warn] No per-seed curves generated for {dataset_name}{label_str}.")


def gather_method_best_metrics(
    dataset_dir: str,
    skip_seeds: set[int] | None = None,
    skip_methods: set[str] | None = None,
    per_method_skip_seeds: Dict[str, set[int]] | None = None,
) -> Dict[str, Dict[str, Tuple[float, str, int, float, float, Dict[str, float]]]]:
    """
    For each method directory, scan all JSON files (seeds) and report the best value for
    each raw metric across all iterations and seeds.

    Returns a mapping:
      method -> {
         'accuracy': (max_value, filename, iter_idx),
         'energy':   (min_value, filename, iter_idx),
         'timing':   (min_value, filename, iter_idx),
         'area':     (min_value, filename, iter_idx),
      }
    """
    # Returns per method -> per metric -> (best_val, filename, iter_idx, mean, std, context_at_best)
    # where context_at_best holds the other metrics' values at that same iteration index.
    result: Dict[str, Dict[str, Tuple[float, str, int, float, float, Dict[str, float]]]] = {}

    for entry in sorted(os.listdir(dataset_dir)):
        method_path = os.path.join(dataset_dir, entry)
        if not os.path.isdir(method_path):
            continue
        if skip_methods and entry in skip_methods:
            continue

        # Track best and its context metrics for each metric
        best_val_file_idx: Dict[str, Tuple[float, str, int]] = {
            'accuracy': (float('-inf'), '', -1),
            'energy':   (float('inf'),  '', -1),
            'timing':   (float('inf'),  '', -1),
            'area':     (float('inf'),  '', -1),
        }
        best_context: Dict[str, Dict[str, float]] = {
            'accuracy': {},
            'energy':   {},
            'timing':   {},
            'area':     {},
        }
        # Accumulate all observed values for mean/std
        accums: Dict[str, List[float]] = {
            'accuracy': [],
            'energy':   [],
            'timing':   [],
            'area':     [],
        }

        json_files = [
            os.path.join(method_path, f)
            for f in os.listdir(method_path)
            if f.endswith('.json') and os.path.isfile(os.path.join(method_path, f))
        ]
        if not json_files:
            continue

        method_seeds_to_skip: set[int] = set(skip_seeds or [])
        if per_method_skip_seeds and entry in per_method_skip_seeds:
            method_seeds_to_skip.update(per_method_skip_seeds[entry])

        for jf in sorted(json_files):
            if method_seeds_to_skip:
                seed = _extract_seed_from_filename(jf)
                if seed is not None and seed in method_seeds_to_skip:
                    continue
            try:
                metrics = read_metrics(jf)
                for metric_name, prefer_max in (
                    ('accuracy', True),
                    ('energy',   False),
                    ('timing',   False),
                    ('area',     False),
                ):
                    arr = metrics.get(metric_name, [])
                    if not arr:
                        continue
                    # Extend accumulator
                    try:
                        accums[metric_name].extend([float(v) for v in arr])
                    except Exception:
                        pass
                    if prefer_max:
                        val = float(np.max(arr))
                        idx = int(np.argmax(arr))
                        if val > best_val_file_idx[metric_name][0]:
                            best_val_file_idx[metric_name] = (val, os.path.basename(jf), idx)
                            # capture context at this iteration
                            ctx: Dict[str, float] = {}
                            for other in ('accuracy', 'energy', 'timing', 'area'):
                                arr2 = metrics.get(other, [])
                                ctx[other] = float(arr2[idx]) if idx < len(arr2) else float('nan')
                            best_context[metric_name] = ctx
                    else:
                        val = float(np.min(arr))
                        idx = int(np.argmin(arr))
                        if val < best_val_file_idx[metric_name][0]:
                            best_val_file_idx[metric_name] = (val, os.path.basename(jf), idx)
                            ctx: Dict[str, float] = {}
                            for other in ('accuracy', 'energy', 'timing', 'area'):
                                arr2 = metrics.get(other, [])
                                ctx[other] = float(arr2[idx]) if idx < len(arr2) else float('nan')
                            best_context[metric_name] = ctx
            except Exception as e:
                print(f"[warn] Skipping metrics for {jf}: {e}")

        # Attach mean/std to results
        enriched: Dict[str, Tuple[float, str, int, float, float, Dict[str, float]]] = {}
        for metric_name in ('accuracy', 'energy', 'timing', 'area'):
            vals = np.array(accums[metric_name], dtype=float)
            if vals.size:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=0))
            else:
                mean = 0.0
                std = 0.0
            bval, bfile, bidx = best_val_file_idx[metric_name]
            enriched[metric_name] = (bval, bfile, bidx, mean, std, best_context.get(metric_name, {}))

        result[entry] = enriched

    return result


def mean_std_curve(series_list: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    # Start from a fixed iteration to emphasize later performance
    sliced = [s[CURVE_START_ITER:] for s in series_list if len(s) > CURVE_START_ITER]
    if not sliced:
        raise ValueError("All series shorter than CURVE_START_ITER")
    # Align to the shortest length to avoid padding artifacts
    L = min(len(s) for s in sliced)
    arr = np.array([s[:L] for s in sliced], dtype=float)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0, ddof=0)
    return mean, std


def plot_mean_curves(
    method_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    out_path: str,
    method_order: List[str] | None = None,
    shade_std: bool = False,
) -> None:
    if method_order:
        methods = [m for m in method_order if m in method_curves]
    else:
        methods = sorted(method_curves.keys())
    if not methods:
        print(f"[warn] No methods available for mean curve plot '{title}'; skipping.")
        return
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in methods:
        mean, std = method_curves[method]
        x = np.arange(len(mean))
        color, linestyle, marker, markevery = _style_for_name(method)
        line, = ax.plot(
            x,
            mean,
            label=method,
            linewidth=2.5,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=markevery,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            alpha=0.95,
        )
        line.set_path_effects([
            pe.Stroke(linewidth=4, foreground="white", alpha=0.8),
            pe.Normal(),
        ])
        if shade_std:
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Hypervolume")
    ax.set_title(title)
    ax.grid(axis="both", linestyle=":", alpha=0.4)
    ax.legend(loc="best")
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
        "--curve_out_name",
        type=str,
        default="hv_mean_curve.pdf",
        help="Filename for the mean HV vs. iteration PDF placed under the dataset directory.",
    )
    parser.add_argument(
        "--curve_shade_std",
        action="store_true",
        help="If set, shade ±std around the mean curves.",
    )
    parser.add_argument(
        "--iter",
        dest="iter_index",
        type=int,
        default=-1,
        help="Iteration index to consider (0-based). Use -1 for final iteration.",
    )
    parser.add_argument(
        "--ignore_seeds",
        type=int,
        nargs="+",
        default=None,
        help="One or more integer seeds to ignore (matched from filenames like '*seed42*.json').",
    )
    parser.add_argument(
        "--ignore_seeds_nehvi",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to ignore only for NEHVI-family methods (linear/static/no-constraint).",
    )
    parser.add_argument(
        "--ignore_seeds_ehvi",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to ignore only for EHVI-family methods (linear/static/no-constraint).",
    )
    parser.add_argument(
        "--ignore_methods",
        type=str,
        nargs="+",
        default=None,
        help="One or more method directory names to ignore (exact names under dataset_dir).",
    )
    parser.add_argument(
        "--no_hv_bar",
        action="store_true",
        help="If set, do not generate the HV bar plot.",
    )
    parser.add_argument(
        "--no_rate_bar",
        action="store_true",
        help="If set, do not generate the eligibility rate bar plot.",
    )
    parser.add_argument(
        "--no_error_bars",
        action="store_true",
        help="Hide error bars (ranges) in HV and eligibility bar charts.",
    )
    # Deprecated alias retained for backward-compatibility: interpret as no error bars
    parser.add_argument(
        "--no_bars",
        action="store_true",
        help="[Deprecated] Same as --no_error_bars: hide error bars (ranges).",
    )

    args = parser.parse_args()

    dataset_dir: str = args.dataset_dir
    ref_point: List[float] = args.ref_point
    constraints: List[float] = args.constraints
    out_name: str = args.out_name
    rate_out_name: str = args.rate_out_name
    curve_out_name: str = args.curve_out_name
    curve_shade_std: bool = args.curve_shade_std
    iter_index: int = args.iter_index
    skip_seeds: set[int] = set(args.ignore_seeds or [])
    skip_seeds_nehvi: set[int] = set(args.ignore_seeds_nehvi or [])
    skip_seeds_ehvi: set[int] = set(args.ignore_seeds_ehvi or [])
    skip_methods: set[str] = set(args.ignore_methods or [])
    # Convenience toggles
    no_hv_bar = bool(args.no_hv_bar)
    no_rate_bar = bool(args.no_rate_bar)
    hide_error_bars = bool(args.no_error_bars or args.no_bars)

    per_method_skip_seeds: Dict[str, set[int]] = {}
    _merge_seed_skips(per_method_skip_seeds, NEHVI_METHOD_ORDER, skip_seeds_nehvi)
    _merge_seed_skips(per_method_skip_seeds, EHVI_METHOD_ORDER, skip_seeds_ehvi)

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

    if skip_seeds:
        print(f"Ignoring seeds (all methods): {sorted(skip_seeds)}")
    if skip_seeds_nehvi:
        print(f"Ignoring NEHVI seeds: {sorted(skip_seeds_nehvi)}")
    if skip_seeds_ehvi:
        print(f"Ignoring EHVI seeds: {sorted(skip_seeds_ehvi)}")
    if skip_methods:
        print(f"Ignoring methods: {sorted(skip_methods)}")
    if args.no_bars and not args.no_error_bars:
        print("[warn] --no_bars is deprecated; use --no_error_bars. Interpreting as --no_error_bars.")

    method_hvs, method_best, method_rates = gather_method_hvs(
        dataset_dir,
        ref_point,
        constraints,
        iter_index,
        skip_seeds=skip_seeds or None,
        skip_methods=skip_methods or None,
        per_method_skip_seeds=per_method_skip_seeds or None,
    )
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

    # Prepare plots output root directory under the dataset directory
    plots_root_dir = os.path.join(dataset_dir, "plots")
    os.makedirs(plots_root_dir, exist_ok=True)

    # Plot and save HV bar (unless disabled)
    if not no_hv_bar:
        for label, order in (("NEHVI", NEHVI_METHOD_ORDER), ("EHVI", EHVI_METHOD_ORDER)):
            group_stats = _filter_dict_by_order(method_stats, order)
            group_values = _filter_dict_by_order(method_hvs, order)
            if not group_stats:
                print(f"[warn] No methods found for {label} HV bar plot; skipping.")
                continue
            group_title = f"{title} ({label})"
            out_path = os.path.join(plots_root_dir, _filename_with_suffix(out_name, label))
            plot_bar_with_error(
                group_stats,
                group_values,
                group_title,
                out_path,
                show_error=(not hide_error_bars),
            )
            print(f"\nSaved: {out_path}")
    else:
        print("\nSkipping HV bar plot (--no_hv_bar)")

    # Plot eligibility rate bar with range (unless disabled)
    if not no_rate_bar:
        rate_title_base = f"{dataset_name} Eligibility Rate"
        for label, order in (("NEHVI", NEHVI_METHOD_ORDER), ("EHVI", EHVI_METHOD_ORDER)):
            group_rate_stats = _filter_dict_by_order(method_rate_stats, order)
            group_rate_values = _filter_dict_by_order(method_rates, order)
            if not group_rate_stats:
                print(f"[warn] No methods found for {label} eligibility rate plot; skipping.")
                continue
            group_title = f"{rate_title_base} ({label})"
            rate_out_path = os.path.join(plots_root_dir, _filename_with_suffix(rate_out_name, label))
            plot_bar_with_error(
                group_rate_stats,
                group_rate_values,
                group_title,
                rate_out_path,
                ylabel="Eligibility Rate",
                show_error=(not hide_error_bars),
            )
            print(f"Saved: {rate_out_path}")
    else:
        print("Skipping eligibility rate bar plot (--no_rate_bar)")

    # Print best raw metric values per method across all seeds/iterations
    print("\nBest raw metric values across seeds (per method):")
    best_metrics = gather_method_best_metrics(
        dataset_dir,
        skip_seeds=skip_seeds or None,
        skip_methods=skip_methods or None,
        per_method_skip_seeds=per_method_skip_seeds or None,
    )
    for method in sorted(best_metrics.keys()):
        bm = best_metrics[method]
        acc_v, acc_f, acc_i, acc_m, acc_s, acc_ctx = bm['accuracy']
        eng_v, eng_f, eng_i, eng_m, eng_s, eng_ctx = bm['energy']
        tim_v, tim_f, tim_i, tim_m, tim_s, tim_ctx = bm['timing']
        area_v, area_f, area_i, area_m, area_s, area_ctx = bm['area']
        print(f"{method}:")
        print(f"  accuracy_max = {acc_v:.6f}  (file={acc_f}, iter={acc_i}) | mean={acc_m:.6f}, std={acc_s:.6f}")
        print(
            f"    at_iter: energy={acc_ctx.get('energy', float('nan')):.6f}, timing={acc_ctx.get('timing', float('nan')):.6f}, area={acc_ctx.get('area', float('nan')):.6f}"
        )
        print(f"  energy_min   = {eng_v:.6f}  (file={eng_f}, iter={eng_i}) | mean={eng_m:.6f}, std={eng_s:.6f}")
        print(
            f"    at_iter: accuracy={eng_ctx.get('accuracy', float('nan')):.6f}, timing={eng_ctx.get('timing', float('nan')):.6f}, area={eng_ctx.get('area', float('nan')):.6f}"
        )
        print(f"  timing_min   = {tim_v:.6f}  (file={tim_f}, iter={tim_i}) | mean={tim_m:.6f}, std={tim_s:.6f}")
        print(
            f"    at_iter: accuracy={tim_ctx.get('accuracy', float('nan')):.6f}, energy={tim_ctx.get('energy', float('nan')):.6f}, area={tim_ctx.get('area', float('nan')):.6f}"
        )
        print(f"  area_min     = {area_v:.6f}  (file={area_f}, iter={area_i}) | mean={area_m:.6f}, std={area_s:.6f}")
        print(
            f"    at_iter: accuracy={area_ctx.get('accuracy', float('nan')):.6f}, energy={area_ctx.get('energy', float('nan')):.6f}, timing={area_ctx.get('timing', float('nan')):.6f}"
        )

    # Plot mean HV vs iteration for all methods
    method_series = gather_method_hv_series(
        dataset_dir,
        ref_point,
        constraints,
        skip_seeds=skip_seeds or None,
        skip_methods=skip_methods or None,
        per_method_skip_seeds=per_method_skip_seeds or None,
    )
    method_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for method, series_list in method_series.items():
        try:
            mean, std = mean_std_curve(series_list)
            method_curves[method] = (mean, std)
        except ValueError:
            continue
    base_curve_title = f"{dataset_name} Mean HV vs Iteration (constrained, from iter {CURVE_START_ITER})"
    for label, order in (("NEHVI", NEHVI_METHOD_ORDER), ("EHVI", EHVI_METHOD_ORDER)):
        group_curves = _filter_dict_by_order(method_curves, order)
        if not group_curves:
            print(f"[warn] No methods found for {label} mean curve plot; skipping.")
            continue
        curve_title = f"{base_curve_title} ({label})"
        curve_out_path = os.path.join(plots_root_dir, _filename_with_suffix(curve_out_name, label))
        plot_mean_curves(
            group_curves,
            curve_title,
            curve_out_path,
            method_order=order,
            shade_std=curve_shade_std,
        )
        print(f"Saved: {curve_out_path}")

    # Always generate per-seed HV vs iteration plots (overlaying all methods per seed)
    seed_to_method_series, _all_methods = gather_seed_hv_series(
        dataset_dir,
        ref_point,
        constraints,
        skip_seeds=skip_seeds or None,
        skip_methods=skip_methods or None,
        per_method_skip_seeds=per_method_skip_seeds or None,
    )
    per_seed_dir = os.path.join(plots_root_dir, "seed_curves")
    plot_per_seed_curves(
        seed_to_method_series,
        dataset_name,
        per_seed_dir,
        method_order=NEHVI_METHOD_ORDER,
        filename_suffix="NEHVI",
    )
    plot_per_seed_curves(
        seed_to_method_series,
        dataset_name,
        per_seed_dir,
        method_order=EHVI_METHOD_ORDER,
        filename_suffix="EHVI",
    )


if __name__ == "__main__":
    main()
