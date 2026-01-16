from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import HV

# Defaults lifted from exps/constraint_scheduling/plot.py
DEFAULT_REF_POINT: List[float] = [0.0, 1.0, 1.0, 1.0]
DEFAULT_CONSTRAINTS: List[float] = [0.40, 0.1, 0.1, 0.2]  # [acc_min, energy_max, timing_max, area_max]
DATASET_CONSTRAINTS: Dict[str, List[float]] = {
    "cifar10": [0.40, 0.1, 0.1, 0.2],
    "cifar10_50iter": [0.40, 0.1, 0.1, 0.2],
    "cifar10_100iter": [0.40, 0.1, 0.1, 0.2],
    "fashion": [0.83, 0.1, 0.1, 0.2],
    "fashion_mnist": [0.83, 0.1, 0.1, 0.2],
    "mnist": [0.97, 0.03, 0.03, 0.2],
    "mnist_50iter": [0.97, 0.03, 0.03, 0.2],
    "mnist_100iter": [0.97, 0.03, 0.03, 0.2],
    "ucihar": [0.79, 0.5, 0.5, 0.2],
    "ucihar_50iter": [0.79, 0.5, 0.5, 0.2],
}


def _constraints_for_dataset(dataset_name: str) -> List[float]:
    normalized = dataset_name.lower().replace("-", "_")
    sanitized = re.sub(r"[^a-z0-9]+", "", normalized)

    candidate_keys = {normalized, sanitized}
    parts = normalized.split("_")
    for idx in range(len(parts), 0, -1):
        candidate_keys.add("_".join(parts[:idx]))

    for candidate in candidate_keys:
        if candidate in DATASET_CONSTRAINTS:
            return list(DATASET_CONSTRAINTS[candidate])

    for key, values in DATASET_CONSTRAINTS.items():
        key_norm = key.lower()
        key_sanitized = re.sub(r"[^a-z0-9]+", "", key_norm)
        if normalized.startswith(key_norm) or sanitized.startswith(key_sanitized):
            return list(values)

    return list(DEFAULT_CONSTRAINTS)


def _read_metrics(path: str) -> Dict[str, List[float]]:
    with open(path, "r") as f:
        return json.load(f)


def _transform_points(
    acc: Sequence[float],
    eng: Sequence[float],
    tim: Sequence[float],
    area: Sequence[float],
) -> np.ndarray:
    return np.array([[-a, e, t, ar] for a, e, t, ar in zip(acc, eng, tim, area)], dtype=float)


def _feasible_mask(
    acc: Sequence[float],
    eng: Sequence[float],
    tim: Sequence[float],
    area: Sequence[float],
    constraints: Sequence[float],
) -> np.ndarray:
    acc_thr, eng_thr, tim_thr, area_thr = constraints
    return np.array(
        [
            (a >= acc_thr) and (e <= eng_thr) and (t <= tim_thr) and (ar <= area_thr)
            for a, e, t, ar in zip(acc, eng, tim, area)
        ],
        dtype=bool,
    )


def compute_hv_series(
    metrics: Dict[str, Sequence[float]],
    ref_point: Sequence[float],
    constraints: Sequence[float] | None,
    constrained: bool,
) -> List[float]:
    acc = metrics.get("accuracy", [])
    eng = metrics.get("energy", [])
    tim = metrics.get("timing", [])
    area = metrics.get("area", [])

    N = min(len(acc), len(eng), len(tim), len(area))
    if N == 0:
        return []

    acc = acc[:N]
    eng = eng[:N]
    tim = tim[:N]
    area = area[:N]

    points_all = _transform_points(acc, eng, tim, area)
    feasible = np.ones(N, dtype=bool)
    if constrained:
        if constraints is None or len(constraints) != 4:
            raise ValueError("constraints must provide four values when constrained=True")
        feasible = _feasible_mask(acc, eng, tim, area, constraints)

    hv_indicator = HV(ref_point=np.asarray(ref_point, dtype=float))
    series: List[float] = []
    for end in range(1, N + 1):
        pts = points_all[:end]
        mask = feasible[:end]
        pts = pts[mask]
        series.append(float(hv_indicator(pts)) if pts.size else 0.0)
    return series


def compute_feasible_hv(
    metrics: Dict[str, Sequence[float]],
    ref_point: Sequence[float],
    constraints: Sequence[float],
) -> float:
    acc = metrics.get("accuracy", [])
    eng = metrics.get("energy", [])
    tim = metrics.get("timing", [])
    area = metrics.get("area", [])

    N = min(len(acc), len(eng), len(tim), len(area))
    if N == 0:
        return 0.0

    acc = acc[:N]
    eng = eng[:N]
    tim = tim[:N]
    area = area[:N]

    points = _transform_points(acc, eng, tim, area)
    feasible = _feasible_mask(acc, eng, tim, area, constraints)
    points = points[feasible]
    if points.size == 0:
        return 0.0
    hv_indicator = HV(ref_point=np.asarray(ref_point, dtype=float))
    return float(hv_indicator(points))


def compute_unconstrained_hv(
    metrics: Dict[str, Sequence[float]],
    ref_point: Sequence[float],
) -> float:
    acc = metrics.get("accuracy", [])
    eng = metrics.get("energy", [])
    tim = metrics.get("timing", [])
    area = metrics.get("area", [])

    N = min(len(acc), len(eng), len(tim), len(area))
    if N == 0:
        return 0.0

    acc = acc[:N]
    eng = eng[:N]
    tim = tim[:N]
    area = area[:N]

    points = _transform_points(acc, eng, tim, area)
    hv_indicator = HV(ref_point=np.asarray(ref_point, dtype=float))
    return float(hv_indicator(points))


def _discover_dataset_name(search_dir: str) -> str:
    for method in sorted(os.listdir(search_dir)):
        method_path = os.path.join(search_dir, method)
        if not os.path.isdir(method_path):
            continue
        for entry in sorted(os.listdir(method_path)):
            if entry.endswith(".json"):
                base = os.path.splitext(entry)[0]
                return base.split("_", 1)[0] or "unknown"
    return "unknown"


def _load_method_series(
    search_dir: str,
    ref_point: Sequence[float],
    constraints: Sequence[float] | None,
    constrained: bool,
) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[str]]]:
    method_to_series: Dict[str, List[List[float]]] = {}
    method_to_files: Dict[str, List[str]] = {}

    for method in sorted(os.listdir(search_dir)):
        method_path = os.path.join(search_dir, method)
        if not os.path.isdir(method_path):
            continue
        json_files = [
            os.path.join(method_path, f)
            for f in sorted(os.listdir(method_path))
            if f.endswith(".json") and os.path.isfile(os.path.join(method_path, f))
        ]
        if not json_files:
            continue

        series_list: List[List[float]] = []
        for jf in json_files:
            metrics = _read_metrics(jf)
            series = compute_hv_series(metrics, ref_point, constraints, constrained=constrained)
            if series:
                series_list.append(series)

        if series_list:
            method_to_series[method] = series_list
            method_to_files[method] = json_files

    return method_to_series, method_to_files


def _mean_series(series_list: List[List[float]], start_iter: int) -> np.ndarray:
    trimmed = [s[start_iter:] for s in series_list if len(s) > start_iter]
    if not trimmed:
        return np.array([], dtype=float)
    min_len = min(len(s) for s in trimmed)
    if min_len <= 0:
        return np.array([], dtype=float)
    arr = np.array([s[:min_len] for s in trimmed], dtype=float)
    return np.mean(arr, axis=0)


def _mean_final_value(series_list: List[List[float]]) -> float:
    finals = [seq[-1] for seq in series_list if seq]
    if not finals:
        return 0.0
    return float(np.mean(np.array(finals, dtype=float)))


def _plot_mean_curves(
    mean_by_method: Dict[str, np.ndarray],
    exhaustive_hv: float,
    output_path: str,
    title: str,
    exhaustive_label: str,
    start_iter: int,
) -> None:
    plt.figure(figsize=(10, 6))
    for method, series in mean_by_method.items():
        if series.size == 0:
            continue
        x = np.arange(start_iter, start_iter + len(series))
        plt.plot(x, series, linewidth=2.5, label=method)

    plt.axhline(exhaustive_hv, color="black", linestyle="--", linewidth=2.0, label=exhaustive_label)
    plt.xlabel("Iteration")
    plt.ylabel("Feasible hypervolume")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot feasible/unconstrained HV vs iteration for optimality runs.")
    parser.add_argument(
        "--start_iter",
        type=int,
        default=10,
        help="Iteration index (0-based) to start plotting from.",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.dirname(__file__))
    exhaustive_path = os.path.join(base_dir, "exhaustive.json")
    search_dir = os.path.join(base_dir, "search")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    dataset_name = _discover_dataset_name(search_dir)
    constraints = _constraints_for_dataset(dataset_name)
    ref_point = list(DEFAULT_REF_POINT)

    exhaustive_metrics = _read_metrics(exhaustive_path)
    exhaustive_hv = compute_feasible_hv(exhaustive_metrics, ref_point, constraints)
    exhaustive_hv_unconstrained = compute_unconstrained_hv(exhaustive_metrics, ref_point)

    method_series, method_files = _load_method_series(search_dir, ref_point, constraints, constrained=True)
    mean_by_method: Dict[str, np.ndarray] = {}
    mean_final_by_method: Dict[str, float] = {}
    for method, series_list in method_series.items():
        mean_by_method[method] = _mean_series(series_list, args.start_iter)
        mean_final_by_method[method] = _mean_final_value(series_list)

    title = f"{dataset_name} Feasible HV vs Iteration (from iter {args.start_iter})"
    output_path = os.path.join(plots_dir, "optimality_feasible_hv.pdf")
    _plot_mean_curves(
        mean_by_method,
        exhaustive_hv,
        output_path,
        title,
        exhaustive_label="exhaustive feasible HV",
        start_iter=args.start_iter,
    )

    method_series_unconstrained, _ = _load_method_series(search_dir, ref_point, None, constrained=False)
    mean_by_method_unconstrained: Dict[str, np.ndarray] = {}
    for method, series_list in method_series_unconstrained.items():
        mean_by_method_unconstrained[method] = _mean_series(series_list, args.start_iter)

    title_unconstrained = f"{dataset_name} HV vs Iteration (from iter {args.start_iter})"
    output_unconstrained = os.path.join(plots_dir, "optimality_hv_unconstrained.pdf")
    _plot_mean_curves(
        mean_by_method_unconstrained,
        exhaustive_hv_unconstrained,
        output_unconstrained,
        title_unconstrained,
        exhaustive_label="exhaustive HV",
        start_iter=args.start_iter,
    )

    print(f"Dataset: {dataset_name}")
    print(f"Constraints [acc_min, energy_max, timing_max, area_max]: {constraints}")
    print(f"Ref point: {ref_point}")
    print(f"Optimality (exhaustive feasible HV): {exhaustive_hv:.6f}")
    for method, files in method_files.items():
        series = mean_by_method.get(method, np.array([]))
        final_mean = mean_final_by_method.get(method, 0.0)
        print(f"{method}: seeds={len(files)}, mean_len={len(series)}, avg_final={final_mean:.6f}")
    print(f"Optimality (exhaustive HV): {exhaustive_hv_unconstrained:.6f}")
    print(f"Saved plot: {output_path}")
    print(f"Saved plot: {output_unconstrained}")


if __name__ == "__main__":
    main()
