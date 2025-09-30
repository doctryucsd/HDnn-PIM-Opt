from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from typing import Dict, Iterable, List, Tuple

# Ensure the repo root is importable so we can reuse helpers from plot.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from exps.constraint_scheduling.plot import (  # type: ignore
    EHVI_METHOD_ORDER,
    NEHVI_METHOD_ORDER,
    _extract_seed_from_filename,
    read_metrics,
)

MetricEntry = Tuple[float, str, int, float, float, Dict[str, float]]
METRIC_ORDER: List[str] = ["accuracy", "energy", "timing", "area"]
PPA_METRICS: List[str] = ["energy", "timing", "area"]
DEFAULT_COMBINED_METHODS: List[str] = ["NEHVI_linear", "EHVI_linear"]
DEFAULT_COMBINATION_LABEL: str = "+".join(DEFAULT_COMBINED_METHODS)

BASELINES: Dict[str, Dict[str, List[float]]] = {
    "fashion_50iter": {
        "default": [0.813, 0.54, 0.1, 0.287],
    },
    "cifar10_50iter": {
        "default": [0.413, 0.727, 0.227, 0.287],
    },
}

CONSTRAINTS: Dict[str, Dict[str, List[float]]] = {
    "fashion_50iter": {
        # [acc_min, energy_max, timing_max, area_max]
        "default": [0.83, 0.1, 0.1, 0.2],
    },
    "cifar10_50iter": {
        "default": [0.40, 0.1, 0.1, 0.2],
    },
}


def _format_percent(value: float) -> str:
    if math.isinf(value):
        return "inf%"
    if math.isnan(value):
        return "nan%"
    return f"{value:+.2f}%"


def _percent_change(new: float, baseline: float) -> float:
    if math.isclose(baseline, 0.0):
        if math.isclose(new, 0.0):
            return 0.0
        return math.inf if new > 0 else -math.inf
    return (new / baseline - 1.0) * 100.0


def _method_iterable(method: str | None, available: Iterable[str]) -> List[str]:
    if method:
        return [method]
    ordered: List[str] = []
    for name in NEHVI_METHOD_ORDER + EHVI_METHOD_ORDER:
        if name in available and name not in ordered:
            ordered.append(name)
    for name in sorted(available):
        if name not in ordered:
            ordered.append(name)
    return ordered


def _resolve_baseline(dataset_dir: str, key: str | None) -> Dict[str, float]:
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    dataset_baselines = BASELINES.get(dataset_name)
    if not dataset_baselines:
        raise KeyError(
            f"No baseline configured for dataset '{dataset_name}'. "
            "Update BASELINES in compare.py to add one."
        )
    baseline_key = key or "default"
    values = dataset_baselines.get(baseline_key)
    if not values:
        raise KeyError(
            f"Baseline key '{baseline_key}' not found for dataset '{dataset_name}'. "
            "Available keys: " + ", ".join(sorted(dataset_baselines.keys()))
        )
    if len(values) != len(METRIC_ORDER):
        raise ValueError(
            f"Baseline for dataset '{dataset_name}' key '{baseline_key}' must have {len(METRIC_ORDER)} entries."
        )
    return dict(zip(METRIC_ORDER, values))


def _resolve_constraints(dataset_dir: str, key: str | None) -> Dict[str, float]:
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    dataset_constraints = CONSTRAINTS.get(dataset_name)
    if not dataset_constraints:
        raise KeyError(
            f"No constraints configured for dataset '{dataset_name}'. "
            "Update CONSTRAINTS in compare.py to add one."
        )
    constraint_key = key or "default"
    values = dataset_constraints.get(constraint_key)
    if not values:
        raise KeyError(
            f"Constraint key '{constraint_key}' not found for dataset '{dataset_name}'. "
            "Available keys: " + ", ".join(sorted(dataset_constraints.keys()))
        )
    if len(values) != len(METRIC_ORDER):
        raise ValueError(
            f"Constraints for dataset '{dataset_name}' key '{constraint_key}' must have {len(METRIC_ORDER)} entries."
        )
    return dict(zip(METRIC_ORDER, values))


def gather_method_best_metrics_feasible(
    dataset_dir: str,
    constraints: Dict[str, float],
    skip_seeds: set[int] | None = None,
    skip_methods: set[str] | None = None,
    per_method_skip_seeds: Dict[str, set[int]] | None = None,
) -> Dict[str, Dict[str, MetricEntry]]:
    result: Dict[str, Dict[str, MetricEntry]] = {}

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

        best_val_file_idx: Dict[str, Tuple[float, str, int]] = {
            "accuracy": (-math.inf, "", -1),
            "energy": (math.inf, "", -1),
            "timing": (math.inf, "", -1),
            "area": (math.inf, "", -1),
        }
        best_context: Dict[str, Dict[str, float]] = {metric: {} for metric in METRIC_ORDER}
        accums: Dict[str, List[float]] = {metric: [] for metric in METRIC_ORDER}

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
            except Exception as exc:
                print(f"[warn] Failed to read metrics for {jf}: {exc}")
                continue

            lengths = [len(metrics.get(metric, [])) for metric in METRIC_ORDER]
            if not lengths or min(lengths) == 0:
                continue
            num_points = min(lengths)

            for idx in range(num_points):
                try:
                    point = {
                        metric: float(metrics[metric][idx]) for metric in METRIC_ORDER
                    }
                except (KeyError, TypeError, ValueError):
                    continue

                feasible = (
                    point["accuracy"] >= constraints["accuracy"]
                    and point["energy"] <= constraints["energy"]
                    and point["timing"] <= constraints["timing"]
                    and point["area"] <= constraints["area"]
                )
                if not feasible:
                    continue

                for metric in METRIC_ORDER:
                    accums[metric].append(point[metric])

                for metric in METRIC_ORDER:
                    current = point[metric]
                    best_val, _, _ = best_val_file_idx[metric]
                    better = False
                    if metric == "accuracy":
                        better = current > best_val
                    else:
                        better = current < best_val
                    if better:
                        best_val_file_idx[metric] = (
                            current,
                            os.path.basename(jf),
                            idx,
                        )
                        best_context[metric] = point.copy()

        enriched: Dict[str, MetricEntry] = {}
        for metric in METRIC_ORDER:
            values = accums[metric]
            if not values:
                continue
            mean = statistics.fmean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            best_val, best_file, best_idx = best_val_file_idx[metric]
            if metric == "accuracy" and best_val == -math.inf:
                continue
            if metric != "accuracy" and best_val == math.inf:
                continue
            enriched[metric] = (
                best_val,
                best_file,
                best_idx,
                mean,
                std,
                best_context.get(metric, {}),
            )

        if enriched:
            result[entry] = enriched

    return result


def _combine_methods(
    per_method: Dict[str, Dict[str, MetricEntry]], method_names: Iterable[str]
) -> Tuple[Dict[str, MetricEntry], Dict[str, str], List[str]]:
    combined: Dict[str, MetricEntry] = {}
    source_map: Dict[str, str] = {}
    present: List[str] = []

    for name in method_names:
        metrics = per_method.get(name)
        if not metrics:
            continue
        present.append(name)
        for metric in METRIC_ORDER:
            entry = metrics.get(metric)
            if not entry:
                continue
            current = combined.get(metric)
            better = False
            if metric == "accuracy":
                better = current is None or entry[0] > current[0]
            else:
                better = current is None or entry[0] < current[0]
            if better:
                best_file = entry[1]
                if best_file and not best_file.startswith(f"{name}/"):
                    best_file = f"{name}/{best_file}"
                combined[metric] = (
                    entry[0],
                    best_file,
                    entry[2],
                    entry[3],
                    entry[4],
                    entry[5],
                )
                source_map[metric] = name

    return combined, source_map, present


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare best observed metrics for constraint scheduling experiments "
            "against preconfigured baselines, considering only constraint-satisfying design points."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Path to a dataset directory, e.g. exps/constraint_scheduling/cifar10_50iter",
    )
    parser.add_argument(
        "--baseline_key",
        default=None,
        help="Optional key for baseline/constraint tables (default: 'default').",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Optional method directory to focus on (e.g. EHVI_linear). If omitted, combine NEHVI_linear and EHVI_linear.",
    )
    parser.add_argument(
        "--ignore_seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of seed integers to ignore across all methods.",
    )
    parser.add_argument(
        "--ignore_methods",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of method directory names to skip entirely.",
    )
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    baseline_values = _resolve_baseline(dataset_dir, args.baseline_key)
    constraint_values = _resolve_constraints(dataset_dir, args.baseline_key)

    per_method = gather_method_best_metrics_feasible(
        dataset_dir=dataset_dir,
        constraints=constraint_values,
        skip_seeds=set(args.ignore_seeds or []),
        skip_methods=set(args.ignore_methods or []),
        per_method_skip_seeds=None,
    )
    if not per_method:
        raise RuntimeError(f"No methods found under {dataset_dir}")

    display_items: List[Tuple[str, Dict[str, MetricEntry], Dict[str, str] | None]] = []

    if args.method:
        requested = _method_iterable(args.method, per_method.keys())
        for method in requested:
            if method not in per_method:
                print(f"[warn] Method '{method}' not present in dataset; skipping.")
                continue
            display_items.append((method, per_method[method], None))
    else:
        combined_metrics, source_map, present = _combine_methods(per_method, DEFAULT_COMBINED_METHODS)
        if combined_metrics:
            missing = [m for m in DEFAULT_COMBINED_METHODS if m not in present]
            if missing:
                print(
                    "[warn] Default combination missing methods: " + ", ".join(missing)
                )
            display_items.append((DEFAULT_COMBINATION_LABEL, combined_metrics, source_map))
        else:
            print(
                "[warn] No data found for default method combination; showing all available methods."
            )
            fallback_methods = _method_iterable(None, per_method.keys())
            for method in fallback_methods:
                display_items.append((method, per_method[method], None))

    if not display_items:
        raise RuntimeError("No methods selected for comparison.")

    print(f"Dataset: {dataset_dir}")
    print(
        "Constraints (accuracy>=, energy<=, timing<=, area<=): "
        + ", ".join(f"{m}={constraint_values[m]:.6f}" for m in METRIC_ORDER)
    )
    print(
        "Baseline (accuracy, energy, timing, area): "
        + ", ".join(f"{m}={baseline_values[m]:.6f}" for m in METRIC_ORDER)
    )
    print()

    for method, metrics, source_map in display_items:
        print(f"Method: {method}")
        for metric in METRIC_ORDER:
            entry: MetricEntry | None = metrics.get(metric)
            if not entry:
                print(f"  {metric:8s}: no data")
                continue
            best_val, best_file, best_idx, mean, std, context = entry

            baseline_metric = baseline_values[metric]
            if metric == "accuracy":
                diff_percent = (best_val - baseline_metric) * 100.0
                print(
                    f"  {metric:8s}: best={best_val:.6f} (file={best_file}, iter={best_idx})"
                    f", baseline={baseline_metric:.6f}, accuracy delta={diff_percent:+.2f}%"
                )
            else:
                pct_delta = _percent_change(best_val, baseline_metric)
                if pct_delta < 0.0:
                    if math.isclose(best_val, 0.0):
                        improvement_factor = math.inf
                    else:
                        improvement_factor = baseline_metric / best_val
                    improvement_str = "inf" if math.isinf(improvement_factor) else f"{improvement_factor:.2f}"
                    print(
                        f"  {metric:8s}: best={best_val:.6f} (file={best_file}, iter={best_idx})"
                        f", baseline={baseline_metric:.6f}, {metric} improvement={improvement_str}x({improvement_str} times)"
                    )
                else:
                    print(
                        f"  {metric:8s}: best={best_val:.6f} (file={best_file}, iter={best_idx})"
                        f", baseline={baseline_metric:.6f}, {metric} ratio delta={_format_percent(pct_delta)}"
                    )

            ctx_parts: List[str] = []
            for name in METRIC_ORDER:
                if name == metric:
                    continue
                if name in context and not math.isnan(context[name]):
                    ctx_parts.append(f"{name}={context[name]:.6f}")
            if ctx_parts:
                print(f"             context: {', '.join(ctx_parts)}")
            if source_map and metric in source_map:
                print(f"             source method: {source_map[metric]}")
        print()

    print(
        "Legend: accuracy delta = (best - baseline) * 100. Negative PPA deltas show improvement factors; positive deltas remain as percentages."
    )


if __name__ == "__main__":
    main()
