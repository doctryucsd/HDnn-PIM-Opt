#!/usr/bin/env python3
"""Generate Pareto-front figures that highlight accuracy feasibility."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 16,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

ACC_THRESHOLD_DEFAULT = 0.90
MINIMIZE_METRICS = ("power", "latency", "area")
METRIC_ALIASES = {
    "energy": "power",
    "energy_w": "power",
    "energy(j)": "power",
    "energy_j": "power",
    "energy_joules": "power",
    "timing": "latency",
    "delay": "latency",
    "runtime": "latency",
    "latency_ns": "latency",
    "latency_ms": "latency",
}
MAX_SCATTER_POINTS = 50_000
INFEASIBLE_COLOR = "#ff0000"  # vivid red
FEASIBLE_COLOR = "#0a6026"  # richer green
INFEASIBLE_MARKER = "^"
FEASIBLE_MARKER = "o"
PARETO_ALL_COLOR = "#888888"
PARETO_FEASIBLE_COLOR = "#006400"
SCATTER_INFEASIBLE_SIZE = 200
SCATTER_FEASIBLE_SIZE = 200
SIZE_MARKER_BASE = 40
SIZE_MARKER_SPAN = 180
RNG = random.Random(1337)


@dataclass
class PlotConfig:
    x_metric: str
    y_metric: str
    size_metric: Optional[str]
    exclude_metric: str
    acc_threshold: float
    acc_threshold_label: str
    suffix: str
    dpi: int
    input_root: Optional[Path]
    output_root: Path
    plot_2d: bool
    show_threshold_plane: bool
    axes_3d: Tuple[str, str, str]
    acc_axis: str
    percentile_low: float
    percentile_high: float
    axis_padding: float
    equal_aspect: bool
    elev: Optional[float]
    azim: Optional[float]
    xlog: bool
    ylog: bool
    show_outliers: bool
    normalize_mode: str
    tick_original: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Pareto-front plots that distinguish feasible and "
            "infeasible points based on an accuracy threshold."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSON file or a directory containing JSON files.",
    )
    parser.add_argument(
        "--exclude-metric",
        default="area",
        choices=MINIMIZE_METRICS,
        help=(
            "Metric to exclude from the minimized objectives "
            "(accuracy is always included)."
        ),
    )
    parser.add_argument(
        "--x",
        default=None,
        help="Metric for the x-axis (from the remaining minimized metrics).",
    )
    parser.add_argument(
        "--y",
        default=None,
        help="Metric for the y-axis (from the remaining minimized metrics).",
    )
    parser.add_argument(
        "--size-metric",
        default=None,
        help=(
            "Optional metric to encode via marker size "
            "(third minimized metric)."
        ),
    )
    parser.add_argument(
        "--suffix",
        default="",
        help=(
            "Additional suffix to append to output filenames "
            "(e.g., _acc0p90)."
        ),
    )
    parser.add_argument(
        "--acc-threshold",
        default=None,
        help="Accuracy feasibility threshold (default: 0.90).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure resolution in DPI (default: 200).",
    )
    parser.add_argument(
        "--plot-2d",
        action="store_true",
        help="Render the Pareto trade-off in 2D (legacy mode).",
    )
    parser.add_argument(
        "--no-separator-plane",
        action="store_true",
        help="Disable the accuracy threshold plane in 3D plots.",
    )
    parser.add_argument(
        "--acc-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Axis on which to plot accuracy in 3D mode (default: z).",
    )
    parser.add_argument(
        "--percentile-clip",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(1.0, 99.0),
        help="Robust percentile bounds (default: 1 99).",
    )
    parser.add_argument(
        "--axis-padding",
        type=float,
        default=0.08,
        help="Fractional padding to apply to each axis range (default: 0.08).",
    )
    parser.add_argument(
        "--normalize",
        choices=("minmax", "none"),
        default="minmax",
        help="Axis normalization strategy (default: minmax).",
    )
    parser.add_argument(
        "--equal-aspect",
        action="store_true",
        help="Force cubic aspect ratio in 3D mode.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=None,
        help="Elevation angle for the 3D camera (degrees).",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=None,
        help="Azimuth angle for the 3D camera (degrees).",
    )
    parser.add_argument(
        "--xlog",
        action="store_true",
        help="Plot the X-axis (power) on a log10 scale when available.",
    )
    parser.add_argument(
        "--ylog",
        action="store_true",
        help="Plot the Y-axis (latency) on a log10 scale when available.",
    )
    parser.add_argument(
        "--show-outliers",
        action="store_true",
        help="Draw points outside the robust percentile bounds instead of clipping them.",
    )
    parser.add_argument(
        "--tick-original",
        action="store_true",
        help="Format ticks using original units when normalization is applied.",
    )
    return parser

def canonical_metric_name(name: str) -> str:
    base = name.strip().lower()
    return METRIC_ALIASES.get(base, base)


def normalize_record_keys(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(key, str):
            key_lower = key.strip().lower()
        else:
            key_lower = str(key).lower()
        canonical = canonical_metric_name(key_lower)
        if canonical in normalized and canonical != key_lower:
            continue
        normalized[canonical] = value
    return normalized


def convert_data_to_records(data: Any) -> List[Dict[str, Any]]:
    """Convert loaded JSON data into a list of per-design dictionaries."""
    if isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            return [normalize_record_keys(item) for item in data]
        raise ValueError("Expected list of dicts in JSON data.")

    if isinstance(data, dict):
        lowered = {str(k).lower(): v for k, v in data.items()}
        for field in ("results", "designs", "records"):
            if field in lowered and isinstance(lowered[field], list):
                return convert_data_to_records(lowered[field])

        sequence_keys = [
            key
            for key, value in data.items()
            if isinstance(value, (list, tuple))
        ]
        if sequence_keys:
            max_length = max(len(data[key]) for key in sequence_keys)
            records: List[Dict[str, Any]] = []
            for idx in range(max_length):
                record: Dict[str, Any] = {}
                for key, value in data.items():
                    if isinstance(value, (list, tuple)):
                        if idx < len(value):
                            record[str(key).lower()] = value[idx]
                    else:
                        record[str(key).lower()] = value
                records.append(record)
            return records

    raise ValueError("Unsupported JSON structure for Pareto plotting.")


def fallback_load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    data = fallback_load_json(path)
    try:
        records = convert_data_to_records(data)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return records


def maybe_rescale_accuracy(
    points: List[Dict[str, Any]],
    origin: Path,
) -> float:
    """Detect percentage-based accuracy and rescale to [0, 1]."""
    accuracies: List[float] = []
    for record in points:
        value = record.get("accuracy")
        if value is None:
            continue
        try:
            accuracies.append(float(value))
        except (TypeError, ValueError):
            continue
    if not accuracies:
        return 1.0
    max_val = max(accuracies)
    if 1.5 < max_val <= 100.0:
        print(
            "[info] Accuracy appears to be in percent for "
            f"{origin}; scaling by 1/100.",
        )
        for record in points:
            if "accuracy" in record:
                try:
                    record["accuracy"] = float(record["accuracy"]) / 100.0
                except (TypeError, ValueError):
                    record["accuracy"] = math.nan
        return 100.0
    return 1.0


def sanitize_records(
    records: Iterable[Dict[str, Any]],
    required_metrics: Sequence[str],
    origin: Path,
) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    skipped = 0
    for index, record in enumerate(records):
        normalized = normalize_record_keys(record)
        candidate = dict(normalized)
        missing = False
        for metric in required_metrics:
            if metric not in normalized:
                missing = True
                break
            value = normalized[metric]
            try:
                candidate[metric] = float(value)
                if not math.isfinite(candidate[metric]):
                    missing = True
                    break
            except (TypeError, ValueError):
                missing = True
                break
        if missing:
            skipped += 1
            continue
        sanitized.append(candidate)
    if skipped:
        print(
            f"[warn] {origin}: skipped {skipped} record(s) missing "
            "required metrics."
        )
    return sanitized


def extract_units(records: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    units: Dict[str, str] = {}
    for record in records:
        units_blob = record.get("units") or record.get("unit")
        if isinstance(units_blob, dict):
            for metric, unit in units_blob.items():
                key = canonical_metric_name(str(metric).lower())
                if key not in units and isinstance(unit, str):
                    units[key] = unit
        for metric in ("accuracy",) + MINIMIZE_METRICS:
            for suffix in ("_unit", "_units"):
                key = f"{metric}{suffix}"
                if key in record and isinstance(record[key], str):
                    units.setdefault(metric, record[key])
                    continue
                for alias, canonical in METRIC_ALIASES.items():
                    if canonical != metric:
                        continue
                    alias_key = f"{alias}{suffix}"
                    if alias_key in record and isinstance(record[alias_key], str):
                        units.setdefault(metric, record[alias_key])
    return units


def format_metric_label(metric: str, unit: Optional[str]) -> str:
    pretty_names = {"power": "Energy"}
    title_case = pretty_names.get(metric, metric.replace("_", " ").title())
    if unit:
        return f"{title_case} ({unit})"
    return title_case


def format_threshold_label(value: float) -> str:
    decimal_value = Decimal(str(value))
    try:
        normalized = decimal_value.normalize()
    except InvalidOperation:
        return str(value)
    text = format(normalized, "f")
    if "." not in text:
        text = f"{text}.0"
    return text


def threshold_tag_from_label(label: str) -> str:
    sanitized = label.strip().replace(" ", "")
    if not sanitized:
        return "0"
    sanitized = sanitized.replace(".", "p")
    sanitized = sanitized.replace("-", "neg")
    sanitized = sanitized.replace("+", "")
    return sanitized


def pareto_front_2d(
    points: Sequence[Dict[str, float]],
    x: str,
    y: str,
) -> List[Dict[str, float]]:
    if not points:
        return []
    sorted_points = sorted(points, key=lambda rec: (rec[x], rec[y]))
    front: List[Dict[str, float]] = []
    best_y = float("inf")
    eps = 1e-12
    for record in sorted_points:
        y_val = record[y]
        if y_val < best_y - eps:
            front.append(record)
            best_y = y_val
    return front


def compute_marker_sizes(
    points: List[Dict[str, float]],
    metric: str,
) -> Optional[List[float]]:
    if metric is None:
        return None
    values = [record.get(metric) for record in points]
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    v_min = min(numeric)
    v_max = max(numeric)
    if not math.isfinite(v_min) or not math.isfinite(v_max):
        return None
    if math.isclose(v_min, v_max):
        return [SIZE_MARKER_BASE for _ in points]
    span = v_max - v_min
    sizes: List[float] = []
    for record in points:
        value = record.get(metric)
        if value is None:
            sizes.append(SIZE_MARKER_BASE)
            continue
        try:
            normalized = (float(value) - v_min) / span
        except (TypeError, ValueError):
            normalized = 0.0
        size = SIZE_MARKER_BASE + normalized * SIZE_MARKER_SPAN
        sizes.append(size)
    return sizes


def downsample_points(
    points: List[Tuple[Dict[str, float], Tuple[float, ...]]],
    sizes: Optional[List[float]],
) -> Tuple[List[Tuple[Dict[str, float], Tuple[float, ...]]], Optional[List[float]]]:
    if len(points) <= MAX_SCATTER_POINTS:
        return points, sizes
    indices = sorted(RNG.sample(range(len(points)), MAX_SCATTER_POINTS))
    sampled_points = [points[idx] for idx in indices]
    sampled_sizes = (
        [sizes[idx] for idx in indices] if sizes is not None else None
    )
    print(
        "[info] Downsampled scatter from "
        f"{len(points)} to {len(sampled_points)} points."
    )
    return sampled_points, sampled_sizes


def compute_axis_limits(
    values: Sequence[float],
    pct_low: float,
    pct_high: float,
    padding: float,
) -> Tuple[float, float]:
    if not values:
        return -0.5, 0.5
    low = float(np.percentile(values, pct_low))
    high = float(np.percentile(values, pct_high))
    if not math.isfinite(low) or not math.isfinite(high):
        low, high = min(values), max(values)
    if math.isclose(low, high, rel_tol=1e-6, abs_tol=1e-9):
        center = (low + high) / 2.0
        span = max(abs(center), 1.0)
        low = center - span * 0.5
        high = center + span * 0.5
    span = max(high - low, 1e-9)
    pad = max(span * padding, 1e-6)
    return low - pad, high + pad


def ensure_accuracy_bounds(
    bounds: List[Tuple[float, float]],
    acc_index: int,
    threshold: float,
    padding: float,
) -> None:
    low, high = bounds[acc_index]
    span = max(high - low, 1e-9)
    margin = max(span * padding, 0.01)
    if threshold <= low + margin:
        low = threshold - margin
    if threshold >= high - margin:
        high = threshold + margin
    if high <= low:
        center = threshold
        half = max(abs(center) * 0.05, 0.01)
        low = center - half
        high = center + half
    bounds[acc_index] = (low, high)


def clamp_aspect(axes_ranges: Sequence[float], equal_aspect: bool) -> Tuple[float, float, float]:
    if equal_aspect:
        return (1.0, 1.0, 1.0)
    ranges = [max(r, 1e-6) for r in axes_ranges]
    median = np.median(ranges)
    if median <= 0:
        return (1.0, 1.0, 1.0)
    clamped = []
    for r in ranges:
        ratio = r / median
        ratio = min(max(ratio, 0.6), 1.6)
        clamped.append(ratio)
    return tuple(clamped)  # type: ignore[return-value]


def within_bounds(
    coords: Tuple[float, ...],
    bounds: Sequence[Tuple[float, float]],
) -> bool:
    for value, (low, high) in zip(coords, bounds):
        if value < low or value > high:
            return False
    return True


def build_axis_labels(
    metrics: Sequence[str],
    units: Dict[str, str],
    log_flags: Sequence[bool],
) -> List[str]:
    labels: List[str] = []
    for metric, logged in zip(metrics, log_flags):
        label = format_metric_label(metric, units.get(metric))
        if logged:
            label = f"{label} (log10)"
        labels.append(label)
    return labels


def transform_records(
    records: Iterable[Dict[str, float]],
    metrics: Sequence[str],
    log_flags: Sequence[bool],
    axis_values: Optional[List[List[float]]] = None,
) -> Tuple[List[Tuple[Dict[str, float], Tuple[float, ...]]], int]:
    transformed: List[Tuple[Dict[str, float], Tuple[float, ...]]] = []
    dropped = 0
    for record in records:
        coords: List[float] = []
        valid = True
        for metric, logged in zip(metrics, log_flags):
            value = float(record[metric])
            if logged:
                if value <= 0.0 or not math.isfinite(value):
                    valid = False
                    break
                value = math.log10(value)
            coords.append(value)
        if not valid:
            dropped += 1
            continue
        coord_tuple = tuple(coords)
        transformed.append((record, coord_tuple))
        if axis_values is not None:
            for idx, val in enumerate(coord_tuple):
                axis_values[idx].append(val)
    return transformed, dropped


def convert_raw_to_original(value: float, log_flag: bool) -> float:
    return 10 ** value if log_flag else value


def convert_bounds_to_original(
    bounds: Sequence[Tuple[float, float]],
    log_flags: Sequence[bool],
) -> List[Tuple[float, float]]:
    original: List[Tuple[float, float]] = []
    for (low, high), log_flag in zip(bounds, log_flags):
        original.append(
            (
                convert_raw_to_original(low, log_flag),
                convert_raw_to_original(high, log_flag),
            )
        )
    return original


def format_value_for_note(metric: str, unit: Optional[str], value: float) -> str:
    if metric == "accuracy":
        return f"{value * 100:.1f}%"
    unit_str = f" {unit}" if unit else ""
    if value == 0:
        formatted = "0"
    elif 0.01 <= abs(value) < 1000:
        formatted = f"{value:.3g}"
    else:
        formatted = f"{value:.2e}"
    return f"{formatted}{unit_str}"


def build_note_line(
    metric: str,
    unit: Optional[str],
    low: float,
    high: float,
) -> str:
    label = metric.replace("_", " ").title()
    return (
        f"{label}: {format_value_for_note(metric, unit, low)}"
        f" – {format_value_for_note(metric, unit, high)}"
    )


def normalize_coord(
    coords: Tuple[float, ...],
    bounds: Sequence[Tuple[float, float]],
    clamp: bool,
) -> Tuple[Tuple[float, ...], bool]:
    normalized: List[float] = []
    for idx, value in enumerate(coords):
        low, high = bounds[idx]
        span = max(high - low, 1e-9)
        norm_val = (value - low) / span
        outside = value < low or value > high
        if outside and not clamp:
            return tuple(), False
        norm_val = min(1.0, max(0.0, norm_val))
        normalized.append(norm_val)
    return tuple(normalized), True


def make_normalized_formatter(
    metric: str,
    unit: Optional[str],
    log_flag: bool,
    raw_low: float,
    raw_high: float,
) -> FuncFormatter:
    span = max(raw_high - raw_low, 1e-9)

    def _fmt(value: float, _pos: int) -> str:
        raw_value = raw_low + value * span
        original = convert_raw_to_original(raw_value, log_flag)
        if metric == "accuracy":
            return f"{original * 100:.0f}%"
        unit_str = f" {unit}" if unit else ""
        if original == 0:
            formatted = "0"
        elif 0.01 <= abs(original) < 1000:
            formatted = f"{original:.3g}"
        else:
            formatted = f"{original:.2e}"
        return f"{formatted}{unit_str}"

    return FuncFormatter(_fmt)


def filter_points_within_bounds(
    points: List[Tuple[Dict[str, float], Tuple[float, ...]]],
    bounds: Sequence[Tuple[float, float]],
    show_outliers: bool,
) -> List[Tuple[Dict[str, float], Tuple[float, ...]]]:
    if show_outliers:
        return list(points)
    filtered: List[Tuple[Dict[str, float], Tuple[float, ...]]] = []
    for record, coords in points:
        if within_bounds(coords, bounds):
            filtered.append((record, coords))
    return filtered


def filter_coords_within_bounds(
    coords_list: List[Tuple[float, ...]],
    bounds: Sequence[Tuple[float, float]],
    show_outliers: bool,
) -> List[Tuple[float, ...]]:
    if show_outliers:
        return list(coords_list)
    return [coords for coords in coords_list if within_bounds(coords, bounds)]


def minimize_flags_for_metrics(metrics: Sequence[str]) -> List[bool]:
    flags: List[bool] = []
    for metric in metrics:
        flags.append(metric not in ("accuracy",))
    return flags


def dominates(
    a: Tuple[float, ...],
    b: Tuple[float, ...],
    minimize_flags: Sequence[bool],
) -> bool:
    better_or_equal = True
    strictly_better = False
    for val_a, val_b, minimize in zip(a, b, minimize_flags):
        if minimize:
            if val_a > val_b + 1e-12:
                better_or_equal = False
                break
            if val_a + 1e-12 < val_b:
                strictly_better = True
        else:
            if val_a < val_b - 1e-12:
                better_or_equal = False
                break
            if val_a > val_b + 1e-12:
                strictly_better = True
    return better_or_equal and strictly_better


def retain_non_dominated(
    feasible_points: List[Tuple[Dict[str, float], Tuple[float, ...]]],
    infeasible_points: List[Tuple[Dict[str, float], Tuple[float, ...]]],
    metrics: Sequence[str],
) -> Tuple[List[Tuple[Dict[str, float], Tuple[float, ...]]], List[Tuple[Dict[str, float], Tuple[float, ...]]]]:
    minimize_flags = minimize_flags_for_metrics(metrics)
    combined: List[Tuple[str, Dict[str, float], Tuple[float, ...]]] = []
    for record, coords in feasible_points:
        combined.append(("feasible", record, coords))
    for record, coords in infeasible_points:
        combined.append(("infeasible", record, coords))

    keep_flags = [True] * len(combined)
    for i, (_, _, coords_i) in enumerate(combined):
        if not keep_flags[i]:
            continue
        for j, (_, _, coords_j) in enumerate(combined):
            if i == j:
                continue
            if dominates(coords_j, coords_i, minimize_flags):
                keep_flags[i] = False
                break

    filtered_feasible: List[Tuple[Dict[str, float], Tuple[float, ...]]] = []
    filtered_infeasible: List[Tuple[Dict[str, float], Tuple[float, ...]]] = []
    for keep, (kind, record, coords) in zip(keep_flags, combined):
        if not keep:
            continue
        if kind == "feasible":
            filtered_feasible.append((record, coords))
        else:
            filtered_infeasible.append((record, coords))
    return filtered_feasible, filtered_infeasible
def draw_threshold_plane(
    ax: Axes3D,
    axis_index: int,
    threshold: float,
    ranges: Sequence[Tuple[float, float]],
) -> None:
    other_indices = [idx for idx in range(3) if idx != axis_index]
    grid_u = np.linspace(
        ranges[other_indices[0]][0],
        ranges[other_indices[0]][1],
        num=20,
    )
    grid_v = np.linspace(
        ranges[other_indices[1]][0],
        ranges[other_indices[1]][1],
        num=20,
    )
    grid_u, grid_v = np.meshgrid(grid_u, grid_v)
    surfaces: List[np.ndarray] = [np.zeros_like(grid_u) for _ in range(3)]
    surfaces[axis_index] = np.full_like(grid_u, threshold)
    surfaces[other_indices[0]] = grid_u
    surfaces[other_indices[1]] = grid_v
    ax.plot_surface(
        surfaces[0],
        surfaces[1],
        surfaces[2],
        color="#ffd400",
        alpha=0.25,
        linewidth=0,
        antialiased=False,
        shade=False,
    )


def plot_pareto_2d(
    feasible_points: List[Tuple[Dict[str, float], Tuple[float, float]]],
    infeasible_points: List[Tuple[Dict[str, float], Tuple[float, float]]],
    pareto_all_points: List[Tuple[float, float]],
    pareto_feasible_points: List[Tuple[float, float]],
    size_feasible: Optional[List[float]],
    size_infeasible: Optional[List[float]],
    metrics: Sequence[str],
    axis_labels: Sequence[str],
    axis_limits: Sequence[Tuple[float, float]],
    raw_bounds: Sequence[Tuple[float, float]],
    log_flags: Sequence[bool],
    units: Dict[str, str],
    tick_original: bool,
    note_lines: Sequence[str],
    normalized: bool,
    force_equal_aspect: bool,
    title_path: Path,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    feasible_points, size_feasible = downsample_points(feasible_points, size_feasible)
    infeasible_points, size_infeasible = downsample_points(
        infeasible_points, size_infeasible
    )

    legend_handles: List[Any] = []
    legend_labels: List[str] = []

    if feasible_points:
        xs = [coords[0] for _, coords in feasible_points]
        ys = [coords[1] for _, coords in feasible_points]
        scatter_feasible = ax.scatter(
            xs,
            ys,
            label="Feasible points",
            c=FEASIBLE_COLOR,
            alpha=0.85,
            marker=FEASIBLE_MARKER,
            edgecolors="white",
            linewidths=0.4,
            s=size_feasible or SCATTER_FEASIBLE_SIZE,
        )
        legend_handles.append(scatter_feasible)
        legend_labels.append("Feasible points")

    if infeasible_points:
        xs = [coords[0] for _, coords in infeasible_points]
        ys = [coords[1] for _, coords in infeasible_points]
        scatter_infeasible = ax.scatter(
            xs,
            ys,
            label="Infeasible points",
            c=INFEASIBLE_COLOR,
            alpha=0.5,
            marker=INFEASIBLE_MARKER,
            edgecolors="none",
            s=size_infeasible or SCATTER_INFEASIBLE_SIZE,
        )
        legend_handles.append(scatter_infeasible)
        legend_labels.append("Infeasible points")

    if pareto_all_points:
        pareto_all_sorted = sorted(pareto_all_points, key=lambda pt: pt[0])
        xs = [pt[0] for pt in pareto_all_sorted]
        ys = [pt[1] for pt in pareto_all_sorted]
        line_all, = ax.plot(
            xs,
            ys,
            label="Unconstrained Pareto",
            color=PARETO_ALL_COLOR,
            linewidth=1.2,
            marker="o",
            markersize=4,
            markerfacecolor="white",
            alpha=0.35,
        )
        legend_handles.append(line_all)
        legend_labels.append("Unconstrained Pareto")

    if pareto_feasible_points:
        pareto_feasible_sorted = sorted(pareto_feasible_points, key=lambda pt: pt[0])
        xs = [pt[0] for pt in pareto_feasible_sorted]
        ys = [pt[1] for pt in pareto_feasible_sorted]
        line_feasible, = ax.plot(
            xs,
            ys,
            label="Feasible Pareto",
            color=PARETO_FEASIBLE_COLOR,
            linewidth=2.0,
            marker="o",
            markersize=5.5,
        )
        legend_handles.append(line_feasible)
        legend_labels.append("Feasible Pareto")

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.grid(True, linestyle=":", alpha=0.4)

    ax.set_xlim(axis_limits[0])
    ax.set_ylim(axis_limits[1])

    if normalized or force_equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

    if normalized and tick_original:
        ax.xaxis.set_major_formatter(
            make_normalized_formatter(
                metrics[0],
                units.get(metrics[0]),
                log_flags[0],
                raw_bounds[0][0],
                raw_bounds[0][1],
            )
        )
        ax.yaxis.set_major_formatter(
            make_normalized_formatter(
                metrics[1],
                units.get(metrics[1]),
                log_flags[1],
                raw_bounds[1][0],
                raw_bounds[1][1],
            )
        )
    elif normalized:
        formatter = FuncFormatter(lambda value, _pos: f"{value:.2f}")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    if note_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(note_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            frameon=True,
            ncol=2,
        )

    fig.subplots_adjust(left=0.12, right=0.92, top=0.9, bottom=0.12)

    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf, dpi=dpi)
    plt.close(fig)
    message = (
        f"[info] Saved figure(s) for {title_path} -> "
        f"{out_png.name}, {out_pdf.name}"
    )
    print(message)


def plot_pareto_3d(
    feasible_points: List[Tuple[Dict[str, float], Tuple[float, float, float]]],
    infeasible_points: List[Tuple[Dict[str, float], Tuple[float, float, float]]],
    size_feasible: Optional[List[float]],
    size_infeasible: Optional[List[float]],
    metrics: Sequence[str],
    axis_labels: Sequence[str],
    axis_limits: Sequence[Tuple[float, float]],
    raw_bounds: Sequence[Tuple[float, float]],
    log_flags: Sequence[bool],
    units: Dict[str, str],
    tick_original: bool,
    note_lines: Sequence[str],
    threshold_value: float,
    show_plane: bool,
    acc_index: int,
    aspect: Tuple[float, float, float],
    elev: float,
    azim: float,
    title_path: Path,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
    normalized: bool,
) -> None:
    fig = plt.figure(figsize=(7.4, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    feasible_points, size_feasible = downsample_points(feasible_points, size_feasible)
    infeasible_points, size_infeasible = downsample_points(
        infeasible_points, size_infeasible
    )

    legend_handles: List[Any] = []
    legend_labels: List[str] = []

    if infeasible_points:
        xs = [coords[0] for _, coords in infeasible_points]
        ys = [coords[1] for _, coords in infeasible_points]
        zs = [coords[2] for _, coords in infeasible_points]
        scatter_infeasible = ax.scatter(
            xs,
            ys,
            zs,
            c=INFEASIBLE_COLOR,
            alpha=0.35,
            marker=INFEASIBLE_MARKER,
            s=size_infeasible or SCATTER_INFEASIBLE_SIZE,
            depthshade=False,
            edgecolors="none",
            label="Infeasible points",
        )
        legend_handles.append(scatter_infeasible)
        legend_labels.append("Infeasible points")

    if feasible_points:
        xs = [coords[0] for _, coords in feasible_points]
        ys = [coords[1] for _, coords in feasible_points]
        zs = [coords[2] for _, coords in feasible_points]
        scatter_feasible = ax.scatter(
            xs,
            ys,
            zs,
            c=FEASIBLE_COLOR,
            alpha=0.85,
            marker=FEASIBLE_MARKER,
            s=size_feasible or SCATTER_FEASIBLE_SIZE,
            depthshade=False,
            edgecolors="white",
            linewidths=0.4,
            label="Feasible points",
        )
        legend_handles.append(scatter_feasible)
        legend_labels.append("Feasible points")

    ax.set_xlim(axis_limits[0])
    ax.set_ylim(axis_limits[1])
    ax.set_zlim(axis_limits[2])

    try:
        ax.set_box_aspect(aspect)
    except AttributeError:  # pragma: no cover - compatibility guard
        pass

    if show_plane:
        draw_threshold_plane(ax, acc_index, threshold_value, axis_limits)
        legend_handles.append(
            Patch(facecolor="#ffd400", alpha=0.25, label="Accuracy threshold")
        )
        legend_labels.append("Accuracy threshold")

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.view_init(elev=elev, azim=azim)

    locators = [MaxNLocator(nbins=5, prune="both") for _ in range(3)]
    ax.xaxis.set_major_locator(locators[0])
    ax.yaxis.set_major_locator(locators[1])
    ax.zaxis.set_major_locator(locators[2])

    axes_seq = [ax.xaxis, ax.yaxis, ax.zaxis]

    if normalized and tick_original:
        for idx, axis in enumerate(axes_seq):
            axis.set_major_formatter(
                make_normalized_formatter(
                    metrics[idx],
                    units.get(metrics[idx]),
                    log_flags[idx],
                    raw_bounds[idx][0],
                    raw_bounds[idx][1],
                )
            )
    elif normalized:
        formatter = FuncFormatter(lambda value, _pos: f"{value:.2f}")
        for axis in axes_seq:
            axis.set_major_formatter(formatter)
    else:
        for idx, axis in enumerate(axes_seq):
            metric = metrics[idx]
            if metric == "accuracy":
                axis.set_major_formatter(FuncFormatter(lambda val, _: f"{val * 100:.0f}%"))
            else:
                formatter = ScalarFormatter(useOffset=False)
                formatter.set_powerlimits((-3, 4))
                axis.set_major_formatter(formatter)

    if note_lines:
        ax.text2D(
            0.02,
            0.98,
            "\n".join(note_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.2),
            frameon=True,
            borderaxespad=0.4,
        )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf, dpi=dpi)
    plt.close(fig)
    message = (
        f"[info] Saved figure(s) for {title_path} -> "
        f"{out_png.name}, {out_pdf.name}"
    )
    print(message)


def process_file(json_path: Path, config: PlotConfig) -> bool:
    try:
        raw_records = load_json_records(json_path)
    except Exception as exc:
        message = f"[error] Failed to load {json_path}: {exc}"
        print(message)
        return False

    required_metrics = {"accuracy"}
    if config.plot_2d:
        required_metrics.update({config.x_metric, config.y_metric})
    else:
        required_metrics.update(config.axes_3d)
        required_metrics.update({config.x_metric, config.y_metric})
    if config.size_metric:
        required_metrics.add(config.size_metric)

    sanitized = sanitize_records(
        raw_records, sorted(required_metrics), json_path
    )
    if not sanitized:
        print(
            f"[warn] {json_path}: no valid records after filtering; skipping."
        )
        return False

    scale_factor = maybe_rescale_accuracy(sanitized, json_path)
    threshold_value = config.acc_threshold
    threshold_label = config.acc_threshold_label
    if scale_factor != 1.0 and threshold_value > 1.5:
        threshold_value = threshold_value / scale_factor
        threshold_label = format_threshold_label(threshold_value)

    feasible = [
        rec for rec in sanitized if rec["accuracy"] >= threshold_value
    ]
    infeasible = [
        rec for rec in sanitized if rec["accuracy"] < threshold_value
    ]

    if not feasible and not infeasible:
        print(f"[warn] {json_path}: no points to plot; skipping.")
        return False

    pareto_all = pareto_front_2d(
        sanitized, config.x_metric, config.y_metric
    )
    pareto_feasible = pareto_front_2d(
        feasible, config.x_metric, config.y_metric
    )

    units = extract_units(sanitized)

    output_root = config.output_root
    if config.input_root:
        try:
            relative = json_path.relative_to(config.input_root)
        except ValueError:
            relative = Path(Path(json_path).name)
        output_dir = output_root / relative.parent
    else:
        output_dir = output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_tag = threshold_tag_from_label(threshold_label)
    filename_base = (
        f"{json_path.stem}_pareto_{config.x_metric}_vs_{config.y_metric}"
        f"_ex{config.exclude_metric}_acc{threshold_tag}"
    )
    if config.suffix:
        filename_base = f"{filename_base}{config.suffix}"

    out_png = output_dir / f"{filename_base}.png"
    out_pdf = output_dir / f"{filename_base}.pdf"

    tick_original_flag = (
        config.normalize_mode == "minmax" or config.tick_original
    )

    if config.plot_2d:
        metrics = (config.x_metric, config.y_metric)
        log_flags = (
            config.xlog and metrics[0] != "accuracy",
            config.ylog and metrics[1] != "accuracy",
        )
        axis_values: List[List[float]] = [[], []]

        feasible_points_raw, dropped_fea = transform_records(
            feasible, metrics, log_flags, axis_values
        )
        infeasible_points_raw, dropped_inf = transform_records(
            infeasible, metrics, log_flags, axis_values
        )
        dropped_total = dropped_fea + dropped_inf
        if dropped_total:
            print(
                f"[warn] {json_path}: skipped {dropped_total} point(s) due to "
                "non-positive values for log-transformed axes."
            )

        raw_bounds = [
            compute_axis_limits(axis, config.percentile_low, config.percentile_high, config.axis_padding)
            for axis in axis_values
        ]
        original_bounds = convert_bounds_to_original(raw_bounds, log_flags)

        pareto_all_raw, _ = transform_records(
            pareto_all, metrics, log_flags
        )
        pareto_feasible_raw, _ = transform_records(
            pareto_feasible, metrics, log_flags
        )

        if config.normalize_mode == "minmax":
            normalized = True
            feasible_points_plot: List[
                Tuple[Dict[str, float], Tuple[float, float]]
            ] = []
            for record, coords in feasible_points_raw:
                norm_coords, keep = normalize_coord(
                    coords, raw_bounds, config.show_outliers
                )
                if keep or config.show_outliers:
                    feasible_points_plot.append((record, norm_coords))

            infeasible_points_plot: List[
                Tuple[Dict[str, float], Tuple[float, float]]
            ] = []
            for record, coords in infeasible_points_raw:
                norm_coords, keep = normalize_coord(
                    coords, raw_bounds, config.show_outliers
                )
                if keep or config.show_outliers:
                    infeasible_points_plot.append((record, norm_coords))

            pareto_all_points = []
            for _, coords in pareto_all_raw:
                norm_coords, keep = normalize_coord(
                    coords, raw_bounds, config.show_outliers
                )
                if keep or config.show_outliers:
                    pareto_all_points.append(norm_coords)

            pareto_feasible_points = []
            for _, coords in pareto_feasible_raw:
                norm_coords, keep = normalize_coord(
                    coords, raw_bounds, config.show_outliers
                )
                if keep or config.show_outliers:
                    pareto_feasible_points.append(norm_coords)

            feasible_points_plot, infeasible_points_plot = retain_non_dominated(
                feasible_points_plot, infeasible_points_plot, metrics
            )

            size_feasible = compute_marker_sizes(
                [record for record, _ in feasible_points_plot], config.size_metric
            )
            size_infeasible = compute_marker_sizes(
                [record for record, _ in infeasible_points_plot], config.size_metric
            )

            axis_limits_plot = [(0.0, 1.0), (0.0, 1.0)]
            axis_labels_base = build_axis_labels(metrics, units, log_flags)
            axis_labels_plot = list(axis_labels_base)
            note_lines = []
        else:
            normalized = False
            feasible_points_plot = filter_points_within_bounds(
                feasible_points_raw, raw_bounds, config.show_outliers
            )
            infeasible_points_plot = filter_points_within_bounds(
                infeasible_points_raw, raw_bounds, config.show_outliers
            )
            feasible_points_plot, infeasible_points_plot = retain_non_dominated(
                feasible_points_plot, infeasible_points_plot, metrics
            )
            size_feasible = compute_marker_sizes(
                [record for record, _ in feasible_points_plot], config.size_metric
            )
            size_infeasible = compute_marker_sizes(
                [record for record, _ in infeasible_points_plot], config.size_metric
            )
            pareto_all_points = filter_coords_within_bounds(
                [coords for _, coords in pareto_all_raw],
                raw_bounds,
                config.show_outliers,
            )
            pareto_feasible_points = filter_coords_within_bounds(
                [coords for _, coords in pareto_feasible_raw],
                raw_bounds,
                config.show_outliers,
            )
            axis_limits_plot = raw_bounds
            axis_labels_plot = build_axis_labels(metrics, units, log_flags)
            note_lines = []

        plot_pareto_2d(
            feasible_points_plot,
            infeasible_points_plot,
            pareto_all_points,
            pareto_feasible_points,
            size_feasible,
            size_infeasible,
            metrics,
            axis_labels_plot,
            axis_limits_plot,
            raw_bounds,
            log_flags,
            units,
            tick_original_flag,
            note_lines,
            normalized,
            config.equal_aspect,
            json_path,
            out_png,
            out_pdf,
            config.dpi,
        )
    else:
        metrics = list(config.axes_3d)
        log_flags = [
            config.xlog and metrics[0] != "accuracy",
            config.ylog and metrics[1] != "accuracy",
            False,
        ]
        axis_values: List[List[float]] = [[], [], []]

        feasible_points_raw, dropped_fea = transform_records(
            feasible, metrics, log_flags, axis_values
        )
        infeasible_points_raw, dropped_inf = transform_records(
            infeasible, metrics, log_flags, axis_values
        )
        dropped_total = dropped_fea + dropped_inf
        if dropped_total:
            print(
                f"[warn] {json_path}: skipped {dropped_total} point(s) due to "
                "non-positive values for log-transformed axes."
            )

        raw_bounds = [
            compute_axis_limits(axis, config.percentile_low, config.percentile_high, config.axis_padding)
            for axis in axis_values
        ]

        acc_index = metrics.index("accuracy")
        ensure_accuracy_bounds(raw_bounds, acc_index, threshold_value, config.axis_padding)
        original_bounds = convert_bounds_to_original(raw_bounds, log_flags)

        if config.normalize_mode == "minmax":
            normalized = True
            feasible_points_plot: List[
                Tuple[Dict[str, float], Tuple[float, float, float]]
            ] = []
            for record, coords in feasible_points_raw:
                norm_coords, keep = normalize_coord(
                    coords, raw_bounds, config.show_outliers
                )
                if keep or config.show_outliers:
                    feasible_points_plot.append((record, norm_coords))

            infeasible_points_plot: List[
                Tuple[Dict[str, float], Tuple[float, float, float]]
            ] = []
            for record, coords in infeasible_points_raw:
                norm_coords, keep = normalize_coord(
                    coords, raw_bounds, config.show_outliers
                )
                if keep or config.show_outliers:
                    infeasible_points_plot.append((record, norm_coords))

            feasible_points_plot, infeasible_points_plot = retain_non_dominated(
                feasible_points_plot, infeasible_points_plot, metrics
            )

            size_feasible = compute_marker_sizes(
                [record for record, _ in feasible_points_plot],
                config.size_metric,
            )
            size_infeasible = compute_marker_sizes(
                [record for record, _ in infeasible_points_plot],
                config.size_metric,
            )

            axis_limits_plot = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
            axis_labels_plot = build_axis_labels(metrics, units, log_flags)
            note_lines = []

            span = max(raw_bounds[acc_index][1] - raw_bounds[acc_index][0], 1e-9)
            threshold_plot_value = (
                (threshold_value - raw_bounds[acc_index][0]) / span
            )
            threshold_plot_value = min(1.0, max(0.0, threshold_plot_value))
            aspect = (1.0, 1.0, 1.0)
        else:
            normalized = False
            feasible_points_plot = filter_points_within_bounds(
                feasible_points_raw, raw_bounds, config.show_outliers
            )
            infeasible_points_plot = filter_points_within_bounds(
                infeasible_points_raw, raw_bounds, config.show_outliers
            )
            feasible_points_plot, infeasible_points_plot = retain_non_dominated(
                feasible_points_plot, infeasible_points_plot, metrics
            )
            size_feasible = compute_marker_sizes(
                [record for record, _ in feasible_points_plot],
                config.size_metric,
            )
            size_infeasible = compute_marker_sizes(
                [record for record, _ in infeasible_points_plot],
                config.size_metric,
            )
            axis_limits_plot = raw_bounds
            axis_labels_plot = build_axis_labels(metrics, units, log_flags)
            note_lines = []
            threshold_plot_value = threshold_value
            ranges = [high - low for (low, high) in raw_bounds]
            aspect = (1.0, 1.0, 1.0) if config.equal_aspect else clamp_aspect(ranges, False)

        elev = config.elev if config.elev is not None else 22.0
        azim = config.azim if config.azim is not None else 38.0

        plot_pareto_3d(
            feasible_points_plot,
            infeasible_points_plot,
            size_feasible,
            size_infeasible,
            metrics,
            axis_labels_plot,
            axis_limits_plot,
            raw_bounds,
            log_flags,
            units,
            tick_original_flag,
            note_lines,
            threshold_plot_value,
            config.show_threshold_plane,
            acc_index,
            aspect,
            elev,
            azim,
            json_path,
            out_png,
            out_pdf,
            config.dpi,
            normalized,
        )

    print(
        f"[info] {json_path}: total={len(sanitized)} feasible={len(feasible)} "
        f"infeasible={len(infeasible)} pareto_all={len(pareto_all)} "
        f"pareto_feasible={len(pareto_feasible)}"
    )
    return True


def collect_json_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*.json") if p.is_file())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    exclude_metric = canonical_metric_name(args.exclude_metric)
    if exclude_metric == "accuracy":
        parser.error("Cannot exclude accuracy.")

    remaining_metrics = [m for m in MINIMIZE_METRICS if m != exclude_metric]
    if len(remaining_metrics) < 2:
        parser.error(
            "Not enough metrics remain after exclusion to plot a Pareto front."
        )

    x_metric = (
        canonical_metric_name(args.x) if args.x else remaining_metrics[0]
    )
    y_metric = (
        canonical_metric_name(args.y) if args.y else remaining_metrics[1]
    )

    for metric_name, label in ((x_metric, "--x"), (y_metric, "--y")):
        if metric_name == "accuracy":
            parser.error(f"{label} cannot be accuracy.")
        if metric_name == exclude_metric:
            parser.error(
                f"{label} cannot match the excluded metric '{exclude_metric}'."
            )
        if metric_name not in MINIMIZE_METRICS:
            parser.error(f"{label} must be one of {MINIMIZE_METRICS}.")

    if x_metric == y_metric:
        parser.error("--x and --y must differ.")

    size_metric = (
        canonical_metric_name(args.size_metric)
        if args.size_metric
        else None
    )
    if size_metric:
        if args.plot_2d and size_metric in {x_metric, y_metric}:
            parser.error("--size-metric must differ from --x and --y.")
        if size_metric == exclude_metric and args.plot_2d:
            parser.error("--size-metric cannot be the excluded metric.")
        if size_metric not in MINIMIZE_METRICS:
            parser.error(f"--size-metric must be one of {MINIMIZE_METRICS}.")

    if args.acc_threshold is None:
        acc_threshold_value = ACC_THRESHOLD_DEFAULT
        acc_threshold_label = f"{ACC_THRESHOLD_DEFAULT:.2f}"
    else:
        try:
            acc_threshold_value = float(args.acc_threshold)
        except ValueError as exc:
            parser.error(f"Invalid --acc-threshold value: {exc}")
        acc_threshold_label = args.acc_threshold.strip()
        if not acc_threshold_label:
            acc_threshold_label = format_threshold_label(acc_threshold_value)

    dpi = max(args.dpi, 1)
    plot_2d = args.plot_2d
    show_threshold_plane = not args.no_separator_plane
    acc_axis_choice = args.acc_axis.lower()

    pct_low, pct_high = args.percentile_clip
    if pct_low < 0 or pct_high > 100 or pct_low >= pct_high:
        parser.error("--percentile-clip must satisfy 0 <= low < high <= 100")
    axis_padding = args.axis_padding
    if axis_padding < 0:
        parser.error("--axis-padding must be non-negative")
    normalize_mode = args.normalize.lower()

    axes_list: List[Optional[str]] = [None, None, None]
    axis_index_map = {"x": 0, "y": 1, "z": 2}
    axes_list[axis_index_map[acc_axis_choice]] = "accuracy"
    remaining_iter = iter(remaining_metrics)
    for idx in range(3):
        if axes_list[idx] is None:
            axes_list[idx] = next(remaining_iter)
    axes_3d = tuple(axes_list)  # type: ignore[arg-type]

    output_root = Path(__file__).resolve().parent / "pareto-front"
    output_root.mkdir(parents=True, exist_ok=True)

    input_root: Optional[Path] = None
    if input_path.is_dir():
        input_root = input_path

    config = PlotConfig(
        x_metric=x_metric,
        y_metric=y_metric,
        size_metric=size_metric,
        exclude_metric=exclude_metric,
        acc_threshold=acc_threshold_value,
        acc_threshold_label=acc_threshold_label,
        suffix=args.suffix,
        dpi=dpi,
        input_root=input_root,
        output_root=output_root,
        plot_2d=plot_2d,
        show_threshold_plane=show_threshold_plane,
        axes_3d=axes_3d,
        acc_axis=acc_axis_choice,
        percentile_low=pct_low,
        percentile_high=pct_high,
        axis_padding=axis_padding,
        equal_aspect=args.equal_aspect,
        elev=args.elev,
        azim=args.azim,
        xlog=args.xlog,
        ylog=args.ylog,
        show_outliers=args.show_outliers,
        normalize_mode=normalize_mode,
        tick_original=args.tick_original,
    )

    json_files = collect_json_files(input_path)
    if not json_files:
        print(f"[warn] No JSON files found under {input_path}.")
        return

    processed_any = False
    for json_file in json_files:
        processed_any |= process_file(json_file, config)

    if not processed_any:
        print("[warn] No figures were generated.")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# Local testing instructions
#
# 1. Create a toy dataset for smoke testing:
#      mkdir -p test_data
#      cat <<'EOF' > test_data/toy_results.json
#      [
#        {"accuracy": 0.85, "power": 120, "latency": 2.3, "area": 1.2},
#        {"accuracy": 0.92, "power": 100, "latency": 2.5, "area": 1.1},
#        {"accuracy": 0.88, "power": 90,  "latency": 3.1, "area": 1.3},
#        {"accuracy": 0.95, "power": 80,  "latency": 1.9, "area": 1.0}
#      ]
#      EOF
#
# 2. Default 3D: robust scaling, percent accuracy ticks, good aspect
#      python make_pareto_figs.py --input test_data/toy_results.json --acc-threshold 0.9
#
# 3. Equal aspect with custom camera angles:
#      python make_pareto_figs.py --input test_data/toy_results.json --equal-aspect --elev 24 --azim 40
#
# 4. Wider clip, include outliers, log-scale power axis:
#      python make_pareto_figs.py --input test_data/toy_results.json --percentile-clip 0.5 99.5 --show-outliers --xlog
#
# 5. Exercise the 2D fallback mode with the same scaling rules:
#      python make_pareto_figs.py --input test_data/toy_results.json --plot-2d
#
# 6. Confirm that `pareto-front/` contains PNG and PDF outputs mirroring
#    the input layout, that feasible (green) and infeasible (red) points
#    are rendered correctly, and that CLI argument errors are surfaced for
#    unsupported metrics (e.g., `--x accuracy`).
# ----------------------------------------------------------------------
