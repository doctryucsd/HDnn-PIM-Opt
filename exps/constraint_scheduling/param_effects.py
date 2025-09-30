#!/usr/bin/env python3
"""Analyze how accuracy varies with experiment parameters."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".json"}
SKIP_DIR_NAMES = {"analysis_plots", "param_effects", "extras"}
EXCLUDED_PARAM_NAMES = {
    "timing",
    "area",
    "energy",
    "hv",
    "hv_constrained",
    "hv_constrainted",
    "eligible_count",
}
MAX_IMPORTANCE_BARS = 10
MARGINAL_TOP_N = 3
INTERACTION_TOP_N = 2
PAIR_MAX_BINS = 8
MARGINAL_MAX_POINTS = 30


@dataclass
class ParamInfo:
    name: str
    kind: str  # "numeric" or "categorical"
    series: pd.Series
    pair_values: pd.Series
    pair_order: List


def discover_files(data_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative_parts = path.relative_to(data_dir).parts
        except ValueError:
            relative_parts = path.parts
        if any(part in SKIP_DIR_NAMES for part in relative_parts[:-1]):
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def expand_structured_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in list(result.columns):
        series = result[col]
        non_null = series.dropna()
        if non_null.empty:
            continue
        sample = non_null.iloc[0]
        if isinstance(sample, dict):
            expanded = pd.json_normalize(non_null)
            expanded.index = non_null.index
            for sub_col in expanded.columns:
                if sub_col in result.columns:
                    sub_name = f"{col}.{sub_col}"
                else:
                    sub_name = sub_col
                result.loc[expanded.index, sub_name] = expanded[sub_col]
            result.drop(columns=col, inplace=True)
    return result


def load_tabular_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        try:
            return pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Failed to read CSV {path}: {exc}") from exc
    if suffix == ".tsv":
        try:
            return pd.read_csv(path, sep="\t")
        except Exception as exc:
            raise ValueError(f"Failed to read TSV {path}: {exc}") from exc
    if suffix == ".json":
        try:
            with open(path, "r") as fh:
                payload = json.load(fh)
        except Exception as exc:
            raise ValueError(f"Failed to parse JSON {path}: {exc}") from exc
        if isinstance(payload, list):
            return pd.json_normalize(payload)
        if isinstance(payload, dict):
            if any(isinstance(v, list) for v in payload.values()):
                try:
                    return pd.DataFrame(payload)
                except ValueError:
                    pass
            if "data" in payload and isinstance(payload["data"], list):
                return pd.json_normalize(payload["data"])
            return pd.json_normalize(payload)
        raise ValueError(f"Unsupported JSON structure in {path}")
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def load_dataset(data_dir: Path) -> Tuple[pd.DataFrame, List[Path]]:
    files = discover_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No supported data files found under {data_dir}")
    frames: List[pd.DataFrame] = []
    for path in files:
        df = load_tabular_file(path)
        if df.empty:
            continue
        df = expand_structured_columns(df)
        df["__source_file__"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("All discovered files were empty")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, files


def normalize_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def detect_accuracy_column(df: pd.DataFrame) -> str:
    candidates = []
    for col in df.columns:
        norm = normalize_column_name(str(col))
        if norm in {"accuracy", "acc", "val_accuracy", "validation_accuracy"} or "accuracy" in norm:
            candidates.append(col)
    if not candidates:
        raise ValueError("Could not find an accuracy column (looked for names containing 'accuracy').")
    # Prefer exact matches
    for preferred in ("accuracy", "Accuracy", "ACC", "acc"):
        if preferred in df.columns:
            return preferred
    return candidates[0]


def classify_parameters(df: pd.DataFrame, accuracy_col: str) -> Dict[str, ParamInfo]:
    params: Dict[str, ParamInfo] = {}
    for col in df.columns:
        if col == accuracy_col or col == "__source_file__":
            continue
        norm_name = normalize_column_name(str(col))
        if norm_name in EXCLUDED_PARAM_NAMES:
            continue
        series = df[col]
        non_null = series.dropna()
        if non_null.nunique() <= 1:
            continue
        if pd.api.types.is_bool_dtype(series):
            kind = "categorical"
        elif pd.api.types.is_numeric_dtype(series):
            kind = "numeric"
        else:
            kind = "categorical"
        pair_values, pair_order = prepare_pair_values(series, kind)
        params[col] = ParamInfo(col, kind, series, pair_values, pair_order)
    if not params:
        raise ValueError("No varying parameter columns found to analyze.")
    return params


def prepare_pair_values(series: pd.Series, kind: str) -> Tuple[pd.Series, List]:
    s = series.copy()
    if kind == "numeric":
        unique_vals = s.dropna().unique()
        if len(unique_vals) <= PAIR_MAX_BINS:
            order = sorted(unique_vals)
            return s, order
        bins = min(PAIR_MAX_BINS, len(unique_vals))
        try:
            cat = pd.qcut(s, q=bins, duplicates="drop")
        except ValueError:
            order = sorted(unique_vals)
            return s, order
        order = [str(c) for c in cat.cat.categories]
        labels = cat.astype(str)
        labels[cat.isna()] = np.nan
        return labels, order
    labels = s.astype(str)
    labels[s.isna()] = np.nan
    order = list(dict.fromkeys(labels.dropna()))
    return labels, order


def compute_numeric_importance(data: pd.DataFrame, param: str, target: str) -> float:
    subset = data[[param, target]].dropna()
    if subset.empty or subset[param].nunique() <= 1:
        return 0.0
    corr = subset[param].corr(subset[target])
    if pd.isna(corr):
        return 0.0
    return float(corr ** 2)


def compute_categorical_importance(data: pd.DataFrame, param: str, target: str) -> float:
    subset = data[[param, target]].dropna()
    if subset.empty or subset[param].nunique() <= 1:
        return 0.0
    total_var = subset[target].var(ddof=0)
    if total_var <= 0:
        return 0.0
    group = subset.groupby(param, dropna=False)[target].agg(["mean", "count"])
    weights = group["count"] / group["count"].sum()
    between = ((group["mean"] - subset[target].mean()) ** 2 * weights).sum()
    return float(between / total_var)


def compute_param_importances(df: pd.DataFrame, accuracy_col: str, params: Dict[str, ParamInfo]) -> pd.DataFrame:
    records = []
    for name, info in params.items():
        if info.kind == "numeric":
            importance = compute_numeric_importance(df, name, accuracy_col)
        else:
            importance = compute_categorical_importance(df, name, accuracy_col)
        records.append({"parameter": name, "importance": importance, "type": info.kind})
    result = pd.DataFrame(records)
    result.sort_values(by="importance", ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def render_param_importance(ax: plt.Axes, importances: pd.DataFrame) -> None:
    top = importances.head(MAX_IMPORTANCE_BARS)
    ax.barh(top["parameter"], top["importance"], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Explained variance (R^2)")
    ax.set_title("Parameter importance ranking")
    ax.grid(axis="x", linestyle=":", alpha=0.3)


def plot_param_importance(importances: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    render_param_importance(ax, importances)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def marginal_series_numeric(series: pd.Series, accuracy: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    data = pd.DataFrame({"x": series, "y": accuracy}).dropna()
    if data.empty:
        return np.array([]), np.array([])
    unique_vals = data["x"].nunique()
    if unique_vals <= MARGINAL_MAX_POINTS:
        grouped = data.groupby("x", as_index=False)["y"].max().sort_values("x")
        return grouped["x"].to_numpy(dtype=float), grouped["y"].to_numpy(dtype=float)
    bins = min(MARGINAL_MAX_POINTS, unique_vals)
    try:
        cat = pd.qcut(data["x"], q=bins, duplicates="drop")
    except ValueError:
        grouped = data.groupby("x", as_index=False)["y"].max().sort_values("x")
        return grouped["x"].to_numpy(dtype=float), grouped["y"].to_numpy(dtype=float)
    grouped = data.groupby(cat)["y"].max()
    mids: List[float] = []
    for interval in cat.cat.categories:
        mids.append(float((interval.left + interval.right) / 2))
    return np.array(mids, dtype=float), grouped.to_numpy(dtype=float)


def marginal_series_categorical(series: pd.Series, accuracy: pd.Series) -> Tuple[List[str], np.ndarray]:
    data = pd.DataFrame({"x": series.astype(str), "y": accuracy}).dropna()
    if data.empty:
        return [], np.array([])
    grouped = data.groupby("x")["y"].max().sort_values(ascending=False)
    return grouped.index.tolist(), grouped.to_numpy(dtype=float)


def plot_marginal_effects(
    importances: pd.DataFrame,
    df: pd.DataFrame,
    params: Dict[str, ParamInfo],
    accuracy_col: str,
    out_path: Path,
) -> List[Dict[str, Any]]:
    selected = importances.head(MARGINAL_TOP_N)["parameter"].tolist()
    selected = [p for p in selected if p in params]
    if not selected:
        return []
    cols = len(selected)
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 4), squeeze=False)
    axes = axes[0]
    entries: List[Dict[str, Any]] = []
    for ax, param in zip(axes, selected):
        info = params[param]
        if info.kind == "numeric":
            x, y = marginal_series_numeric(info.series, df[accuracy_col])
            if x.size == 0:
                ax.set_visible(False)
                continue
            ax.plot(x, y, marker="o", linestyle="-", color="darkorange")
            ax.set_xlabel(param)
            entries.append({
                "parameter": param,
                "kind": "numeric",
                "x": x.tolist(),
                "y": y.tolist(),
            })
        else:
            labels, y = marginal_series_categorical(info.series, df[accuracy_col])
            if not labels:
                ax.set_visible(False)
                continue
            positions = np.arange(len(labels))
            ax.bar(positions, y, color="mediumpurple")
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_xlabel(param)
            entries.append({
                "parameter": param,
                "kind": "categorical",
                "labels": labels,
                "y": y.tolist(),
            })
        ax.set_ylabel("Top accuracy")
        ax.set_title(f"Top accuracy vs {param}")
        ax.grid(axis="both", linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return entries


def compute_pair_interactions(
    df: pd.DataFrame,
    accuracy_col: str,
    params: Dict[str, ParamInfo],
    single_importances: pd.Series,
) -> List[Dict[str, object]]:
    names = list(params.keys())
    results: List[Dict[str, object]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p1, p2 = names[i], names[j]
            info1, info2 = params[p1], params[p2]
            pair_df = pd.DataFrame(
                {
                    "p1": info1.pair_values,
                    "p2": info2.pair_values,
                    "accuracy": df[accuracy_col],
                }
            ).dropna()
            if pair_df.empty:
                continue
            if pair_df["p1"].nunique() <= 1 or pair_df["p2"].nunique() <= 1:
                continue
            total_var = pair_df["accuracy"].var(ddof=0)
            if total_var <= 0:
                continue
            grouped = pair_df.groupby(["p1", "p2"])["accuracy"].agg(mean="mean", count="count", top="max")
            weights = grouped["count"] / grouped["count"].sum()
            between = ((grouped["mean"] - pair_df["accuracy"].mean()) ** 2 * weights).sum()
            pair_r2 = float(between / total_var)
            interaction = pair_r2 - max(single_importances.get(p1, 0.0), single_importances.get(p2, 0.0))
            interaction = max(interaction, 0.0)
            order1 = info1.pair_order or sorted(grouped.index.get_level_values(0).unique())
            order2 = info2.pair_order or sorted(grouped.index.get_level_values(1).unique())
            results.append(
                {
                    "pair": (p1, p2),
                    "interaction": interaction,
                    "pair_r2": pair_r2,
                    "group": grouped.reset_index(),
                    "order1": list(order1),
                    "order2": list(order2),
                }
            )
    results.sort(key=lambda item: item["interaction"], reverse=True)
    return results


def plot_interactions(
    interactions: Sequence[Dict[str, object]],
    out_path: Path,
    max_pairs: int = INTERACTION_TOP_N,
) -> Tuple[List[Tuple[str, str]], Optional[Path], List[Dict[str, Any]]]:
    top = [item for item in interactions if item["interaction"] > 0]
    if not top:
        return [], None, []
    top = top[:max_pairs]
    cols = len(top)
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5), squeeze=False)
    axes = axes[0]
    used_pairs: List[Tuple[str, str]] = []
    entries: List[Dict[str, Any]] = []
    for ax, item in zip(axes, top):
        p1, p2 = item["pair"]
        group_df = item["group"]
        order1 = item["order1"]
        order2 = item["order2"]
        pivot = group_df.pivot(index="p1", columns="p2", values="top")
        if order1:
            pivot = pivot.reindex(order1)
        if order2:
            pivot = pivot.reindex(columns=order2)
        im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel(p2)
        ax.set_ylabel(p1)
        ax.set_title(f"Interaction: {p1} vs {p2}")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Top accuracy")
        used_pairs.append((p1, p2))
        entries.append(
            {
                "pair": (p1, p2),
                "matrix": pivot.to_numpy(),
                "x_labels": list(pivot.columns),
                "y_labels": list(pivot.index),
            }
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return used_pairs, out_path, entries


def write_summary(
    out_path: Path,
    data_dir: Path,
    row_count: int,
    file_count: int,
    accuracy_col: str,
    importances: pd.DataFrame,
    top_marginals: Iterable[str],
    top_pairs: Iterable[Tuple[str, str]],
    interaction_available: bool,
) -> Dict[str, Any]:
    summary = {
        "data_dir": str(data_dir),
        "row_count": int(row_count),
        "file_count": int(file_count),
        "accuracy_column": accuracy_col,
        "parameter_importances": importances[["parameter", "importance"]].to_dict(orient="records"),
        "top_marginal_parameters": list(top_marginals),
        "top_interacting_pairs": [list(pair) for pair in top_pairs],
        "interactions_plotted": interaction_available,
    }
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def print_report(
    data_dir: Path,
    row_count: int,
    file_count: int,
    accuracy_col: str,
    importances: pd.DataFrame,
    marginals: Sequence[str],
    interactions: Sequence[Tuple[str, str]],
    outputs: Sequence[Path],
    interaction_available: bool,
) -> None:
    print("=== Accuracy Analysis Report ===")
    print(f"Data directory  : {data_dir}")
    print(f"Files processed : {file_count}")
    print(f"Total rows      : {row_count}")
    print(f"Accuracy column : {accuracy_col}")
    print("Top-5 parameter importances:")
    for _, row in importances.head(5).iterrows():
        print(f"  {row['parameter']:<20} R^2={row['importance']:.4f}")
    print("Top marginal parameters:")
    if marginals:
        for name in marginals:
            print(f"  {name}")
    else:
        print("  (none)")
    print("Top interacting pairs:")
    if interactions:
        for p1, p2 in interactions:
            print(f"  {p1} x {p2}")
    else:
        note = "none (insufficient interaction signal)" if not interaction_available else "(not computed)"
        print(f"  {note}")
    print("Saved files:")
    for path in outputs:
        print(f"  {path.name}")


def generate_summary_pdf(
    out_dir: Path,
    data_dir: Path,
    dataset_name: str,
    importances: pd.DataFrame,
    marginal_entries: Sequence[Dict[str, Any]],
    interaction_entries: Sequence[Dict[str, Any]],
    summary_info: Dict[str, Any],
    interaction_available: bool,
) -> Path:
    safe_dataset = dataset_name.replace(" ", "_") or "dataset"
    summary_path = out_dir / f"{safe_dataset}_accuracy_summary.pdf"

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.14, right=0.94, top=0.9, bottom=0.12, hspace=0.65, wspace=0.45)

    fig.text(
        0.5,
        0.96,
        f"{dataset_name} Accuracy Analysis Report",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.1, 1])

    # Parameter importance panel
    ax_imp = fig.add_subplot(gs[0, 0])
    render_param_importance(ax_imp, importances)

    # Marginal effects panel
    if marginal_entries:
        subgrid = gs[0, 1].subgridspec(len(marginal_entries), 1, hspace=0.45)
        for idx, entry in enumerate(marginal_entries):
            ax = fig.add_subplot(subgrid[idx, 0])
            if entry["kind"] == "numeric":
                ax.plot(entry["x"], entry["y"], marker="o", linestyle="-", color="darkorange", linewidth=1.3)
                ax.tick_params(axis="x", labelrotation=20, labelsize=8)
            else:
                positions = np.arange(len(entry["labels"]))
                ax.bar(positions, entry["y"], color="mediumpurple")
                ax.set_xticks(positions)
                ax.set_xticklabels(entry["labels"], rotation=20, ha="right", fontsize=8)
            ax.set_ylabel("Top accuracy", fontsize=8)
            ax.set_title(entry["parameter"], fontsize=10, pad=4, loc="left")
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.3, linestyle=":")
    else:
        ax = fig.add_subplot(gs[0, 1])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No marginal plots available",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#f0f0f0"),
        )

    # Interaction panel
    if interaction_entries:
        subgrid = gs[1, :].subgridspec(1, len(interaction_entries), wspace=0.35)
        for idx, entry in enumerate(interaction_entries):
            ax = fig.add_subplot(subgrid[0, idx])
            im = ax.imshow(entry["matrix"], aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(np.arange(len(entry["x_labels"])))
            ax.set_xticklabels(entry["x_labels"], rotation=30, ha="right", fontsize=8)
            ax.set_yticks(np.arange(len(entry["y_labels"])))
            ax.set_yticklabels(entry["y_labels"], fontsize=8)
            ax.set_xlabel(entry['pair'][1], fontsize=9, labelpad=4)
            ax.set_ylabel(entry['pair'][0], fontsize=9, labelpad=4)
            ax.set_title(f"{entry['pair'][0]} × {entry['pair'][1]}", fontsize=10, pad=6)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel("Top accuracy", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
    else:
        ax = fig.add_subplot(gs[1, :])
        ax.axis("off")
        placeholder = "No interaction plots generated" if not interaction_available else "Interaction data unavailable"
        ax.text(
            0.5,
            0.5,
            placeholder,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#f0f0f0"),
        )

    fig.text(0.5, 0.06, "Page 1 of 1", ha="center", fontsize=8)

    fig.savefig(summary_path)
    extras_preview = out_dir / "extras" / "accuracy_summary_preview.png"
    fig.savefig(extras_preview, dpi=200)
    plt.close(fig)
    print(summary_path)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize accuracy trends across experiment parameters.")
    parser.add_argument("--data-dir", dest="data_dir", default=None, help="Directory containing result files (CSV/TSV/JSON)")
    parser.add_argument("--data", dest="data_dir_legacy", default=None, help="Deprecated alias for --data-dir")
    parser.add_argument("--out-dir", dest="out_dir", default=None, help="Output directory for figures and summary")
    parser.add_argument("--outname", dest="out_name", default=None, help="Name of subfolder inside data directory for outputs")
    args = parser.parse_args()

    data_raw = args.data_dir or args.data_dir_legacy
    if not data_raw:
        parser.error("Must provide --data-dir (or --data).")

    data_dir = Path(data_raw).expanduser().resolve()
    out_dir: Path
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    elif args.out_name:
        out_dir = data_dir / args.out_name
    else:
        out_dir = data_dir / "analysis_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    extras_dir = out_dir / "extras"
    extras_dir.mkdir(exist_ok=True)

    df, files = load_dataset(data_dir)
    accuracy_col = detect_accuracy_column(df)
    params = classify_parameters(df, accuracy_col)

    importances = compute_param_importances(df, accuracy_col, params)
    importance_path = out_dir / "01_param_importance.pdf"
    plot_param_importance(importances, importance_path)

    marginals_path = out_dir / "02_top_params_marginals.pdf"
    marginal_entries = plot_marginal_effects(importances, df, params, accuracy_col, marginals_path)

    interactions_info = compute_pair_interactions(df, accuracy_col, params, importances.set_index("parameter")["importance"])
    interactions_path = out_dir / "03_top_interactions.pdf"
    interaction_pairs, figure_path, interaction_entries = plot_interactions(interactions_info, interactions_path)
    interaction_available = figure_path is not None

    outputs = [importance_path, marginals_path]
    if interaction_available and figure_path is not None:
        outputs.append(figure_path)
    summary_path = out_dir / "summary_insights.json"
    marginal_names = [entry["parameter"] for entry in marginal_entries]
    summary_dict = write_summary(
        summary_path,
        data_dir,
        len(df),
        len(files),
        accuracy_col,
        importances,
        marginal_names,
        interaction_pairs,
        interaction_available,
    )
    outputs.append(summary_path)

    summary_pdf_path = generate_summary_pdf(
        out_dir=out_dir,
        data_dir=data_dir,
        dataset_name=data_dir.name,
        importances=importances,
        marginal_entries=marginal_entries,
        interaction_entries=interaction_entries,
        summary_info=summary_dict,
        interaction_available=interaction_available,
    )
    outputs.append(summary_pdf_path)

    print_report(
        data_dir,
        len(df),
        len(files),
        accuracy_col,
        importances,
        marginal_names,
        interaction_pairs,
        outputs,
        interaction_available,
    )


if __name__ == "__main__":
    main()
