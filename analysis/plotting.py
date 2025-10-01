#!/usr/bin/env python3
"""Visualization utilities for accuracy proxy analyses."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from .accuracy_proxy import AccuracyProxy
from .data_loader import load_all_results
from .features import build_features

LOGGER = logging.getLogger(__name__)


def _ensure_out_dir(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)


def _spearman(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if true.size < 2 or pred.size < 2:
        return float("nan")
    true_rank = pd.Series(true).rank(method="average")
    pred_rank = pd.Series(pred).rank(method="average")
    corr = np.corrcoef(true_rank, pred_rank)[0, 1]
    return float(corr)


def _load_dataset(dataset: str, seeds: Sequence[int] | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_all_results(dataset, seeds=seeds)
    df = df.sort_values(["method", "seed", "iter"], kind="stable")
    features = build_features(df)
    return df, features


# ---------------------------------------------------------------------------
# Predicted vs. True scatter


def plot_pred_true(dataset: str, out_path: Path, seeds: Sequence[int] | None = None) -> None:
    df, features = _load_dataset(dataset, seeds=seeds)
    y = df["acc"].to_numpy(dtype=float)

    model = AccuracyProxy(alpha=1.0, use_isotonic=True)
    model.fit(features, y)
    preds = model.predict(features)

    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    spear = _spearman(y, preds)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, preds, s=15, alpha=0.6)

    min_val = float(min(np.min(y), np.min(preds)))
    max_val = float(max(np.max(y), np.max(preds)))
    ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1)

    ax.set_xlabel("True accuracy")
    ax.set_ylabel("Predicted accuracy")
    ax.set_title(f"{dataset} proxy fit")
    text = f"R$^2$ = {r2:.3f}\nMAE = {mae:.4f}\nSpearman = {spear:.3f}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, verticalalignment="top", fontsize=10)
    fig.tight_layout()

    _ensure_out_dir(out_path)
    fig.savefig(out_path, format=out_path.suffix.lstrip("."))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Counterfactual early efficiency plot


def _compute_best_curves(
    dataset: str,
    method: str,
    early: int,
    seeds: Sequence[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df, features = _load_dataset(dataset, seeds=seeds)
    if method not in df["method"].unique():
        raise ValueError(f"Method '{method}' not found for dataset {dataset}")

    use_feasible = "feasible" in df.columns

    best_orig_list: List[np.ndarray] = []
    best_re_list: List[np.ndarray] = []
    delta_tff: List[float] = []

    grouped = df[df["method"] == method].groupby("seed", sort=False)
    for seed, group in grouped:
        group_features = features.loc[group.index]
        model = AccuracyProxy(alpha=1.0, use_isotonic=True)
        model.fit(group_features, group["acc"])
        preds = pd.Series(model.predict(group_features), index=group.index)

        order = list(group.index)
        early_count = min(early, len(order))
        early_idx = order[:early_count]
        rest_idx = order[early_count:]
        reordered_early = sorted(early_idx, key=lambda idx: preds[idx], reverse=True)
        reordered_order = reordered_early + rest_idx

        acc_orig = group.loc[order, "acc"].to_numpy(dtype=float)
        acc_re = group.loc[reordered_order, "acc"].to_numpy(dtype=float)
        best_orig = np.maximum.accumulate(acc_orig)
        best_re = np.maximum.accumulate(acc_re)

        best_orig_list.append(best_orig)
        best_re_list.append(best_re)

        if use_feasible:
            feas_orig = group.loc[order, "feasible"].astype(bool).to_numpy()
            feas_re = group.loc[reordered_order, "feasible"].astype(bool).to_numpy()
            tff_orig = np.argmax(feas_orig) + 1 if feas_orig.any() else np.nan
            tff_re = np.argmax(feas_re) + 1 if feas_re.any() else np.nan
            if not (np.isnan(tff_orig) or np.isnan(tff_re)):
                delta_tff.append(tff_re - tff_orig)

    if not best_orig_list:
        raise ValueError(f"No seeds available for method {method}")

    best_orig_mat = np.vstack(best_orig_list)
    best_re_mat = np.vstack(best_re_list)

    median_orig = np.median(best_orig_mat, axis=0)
    median_re = np.median(best_re_mat, axis=0)
    median_delta_tff = float(np.median(delta_tff)) if delta_tff else float("nan")

    return median_orig, median_re, median_delta_tff


def plot_early_eff(
    dataset: str,
    method: str,
    out_path: Path,
    early: int = 30,
    seeds: Sequence[int] | None = None,
) -> None:
    median_orig, median_re, median_delta_tff = _compute_best_curves(dataset, method, early, seeds=seeds)
    iterations = np.arange(1, len(median_orig) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(iterations, median_orig, label="Original", linewidth=2)
    ax.plot(iterations, median_re, label=f"Reordered (first {early})", linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Median best-so-far accuracy")
    ax.set_title(f"{dataset} {method} counterfactual reorder")
    if not np.isnan(median_delta_tff):
        ax.text(0.02, 0.05, f"Median ΔTFF: {median_delta_tff:.1f}", transform=ax.transAxes, fontsize=10)
    ax.legend()
    ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()

    _ensure_out_dir(out_path)
    fig.savefig(out_path, format=out_path.suffix.lstrip("."))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Filter efficacy bar charts


def plot_filter_bars(dataset: str, table_path: Path, out_path: Path) -> None:
    data = pd.read_csv(table_path)
    if "dataset" in data.columns:
        data = data[data["dataset"] == dataset]
    if data.empty:
        raise ValueError(f"No rows found for dataset {dataset} in {table_path}")

    data = data.sort_values("K")
    ks = data["K"].to_numpy(dtype=int)
    precision = data["Precision@K"].to_numpy(dtype=float)
    recall = data["Recall@K"].to_numpy(dtype=float) if "Recall@K" in data.columns else None

    positions = np.arange(len(ks))
    width = 0.35 if recall is not None else 0.5

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(positions - width / 2 if recall is not None else positions, precision, width, label="Precision@K")

    if recall is not None:
        ax.bar(positions + width / 2, recall, width, label="Recall@K")

    ax.set_xticks(positions)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{dataset} proxy filter efficacy")
    if recall is not None:
        ax.legend()
    fig.tight_layout()

    _ensure_out_dir(out_path)
    fig.savefig(out_path, format=out_path.suffix.lstrip("."))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plotting utilities for analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pred = subparsers.add_parser("pred_true", help="Scatter plot of predicted vs true accuracy")
    pred.add_argument("--dataset", required=True, help="Dataset name")
    pred.add_argument("--out", required=True, type=Path, help="Output figure path")
    pred.add_argument("--seeds", nargs="+", type=int, help="Override default seed list")

    early_eff = subparsers.add_parser("early_eff", help="Median best-so-far accuracy curves")
    early_eff.add_argument("--dataset", required=True, help="Dataset name")
    early_eff.add_argument("--method", required=True, help="Method name to analyze")
    early_eff.add_argument("--out", required=True, type=Path, help="Output figure path")
    early_eff.add_argument("--early", type=int, default=30, help="Number of early iterations to reorder (default 30)")
    early_eff.add_argument("--seeds", nargs="+", type=int, help="Override default seed list")

    filt = subparsers.add_parser("filter_bars", help="Filter efficacy bar chart")
    filt.add_argument("--dataset", required=True, help="Dataset name")
    filt.add_argument("--table", required=True, type=Path, help="CSV with filter metrics")
    filt.add_argument("--out", required=True, type=Path, help="Output figure path")
    filt.add_argument("--seeds", nargs="+", type=int, help="(Unused) seed override for API symmetry", default=None)

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    if args.command == "pred_true":
        plot_pred_true(args.dataset, args.out, seeds=args.seeds)
    elif args.command == "early_eff":
        plot_early_eff(args.dataset, args.method, args.out, early=args.early, seeds=args.seeds)
    elif args.command == "filter_bars":
        plot_filter_bars(args.dataset, args.table, args.out)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
