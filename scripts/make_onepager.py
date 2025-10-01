#!/usr/bin/env python3
"""Compose a one-page PDF summarizing CIFAR-10 proxy performance."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.accuracy_proxy import AccuracyProxy
from analysis.data_loader import load_all_results
from analysis.features import build_features
from analysis.filter_efficacy import evaluate_filter

PDF_PATH = ROOT_DIR / "results" / "Proxy_Summary_1pager.pdf"
DATASET = "cifar10"
FLAGSHIP_METHOD = "NEHVI_no-constraint"
EARLY_REORDER = 30
FILTER_KS: Sequence[int] = (10, 20, 30)


def compute_predictions(df, features) -> tuple[np.ndarray, np.ndarray, float, float]:
    model = AccuracyProxy(alpha=1.0, use_isotonic=True)
    y = df["acc"].to_numpy(dtype=float)
    model.fit(features, y)
    preds = model.predict(features)
    r2 = r2_score(y, preds) if y.size else float("nan")
    mae = mean_absolute_error(y, preds) if y.size else float("nan")
    return y, preds, r2, mae


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    ranks_true = np.argsort(np.argsort(y_true))
    ranks_pred = np.argsort(np.argsort(y_pred))
    return float(np.corrcoef(ranks_true, ranks_pred)[0, 1])


def compute_best_curves(df, features, method: str, early: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    model_seeds = []
    best_orig_curves = []
    best_re_curves = []
    delta_tff_values = []

    grouped = df[df["method"] == method].groupby("seed", sort=False)
    for seed, group in grouped:
        feats = features.loc[group.index]
        model = AccuracyProxy(alpha=1.0, use_isotonic=True)
        model.fit(feats, group["acc"])
        preds = model.predict(feats)
        preds_series = dict(zip(group.index, preds))

        order = list(group.index)
        early_n = min(early, len(order))
        early_idx = order[:early_n]
        rest_idx = order[early_n:]
        reordered_early = sorted(early_idx, key=lambda idx: preds_series[idx], reverse=True)
        reordered_order = reordered_early + rest_idx

        acc_orig = group.loc[order, "acc"].to_numpy(dtype=float)
        acc_re = group.loc[reordered_order, "acc"].to_numpy(dtype=float)
        best_orig_curves.append(np.maximum.accumulate(acc_orig))
        best_re_curves.append(np.maximum.accumulate(acc_re))

        if "feasible" in group.columns:
            feas_orig = group.loc[order, "feasible"].to_numpy(dtype=bool)
            feas_re = group.loc[reordered_order, "feasible"].to_numpy(dtype=bool)
            tff_orig = np.argmax(feas_orig) + 1 if feas_orig.any() else np.nan
            tff_re = np.argmax(feas_re) + 1 if feas_re.any() else np.nan
            if not np.isnan(tff_orig) and not np.isnan(tff_re):
                delta_tff_values.append(tff_re - tff_orig)

        model_seeds.append(seed)

    if not best_orig_curves:
        raise ValueError(f"No seeds found for method {method}")

    best_orig_mat = np.vstack(best_orig_curves)
    best_re_mat = np.vstack(best_re_curves)
    median_orig = np.median(best_orig_mat, axis=0)
    median_re = np.median(best_re_mat, axis=0)
    delta_auc_pct = ((median_re[-1] - median_orig[-1]) / median_orig[-1] * 100.0) if median_orig[-1] else float("nan")
    median_delta_tff = float(np.median(delta_tff_values)) if delta_tff_values else float("nan")
    return median_orig, median_re, median_delta_tff, delta_auc_pct


def plot_scatter(ax, y_true: np.ndarray, y_pred: np.ndarray, r2: float, mae: float, spear: float) -> None:
    ax.scatter(y_true, y_pred, s=12, alpha=0.6)
    bounds = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(bounds, bounds, color="black", linestyle="--", linewidth=1)
    ax.set_title("Proxy predictions vs. observed accuracy (CIFAR-10)")
    ax.set_xlabel("True accuracy")
    ax.set_ylabel("Predicted accuracy")
    ax.text(
        0.02,
        0.98,
        f"R$^2$={r2:.3f}\nMAE={mae:.4f}\nSpearman={spear:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    ax.grid(alpha=0.2, linestyle=":")


def plot_early_efficiency(ax, median_orig: np.ndarray, median_re: np.ndarray, median_delta_tff: float) -> None:
    iterations = np.arange(1, len(median_orig) + 1)
    ax.plot(iterations, median_orig, label="Original", linewidth=2)
    ax.plot(iterations, median_re, label=f"Reordered first {EARLY_REORDER}", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Median best-so-far accuracy")
    ax.set_title(f"Counterfactual early efficiency ({FLAGSHIP_METHOD})")
    if not np.isnan(median_delta_tff):
        ax.text(
            0.02,
            0.92,
            f"Median ΔTFF: {median_delta_tff:.1f}",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
    ax.legend()
    ax.grid(alpha=0.2, linestyle=":")


def plot_filter_bars(ax, filter_df: pd.DataFrame) -> None:
    filter_df = filter_df.sort_values("K")
    ks = filter_df["K"].to_numpy(dtype=int)
    precision = filter_df["Precision@K"].to_numpy(dtype=float)
    recall = filter_df["Recall@K"].to_numpy(dtype=float)
    width = 0.35
    x = np.arange(len(ks))
    ax.bar(x - width / 2, precision, width, label="Precision@K")
    ax.bar(x + width / 2, recall, width, label="Recall@K")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("K")
    ax.set_ylabel("Score")
    ax.set_title("Filter efficacy (CIFAR-10)")
    ax.legend()
    ax.grid(alpha=0.2, axis="y", linestyle=":")


def main() -> None:
    df = load_all_results(DATASET)
    df = df.sort_values(["method", "seed", "iter"], kind="stable")
    features = build_features(df)

    y_true, y_pred, r2_est, mae_est = compute_predictions(df, features)
    spear_est = spearman(y_true, y_pred)

    median_orig, median_re, median_delta_tff, delta_auc_pct = compute_best_curves(
        df, features, FLAGSHIP_METHOD, EARLY_REORDER
    )

    filter_df = evaluate_filter(DATASET, FILTER_KS)
    if "dataset" in filter_df.columns:
        filter_df = filter_df[filter_df["dataset"] == DATASET]

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("CIFAR-10 Proxy Optimization Snapshot", fontsize=18, fontweight="bold")

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.4, wspace=0.3)
    ax_scatter = fig.add_subplot(gs[0, :])
    ax_early = fig.add_subplot(gs[1, 0])
    ax_filter = fig.add_subplot(gs[1, 1])

    plot_scatter(ax_scatter, y_true, y_pred, r2_est, mae_est, spear_est)
    plot_early_efficiency(ax_early, median_orig, median_re, median_delta_tff)
    plot_filter_bars(ax_filter, filter_df)

    bullet_text = [
        f"• Proxy fit: R$^2$={r2_est:.3f}, MAE={mae_est:.4f}, Spearman={spear_est:.3f}",
        f"• {FLAGSHIP_METHOD} reorder: ΔAUAccI≈{delta_auc_pct:.2f}%",
    ]
    fig.text(0.02, 0.9, "Summary", fontsize=12, fontweight="bold")
    for i, line in enumerate(bullet_text):
        fig.text(0.04, 0.86 - i * 0.04, line, fontsize=11)

    fig.savefig(PDF_PATH, format="pdf")
    plt.close(fig)
    print(f"Saved one-pager to {PDF_PATH}")


if __name__ == "__main__":
    main()
