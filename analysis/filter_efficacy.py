#!/usr/bin/env python3
"""Evaluate how well the accuracy proxy filters high-performing designs."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from .accuracy_proxy import AccuracyProxy
from .data_loader import load_all_results
from .features import build_features

LOGGER = logging.getLogger(__name__)


def _prepare_labels(target: np.ndarray, top_fraction: float = 0.25) -> np.ndarray:
    if target.ndim != 1:
        raise ValueError("Target must be 1-D")
    count = int(np.ceil(len(target) * top_fraction))
    if count <= 0:
        raise ValueError("No positive examples available")
    sorted_indices = np.argsort(target)[::-1]
    threshold_value = target[sorted_indices[count - 1]]
    labels = (target >= threshold_value).astype(int)
    positives = labels.sum()
    if positives == 0:
        raise ValueError("Could not determine positive examples for threshold")
    return labels


def _precision_recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    if k <= 0:
        raise ValueError("K must be positive")
    order = np.argsort(scores)[::-1]
    selected = order[:k]
    hits = labels[selected].sum()
    precision = hits / k
    recall = hits / labels.sum()
    return precision, recall


def _enrichment_factor(precision: float, baseline_positive_rate: float) -> float:
    if baseline_positive_rate <= 0:
        return float("nan")
    return precision / baseline_positive_rate


def evaluate_filter(
    dataset: str,
    ks: Sequence[int],
    min_history: int = 12,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    LOGGER.info("Loading dataset %s", dataset)
    df = load_all_results(dataset, seeds=seeds)
    df = df.sort_values(["method", "seed", "iter"], kind="stable")
    features = build_features(df)
    y = df["acc"].to_numpy(dtype=float)

    LOGGER.info("Fitting proxy on %d samples", len(df))
    proxy = AccuracyProxy(alpha=1.0, use_isotonic=True)
    proxy.fit(features, y)
    scores = proxy.predict(features)

    labels = _prepare_labels(y)
    positives = labels.sum()
    base_rate = positives / len(labels)
    LOGGER.info("Identified %d good samples (base rate %.3f)", positives, base_rate)

    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    records = []
    for k in ks:
        if k > len(scores):
            LOGGER.warning("Requested K=%d exceeds dataset size (%d); trimming.", k, len(scores))
            k = len(scores)
        prec_at_k, rec_at_k = _precision_recall_at_k(labels, scores, k)
        ef_at_k = _enrichment_factor(prec_at_k, base_rate)
        records.append(
            {
                "dataset": dataset,
                "K": int(k),
                "Precision@K": float(prec_at_k),
                "Recall@K": float(rec_at_k),
                "EF@K": float(ef_at_k),
                "PR_AUC": float(pr_auc),
            }
        )
    result = pd.DataFrame.from_records(records)
    result.sort_values("K", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate proxy filtering efficacy")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path")
    parser.add_argument("--k", nargs="+", type=int, required=True, help="K values for metrics")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Override default seeds",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    result = evaluate_filter(args.dataset, args.k, seeds=args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    LOGGER.info("Saved efficacy metrics to %s", args.out)


if __name__ == "__main__":
    main()
