#!/usr/bin/env python3
"""Counterfactual reordering analysis for proxy-guided selection."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .accuracy_proxy import AccuracyProxy
from .data_loader import load_all_results
from .features import build_features

LOGGER = logging.getLogger(__name__)


def _compute_tff(feasible: Iterable[bool]) -> float:
    for idx, value in enumerate(feasible, start=1):
        if bool(value):
            return float(idx)
    return float("nan")


def _compute_auacci(accuracy: Iterable[float]) -> float:
    arr = np.asarray(list(accuracy), dtype=float)
    if arr.size == 0:
        return float("nan")
    best = np.maximum.accumulate(arr)
    return float(best.mean())


def _prepare_sequences(
    group_df: pd.DataFrame,
    preds: pd.Series,
    early: int,
) -> Tuple[List[int], List[int]]:
    orig_order = list(group_df.index)
    early_n = min(early, len(orig_order))
    early_indices = orig_order[:early_n]
    rest_indices = orig_order[early_n:]
    reordered_early = sorted(early_indices, key=lambda idx: preds[idx], reverse=True)
    reorder_order = reordered_early + rest_indices
    return orig_order, reorder_order


def _analyze_group(
    group_df: pd.DataFrame,
    features: pd.DataFrame,
    early: int,
    use_feasible: bool,
) -> dict:
    method = group_df["method"].iloc[0]
    seed = int(group_df["seed"].iloc[0])

    model = AccuracyProxy(alpha=1.0, use_isotonic=True)
    model.fit(features, group_df["acc"])
    preds = pd.Series(model.predict(features), index=group_df.index)

    orig_order, reorder_order = _prepare_sequences(group_df, preds, early)

    acc_orig = group_df.loc[orig_order, "acc"].to_numpy(dtype=float)
    acc_re = group_df.loc[reorder_order, "acc"].to_numpy(dtype=float)

    auacci_orig = _compute_auacci(acc_orig)
    auacci_re = _compute_auacci(acc_re)
    delta_auacci_pct = (
        (auacci_re - auacci_orig) / auacci_orig * 100.0 if auacci_orig != 0 else float("nan")
    )

    result = {
        "method": method,
        "seed": seed,
        "AUAccI_orig": auacci_orig,
        "AUAccI_re": auacci_re,
        "Delta_AUAccI_pct": delta_auacci_pct,
    }

    if use_feasible:
        feas_orig = group_df.loc[orig_order, "feasible"].astype(bool).to_list()
        feas_re = group_df.loc[reorder_order, "feasible"].astype(bool).to_list()
        tff_orig = _compute_tff(feas_orig)
        tff_re = _compute_tff(feas_re)
        delta_tff = tff_re - tff_orig if not np.isnan(tff_orig) and not np.isnan(tff_re) else float("nan")
        result.update(
            {
                "TFF_orig": tff_orig,
                "TFF_re": tff_re,
                "Delta_TFF": delta_tff,
            }
        )

    return result


def evaluate_reordering(dataset: str, early: int, seeds: Sequence[int] | None = None) -> pd.DataFrame:
    LOGGER.info("Loading dataset %s", dataset)
    df = load_all_results(dataset, seeds=seeds)
    df = df.sort_values(["method", "seed", "iter"], kind="stable")
    features = build_features(df)

    use_feasible = "feasible" in df.columns
    if not use_feasible:
        LOGGER.warning("Dataset %s lacks feasibility information; skipping TFF metrics", dataset)

    records: List[dict] = []

    grouped = df.groupby(["method", "seed"], sort=False)
    for (method, seed), group in grouped:
        LOGGER.info("Analyzing method=%s seed=%s", method, seed)
        group_features = features.loc[group.index]
        record = _analyze_group(group, group_features, early=early, use_feasible=use_feasible)
        record.update({"dataset": dataset})
        records.append(record)

    result = pd.DataFrame.from_records(records)
    order_cols = ["dataset", "method", "seed"]
    if use_feasible:
        order_cols += ["TFF_orig", "TFF_re", "Delta_TFF"]
    order_cols += ["AUAccI_orig", "AUAccI_re", "Delta_AUAccI_pct"]
    result = result[order_cols]
    result.sort_values(["method", "seed"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual reordering evaluation")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--early", type=int, required=True, help="Number of early iterations to reorder")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
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
    result = evaluate_reordering(args.dataset, args.early, seeds=args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    LOGGER.info("Saved counterfactual metrics to %s", args.out)


if __name__ == "__main__":
    main()
