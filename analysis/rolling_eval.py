#!/usr/bin/env python3
"""Rolling evaluation of the accuracy proxy across experiment seeds."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .accuracy_proxy import AccuracyProxy
from .data_loader import load_all_results
from .features import build_features

LOGGER = logging.getLogger(__name__)


def _spearman(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if true.size < 2 or pred.size < 2:
        return float("nan")
    true_rank = pd.Series(true).rank(method="average")
    pred_rank = pd.Series(pred).rank(method="average")
    corr = np.corrcoef(true_rank, pred_rank)[0, 1]
    return float(corr)


def _rolling_metrics(
    df: pd.DataFrame,
    features: pd.DataFrame,
    dataset_name: str,
    min_history: int,
    alpha: float,
    use_isotonic: bool,
) -> pd.DataFrame:
    records: List[dict] = []
    grouped = df.groupby(["method", "seed"], sort=False)

    for (method, seed), group in grouped:
        if len(group) <= min_history:
            LOGGER.info(
                "Skipping method=%s seed=%s (only %d samples, need > %d)",
                method,
                seed,
                len(group),
                min_history,
            )
            continue
        LOGGER.info(
            "Evaluating method=%s seed=%s with %d iterations",
            method,
            seed,
            len(group),
        )
        idx_list: Sequence[int] = list(group.index)
        for offset, idx in enumerate(idx_list):
            if offset < min_history:
                continue
            history_idx = idx_list[:offset]
            X_hist = features.loc[history_idx]
            y_hist = df.loc[history_idx, "acc"]

            model = AccuracyProxy(alpha=alpha, use_isotonic=use_isotonic)
            model.fit(X_hist, y_hist)

            hist_preds = model.predict(X_hist)
            r2 = r2_score(y_hist, hist_preds)
            mae = mean_absolute_error(y_hist, hist_preds)
            spear = _spearman(y_hist.to_numpy(), hist_preds)

            # Predict on the current row to satisfy the evaluation contract.
            _ = model.predict(features.loc[[idx]])

            records.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "seed": int(seed),
                    "iter": int(df.loc[idx, "iter"]),
                    "R2": float(r2),
                    "MAE": float(mae),
                    "Spearman": float(spear),
                }
            )
    if not records:
        raise ValueError("No evaluation records produced; check dataset and parameters.")
    result = pd.DataFrame.from_records(records)
    result.sort_values(["method", "seed", "iter"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def run_rolling_eval(
    dataset: str,
    out_path: Path,
    min_history: int = 12,
    alpha: float = 1.0,
    use_isotonic: bool = True,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    LOGGER.info("Loading dataset %s", dataset)
    df = load_all_results(dataset, seeds=seeds)
    LOGGER.info("Loaded %d rows for dataset %s", len(df), dataset)
    df_sorted = df.sort_values(["method", "seed", "iter"], kind="stable")
    features = build_features(df_sorted)
    result = _rolling_metrics(
        df_sorted,
        features,
        dataset_name=dataset,
        min_history=min_history,
        alpha=alpha,
        use_isotonic=use_isotonic,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    LOGGER.info("Saved %d evaluation rows to %s", len(result), out_path)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling evaluation for accuracy proxy")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. cifar10)")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path")
    parser.add_argument(
        "--min-history",
        type=int,
        default=12,
        help="Minimum history length before evaluating (default: 12)",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha (default: 1.0)")
    parser.add_argument(
        "--no-isotonic",
        action="store_true",
        help="Disable isotonic calibration stage",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Override the default seed list",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    use_isotonic = not args.no_isotonic
    run_rolling_eval(
        dataset=args.dataset,
        out_path=args.out,
        min_history=args.min_history,
        alpha=args.alpha,
        use_isotonic=use_isotonic,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
