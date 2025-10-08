#!/usr/bin/env python3
"""Aggregate proxy quality metrics for inline LaTeX reporting."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - import shim for script execution
    from analysis.accuracy_proxy import AccuracyProxy
    from analysis.data_loader import DEFAULT_SEEDS, load_all_results
    from analysis.features import build_features
except ImportError:  # pragma: no cover - script execution fallback
    import sys

    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from analysis.accuracy_proxy import AccuracyProxy
    from analysis.data_loader import DEFAULT_SEEDS, load_all_results
    from analysis.features import build_features


DATASETS = ("mnist", "fashion", "cifar10")
TOP_FRACTION = 0.25
DEFAULT_K = 30


def _compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or y_pred.size < 2:
        return float("nan")
    true_rank = pd.Series(y_true).rank(method="average")
    pred_rank = pd.Series(y_pred).rank(method="average")
    matrix = np.corrcoef(true_rank, pred_rank)
    return float(matrix[0, 1])


def _top_quartile_labels(values: np.ndarray, top_fraction: float = TOP_FRACTION) -> tuple[np.ndarray, float]:
    if values.ndim != 1:
        values = np.ravel(values)
    if values.size == 0:
        raise ValueError("Cannot determine quartile labels without values")
    count = max(1, math.ceil(values.size * top_fraction))
    sorted_vals = np.sort(values)[::-1]
    threshold = float(sorted_vals[count - 1])
    labels = (values >= threshold).astype(int)
    return labels, threshold


def _enrichment_factor_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if labels.size == 0 or scores.size == 0:
        return float("nan")
    order = np.argsort(scores)[::-1]
    k_eff = min(k, order.size)
    if k_eff == 0:
        return float("nan")
    selected = order[:k_eff]
    hits = labels[selected].sum()
    precision = hits / k_eff
    base_rate = labels.mean()
    if base_rate <= 0:
        return float("nan")
    return float(precision / base_rate)


def _compute_tff(flags: Iterable[bool]) -> float:
    for idx, value in enumerate(flags, start=1):
        if bool(value):
            return float(idx)
    return float("nan")


def _compute_tqf(values: Iterable[float], threshold: float) -> float:
    for idx, value in enumerate(values, start=1):
        if float(value) >= threshold:
            return float(idx)
    return float("nan")


def _compute_auacci(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    running_best = np.maximum.accumulate(arr)
    return float(running_best.mean())


def _detect_flagship_method(df: pd.DataFrame, override: str | None) -> str:
    methods = df["method"].astype(str)
    unique = methods.unique()
    if override is not None:
        if override not in unique:
            valid = ", ".join(sorted(unique))
            raise ValueError(f"Flagship override '{override}' not found. Available: {valid}")
        return override
    lower_map = {m: m.lower() for m in unique}
    for method, lowered in lower_map.items():
        if "scheduling-nevhi" in lowered:
            return method
    scheduling_candidates = {m: methods.eq(m).sum() for m, lowered in lower_map.items() if "scheduling" in lowered}
    if scheduling_candidates:
        return max(scheduling_candidates, key=scheduling_candidates.get)
    counts = methods.value_counts()
    return counts.idxmax()


def _evaluate_flagship_counterfactual(
    df: pd.DataFrame,
    preds: pd.Series,
    method: str,
    early: int,
    top_quartile_threshold: float,
) -> dict:
    method_df = df[df["method"] == method]
    if method_df.empty:
        raise ValueError(f"No rows found for flagship method '{method}'")
    method_df = method_df.sort_values(["seed", "iter"], kind="stable")
    has_feasible = "feasible" in method_df.columns

    records: list[dict] = []
    for seed, group in method_df.groupby("seed", sort=False):
        group = group.sort_values("iter", kind="stable")
        indices = list(group.index)
        early_n = min(early, len(indices))
        early_indices = indices[:early_n]
        rest_indices = indices[early_n:]
        reordered_early = sorted(early_indices, key=lambda idx: preds.loc[idx], reverse=True)
        reordered_indices = reordered_early + rest_indices

        acc_orig = group.loc[indices, "acc"].to_numpy(dtype=float)
        acc_re = group.loc[reordered_indices, "acc"].to_numpy(dtype=float)
        auacci_orig = _compute_auacci(acc_orig)
        auacci_re = _compute_auacci(acc_re)
        delta_auacci_pct = (
            (auacci_re - auacci_orig) / auacci_orig * 100.0 if auacci_orig != 0 else float("nan")
        )
        record = {
            "seed": int(seed),
            "AUAccI_orig": auacci_orig,
            "AUAccI_re": auacci_re,
            "Delta_AUAccI_pct": delta_auacci_pct,
        }

        if has_feasible:
            feas_orig = group.loc[indices, "feasible"].astype(bool).to_list()
            feas_re = group.loc[reordered_indices, "feasible"].astype(bool).to_list()
            tff_orig = _compute_tff(feas_orig)
            tff_re = _compute_tff(feas_re)
            delta_tff = (
                tff_re - tff_orig if not (np.isnan(tff_orig) or np.isnan(tff_re)) else float("nan")
            )
            record.update({"T_orig": tff_orig, "T_re": tff_re, "Delta_T": delta_tff})
        else:
            tqf_orig = _compute_tqf(acc_orig, top_quartile_threshold)
            tqf_re = _compute_tqf(acc_re, top_quartile_threshold)
            delta_tqf = (
                tqf_re - tqf_orig if not (np.isnan(tqf_orig) or np.isnan(tqf_re)) else float("nan")
            )
            record.update({"T_orig": tqf_orig, "T_re": tqf_re, "Delta_T": delta_tqf})

        records.append(record)

    flag_df = pd.DataFrame.from_records(records)
    delta_time = float(flag_df["Delta_T"].mean(skipna=True)) if not flag_df.empty else float("nan")
    delta_auacci_pct = (
        float(flag_df["Delta_AUAccI_pct"].mean(skipna=True)) if not flag_df.empty else float("nan")
    )
    return {
        "flagship_method": method,
        "delta_time": delta_time,
        "delta_auacci_pct": delta_auacci_pct,
    }


def _analyze_dataset(
    dataset: str,
    seeds: Sequence[int],
    flagship_override: str | None,
    early: int,
) -> dict:
    df = load_all_results(dataset, seeds=seeds)
    df = df.sort_values(["method", "seed", "iter"], kind="stable")
    features = build_features(df)
    proxy = AccuracyProxy(alpha=1.0, use_isotonic=True)
    proxy.fit(features, df["acc"])
    preds = pd.Series(proxy.predict(features), index=df.index)

    y = df["acc"].to_numpy(dtype=float)
    rho = _compute_spearman(y, preds.to_numpy(dtype=float))

    labels, threshold = _top_quartile_labels(y, top_fraction=TOP_FRACTION)
    ef30 = _enrichment_factor_at_k(labels, preds.to_numpy(dtype=float), DEFAULT_K)

    result = {
        "dataset": dataset,
        "rho": rho,
        "ef30": ef30,
        "threshold": threshold,
    }

    if dataset == "cifar10":
        flagship = _detect_flagship_method(df, flagship_override)
        flag_metrics = _evaluate_flagship_counterfactual(
            df, preds, flagship, early=early, top_quartile_threshold=threshold
        )
        result.update(flag_metrics)
    return result


def _format_float(value: float, digits: int = 3, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "nan"
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return fmt.format(value)


def _build_macro_lines(metrics: dict) -> list[str]:
    delta_pct = _format_float(metrics["cifar10"]["delta_auacci_pct"], digits=2, signed=True)
    tex_lines = [
        f"\\newcommand{{\\ProxyRhoMNIST}}{{{_format_float(metrics['mnist']['rho'])}}}",
        f"\\newcommand{{\\ProxyRhoFashion}}{{{_format_float(metrics['fashion']['rho'])}}}",
        f"\\newcommand{{\\ProxyRhoCIFAR}}{{{_format_float(metrics['cifar10']['rho'])}}}",
        f"\\newcommand{{\\ProxyEFThirtyMNIST}}{{{_format_float(metrics['mnist']['ef30'])}}}",
        f"\\newcommand{{\\ProxyEFThirtyFashion}}{{{_format_float(metrics['fashion']['ef30'])}}}",
        f"\\newcommand{{\\ProxyEFThirtyCIFAR}}{{{_format_float(metrics['cifar10']['ef30'])}}}",
        f"\\newcommand{{\\ProxyDeltaTFFCF}}{{{_format_float(metrics['cifar10']['delta_time'])}}}",
        f"\\newcommand{{\\ProxyDeltaAUAccICF}}{{{delta_pct}\\%}}",
    ]
    return tex_lines


def _write_outputs(
    tex_path: Path,
    csv_path: Path,
    metrics: dict,
    summary_rows: list[dict],
) -> None:
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    lines = _build_macro_lines(metrics)
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    df = pd.DataFrame(summary_rows)
    df.to_csv(csv_path, index=False)

    for line in lines:
        print(line)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate proxy inline metrics")
    parser.add_argument(
        "--flagship-method",
        help="Override flagship method detection for CIFAR-10",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Override the default seed list (defaults to 144-148)",
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=Path("results/tex/proxy_inline.tex"),
        help="TeX output path",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("results/tables/proxy_inline_summary.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--reorder-early",
        type=int,
        default=DEFAULT_K,
        help="Number of early iterations subject to reordering",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    seeds = tuple(args.seeds) if args.seeds else tuple(DEFAULT_SEEDS)

    metrics: dict[str, dict] = {}
    summary_rows: list[dict] = []
    for dataset in DATASETS:
        override = args.flagship_method if dataset == "cifar10" else None
        ds_metrics = _analyze_dataset(dataset, seeds=seeds, flagship_override=override, early=args.reorder_early)
        metrics[dataset] = ds_metrics
        summary_rows.append(
            {
                "dataset": dataset,
                "spearman_rho": ds_metrics["rho"],
                "ef_at_30": ds_metrics["ef30"],
                "flagship_method": ds_metrics.get("flagship_method", ""),
                "delta_time": ds_metrics.get("delta_time", float("nan")),
                "delta_auacci_pct": ds_metrics.get("delta_auacci_pct", float("nan")),
            }
        )

    _write_outputs(args.tex_out, args.csv_out, metrics, summary_rows)


if __name__ == "__main__":
    main()
