#!/usr/bin/env python3
"""Feature engineering helpers for constraint-scheduling results."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

MAIN_EFFECTS = [
    "hd_dim",
    "inner_dim",
    "reram_size",
    "out_channels_1",
    "out_channels_2",
    "kernel_size_1",
    "kernel_size_2",
    "stride_1",
    "stride_2",
    "cnn_x_dim_1",
    "cnn_y_dim_1",
    "cnn_x_dim_2",
    "cnn_y_dim_2",
    "encoder_x_dim",
    "encoder_y_dim",
    "frequency",
    "kron",
    "padding_1",
    "padding_2",
    "dilation_1",
    "dilation_2",
]

INTERACTIONS: Dict[str, Tuple[str, str]] = {
    "kernel_size_pair": ("kernel_size_1", "kernel_size_2"),
    "stride_pair": ("stride_1", "stride_2"),
    "channel_pair": ("out_channels_1", "out_channels_2"),
    "tile_area_1": ("cnn_x_dim_1", "cnn_y_dim_1"),
    "tile_area_2": ("cnn_x_dim_2", "cnn_y_dim_2"),
    "encoder_area": ("encoder_x_dim", "encoder_y_dim"),
}

LOG_FEATURES = {
    "hd_dim": "log_hd_dim",
    "inner_dim": "log_inner_dim",
    "reram_size": "log_reram_size",
    "frequency": "log_frequency",
    "tile_area_1": "log_tile_area_1",
    "tile_area_2": "log_tile_area_2",
    "encoder_area": "log_encoder_area",
}


def _ensure_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype(float, copy=False)


def _safe_log1p(series: pd.Series) -> pd.Series:
    clipped = series.clip(lower=0)
    return np.log1p(clipped)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create numeric features with hierarchy-respecting interactions."""
    features = pd.DataFrame(index=df.index)

    for name in MAIN_EFFECTS:
        if name in df.columns:
            series = _ensure_numeric(df[name])
            if series.notna().any():
                features[name] = series.fillna(0.0)

    for feat_name, (a, b) in INTERACTIONS.items():
        if a in features.columns and b in features.columns:
            interaction = features[a] * features[b]
            features[feat_name] = interaction

    for source, output in LOG_FEATURES.items():
        if source in features.columns:
            features[output] = _safe_log1p(features[source])

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features.astype(float)


def _run_self_test() -> None:
    try:
        from .data_loader import load_all_results
    except ImportError:  # pragma: no cover - direct execution fallback
        from analysis.data_loader import load_all_results

    df = load_all_results("cifar10")
    feats = build_features(df)
    seeds = sorted(df["seed"].unique())
    print(f"features shape: {feats.shape[0]} rows x {feats.shape[1]} cols (seeds={seeds})")


if __name__ == "__main__":
    _run_self_test()
