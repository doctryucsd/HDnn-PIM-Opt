#!/usr/bin/env python3
"""Accuracy surrogate model utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - package import
    from .data_loader import load_all_results
    from .features import build_features
except ImportError:  # pragma: no cover - script execution fallback
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from analysis.data_loader import load_all_results
    from analysis.features import build_features


def _prepare_matrix(
    data: Iterable,
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Sequence[str]]:
    if isinstance(data, pd.DataFrame):
        if feature_names is not None:
            frame = data.reindex(columns=feature_names, fill_value=0.0)
            return frame.to_numpy(dtype=float), tuple(frame.columns)
        return data.to_numpy(dtype=float), tuple(data.columns)
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected 2D feature matrix")
    if feature_names is not None and array.shape[1] != len(feature_names):
        raise ValueError("Feature count mismatch")
    names = feature_names if feature_names is not None else tuple(f"x{i}" for i in range(array.shape[1]))
    return array, names


def _prepare_target(target: Iterable) -> np.ndarray:
    y = np.asarray(target, dtype=float)
    if y.ndim != 1:
        y = np.ravel(y)
    return y


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or y_pred.size < 2:
        return float("nan")
    true_rank = pd.Series(y_true).rank(method="average")
    pred_rank = pd.Series(y_pred).rank(method="average")
    corr = np.corrcoef(true_rank, pred_rank)[0, 1]
    return float(corr)


@dataclass
class AccuracyProxy:
    alpha: float = 1.0
    use_isotonic: bool = True
    min_isotonic_samples: int = 40
    _scaler: StandardScaler = field(init=False, repr=False)
    _ridge: Ridge = field(init=False, repr=False)
    _iso: Optional[IsotonicRegression] = field(init=False, default=None, repr=False)
    _feature_names: Tuple[str, ...] = field(init=False, default_factory=tuple, repr=False)
    _fitted: bool = field(init=False, default=False, repr=False)

    def fit(self, X: Iterable, y: Iterable) -> "AccuracyProxy":
        X_values, names = _prepare_matrix(X)
        y_values = _prepare_target(y)
        if X_values.shape[0] != y_values.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X_values.shape[0] < 2:
            raise ValueError("Need at least two samples to fit")
        if not np.isfinite(X_values).all():
            raise ValueError("Features must be finite")
        if not np.isfinite(y_values).all():
            raise ValueError("Targets must be finite")

        self._feature_names = tuple(names)

        self._scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = self._scaler.fit_transform(X_values)

        self._ridge = Ridge(alpha=self.alpha, solver="auto")
        self._ridge.fit(X_scaled, y_values)

        self._iso = None
        if self.use_isotonic and X_values.shape[0] >= self.min_isotonic_samples:
            ridge_preds = self._ridge.predict(X_scaled)
            self._iso = IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(ridge_preds, y_values)

        self._fitted = True
        return self

    def predict(self, X: Iterable) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("AccuracyProxy must be fit before predicting")
        X_values, _ = _prepare_matrix(X, self._feature_names)
        X_scaled = self._scaler.transform(X_values)
        preds = self._ridge.predict(X_scaled)
        if self._iso is not None:
            preds = self._iso.predict(preds)
        return preds


def _demo() -> None:
    df = load_all_results("cifar10")
    X = build_features(df)
    y = df["acc"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = AccuracyProxy(alpha=1.0, use_isotonic=True, min_isotonic_samples=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    spear = _spearman(y_test, preds)

    seeds = sorted(df["seed"].unique())
    print(f"AccuracyProxy demo (CIFAR-10, seeds={seeds})")
    print(f"R^2: {r2:.3f}  MAE: {mae:.4f}  Spearman: {spear:.3f}")


if __name__ == "__main__":
    _demo()
