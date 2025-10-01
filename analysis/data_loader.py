#!/usr/bin/env python3
"""Utilities to assemble constraint-scheduling experiment results."""
from __future__ import annotations

import re
import importlib.util
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

import pandas as pd

DEFAULT_SEEDS: Sequence[int] = tuple(range(144, 149))
SEED_PATTERN = re.compile(r"(?:^|_)seed(\d+)\b")
_DATA_ROOT = Path(__file__).resolve().parents[1] / "exps" / "constraint_scheduling"

_PARAM_EFFECTS_PATH = _DATA_ROOT / "param_effects.py"
_PARAM_EFFECTS_SPEC = importlib.util.spec_from_file_location(
    "constraint_param_effects", _PARAM_EFFECTS_PATH
)
if _PARAM_EFFECTS_SPEC is None or _PARAM_EFFECTS_SPEC.loader is None:
    raise ImportError(f"Unable to load param_effects.py from {_PARAM_EFFECTS_PATH}")
_PARAM_EFFECTS_MODULE = importlib.util.module_from_spec(_PARAM_EFFECTS_SPEC)
sys.modules[_PARAM_EFFECTS_SPEC.name] = _PARAM_EFFECTS_MODULE
_PARAM_EFFECTS_SPEC.loader.exec_module(_PARAM_EFFECTS_MODULE)

discover_files = _PARAM_EFFECTS_MODULE.discover_files
detect_accuracy_column = _PARAM_EFFECTS_MODULE.detect_accuracy_column
expand_structured_columns = _PARAM_EFFECTS_MODULE.expand_structured_columns
load_tabular_file = _PARAM_EFFECTS_MODULE.load_tabular_file


def _resolve_dataset_dir(dataset: str) -> Path:
    """Locate the dataset directory, preferring iter-tagged folders when present."""
    candidates = []
    if dataset.endswith(("_50iter", "_100iter")):
        candidates.append(dataset)
    else:
        candidates.extend([f"{dataset}_50iter", f"{dataset}_100iter", dataset])
    # Ensure we always check the user-specified name first.
    if dataset not in candidates:
        candidates.insert(0, dataset)
    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        path = _DATA_ROOT / name
        if not path.exists():
            continue
        files = [p for p in discover_files(path) if SEED_PATTERN.search(p.stem)]
        if files:
            return path
    raise FileNotFoundError(f"Could not find dataset '{dataset}' under {_DATA_ROOT}")


def _iter_result_files(dataset_dir: Path, seeds: Sequence[int]) -> Iterable[Path]:
    allowed = set(seeds)
    for path in discover_files(dataset_dir):
        match = SEED_PATTERN.search(path.stem)
        if not match:
            continue
        seed = int(match.group(1))
        if seed not in allowed:
            continue
        yield path


def _load_single_file(path: Path, dataset_name: str, method: str, seed: int) -> pd.DataFrame:
    raw = load_tabular_file(path)
    if raw.empty:
        return raw
    frame = expand_structured_columns(raw)
    frame = frame.reset_index(drop=True)
    frame.insert(0, "iter", frame.index)
    frame["dataset"] = dataset_name
    frame["method"] = method
    frame["seed"] = seed
    frame["__source_file__"] = str(path)

    try:
        accuracy_col = detect_accuracy_column(frame)
    except ValueError as exc:
        raise ValueError(f"{path} is missing an accuracy field") from exc
    frame["acc"] = frame[accuracy_col]
    if "eligible_count" in frame.columns and "feasible" not in frame.columns:
        frame["feasible"] = frame["eligible_count"].astype(float) > 0
    return frame


def _normalize_seeds(seeds: Optional[Sequence[int]]) -> Sequence[int]:
    if seeds is None:
        return tuple(DEFAULT_SEEDS)
    if isinstance(seeds, np.ndarray):
        seed_list = [int(s) for s in seeds.tolist()]
    else:
        seed_list = [int(s) for s in seeds]
    if not seed_list:
        raise ValueError("Seed list may not be empty")
    return tuple(seed_list)


def load_all_results(dataset: str, seeds: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Load a tidy DataFrame combining all seeds/methods for the requested dataset."""
    dataset_dir = _resolve_dataset_dir(dataset)
    selected_seeds = _normalize_seeds(seeds)
    frames = []
    for path in _iter_result_files(dataset_dir, selected_seeds):
        match = SEED_PATTERN.search(path.stem)
        if not match:
            continue
        seed = int(match.group(1))
        method = path.parent.name
        frame = _load_single_file(path, dataset_dir.name, method, seed)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError(
            f"No result files found for dataset '{dataset_dir.name}' with seeds {list(selected_seeds)}"
        )
    combined = pd.concat(frames, ignore_index=True, sort=False)

    actual_seeds = set(int(s) for s in combined["seed"].unique())
    expected = set(int(s) for s in selected_seeds)
    if seeds is None:
        if actual_seeds != set(DEFAULT_SEEDS):
            raise AssertionError(
                f"Default seed filter mismatch for {dataset}: expected {sorted(DEFAULT_SEEDS)}, got {sorted(actual_seeds)}"
            )
    else:
        if not actual_seeds.issubset(expected):
            raise AssertionError(
                f"Found seeds {sorted(actual_seeds)} outside requested set {sorted(expected)}"
            )
    return combined


if __name__ == "__main__":
    for ds in ("mnist", "fashion", "cifar10"):
        try:
            df = load_all_results(ds)
        except Exception as exc:  # pragma: no cover - smoke output only
            print(f"{ds}: failed ({exc})")
        else:
            unique_seeds = sorted(df["seed"].unique())
            print(f"{ds}: {df.shape[0]} rows x {df.shape[1]} cols (seeds={unique_seeds})")
