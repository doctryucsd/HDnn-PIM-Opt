from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from torch.utils.data import DataLoader


@dataclass
class MetricArgs:
    model_args: Dict[str, Any]
    training_args: Dict[str, Any]
    hardware_args: Dict[str, Any]
    train_loader: DataLoader
    test_loader: DataLoader
    cwd: str
