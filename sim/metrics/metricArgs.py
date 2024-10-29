from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from torch.utils.data import Dataset


@dataclass
class MetricArgs:
    model_args: Dict[str, Any]
    training_args: Dict[str, Any]
    hardware_args: Dict[str, Any]
    data_args: Dict[str, Any]
    cwd: str
