from __future__ import annotations

from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Dict, List, Tuple

from torch import Tensor


class MetricManager(ABC):
    @abstractmethod
    def evaluate(
        self, params: Dict[str, Any], rank: int, shared_tensor: Tensor, logger: Logger
    ) -> None: ...

    @abstractmethod
    def get_optimizations_prop(self) -> List[Tuple[str, str, float]]: ...

    @abstractmethod
    def get_metric_names(self) -> List[str]: ...
