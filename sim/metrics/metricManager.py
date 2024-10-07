from __future__ import annotations

from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Dict, List, Tuple


class MetricManager(ABC):
    @abstractmethod
    def evaluate(
        self, params: Dict[str, Any], logger: Logger
    ) -> Dict[str, Tuple[float, float]]: ...

    @abstractmethod
    def get_optimizations_prop(self) -> List[Tuple[str, str, float]]: ...
