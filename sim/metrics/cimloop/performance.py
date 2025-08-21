from __future__ import annotations

import math
from logging import Logger
from typing import Any, Dict, List, Tuple

from ..metricArgs import MetricArgs

UNIT: float = 1  # us -> us


class Performance:
    def __init__(self, args: MetricArgs):
        # unpack args
        model_args = args.model_args

        # model args
        # self.num_classes: int = model_args["num_classes"]
        # self.input_size: int = model_args["input_size"]

    def evaluate(self, asic_delay: float, reram_delay: float, logger: Logger):
        total_delay = asic_delay + reram_delay

        performance = total_delay * UNIT

        logger.info(f"{self.name()}: {performance}us")

        ret = self._normalize(performance)
        return (ret, 0.0)

    @staticmethod
    def name() -> str:
        return "performance"

    @staticmethod
    def optimization_type() -> str:
        return "min"

    @staticmethod
    def get_params() -> List[str]:
        return [
            "hd_dim",
            "npe1",
            "npe2",
            "reram_size",
            "kron",
            "f1",
            "d1",
            "frequency",
        ]

    @staticmethod
    def ref_point() -> float:
        return 1.0

    def _get_kron_cycle(
        self, f1: int, f2: int, d1: int, d2: int, npe1: int, npe2: int
    ) -> Tuple[int, int, int]:
        pe1_cycle = f2 * math.ceil(f1 / npe1)
        pe2_cycle = d2 * math.ceil(f2 / npe2) + math.ceil(d2 / npe2)
        dominate = max(pe1_cycle, pe2_cycle)
        total_cycle = pe1_cycle + pe2_cycle + (d1 - 1) * dominate
        return total_cycle, pe1_cycle * d1, pe2_cycle * d1

    def _normalize(self, performance: float) -> float:
        BASE: float = 3000.0
        ret: float = performance / BASE
        if ret > self.ref_point():
            ret = self.ref_point()
        # assert (
        #     ret <= self.ref_point()
        # ), f"performance: {ret} > ref_point: {self.ref_point()}"
        return ret
