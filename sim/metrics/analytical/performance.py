from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from ..metricArgs import MetricArgs

UNIT: float = 1e6  # s -> mus


class Performance:
    def __init__(self, args: MetricArgs):
        # unpack args
        model_args = args.model_args

        # model args
        self.num_classes: int = model_args["num_classes"]
        self.input_size: int = model_args["input_size"]

    def evaluate(self, params: Dict[str, Any], num_tiles_per_reram: int):
        # parameters
        hd_dim = params["hd_dim"]
        npe1: int = params["npe1"]
        npe2: int = params["npe2"]
        reram_size: int = params["reram_size"]
        frequency: int = params["frequency"]
        kron: bool = params["kron"]
        f1: int = params["f1"]
        d1: int = params["d1"]
        d2 = hd_dim // d1
        f2 = self.input_size // f1

        ops: List[int] = []

        # encoder
        if kron:
            encoder_total_cycle, pe1_ops, pe2_ops = self._get_kron_cycle(
                f1, f2, d1, d2, npe1, npe2
            )
            ops.append(pe1_ops)
            ops.append(pe2_ops)
        else:
            encoder_total_cycle = self._get_reram_cycle(
                self.input_size, hd_dim, reram_size, num_tiles_per_reram
            )
            ops.append(encoder_total_cycle)

        # hd inference
        inf_total_cycle = self._get_reram_cycle(
            hd_dim, self.num_classes, reram_size, num_tiles_per_reram
        )
        ops.append(inf_total_cycle)
        total_cycle = max(encoder_total_cycle, inf_total_cycle)

        performance = total_cycle / frequency * UNIT
        return {self.name(): (performance, 0.0)}, ops

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
        return 500.0

    def _get_reram_cycle(
        self, row_size: int, column_size: int, reram_size: int, num_tiles_per_reram: int
    ) -> int:
        # try to fit all numbers in hypervector into tiles, a tile is num_BL x num_WL
        # 8 is the number of ADCs in a reram subarray
        tile_size = reram_size * 8
        num_tiles_per_row = math.ceil(row_size / tile_size)
        num_cycles_all_rows = num_tiles_per_row * column_size
        return min(num_cycles_all_rows, num_tiles_per_reram) * 2

    def _get_hd_inference_reram_cycle(self, hd_dim: int, reram_size: int) -> int:
        # try to fit all numbers in hypervector into tiles, a tile is num_BL x num_WL
        # 8 is the number of ADCs in a reram subarray
        tile_size = reram_size * 8
        num_tiles_per_class = math.ceil(hd_dim / tile_size)
        num_cycles_all_classes = num_tiles_per_class * self.input_size
        return num_cycles_all_classes

    def _get_kron_cycle(
        self, f1: int, f2: int, d1: int, d2: int, npe1: int, npe2: int
    ) -> Tuple[int, int, int]:
        pe1_cycle = f2 * math.ceil(f1 / npe1)
        pe2_cycle = d2 * math.ceil(f2 / npe2) + math.ceil(d2 / npe2)
        dominate = max(pe1_cycle, pe2_cycle)
        total_cycle = pe1_cycle + pe2_cycle + (d1 - 1) * dominate
        return total_cycle, pe1_cycle * d1, pe2_cycle * d1
