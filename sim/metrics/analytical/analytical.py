from __future__ import annotations

import math
from logging import Logger
from typing import Any, Dict, List, Tuple

from ..metricArgs import MetricArgs
from ..metricManager import MetricManager
from .accuracy import Accuracy
from .area import Area
from .performance import Performance
from .power import Power


class Analytical(MetricManager):
    def __init__(self, args: MetricArgs, device: str):
        self.args: MetricArgs = args
        self.area_evaluator = Area(args)
        self.performance_evaluator = Performance(args)
        self.power_evaluator = Power(args)
        self.accuracy_evaluator = Accuracy(args, device)

        # model args
        model_args = args.model_args
        self.input_size: int = model_args["input_size"]
        self.num_classes: int = model_args["num_classes"]

    def evaluate(
        self, params: Dict[str, Any], logger: Logger
    ) -> Dict[str, Tuple[float, float]]:
        # parameters
        hd_dim: int = params["hd_dim"]
        reram_size: int = params["reram_size"]
        kron: bool = params["kron"]

        # calculate number of ReRAMs for inference
        num_columns_per_class = math.ceil(hd_dim / reram_size)
        num_columns_all_classes = num_columns_per_class * self.num_classes
        num_rerams_inf = math.ceil(num_columns_all_classes / reram_size)

        if kron:
            num_rerams_enc = 0
        else:
            num_rram_cols_per_enc_row = math.ceil(self.input_size / reram_size)
            num_rram_cols_all_enc_rows = num_rram_cols_per_enc_row * hd_dim
            num_rerams_enc = math.ceil(num_rram_cols_all_enc_rows / reram_size)

        num_tiles_per_reram = reram_size // 8

        ret: Dict[str, Tuple[float, float]] = {}
        ret.update(self.area_evaluator.evaluate(params, num_rerams_inf, num_rerams_enc))
        peformance_dict, ops = self.performance_evaluator.evaluate(
            params, num_tiles_per_reram
        )
        ret.update(peformance_dict)
        ret.update(
            self.power_evaluator.evaluate(params, num_rerams_inf, num_rerams_enc, ops)
        )
        ret.update(self.accuracy_evaluator.evaluate(params, logger))
        return ret

    def get_optimizations_prop(self):
        ret: List[Tuple[str, str, float]] = []
        ret.append((Area.name(), Area.optimization_type(), Area.ref_point()))
        ret.append(
            (
                Performance.name(),
                Performance.optimization_type(),
                Performance.ref_point(),
            )
        )
        ret.append((Power.name(), Power.optimization_type(), Power.ref_point()))
        ret.append(
            (Accuracy.name(), Accuracy.optimization_type(), Accuracy.ref_point())
        )
        return ret
