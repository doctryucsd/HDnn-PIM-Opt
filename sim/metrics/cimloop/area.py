from __future__ import annotations

import os
from logging import Logger
from typing import List

import pandas as pd

from ..metricArgs import MetricArgs

UNIT: float = 1.0  # mm^2


def get_area_df(cwd: str):
    lib_file = os.path.join(cwd, "lib/area.csv")
    return pd.read_csv(lib_file)


class Area:
    def __init__(self, args: MetricArgs):
        # unpack args
        model_args = args.model_args
        cwd = args.cwd

        # model args
        # self.num_classes: int = model_args["num_classes"]
        # self.input_size: int = model_args["input_size"]

    def evaluate(self, asic_area: float, reram_area: float, logger: Logger):
        # parameters
        area = asic_area + reram_area

        logger.info(f"{self.name()}: {area}mm^2")

        ret = self._normalize(area)

        metrics = {self.name(): (ret, 0.0)}

        return metrics

    @staticmethod
    def name() -> str:
        return "area"

    @staticmethod
    def optimization_type() -> str:
        return "min"

    @staticmethod
    def get_params() -> List[str]:
        return ["hd_dim", "npe1", "npe2", "reram_size", "kron"]

    @staticmethod
    def ref_point() -> float:
        return 1.0

    def _get_pe_area(self, num_pe: int, rank: int) -> float:
        return self.mac_area * num_pe * rank + self.adder_area * num_pe * rank

    def _normalize(self, area: float) -> float:
        BASE = 3000.0
        ret = area / BASE
        assert ret <= self.ref_point(), f"{ret} > {self.ref_point()}"
        return ret

    def _get_data(self, cwd: str):
        # look up table
        area_lut = get_area_df(cwd)

        def get_area_number(component: str) -> float:
            assert component in area_lut, f"{component} not found in reram LUT"
            return float(area_lut[component][0])

        self.mac_area = get_area_number("mac")
        self.adder_area = get_area_number("adder")
        self.array_area = get_area_number("array")
        self.wl_switch_area = get_area_number("wl_switch")
        self.sl_switch_area = get_area_number("sl_switch")
        self.decoder_area = get_area_number("decoder")
        self.adc_area = get_area_number("adc")
        self.shift_add_area = get_area_number("shift_add")
