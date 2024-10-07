from __future__ import annotations

import os
from typing import Any, Dict, List

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
        self.num_classes: int = model_args["num_classes"]
        self.input_size: int = model_args["input_size"]

        self._get_data(cwd)

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

    def evaluate(
        self, params: Dict[str, Any], num_rerams_inf: int, num_rerams_enc: int | None
    ):
        # parameters
        kron: bool = params["kron"]
        npe1: int = params["npe1"]
        npe2: int = params["npe2"]
        reram_size: int = params["reram_size"]

        # calculate area
        encoder_area = self._get_encoder_area(
            kron, npe1, npe2, 1, num_rerams_enc, reram_size
        )
        hd_inference_area = self._get_hd_inference_area(reram_size, num_rerams_inf)
        area = encoder_area + hd_inference_area
        metrics = {self.name(): (area, 0.0)}

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
        return 10.0

    def _get_reram_area(self, dim: int) -> float:
        array_area = self.array_area * (dim**2) / (128**2)
        adc_area = self.adc_area * dim / 128
        shift_add_area = self.shift_add_area * dim / 128

        return (
            array_area
            + self.wl_switch_area
            + self.sl_switch_area
            + self.decoder_area
            + adc_area
            + shift_add_area
        )

    def _get_pe_area(self, num_pe: int, rank: int) -> float:
        return self.mac_area * num_pe * rank + self.adder_area * num_pe * rank

    def _get_encoder_area(
        self,
        kron: bool,
        npe1: int,
        npe2: int,
        rank: int,
        num_rerams_enc: int | None,
        reram_size: int,
    ) -> float:
        if kron:
            # kronecker product of PE1 and PE2
            pe1_area = self._get_pe_area(npe1, rank)
            pe2_area = self._get_pe_area(npe2, rank)
            return pe1_area + pe2_area
        else:
            assert (
                num_rerams_enc is not None
            ), "num_rerams_enc must be provided for non-encoding"
            reram_area = self._get_reram_area(reram_size) * num_rerams_enc
            return reram_area

    def _get_hd_inference_area(self, reram_size: int, num_reram_inf: int) -> float:
        return self._get_reram_area(reram_size) * num_reram_inf
