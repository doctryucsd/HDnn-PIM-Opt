from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..metricArgs import MetricArgs

POWER_UNIT: float = 1e12  # pW -> W
ENERGY_UNIT: float = 1e6  # pJ -> muJ


def get_power_df(cwd: str):
    lib_file = os.path.join(cwd, "lib/power.csv")
    return pd.read_csv(lib_file)


class Power:
    def __init__(self, args: MetricArgs):
        # unpack args
        model_args = args.model_args
        hardware_args = args.hardware_args
        cwd = args.cwd

        # model args
        self.num_classes: int = model_args["num_classes"]
        self.input_size: int = model_args["input_size"]

        # hardware args
        self.power: str = hardware_args["power"]

        self._get_data(cwd)

    def _get_data(self, cwd: str):
        # look up table
        power_lut = get_power_df(cwd)

        def get_power_number(component: str) -> float:
            assert component in power_lut, f"{component} not found in reram LUT"
            return float(power_lut[component][0])

        self.mac_power = get_power_number("mac")
        self.adder_power = get_power_number("adder")
        self.array_power = get_power_number("array")
        self.wl_switch_power = get_power_number("wl_switch")
        self.sl_switch_power = get_power_number("sl_switch")
        self.decoder_power = get_power_number("decoder")
        self.adc_power = get_power_number("adc")
        self.shift_add_power = get_power_number("shift_add")

    def evaluate(
        self,
        params: Dict[str, Any],
        num_rerams_inf: int,
        num_rerams_enc: int,
        ops: List[int],
    ):
        # parameters
        kron: bool = params["kron"]
        npe1: int = params["npe1"]
        npe2: int = params["npe2"]
        reram_size: int = params["reram_size"]
        frequency: int = params["frequency"]

        # calculate power
        encoder_power, encoder_energy = self._get_encoder_power(
            kron, npe1, npe2, 1, num_rerams_enc, reram_size, ops
        )
        hd_inference_power, hd_inference_energy = self._get_hd_inference_power(
            reram_size, num_rerams_inf, ops
        )
        power_per_op = encoder_power + hd_inference_power
        power = power_per_op * frequency / POWER_UNIT
        energy = (encoder_energy + hd_inference_energy) / ENERGY_UNIT

        if self.power == "energy":
            metrics = {self.name(): (energy, 0.0)}
        elif self.power == "power":
            metrics = {self.name(): (power, 0.0)}
        return metrics

    @staticmethod
    def name() -> str:
        return "power"

    @staticmethod
    def optimization_type() -> str:
        return "min"

    @staticmethod
    def get_params() -> List[str]:
        return ["hd_dim", "npe1", "npe2", "reram_size", "kron", "frequency"]

    @staticmethod
    def ref_point() -> float:
        return 2.0

    def _get_reram_power(self, dim: int) -> float:
        array_power = self.array_power * (dim**2) / (128**2)
        adc_power = self.adc_power * dim / 128
        shift_add_power = self.shift_add_power * dim / 128

        return (
            array_power
            + self.wl_switch_power
            + self.sl_switch_power
            + self.decoder_power
            + adc_power
            + shift_add_power
        )

    def _get_pe_power(self, num_pe: int, rank: int) -> float:
        return self.mac_power * num_pe * rank + self.adder_power * num_pe * rank

    def _get_encoder_power(
        self,
        kron: bool,
        npe1: int,
        npe2: int,
        rank: int,
        num_rerams_enc: int,
        reram_size: int,
        ops: List[int],
    ) -> Tuple[float, float]:
        if kron:
            # kronecker product of PE1 and PE2
            pe1_power = self._get_pe_power(npe1, rank)
            pe2_power = self._get_pe_power(npe2, rank)

            assert len(ops) == 3, "ops must be provided for different kron stages"
            pe1_energy = pe1_power * ops[0]
            pe2_energy = pe2_power * ops[1]
            return pe1_power + pe2_power, pe1_energy + pe2_energy
        else:
            assert len(ops) == 2, "ops must be provided for ReRAM encoding stage"
            assert num_rerams_enc > 0, "num_rerams_enc must be provided for non-kron"
            reram_power = self._get_reram_power(reram_size) * num_rerams_enc
            reram_energy = reram_power * ops[0]
            return reram_power, reram_energy

    def _get_hd_inference_power(
        self, reram_size: int, num_reram_inf: int, ops: List[int]
    ) -> Tuple[float, float]:
        energy_per_op = self._get_reram_power(reram_size) * num_reram_inf
        energy = energy_per_op * ops[-1]
        return energy_per_op, energy
