from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd

from ..metricArgs import MetricArgs

ENERGY_UNIT: float = 1e6  # pJ -> uJ


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
        assert (
            self.power == "energy"
        ), "only energy is supported for NeuroSim power metric"

        self._get_data(cwd)

    def _get_data(self, cwd):
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
        ops: List[int],
        reram_energy: float,
    ):
        # parameters
        kron: bool = params["kron"]
        npe1: int = params["npe1"]
        npe2: int = params["npe2"]

        if kron:
            # kronecker product of PE1 and PE2
            pe1_power = self._get_pe_power(npe1, 1)
            pe2_power = self._get_pe_power(npe2, 1)

            assert len(ops) == 2, "ops must be provided for different kron stages"
            pe1_energy = pe1_power * ops[0]
            pe2_energy = pe2_power * ops[1]
            encoder_energy = (pe1_energy + pe2_energy) / ENERGY_UNIT
            energy = encoder_energy + reram_energy
        else:
            assert (
                len(ops) == 0
            ), "no ops should be provided for ReRAM encoding stage if not kron"
            energy = reram_energy

        metrics = {self.name(): (energy, 0.0)}
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

    def _get_pe_power(self, num_pe: int, rank: int) -> float:
        return self.mac_power * num_pe * rank + self.adder_power * num_pe * rank
