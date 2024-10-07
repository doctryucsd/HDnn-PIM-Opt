from __future__ import annotations

import os
from logging import Logger
from typing import List

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

    def evaluate(self, asic_energy: float, reram_energy: float, logger: Logger):
        total_energy = asic_energy + reram_energy

        logger.info(f"{self.name()}: {total_energy}uJ")

        ret = self._normalize(total_energy)

        metrics = {self.name(): (ret, 0.0)}
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
        return 1.0

    def _get_pe_power(self, num_pe: int, rank: int) -> float:
        return self.mac_power * num_pe * rank + self.adder_power * num_pe * rank

    def _normalize(self, power: float) -> float:
        BASE = 3000.0
        ret = power / BASE
        assert ret <= self.ref_point(), f"power: {ret} > ref_point: {self.ref_point()}"
        return ret
