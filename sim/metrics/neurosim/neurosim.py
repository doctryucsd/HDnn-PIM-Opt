from __future__ import annotations

from logging import Logger
from typing import Any, Dict, List, Tuple

from neurosim.Inference_pytorch import neurosim_ppa

from ..metricArgs import MetricArgs
from ..metricManager import MetricManager
from .accuracy import Accuracy
from .area import Area
from .performance import Performance
from .power import Power


class NeuroSim(MetricManager):
    def __init__(self, args: MetricArgs, device: str):
        self.args: MetricArgs = args
        self.accuracy_evaluator = Accuracy(args, device)
        self.performance_evaluator = Performance(args)
        self.power_evaluator = Power(args)
        self.area_evaluator = Area(args)

        # unpack args
        training_args = args.training_args
        model_args = args.model_args
        hardware_args = args.hardware_args
        self.test_loader = args.test_loader

        # training args
        self.device: str = training_args["device"]

        # model args
        self.input_size: int = model_args["input_size"]
        self.num_classes: int = model_args["num_classes"]

        # hardware args
        self.temperature: int = hardware_args["temperature"]

    def evaluate(
        self, params: Dict[str, Any], logger: Logger
    ) -> Dict[str, Tuple[float, float]]:
        # parameters
        kron: bool = params["kron"]
        reram_size: int = params["reram_size"]
        freqeuency: int = params["frequency"]

        accuracy_dict = self.accuracy_evaluator.evaluate(params, logger)
        hd_model = self.accuracy_evaluator.hd_model_non_noisy

        if kron:
            reram_energy, reram_delay, reram_area, clk_period = neurosim_ppa(
                "HDInference",
                hd_model.hd_inference,
                hd_model.encoder(next(iter(self.test_loader))[0].to(self.device)),
                reram_size,
                freqeuency,
                self.temperature,
                5,
            )
        else:
            reram_energy, reram_delay, reram_area, clk_period = neurosim_ppa(
                "HD",
                hd_model,
                next(iter(self.test_loader))[0].to(self.device),
                reram_size,
                freqeuency,
                self.temperature,
                5,
            )

        area_dict = self.area_evaluator.evaluate(params, reram_area)
        performance_dict, ops = self.performance_evaluator.evaluate(
            params, reram_delay, clk_period
        )
        power_dict = self.power_evaluator.evaluate(params, ops, reram_energy)

        ret: Dict[str, Tuple[float, float]] = {}
        ret.update(accuracy_dict)
        ret.update(area_dict)
        ret.update(performance_dict)
        ret.update(power_dict)

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
