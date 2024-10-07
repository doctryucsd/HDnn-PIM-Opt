from __future__ import annotations

from logging import Logger
from typing import Any, Dict, List, Tuple

from torch import nn

from cimloop.workspace import cimloop_ppa
from timeloop.workspace import timeloop_ppa

from ..metricArgs import MetricArgs
from ..metricManager import MetricManager
from .accuracy import Accuracy
from .area import Area
from .performance import Performance
from .power import Power


class CIMLoop(MetricManager):
    def __init__(self, args: MetricArgs):
        self.args: MetricArgs = args
        self.accuracy_evaluator = Accuracy(args)
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
        self.cnn: bool = hardware_args["cnn"]
        # self.temperature: int = hardware_args["temperature"]

    def evaluate(self, params: Dict[str, Any], logger: Logger):
        # parameters
        reram_size: int = params["reram_size"]
        freqeuency: int = params["frequency"]
        cnn_x_dim_1: int = params["cnn_x_dim_1"]
        cnn_y_dim_1: int = params["cnn_y_dim_1"]
        cnn_x_dim_2: int = params["cnn_x_dim_2"]
        cnn_y_dim_2: int = params["cnn_y_dim_2"]
        encoder_x_dim: int = params["encoder_x_dim"]
        encoder_y_dim: int = params["encoder_y_dim"]

        accuracy_dict = self.accuracy_evaluator.evaluate(params, logger)
        hd_model = self.accuracy_evaluator.hd_model_non_noisy

        if self.cnn:
            assert hd_model.cnn is not None
            asic_energy, asic_delay, asic_area, _ = timeloop_ppa(
                nn.Sequential(hd_model.cnn, hd_model.encoder),
                next(iter(self.test_loader))[0][0][None, :].to(self.device),
                cnn_x_dim_1,
                cnn_y_dim_1,
                cnn_x_dim_2,
                cnn_y_dim_2,
                encoder_x_dim,
                encoder_y_dim,
                freqeuency,
                5,
            )
            reram_energy, reram_delay, reram_area, _ = cimloop_ppa(
                "HD",
                hd_model.hd_inference,
                hd_model.feature_encode(
                    next(iter(self.test_loader))[0][0][None, :].to(self.device)
                ),
                reram_size,
                freqeuency,
                5,
            )
        else:
            kron: bool = params["kron"]
            if kron:
                asic_energy, asic_delay, asic_area, _ = timeloop_ppa(
                    hd_model.encoder,
                    next(iter(self.test_loader))[0][0][None, :].to(self.device),
                    cnn_x_dim_1,
                    cnn_y_dim_1,
                    cnn_x_dim_2,
                    cnn_y_dim_2,
                    encoder_x_dim,
                    encoder_y_dim,
                    freqeuency,
                    5,
                )
                reram_energy, reram_delay, reram_area, _ = cimloop_ppa(
                    "HD",
                    hd_model.hd_inference,
                    hd_model.encoder(
                        next(iter(self.test_loader))[0][0][None, :].to(self.device)
                    ),
                    reram_size,
                    freqeuency,
                    5,
                )
            else:
                asic_energy, asic_delay, asic_area = 0, 0, 0
                reram_energy, reram_delay, reram_area, _ = cimloop_ppa(
                    "HD",
                    hd_model,
                    next(iter(self.test_loader))[0].to(self.device),
                    reram_size,
                    freqeuency,
                    5,
                )

        area_dict = self.area_evaluator.evaluate(asic_area, reram_area, logger)
        performance_dict = self.performance_evaluator.evaluate(
            asic_delay, reram_delay, logger
        )
        power_dict = self.power_evaluator.evaluate(asic_energy, reram_energy, logger)

        ret: Dict[str, Tuple[float, float]] = {}
        ret.update(accuracy_dict)
        ret.update(area_dict)
        ret.update(performance_dict)
        ret.update(power_dict)

        return ret

    def get_optimizations_prop(self):
        ret: List[Tuple[str, str, float]] = []
        ret.append(
            (Accuracy.name(), Accuracy.optimization_type(), Accuracy.ref_point())
        )
        ret.append((Power.name(), Power.optimization_type(), Power.ref_point()))
        ret.append(
            (
                Performance.name(),
                Performance.optimization_type(),
                Performance.ref_point(),
            )
        )
        ret.append((Area.name(), Area.optimization_type(), Area.ref_point()))
        return ret
