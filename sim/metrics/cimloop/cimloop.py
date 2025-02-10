from __future__ import annotations

from logging import Logger
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor, nn

from cimloop.workspace import cimloop_ppa
from sim.datasets import load_dataloader
from timeloop.workspace import timeloop_ppa

from ..metricArgs import MetricArgs
from ..metricManager import MetricManager
from .accuracy import Accuracy
from .area import Area
from .performance import Performance
from .power import Power


class CiMLoop(MetricManager):
    def __init__(self, args: MetricArgs, device: str | int):
        self.accuracy_evaluator = Accuracy(args, device)
        self.performance_evaluator = Performance(args)
        self.power_evaluator = Power(args)
        self.area_evaluator = Area(args)

        # unpack args
        model_args = args.model_args
        hardware_args = args.hardware_args
        data_args = args.data_args

        # training args
        self.device: str | int = device
        self.dataset: str = data_args["dataset"]

        # model args
        self.input_size: int = model_args["input_size"]
        self.num_classes: int = model_args["num_classes"]

        # hardware args
        self.cnn: bool = hardware_args["cnn"]
        # self.temperature: int = hardware_args["temperature"]

        # data args
        self.data_args: Dict[str, Any] = data_args
        self.dataset_name: str = data_args["dataset"]
        self.cwd: str = args.cwd

    def evaluate(
        self, params: Dict[str, Any], rank: int, shared_tensor: Tensor, logger: Logger
    ):
        # parameters
        reram_size: int = params["reram_size"]
        frequency: int = params["frequency"]
        cnn_x_dim_1: int = params["cnn_x_dim_1"]
        cnn_y_dim_1: int = params["cnn_y_dim_1"]
        cnn_x_dim_2: int = params["cnn_x_dim_2"]
        cnn_y_dim_2: int = params["cnn_y_dim_2"]
        encoder_x_dim: int = params["encoder_x_dim"]
        encoder_y_dim: int = params["encoder_y_dim"]

        train_loader, _, test_loader = load_dataloader(
            self.dataset_name, self.cwd, self.data_args, True
        )

        accuracy_tuple = self.accuracy_evaluator.evaluate(
            params, train_loader, test_loader, logger
        )
        hd_model = self.accuracy_evaluator.hd_model_non_noisy

        if self.cnn:
            assert hd_model.cnn is not None
            asic_energy, asic_delay, asic_area, _ = timeloop_ppa(
                nn.Sequential(hd_model.cnn, hd_model.encoder),
                next(iter(test_loader))[0][0][None, :].to(self.device),
                cnn_x_dim_1,
                cnn_y_dim_1,
                cnn_x_dim_2,
                cnn_y_dim_2,
                encoder_x_dim,
                encoder_y_dim,
                frequency,
                5,
            )
            reram_energy, reram_delay, reram_area, _ = cimloop_ppa(
                "HD",
                hd_model.hd_inference,
                hd_model.feature_encode(
                    next(iter(test_loader))[0][0][None, :].to(self.device)
                ),
                reram_size,
                frequency,
                5,
            )
        else:
            kron: bool = params["kron"]
            if kron:
                asic_energy, asic_delay, asic_area, _ = timeloop_ppa(
                    hd_model.encoder,
                    next(iter(test_loader))[0][0][None, :].to(self.device),
                    cnn_x_dim_1,
                    cnn_y_dim_1,
                    cnn_x_dim_2,
                    cnn_y_dim_2,
                    encoder_x_dim,
                    encoder_y_dim,
                    frequency,
                    5,
                )
                reram_energy, reram_delay, reram_area, _ = cimloop_ppa(
                    "HD",
                    hd_model.hd_inference,
                    hd_model.encoder(
                        next(iter(test_loader))[0][0][None, :].to(self.device)
                    ),
                    reram_size,
                    frequency,
                    5,
                )
            else:
                asic_energy, asic_delay, asic_area = 0, 0, 0
                reram_energy, reram_delay, reram_area, _ = cimloop_ppa(
                    "HD",
                    hd_model,
                    next(iter(test_loader))[0].to(self.device),
                    reram_size,
                    frequency,
                    5,
                )

        area_tuple = self.area_evaluator.evaluate(asic_area, reram_area, logger)
        performance_tuple = self.performance_evaluator.evaluate(
            asic_delay, reram_delay, logger
        )
        power_tuple = self.power_evaluator.evaluate(asic_energy, reram_energy, logger)

        ret_list: List[Tuple[float, float]] = [
            accuracy_tuple,
            power_tuple,
            performance_tuple,
            area_tuple,
        ]
        ret_tensor = torch.tensor(ret_list, device=self.device)

        shared_tensor[rank] = ret_tensor

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

    def get_metric_names(self) -> List[str]:
        return [Accuracy.name(), Power.name(), Performance.name(), Area.name()]
