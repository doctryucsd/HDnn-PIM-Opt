from __future__ import annotations

from logging import Logger
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

from sim.datasets import load_dataloader
from timeloop.workspace import timeloop_ppa_eyeriss

from ..metricArgs import MetricArgs
from ..metricManager import MetricManager
from .accuracy import Accuracy
from .area import Area
from .performance import Performance
from .power import Power


class TimeLoop(MetricManager):
    def __init__(self, args: MetricArgs, device: str | int):
        self.accuracy_evaluator = Accuracy(args, device)
        self.performance_evaluator = Performance(args)
        self.power_evaluator = Power(args)
        self.area_evaluator = Area(args)

        # unpack args
        model_args = args.model_args
        data_args = args.data_args

        # training args
        self.device: str | int = device
        self.dataset: str = data_args["dataset"]

        # model args
        self.input_size: int = model_args["input_size"]
        self.num_classes: int = model_args["num_classes"]

        # data args
        self.data_args: Dict[str, Any] = data_args
        self.dataset_name: str = data_args["dataset"]
        self.cwd: str = args.cwd

    def evaluate(
        self, params: Dict[str, Any], rank: int, shared_tensor: Tensor, logger: Logger
    ):
        # parameters
        frequency: int = params["frequency"]
        # Compute array parameters
        mesh_x: int = params["mesh_x"]
        mesh_y: int = params["mesh_y"]
        # Global buffer parameters
        glb_depth: int = params["glb_depth"]
        glb_width: int = params["glb_width"]
        glb_n_banks: int = params["glb_n_banks"]
        glb_read_bw: int = params["glb_read_bw"]
        glb_write_bw: int = params["glb_write_bw"]
        # RF parameters
        rf_depth: int = params["rf_depth"]
        psum_rf_depth: int = params["psum_rf_depth"]
        rf_width: int = params["rf_width"]
        rf_read_bw: int = params["rf_read_bw"]
        rf_write_bw: int = params["rf_write_bw"]
        # MAC parameters
        mac_mult_width: int = params["mac_mult_width"]
        mac_adder_width: int = params["mac_adder_width"]

        train_loader, _, test_loader = load_dataloader(
            self.dataset_name, self.cwd, self.data_args, True
        )

        accuracy_tuple = self.accuracy_evaluator.evaluate(
            params, train_loader, test_loader, logger
        )
        hd_model = self.accuracy_evaluator.hd_model_non_noisy

        assert hd_model.cnn is not None
        asic_energy, asic_delay, asic_area, _ = timeloop_ppa_eyeriss(
            hd_model,
            next(iter(test_loader))[0][0][None, :].to(self.device),
            mesh_x,
            mesh_y,
            glb_depth,
            glb_width,
            glb_n_banks,
            glb_read_bw,
            glb_write_bw,
            rf_depth,
            psum_rf_depth,
            rf_width,
            rf_read_bw,
            rf_write_bw,
            mac_mult_width,
            mac_adder_width,
            frequency,
        )

        area_tuple = self.area_evaluator.evaluate(asic_area, logger)
        performance_tuple = self.performance_evaluator.evaluate(
            asic_delay, logger
        )
        power_tuple = self.power_evaluator.evaluate(asic_energy, logger)

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
