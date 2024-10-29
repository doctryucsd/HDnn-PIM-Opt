from __future__ import annotations

import multiprocessing as mp
import os
from logging import Logger
from typing import Any, Dict, List, Tuple

import torch
import torch.multiprocessing as mp
from torch import Tensor

from sim.datasets import get_dataset
from sim.utils import get_params_from_dataset

from ..metrics.metricArgs import MetricArgs
from ..metrics.metricManager import MetricManager
from ..metrics.metricManagerFactory import metric_manager_factory


class Evaluator:
    def __init__(
        self,
        data_args: Dict[str, Any],
        training_args: Dict[str, Any],
        hardware_args: Dict[str, Any],
        cwd: str,
        logger: Logger,
    ) -> None:
        self.devices: List[str] = training_args["devices"]
        self.metric_args = self._get_metric_args(
            data_args, training_args, hardware_args, cwd, logger
        )
        self.metric_managers: List[MetricManager] = [
            metric_manager_factory(hardware_args["type"], self.metric_args, device)
            for device in self.devices
        ]

    def _get_metric_args(
        self,
        data_args: Dict[str, Any],
        training_args: Dict[str, Any],
        hardware_args: Dict[str, Any],
        cwd: str,
        logger: Logger | None = None,
    ) -> MetricArgs:
        dataset: str = data_args["dataset"]
        train_set, _, _ = get_dataset(dataset, cwd, data_args, True, logger)

        # set args
        num_classes, input_size, channels = get_params_from_dataset(train_set)
        ckpt_path: str = os.path.join(cwd, f"models/{dataset}.ckpt")
        model_args: Dict[str, Any] = {
            "num_classes": num_classes,
            # "cnn_output_dim": cnn_output_dim,
            "input_size": input_size,
            "input_channels": channels,
            "ckpt_path": ckpt_path,
        }
        metric_args = MetricArgs(
            model_args, training_args, hardware_args, data_args, cwd
        )

        return metric_args

    def evaluate(
        self, params: List[Dict[str, Any]], logger: Logger
    ) -> List[Dict[str, Any]]:

        assert len(params) == len(
            self.metric_managers
        ), f"Number of params ({len(params)}) does not match number of evaluators ({len(self.metric_managers)})"

        # run parallel evaluation
        n_gpus = torch.cuda.device_count()
        shared_tensor = torch.zeros((n_gpus, 4, 2), dtype=torch.float32)
        shared_tensor.share_memory_()
        mp.spawn(self._evaluate, args=(params, shared_tensor, logger), nprocs=n_gpus)  # type: ignore

        # process results
        def tensor2result(eval_results: Tensor) -> Dict[str, Any]:
            assert eval_results.shape == (
                4,
                2,
            ), f"expected shape (4, 2), got {eval_results.shape}"
            ret: Dict[str, Any] = {}
            metric_names = ["accuracy", "performance", "power", "area"]
            for name, eval_value in zip(metric_names, eval_results):
                ret[name] = (eval_value[0].item(), eval_value[1].item())
            return ret

        results = [tensor2result(shared_tensor[i]) for i in range(n_gpus)]

        return results

    def _evaluate(
        self, rank: int, params: List[Dict[str, Any]], shared_tensor: Tensor, logger
    ) -> None:
        evaluator = self.metric_managers[rank]
        return evaluator.evaluate(params[rank], rank, shared_tensor, logger)

    def get_optimizations_prop(self) -> List[Tuple[str, str, float]]:
        return self.metric_managers[0].get_optimizations_prop()
