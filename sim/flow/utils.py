from __future__ import annotations

import os
from logging import Logger
from typing import Any, Dict, List

import numpy as np
import torch

from sim.datasets import load_dataset
from sim.metrics import MetricArgs, MetricManager, metric_manager_factory
from sim.utils import get_params_from_loader


def process_params_prop(props: List[Dict[str, Any]]):
    params_prop: List[Dict[str, Any]] = []
    for prop in props:
        new_prop = dict(prop)
        if "bounds" in prop:
            bounds: List[float] = [float(prop["bounds"][0]), float(prop["bounds"][1])]
            new_prop["bounds"] = bounds
        elif "values" in prop:
            if prop["value_type"] == "int":
                values: List[int | bool] = [int(v) for v in prop["values"]]
            elif prop["value_type"] == "bool":
                values: List[int | bool] = [bool(v) for v in prop["values"]]
            new_prop["values"] = values
        elif "value" in prop:
            if prop["value_type"] == "int":
                new_prop["value"] = int(prop["value"])
            elif prop["value_type"] == "bool":
                new_prop["value"] = bool(prop["value"])
        else:
            raise ValueError(f"Unknown parameter property: {prop}")
        params_prop.append(new_prop)
    return params_prop


def get_evaluator(
    data_args: Dict[str, Any],
    training_args: Dict[str, Any],
    hardware_args: Dict[str, Any],
    cwd: str,
    logger: Logger,
) -> MetricManager:
    dataset: str = data_args["dataset"]
    train_loader, _, test_loader = load_dataset(dataset, cwd, data_args, True, logger)

    # set args
    num_classes, input_size, channels = get_params_from_loader(train_loader)
    ckpt_path: str = os.path.join(cwd, f"models/{dataset}.ckpt")
    model_args: Dict[str, Any] = {
        "num_classes": num_classes,
        # "input_channel": input_channel,
        # "cnn_output_dim": cnn_output_dim,
        "input_size": input_size,
        "input_channels": channels,
        "ckpt_path": ckpt_path,
    }
    metric_args = MetricArgs(
        model_args, training_args, hardware_args, train_loader, test_loader, cwd
    )
    # noise_training: bool = args["optimization"]["noise_training"]
    # assert not (not noise and noise_training), "noise is false and noise_training is true"

    # evaluator
    evaluator = metric_manager_factory(hardware_args["type"], metric_args)
    return evaluator


def metric_type2bool(metric_type: str) -> bool:
    if metric_type == "min":
        return True
    elif metric_type == "max":
        return False
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
