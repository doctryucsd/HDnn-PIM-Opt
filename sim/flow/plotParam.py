from __future__ import annotations

import logging
from typing import Any, Dict, List

import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from .utils import get_evaluator, set_seed


def plot_metrics(
    param_name: str,
    param_list: List[float],
    accuracy_list: List[float],
    energy_list: List[float],
    timing_list: List[float],
    area_list: List[float],
) -> None:
    # enable latex rendering
    # plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 12

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    display_accuracy_list = [accuracy * 100 for accuracy in accuracy_list]
    axs[0, 0].plot(param_list, display_accuracy_list)
    axs[0, 0].set_xlabel(param_name)
    axs[0, 0].set_ylabel("Accuracy (%)")
    axs[0, 0].set_title("(a)")
    # axs[0, 0].set_aspect(3)

    axs[0, 1].plot(param_list, energy_list)
    axs[0, 1].set_xlabel(param_name)
    axs[0, 1].set_ylabel("Total Energy (uJ)")
    axs[0, 1].set_title("(b)")
    # axs[0, 1].set_aspect("equal")

    axs[1, 0].plot(param_list, timing_list)
    axs[1, 0].set_xlabel(param_name)
    axs[1, 0].set_ylabel("Timing (us)")
    axs[1, 0].set_title("(c)")
    # axs[1, 0].set_aspect("equal")

    axs[1, 1].plot(param_list, area_list)
    axs[1, 1].set_xlabel(param_name)
    axs[1, 1].set_ylabel("Area ($mm2$)")
    axs[1, 1].set_title("(d)")
    # axs[1, 1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"{param_name}.png")


def plot_param(args: DictConfig) -> None:
    logger = logging.getLogger("plot_param")

    # set seed
    set_seed(args["seed"])

    evaluator = get_evaluator(
        args["data"],
        args["training"],
        args["hardware"],
        hydra.utils.get_original_cwd(),
        logger,
    )

    param: Dict[str, Any] = args["evaluator"]["param"]
    var_param: Dict[str, List] = args["plot_param"]["var_param"]

    param_name = "hd_dim"
    param_list = var_param[param_name]

    # HACK
    d1_list: List[float] = var_param["d1"]

    accuracy_list: List[float] = []
    energy_list: List[float] = []
    timing_list: List[float] = []
    area_list: List[float] = []

    for param_value, d1_value in zip(param_list, d1_list):
        param[param_name] = param_value

        # HACK
        param["d1"] = d1_value

        evals = evaluator.evaluate([param], logger)
        logger.info(eval)

        # update metrcis
        for eval in evals:
            accuracy_list.append(eval["accuracy"][0])
            energy_list.append(eval["power"][0])
            timing_list.append(eval["performance"][0])
            area_list.append(eval["area"][0])

    plot_metrics(
        param_name, param_list, accuracy_list, energy_list, timing_list, area_list
    )
