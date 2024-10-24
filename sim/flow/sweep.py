import itertools
import logging
import random
from logging import Logger
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import torch
from ax.modelbridge import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax.service.ax_client import AxClient, ObjectiveProperties
from omegaconf import DictConfig
from tqdm import tqdm

from sim.utils import dump_metrics, generate_arithmetic_sequence

from .utils import get_evaluator, metric_type2bool, process_params_prop, set_seed


def generate_param(prop: Dict[str, Any], interval: int):
    name: str = prop["name"]
    bounds: List[float] = list(prop["bounds"])
    assert len(bounds) == 2
    value_type: str = prop["value_type"]

    trials: List[float] = generate_arithmetic_sequence(bounds[0], bounds[1], interval)
    if value_type == "int":
        trials = [int(trial) for trial in trials]
    return (name, trials)


def generate_params(props: List[Dict[str, Any]], interval: int):
    names: List[str] = []
    trials_list: List[List[float]] = []
    for prop in props:
        prop_type = prop["type"]
        if prop_type == "range":
            name, trial = generate_param(prop, interval)
        elif prop_type == "choice":
            name = prop["name"]
            trial = prop["values"]
        names.append(name)
        trials_list.append(trial)
    return names, trials_list


def get_model(
    params_prop: List[Dict[str, Any]],
    metrics_type: List[Tuple[str, str, float]],
    logger: Logger,
) -> AxClient:
    cli = AxClient(GenerationStrategy([GenerationStep(Models.SOBOL, num_trials=-1)]))

    logger.info(f"parameter properties: {params_prop}")
    logger.info(f"metric properties: {metrics_type}")

    # for each element in objctives_prop, add it to the objectives
    objectives = {
        metric_name: ObjectiveProperties(
            minimize=metric_type2bool(metric_type), threshold=ref_point
        )
        for (metric_name, metric_type, ref_point) in metrics_type
    }

    # FIXME: objectives
    cli.create_experiment(
        parameters=params_prop,
        objectives=objectives,
    )

    return cli


def sweep(args: DictConfig) -> None:
    logger = logging.getLogger("sweep")

    # set seed
    set_seed(args["seed"])

    # process params_prop
    params_prop = process_params_prop(args["params_prop"])

    # get evaluator
    evaluator = get_evaluator(
        args["data"],
        args["training"],
        args["hardware"],
        hydra.utils.get_original_cwd(),
        logger,
    )
    metrics_prop = evaluator.get_optimizations_prop()
    # model for hypervoolume
    cli = get_model(params_prop, metrics_prop, logger)

    interval: int = args["sweep"]["interval"]
    names, trials_list = generate_params(params_prop, interval)

    # all kinds of parameters and metrics of interest during BO
    accuracy_list: List[float] = []
    energy_list: List[float] = []
    timing_list: List[float] = []
    area_list: List[float] = []
    hv_list: List[float] = []
    param_list: List[Dict[str, Any]] = []

    # loop
    params: List[float] = []
    bests: List[float] = []
    dbg: int = 0
    accept_rate = args["sweep"]["accept_rate"]
    for _, trial in enumerate(tqdm(itertools.product(*trials_list))):
        param = dict(zip(names, trial))

        if random.random() > accept_rate:
            continue

        # HACK
        # if dbg > 10:
        #     break

        _, idx = cli.get_next_trial()
        evals = evaluator.evaluate([param], logger)

        cli.complete_trial(idx, raw_data=eval)  # type: ignore

        for eval in evals:
            accuracy_list.append(eval["accuracy"][0])
            energy_list.append(eval["power"][0])
            timing_list.append(eval["performance"][0])
            area_list.append(eval["area"][0])
            param_list.append(param)

        model = Models.BOTORCH_MODULAR(
            experiment=cli.experiment,
            data=cli.experiment.fetch_data(),
        )

        hv = observed_hypervolume(model)
        hv_list.append(hv)

        logger.info(f"{idx}: {param}, {eval}")
        dbg += 1

        params.append(param["hd_dim"])  # type: ignore
        bests.append(eval["accuracy"][0])

    data = {
        "accuracy": accuracy_list,
        "power": energy_list,
        "performance": timing_list,
        "area": area_list,
        "hv": hv_list,
        "param": param_list,
    }
    dump_metrics(data, args["sweep"]["metrics_file"])
    logger.info("metrics dumped")
