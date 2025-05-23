from __future__ import annotations

import logging
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
from pymoo.indicators.hv import HV
from tqdm import tqdm

from sim.evaluator import Evaluator
from sim.utils import dump_metrics

from .acqfManagerFactory import acqf_factory
from .utils import metric_type2bool, process_params_prop, set_seed


def process_metrics(metrics_prop: Dict[str, str]):
    ret: List[Tuple[str, bool]] = []
    for obj, minimize in metrics_prop.items():
        if minimize == "min":
            ret.append((obj, True))
        elif minimize == "max":
            ret.append((obj, False))
        else:
            raise ValueError(f"Unknown minimization: {minimize}")
    return ret

def get_constraint_values(
    constraints: Dict[str, float]
) -> Tuple[float, float, float, float]:
    accuracy_lower_bound = constraints["accuracy"]
    power_upper_bound = constraints["power"]
    performance_upper_bound = constraints["performance"]
    area_upper_bound = constraints["area"]
    return (
        accuracy_lower_bound,
        power_upper_bound,
        performance_upper_bound,
        area_upper_bound,
    )


def get_constraint_matrix(
    constraints: Dict[str, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    (
        accuracy_lower_bound,
        power_upper_bound,
        performance_upper_bound,
        area_upper_bound,
    ) = get_constraint_values(constraints)

    # Create A and b for the linear constraints
    A = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],  # accuracy
            [0.0, 1.0, 0.0, 0.0],  # power
            [0.0, 0.0, 1.0, 0.0],  # performance
            [0.0, 0.0, 0.0, 1.0],  # area
        ],
        dtype=torch.float64,
    )

    b = torch.tensor(
        [
            -accuracy_lower_bound,
            power_upper_bound,
            performance_upper_bound,
            area_upper_bound,
        ],
        dtype=torch.float64,
    )
    return A, b


def get_model(
    num_trials: int,
    acqf: str,
    params_prop: List[Dict[str, Any]],
    metrics_type: List[Tuple[str, str, float]],
    constraints: Dict[str, float],
    constrained: bool,
    logger: Logger,
) -> AxClient:
    botorch_acqf_class = acqf_factory(acqf)
    A, b = get_constraint_matrix(constraints)  # Bound values
    if constrained:
        model_kwargs = {
            "torch_device": "cpu",
            "botorch_acqf_class": botorch_acqf_class,
            "acquisition_options": {
                "outcome_constraints": (A, b),
            },
        }
    else:
        model_kwargs = {"torch_device": "cpu", "botorch_acqf_class": botorch_acqf_class}
    cli = AxClient(
        GenerationStrategy(
            [
                GenerationStep(Models.SOBOL, num_trials=num_trials),
                GenerationStep(
                    Models.BOTORCH_MODULAR,
                    num_trials=-1,
                    model_kwargs=model_kwargs,
                ),
            ]
        )
    )

    logger.info(f"parameter properties: {params_prop}")
    logger.info(f"metric properties: {metrics_type}")

    # for each element in objectives_prop, add it to the objectives
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


def is_eligible(
    accuracy: float,
    energy: float,
    timing: float,
    area: float,
    constraints: Dict[str, float],
):
    accuracy_lower_bound, power_upper_bound, timing_upper_bound, area_upper_bound = (
        get_constraint_values(constraints)
    )

    return (
        True
        and accuracy >= accuracy_lower_bound
        and energy <= power_upper_bound
        and timing <= timing_upper_bound
        and area <= area_upper_bound
    )


def optimization(args: DictConfig) -> None:
    logger = logging.getLogger("optimization")

    # set seed
    set_seed(args["seed"])

    print(args["params_prop"])

    # process props
    params_prop = process_params_prop(args["params_prop"])

    # Load dataset
    evaluator = Evaluator(
        args["data"],
        args["training"],
        args["hardware"],
        hydra.utils.get_original_cwd(),
        logger,
    )
    metrics_prop = evaluator.get_optimizations_prop()
    constraints: Dict[str, float] = args["optimization"].get("constraints", {})
    constrained: bool = args["optimization"]["constrained"]
    # BO framework
    num_trials = args["optimization"]["num_trials"]
    cli = get_model(
        num_trials,
        args["optimization"]["acqf"],
        params_prop,
        metrics_prop,
        constraints,
        constrained,
        logger,
    )

    # all kinds of parameters and metrics of interest during BO
    accuracy_list: List[float] = []
    energy_list: List[float] = []
    timing_list: List[float] = []
    area_list: List[float] = []
    hv_list: List[float] = []
    param_list: List[Dict[str, Any]] = []
    eligible_points: List[Dict[str, Any]] = []
    hv_constrained_list: List[float] = []

    # BO loop
    num_epochs: int = args["optimization"]["num_epochs"]
    for iter in tqdm(range(num_epochs)):
        # FIXME, TODO: collect params

        param, idx = cli.get_next_trial()

        evals = evaluator.evaluate([param], logger)

        # collect metrics
        for eval in evals:
            cli.complete_trial(idx, raw_data=eval)  # type: ignore
            accuracy, energy, timing, area = (
                eval["accuracy"][0],
                eval["power"][0],
                eval["performance"][0],
                eval["area"][0],
            )
            accuracy_list.append(accuracy)
            energy_list.append(energy)
            timing_list.append(timing)
            area_list.append(area)
            param_list.append(param)
            if is_eligible(accuracy, energy, timing, area, constraints):
                eligible_points.append(
                    {
                        "accuracy": accuracy,
                        "energy": energy,
                        "timing": timing,
                        "area": area,
                    }
                )

        # Hypervolume calculation
        if iter >= num_trials:
            model = cli.generation_strategy.model
        else:
            model = Models.BOTORCH_MODULAR(
                experiment=cli.experiment,
                data=cli.experiment.fetch_data(),
            )
        hv = observed_hypervolume(model)
        hv_list.append(hv)

        # Hypervolume calculation with constraints
        if len(eligible_points) > 0:
            # ref_point: [accuracy, energy, timing, area], pymoo default is minimization
            ref_point = np.array([0.0, 1.0, 1.0, 1.0])
            hv_constrained_calculation = HV(ref_point=ref_point)
            points = np.array(
                [
                    [-x["accuracy"], x["energy"], x["timing"], x["area"]]
                    for x in eligible_points
                ]
            )
            hv_constrained = hv_constrained_calculation(points)
            assert (
                hv_constrained is not None and hv_constrained >= 0
            ), f"Invalid hypervolume: {hv_constrained}"
        else:
            hv_constrained = 0.0
        hv_constrained_list.append(hv_constrained)

    data = {
        "accuracy": accuracy_list,
        "energy": energy_list,
        "timing": timing_list,
        "area": area_list,
        "hv": hv_list,
        "hv_constrained": hv_constrained_list,
        "param": param_list,
    }
    dump_metrics(data, args["optimization"]["metrics_file"])
    logger.info("metrics dumped")
