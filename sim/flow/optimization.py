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
from .constraint_scheduler import make_constraint_scheduler
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
    # Note: For dynamic constraints we will construct model per-iteration.
    # Here we keep a simple GS config; dynamic outcome constraints are handled in the loop.
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
    acqf_name: str = args["optimization"]["acqf"]
    botorch_acqf_class = acqf_factory(acqf_name)
    cli = get_model(
        num_trials,
        acqf_name,
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
    hv_constrained_list: List[float] = []
    eligible_count_list: List[int] = []

    # Setup constraint scheduler (only used if constrained)
    schedule_type: str = args["optimization"].get("threshold_schedule", "static")
    schedule_final_iter_cfg = args["optimization"].get("threshold_schedule_final_iter", None)
    # Determine scheduling window: starts after Sobol (num_trials)
    schedule_start_iter = int(num_trials)
    if schedule_final_iter_cfg is None:
        schedule_end_iter = int(args["optimization"]["num_epochs"]) - 1
    else:
        # Clamp the configured final iter within [start_iter, num_epochs-1]
        schedule_end_iter = int(schedule_final_iter_cfg)
        schedule_end_iter = max(schedule_start_iter, schedule_end_iter)
        schedule_end_iter = min(schedule_end_iter, int(args["optimization"]["num_epochs"]) - 1)
    schedule_total_steps = max(1, schedule_end_iter - schedule_start_iter + 1)

    if constrained:
        scheduler = make_constraint_scheduler(
            schedule_type, constraints, schedule_total_steps=schedule_total_steps
        )
    else:
        scheduler = None  # type: ignore

    # BO loop
    num_epochs: int = args["optimization"]["num_epochs"]
    for iter in tqdm(range(num_epochs)):
        # Decide generation strategy and current constraints
        if constrained and iter >= num_trials:
            # Scheduled constraints over the remaining iterations
            step_idx = iter - num_trials
            # Use scheduler's configured total steps (derived from schedule_final_iter)
            current_constraints = scheduler.get(step_idx, None)  # type: ignore

            # Create a fresh BoTorch modular model with updated outcome constraints
            A, B = get_constraint_matrix(current_constraints)
            model = Models.BOTORCH_MODULAR(
                experiment=cli.experiment,
                data=cli.experiment.fetch_data(),
                botorch_acqf_class=botorch_acqf_class,
                acquisition_options={
                    "outcome_constraints": (A, B),
                },
                torch_device=torch.device("cpu"),
            )
            gr = model.gen(1)
            # Ax 0.2+/0.3+: access parameters via arms
            param = gr.arms[0].parameters
            _, idx = cli.attach_trial(param)  # type: ignore
        else:
            # Sobol phase or unconstrained: pure random/Sobol generation
            model = Models.SOBOL(
                experiment=cli.experiment,
                data=cli.experiment.fetch_data(),
            )
            gr = model.gen(1)
            param = gr.arms[0].parameters
            _, idx = cli.attach_trial(param)  # type: ignore

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
            # We'll recompute eligibility each iteration using the final constraints when
            # calculating constrained HV, so we don't persist eligibility here.

        # Hypervolume calculation
        # Recompute a BoTorch modular model from accumulated data for HV
        model = Models.BOTORCH_MODULAR(
            experiment=cli.experiment,
            data=cli.experiment.fetch_data(),
        )
        hv = observed_hypervolume(model)
        hv_list.append(hv)

        # Hypervolume calculation with constraints
        # Compute eligibility based on the FINAL target constraints (fixed across iterations)
        if constrained and len(accuracy_list) > 0:
            # Always evaluate constrained HV against final constraints
            current_constraints = constraints

            # Select eligible points among all evaluated so far
            eligible_mask = [
                is_eligible(a, e, t, ar, current_constraints)
                for (a, e, t, ar) in zip(
                    accuracy_list, energy_list, timing_list, area_list
                )
            ]
            eligible_count = int(sum(1 for ok in eligible_mask if ok))
            eligible_count_list.append(eligible_count)
            if eligible_count > 0:
                # ref_point: [accuracy, energy, timing, area], pymoo default is minimization
                ref_point = np.array([0.0, 1.0, 1.0, 1.0])
                hv_constrained_calculation = HV(ref_point=ref_point)
                points = np.array(
                    [
                        [-a, e, t, ar]
                        for a, e, t, ar, ok in zip(
                            accuracy_list, energy_list, timing_list, area_list, eligible_mask
                        )
                        if ok
                    ]
                )
                hv_constrained = hv_constrained_calculation(points)
                assert (
                    hv_constrained is not None and hv_constrained >= 0
                ), f"Invalid hypervolume: {hv_constrained}"
            else:
                hv_constrained = 0.0
        else:
            hv_constrained = 0.0
            eligible_count_list.append(0)
        hv_constrained_list.append(hv_constrained)

    data = {
        "accuracy": accuracy_list,
        "energy": energy_list,
        "timing": timing_list,
        "area": area_list,
        "hv": hv_list,
        "hv_constrained": hv_constrained_list,
        "eligible_count": eligible_count_list,
        "param": param_list,
    }
    dump_metrics(data, args["optimization"]["metrics_file"])
    logger.info("\nmetrics dumped")
