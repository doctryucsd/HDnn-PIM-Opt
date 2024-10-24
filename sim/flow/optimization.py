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
from botorch.models.model import Model
from omegaconf import DictConfig
from pymoo.indicators.hv import HV
from torch import nn
from tqdm import tqdm

from sim.utils import dump_metrics

from .acqfManagerFactory import acqf_factory
from .utils import get_evaluator, metric_type2bool, process_params_prop, set_seed


class PreferenceModel(Model):
    """
    A simple preference model that maps outcomes (Y) to a scalar utility based on weighted preferences.
    This model assumes that the outcomes are provided as a tensor and are reduced to a scalar value
    using a set of weights provided in the preference dictionary.
    """

    def __init__(self, preference: Dict[str, float]):
        """
        Initialize the PreferenceModel.

        Args:
            preference (Dict[str, float]): A dictionary where keys represent outcome metrics (e.g., 'accuracy',
                                           'energy', etc.) and values represent the corresponding preferences/weights.
        """
        super().__init__()
        self.preference = preference
        # Convert the preference dictionary into a tensor of weights.
        self.weights = torch.tensor(list(preference.values()), dtype=torch.float32)
        # Make sure the number of weights matches the number of expected outcomes.
        assert (
            len(self.weights) == 4
        ), "Expected 4-dimensional outcomes, adjust your preference dictionary accordingly."

    @property
    def num_outputs(self):
        # Define how many outputs your model has.
        # This could be the number of preference dimensions or objectives.
        return 1

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Forward method to map the 4-dimensional outcome Y to a scalar value.

        Args:
            Y (Tensor): A tensor of shape (batch_size, 4) representing the outcomes.

        Returns:
            Tensor: A scalar tensor of shape (batch_size,) representing the weighted preference utility.
        """
        # Compute the weighted sum of the outcomes
        return (Y * self.weights).sum(dim=-1)

    def posterior(self, X: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes a posterior over the model outputs. In this case, it's a simple deterministic model.

        Args:
            X (Tensor): A tensor representing the inputs to the model.

        Returns:
            Posterior: A deterministic posterior representing the scalar utility.
        """
        # Just return the scalar utility using forward pass
        Y = self.forward(X)
        return Y


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


def metric_type2bool(metric_type: str) -> bool:
    if metric_type == "min":
        return True
    elif metric_type == "max":
        return False
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


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
    preference: Dict[str, float],
    preferenced: bool,
    logger: Logger,
) -> AxClient:
    botorch_acqf_class = acqf_factory(acqf)
    A, b = get_constraint_matrix(constraints)  # Bound values
    pref_model = PreferenceModel(preference)
    if constrained:
        model_kwargs = {
            "torch_device": "cpu",
            "botorch_acqf_class": botorch_acqf_class,
            "acquisition_options": {
                "outcome_constraints": (A, b),
            },
        }
    elif preferenced:
        model_kwargs = {
            "torch_device": "cpu",
            "botorch_acqf_class": botorch_acqf_class,
            "acquisition_options": {
                "pref_model": pref_model,
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

    # process props
    params_prop = process_params_prop(args["params_prop"])

    # Load dataset
    evaluator = get_evaluator(
        args["data"],
        args["training"],
        args["hardware"],
        hydra.utils.get_original_cwd(),
        logger,
    )
    metrics_prop = evaluator.get_optimizations_prop()
    constraints: Dict[str, float] = args["optimization"].get("constraints", {})
    constrained: bool = args["optimization"]["constrained"]
    preference: Dict[str, float] = args["optimization"].get("preference", {})
    preferenced: bool = args["optimization"]["preferenced"]
    # BO framework
    num_trials = args["optimization"]["num_trials"]
    cli = get_model(
        num_trials,
        args["optimization"]["acqf"],
        params_prop,
        metrics_prop,
        constraints,
        constrained,
        preference,
        preferenced,
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

        evals = evaluator.evaluate(param, logger)
        cli.complete_trial(idx, raw_data=eval)  # type: ignore


        # collect metrics
        for eval in evals:
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
                    {"accuracy": accuracy, "energy": energy, "timing": timing, "area": area}
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
