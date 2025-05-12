from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from ax.modelbridge import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax.service.ax_client import AxClient, ObjectiveProperties
from pymoo.indicators.hv import HV
from tqdm import tqdm

from evaluator import Evaluator
import json

def acqf_factory(acqf: str) -> Any:
    from botorch.acquisition.multi_objective import MOMF  # multi-fidelity
    from botorch.acquisition.multi_objective import (
        MultiObjectiveMCAcquisitionFunction,
        qExpectedHypervolumeImprovement,
        qNoisyExpectedHypervolumeImprovement,
    )
    from botorch.acquisition.preference import (
        AnalyticExpectedUtilityOfBestOption,
    )  # preference based

    if acqf == "qExpectedHypervolumeImprovement":
        botorch_acqf_class = qExpectedHypervolumeImprovement
    elif acqf == "qNoisyExpectedHypervolumeImprovement":
        botorch_acqf_class = qNoisyExpectedHypervolumeImprovement
    elif acqf == "MultiObjectiveMCAcquisitionFunction":
        botorch_acqf_class = MultiObjectiveMCAcquisitionFunction
    elif acqf == "EUBO":
        botorch_acqf_class = AnalyticExpectedUtilityOfBestOption
    elif acqf == "MOMF":
        botorch_acqf_class = MOMF  # multi-fidelity to tryout if we have time in the end

    else:
        raise ValueError(f"Unknown acqf: {acqf}")

    return botorch_acqf_class

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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
) -> Tuple[float, ...]:
    # TODO: Define the constraints for the metrics
    metric_0_bound = constraints["metric_0"]
    metric_1_bound = constraints["metric_1"]
    metric_2_bound = constraints["metric_2"]
    return (
        metric_0_bound,
        metric_1_bound,
        metric_2_bound,
    )


def get_constraint_matrix(
    constraints: Dict[str, float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    (
        metric_0_bound,
        metric_1_bound,
        metric_2_bound,
    ) = get_constraint_values(constraints)

    # Create A and b for the linear constraints
    A = torch.tensor(
        [
            [-1.0, 0.0, 0.0],  # metric_0: max
            [0.0, 1.0, 0.0],  # metric_1: min
            [0.0, 0.0, -1.0],  # metric_2: max
        ],
        dtype=torch.float64,
    )

    b = torch.tensor(
        [
            -metric_0_bound,
            metric_1_bound,
            -metric_2_bound,
            # Add other constraints if needed
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

    print(f"parameter properties: {params_prop}")
    print(f"metric properties: {metrics_type}")

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
    metric_0: float,
    metric_1: float,
    metric_2: float,
    constraints: Dict[str, float],
):
    # TODO: Define the constraints for the metrics
    metric_0_bound, metric_1_bound, metric_2_bound = (
        get_constraint_values(constraints)
    )

    return (
        True
        and metric_0 >= metric_0_bound # metric_0: max
        and metric_1 <= metric_1_bound # metric_1: min
        and metric_2 >= metric_2_bound # metric_2: max
    )

def dump_metrics(metrics: Dict[str, List[float]], filename: str) -> None:
    with open(f"{filename}.json", "w") as f:
        json.dump(metrics, f)

def optimization(seed: int, processed_params_prop: List[Dict[str, Any]], constrained: bool, constraints: Dict[str, float], num_trials: int, num_epochs: int, acqf: str, output_file: str) -> None:
    # set seed
    set_seed(seed)

    # process props
    processed_params_prop = process_params_prop(params_prop)

    # TODO: initialize evaluator
    evaluator = Evaluator()
    metrics_prop = evaluator.get_optimizations_prop()

    # BO framework
    cli = get_model(
        num_trials,
        acqf,
        processed_params_prop,
        metrics_prop,
        constraints,
        constrained,
    )

    # all kinds of parameters and metrics of interest during BO
    # TODO: Define the metric names
    metric_0_list: List[float] = []
    metric_1_list: List[float] = []
    metric_2_list: List[float] = []
    hv_list: List[float] = []
    param_list: List[Dict[str, Any]] = []
    eligible_points: List[Dict[str, Any]] = []
    hv_constrained_list: List[float] = []

    # BO loop
    for iter in tqdm(range(num_epochs)):
        param, idx = cli.get_next_trial()

        eval = evaluator.evaluate(param)
        cli.complete_trial(idx, raw_data=eval)  # type: ignore

        # TODO: Change the metric names to the ones used in the evaluator
        # collect metrics
        metric_0, metric_1, metric_2 = (
            eval["metric_0"][0],
            eval["metric_1"][0],
            eval["metric_2"][0],
        )
        metric_0_list.append(metric_0)
        metric_1_list.append(metric_1)
        metric_2_list.append(metric_2)
        param_list.append(param)
        if is_eligible(metric_0, metric_1, metric_2, constraints):
            eligible_points.append(
                {"metric_0": metric_0, "metric_1": metric_1, "metric_2": metric_2}
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
        "accuracy": metric_0_list,
        "energy": metric_1_list,
        "timing": metric_2_list,
        "hv": hv_list,
        "hv_constrained": hv_constrained_list,
        "param": param_list,
    }
    dump_metrics(data, output_file)
    print("metrics dumped")

if __name__ == "__main__":
    # TODO: Define the seed for reproducibility
    seed: int = 42

    # TODO: An example for the prameter definition. Define the parameter properties.
    params_prop: List[Dict[str, Any]] = [
        {"name": "hd_dim", "type": "choice", "values": [1024, 2048, 4096, 8192], "value_type": "int"},
        {"name": "reram_size", "type": "choice", "values": [64, 128, 256], "value_type": "int"},
        {"name": "frequency", "type": "range", "bounds": [1e8, 1e9], "value_type": "int"}
    ]

    # TODO: Define if the optimization considers constraints or not
    constrained: bool = True
    # TODO: Define the metric constraints
    constraints: Dict[str, float] = {
        "metric_0": 0.2,
        "metric_1": 0.5,
        "metric_2": 0.3,
    }

    # TODO: Define the number of random samples
    num_trials: int = 10
    # TODO: Define the number of total epochs for optimization (including random samples)
    num_epochs: int = 50

    # TODO: Define the acquisition function
    acqf: str = "qNoisyExpectedHypervolumeImprovement"

    # TODO: Define the output file for the optimization results
    output_file: str = "optimization_results.json"

    optimization(
        seed,
        params_prop,
        constrained,
        constraints,
        num_trials,
        num_epochs,
        acqf,
        output_file
    )