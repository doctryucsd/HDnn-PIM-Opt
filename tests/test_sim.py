from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ax.modelbridge import Models
from ax.modelbridge.modelbridge_utils import observed_hypervolume
import torch
import numpy as np

from neurosim.Inference_pytorch import neurosim_ppa
from sim.datasets import load_dataloader
from sim.flow.optimization import get_model
from sim.models import HDFactory
from sim.evaluator import Evaluator

NEUROSIM_PARAMS = [
        {
            "hd_dim": 8192,
            "npe1": 214,
            "npe2": 60,
            "f1": 56,
            "d1": 64,
            "frequency": 555647552,
            "reram_size": 64,
            "kron": False,
        },
        {
            "hd_dim": 1024,
            "npe1": 190,
            "npe2": 89,
            "f1": 56,
            "d1": 32,
            "frequency": 523438682,
            "reram_size": 64,
            "kron": False,
        },
        {
            "hd_dim": 1024,
            "npe1": 182,
            "npe2": 217,
            "f1": 49,
            "d1": 128,
            "frequency": 743736533,
            "reram_size": 64,
            "kron": False,
        },
        {
            "hd_dim": 1024,
            "npe1": 237,
            "npe2": 201,
            "f1": 28,
            "d1": 32,
            "frequency": 345793734,
            "reram_size": 64,
            "kron": False,
        },
        {
            "hd_dim": 8192,
            "npe1": 73,
            "npe2": 231,
            "f1": 28,
            "d1": 64,
            "frequency": 531709616,
            "reram_size": 64,
            "kron": False,
        },
    ]

NEUROSIM_EVALS = [
        {
            "accuracy": (0.90035, 0.00105),
            "area": (289.24305, 0.0),
            "performance": (2.162332, 0.0),
            "power": (0.306458, 0.0),
        },
        {
            "accuracy": (0.8501, 0.0028),
            "area": (39.488375, 0.0),
            "performance": (1.146087, 0.0),
            "power": (0.035177, 0.0),
        },
        {
            "accuracy": (0.8175, 0.0006),
            "area": (39.555416, 0.0),
            "performance": (1.33299, 0.0),
            "power": (0.035394, 0.0),
        },
        {
            "accuracy": (0.85625, 5e-05),
            "area": (39.563796, 0.0),
            "performance": (1.33299, 0.0),
            "power": (0.035404, 0.0),
        },
        {
            "accuracy": (0.91815, 0.00135),
            "area": (289.460082, 0.0),
            "performance": (1.859144, 0.0),
            "power": (0.30477, 0.0),
        },
    ]

def test_optimization():
    # set seed
    np.random.seed(45)
    torch.manual_seed(45)

    # process props
    params_prop: List[Dict[str, Any]] = [
        {
            "name": "hd_dim",
            "type": "choice",
            "values": [1024, 2048, 4096, 8192],
            "value_type": "int",
        },
        {
            "name": "npe1",
            "type": "range",
            "bounds": [8, 256],
            "value_type": "int",
        },
        {
            "name": "npe2",
            "type": "range",
            "bounds": [8, 256],
            "value_type": "int",
        },
        {
            "name": "reram_size",
            "type": "fixed",
            "value": 64,
            "value_type": "int",
        },
        {
            "name": "f1",
            "type": "choice",
            "values": [28, 56, 49],
            "value_type": "int",
        },
        {
            "name": "d1",
            "type": "choice",
            "values": [32, 64, 128],
            "value_type": "int",
        },
        {
            "name": "kron",
            "type": "choice",
            "values": [False],
            "value_type": "bool",
        },
        {
            "name": "frequency",
            "type": "range",
            "bounds": [1e7, 1e9],
            "value_type": "int",
        },
    ]

    # Load dataset
    data_args: Dict[str, Any] = {
        "dataset": "mnist",
        "train_batch_size": 2048,
        "test_batch_size": 16,
        "num_workers": 4,
        "train_ratio": 0.8,
    }
    training_args: Dict[str, Any] = {
        "device": "cuda:0",
        "epochs": 1,
        "lr": 0.01,
        "num_tests": 1,
    }
    hardware_args: Dict[str, Any] = {
        "type": "neurosim",
        "noise": True,
        "temperature": 300,
        "power": "energy",
    }
    evaluator = Evaluator(data_args, training_args, hardware_args, ".")
    metrics_prop = evaluator.get_optimizations_prop()

    # BO framework
    num_trials = 5
    cli = get_model(
        num_trials,
        "qExpectedHypervolumeImprovement",
        params_prop,
        metrics_prop,
    )

    # all kinds of parameters and metrics of interest during BO
    trial_params: List[Dict[str, float]] = []
    trial_metrics: List[Dict[str, Tuple[float, float]]] = []

    accuracy_list: List[float] = []
    energy_list: List[float] = []
    timing_list: List[float] = []
    area_list: List[float] = []
    hv_list: List[float] = []
    param_list: List[Dict[str, Tuple[float, float]]] = []

    params = NEUROSIM_PARAMS

    evals = NEUROSIM_EVALS

    # BO loop
    for iter, (param, eval) in enumerate(zip(params, evals)):
        _, idx = cli.get_next_trial()

        cli.complete_trial(idx, raw_data=eval)  # type: ignore

        trial_params.append(param)
        trial_metrics.append(eval)

        accuracy_list.append(eval["accuracy"][0])
        energy_list.append(eval["power"][0])
        timing_list.append(eval["performance"][0])
        area_list.append(eval["area"][0])
        param_list.append(param)

        if iter >= num_trials:
            model = cli.generation_strategy.model
        else:
            model = Models.BOTORCH_MODULAR(
                experiment=cli.experiment,
                data=cli.experiment.fetch_data(),
            )

        hv = observed_hypervolume(model)
        print(hv)
        hv_list.append(hv)

    data = {
        "accuracy": accuracy_list,
        "power": energy_list,
        "performance": timing_list,
        "area": area_list,
        "hv": hv_list,
        "param": param_list,
    }


def test_neurosim_ppa():
    data_args: Dict[str, Any] = {
        "dataset": "mnist",
        "train_batch_size": 2048,
        "test_batch_size": 16,
        "num_workers": 4,
        "train_ratio": 0.8,
    }
    train_loader, _, test_loader = load_dataloader("mnist", ".", data_args, True)
    # params
    hd_dim: int = 512
    f1: int = 49
    d1: int = 64
    reram_size: int = 64
    frequency: int = int(1e7)
    binarize_type: bool = False
    kron: bool = False

    # construct hd
    hd_factory = HDFactory(28 * 28, hd_dim, 10, binarize_type, "cpu")
    if kron:
        # pass
        hd_factory.set_kronecker(d1, f1)
    hd_factory.bernoulli()
    hd_factory.binarize(binarize_type)
    hd_factory.init_buffer(train_loader)

    model = hd_factory.create_neurosim()

    from rich import print

    print(
        neurosim_ppa(
            "HD",
            model,
            next(iter(test_loader))[0],
            reram_size,
            frequency,
            300,
            1,
            "cuda:0",
        )
    )
