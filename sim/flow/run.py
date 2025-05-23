from __future__ import annotations

from omegaconf import DictConfig

from .data_collect import data_collect
from .evaluator import evaluator
from .optimization import optimization
from .plotParam import plot_param
from .sweep import sweep


def run(args: DictConfig) -> None:
    if "evaluator" in args["flow"]:
        evaluator(args)
    if "optimization" in args["flow"]:
        optimization(args)
    if "sweep" in args["flow"]:
        sweep(args)
    if "plot_param" in args["flow"]:
        plot_param(args)
    if "data_collect" in args["flow"]:
        data_collect(args)
