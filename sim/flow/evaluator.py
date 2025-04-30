from __future__ import annotations

import logging
from typing import Any, Dict

import hydra
from omegaconf import DictConfig

from sim.evaluator import Evaluator

from .utils import set_seed


def evaluator(args: DictConfig) -> None:
    logger = logging.getLogger("evaluator")

    # set seed
    set_seed(args["seed"])

    evaluator = Evaluator(
        args["data"],
        args["training"],
        args["hardware"],
        hydra.utils.get_original_cwd(),
        logger,
    )

    param: Dict[str, Any] = args["evaluator"]["param"]
    eval = evaluator.evaluate([param], logger)

    logger.info(eval)
