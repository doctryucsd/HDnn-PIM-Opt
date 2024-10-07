from __future__ import annotations

import logging
from typing import Any, Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from .utils import get_evaluator, set_seed


def evaluator(args: DictConfig) -> None:
    logger = logging.getLogger("evaluator")

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
    eval = evaluator.evaluate(param, logger)

    logger.info(eval)
