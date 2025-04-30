import csv
import itertools
import logging
import os
from typing import Any, Dict, List

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from sim.evaluator import Evaluator

from .utils import process_params_prop, set_seed


def generate_param(prop: Dict[str, Any]):
    name: str = prop["name"]
    bounds: List[float] = list(prop["bounds"])
    assert len(bounds) == 2
    value_type: str = prop["value_type"]

    trials: List[int] = list(range(int(bounds[0]), int(bounds[1]) + 1))
    if value_type == "int":
        trials = [int(trial) for trial in trials]
    return (name, trials)


def generate_params(props: List[Dict[str, Any]]):
    names: List[str] = []
    trials_list: List[List[Any]] = []
    for prop in props:
        prop_type = prop["type"]
        if prop_type == "range":
            name, trial = generate_param(prop)
        elif prop_type == "choice":
            name = prop["name"]
            trial = prop["values"]
        elif prop_type == "fixed":
            name = prop["name"]
            trial = [prop["value"]]
        names.append(name)
        trials_list.append(trial)
    return names, trials_list


def data_collect(args: DictConfig) -> None:
    logger = logging.getLogger("data_collect")

    # set seed
    set_seed(args["seed"])

    # process params_prop
    params_prop = process_params_prop(args["params_prop"])

    # get evaluator
    evaluator = Evaluator(
        args["data"],
        args["training"],
        args["hardware"],
        hydra.utils.get_original_cwd(),
        logger,
    )

    # define the num_elements for continuous hyperparameter search
    param_names, trials_list = generate_params(params_prop)
    param_product = list(itertools.product(*trials_list))

    # Determine the starting index by checking how many rows are already in the CSV The CSV file has a header row.
    output_file: str = args["data_collect"]["output_file"]

    # HACK: Not considering parallel evaluation
    start_index = 0
    if os.path.exists(output_file):
        with open(output_file, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            # read the header row
            header = next(reader, None)
            if header is not None:
                # Count the already processed rows
                start_index = sum(1 for _ in reader)
    else:
        # If file does not exist, create it and write the header.
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write CSE header
            metric_names: List[str] = ["accuracy", "energy", "performance", "area"]
            header_list = param_names + metric_names
            writer.writerow(header_list)

    print(
        f"Resuming from combination index: {start_index} out of {len(param_product)} total combinations."
    )

    # # Process each element of param_product starting from the resume point.
    # Each result is appended as a new row in the CSV file.
    with open(output_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for idx, trial in enumerate(tqdm(param_product, desc="Evaluating")):
            param = dict(zip(param_names, trial))

            # skip combinations that have already been processed
            if idx < start_index:
                continue

            evals = evaluator.evaluate([param], logger)

            for eval in evals:
                metric_results = [
                    eval["accuracy"][0],
                    eval["power"][0],
                    eval["performance"][0],
                    eval["area"][0],
                ]
                row_list = list(param.values()) + metric_results
                writer.writerow(row_list)
                csvfile.flush()
                logger.info(f"{idx}: params: {param}, results: {eval}")

    logger.info("Data collection completed.")
