from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sim.models import HD, HDFactory, HDTrainer

from ..metricArgs import MetricArgs


class Accuracy:
    def __init__(self, args: MetricArgs):
        # unpack args
        model_args = args.model_args
        training_args = args.training_args
        hardware_args = args.hardware_args
        train_loader = args.train_loader
        test_loader = args.test_loader

        # model args
        self.num_classes: int = model_args["num_classes"]
        self.input_size: int = model_args["input_size"]

        # training args
        self.trainer = HDTrainer(training_args)
        self.num_tests: int = training_args["num_tests"]
        self.epochs: int = training_args["epochs"]
        self.lr: float = training_args["lr"]
        self.device: str = training_args["device"]

        # hardware args
        self.noisy: bool = hardware_args["noise"]
        self.temperature: int = hardware_args["temperature"]

        # dataloaders
        self.train_loader: DataLoader = train_loader
        self.test_loader: DataLoader = test_loader

    def evaluate(self, params: Dict[str, Any]):
        # params
        hd_dim: int = params["hd_dim"]
        kron: bool = params["kron"]
        f1: int = params["f1"]
        d1: int = params["d1"]
        reram_size: int = params["reram_size"]
        frequency: int = params["frequency"]
        binarize_type: bool = False

        # construct hd
        hd_factory = HDFactory(
            self.input_size, hd_dim, self.num_classes, binarize_type, self.device
        )
        if kron:
            # pass
            hd_factory.set_kronecker(d1, f1)
        hd_factory.bernoulli()
        hd_factory.binarize(binarize_type)
        hd_factory.init_buffer(self.train_loader)
        hd_factory.retrain(self.train_loader, self.epochs, self.lr)

        # finish training
        if not kron:
            hd_factory.noisy_encoder(reram_size, frequency, self.temperature)
        if self.noisy:
            hd_factory.noisy_inference(reram_size, frequency, self.temperature)
        hd = hd_factory.create().to(self.device)

        # testing
        (acc_avg, acc_std) = self._inference(hd)
        metrics = {"accuracy": (acc_avg, acc_std)}

        return metrics

    @torch.no_grad()
    def _inference(self, hd: HD):
        accs: List[float] = []
        for _ in tqdm(range(self.num_tests), desc="Inference"):
            acc = self.trainer.test(hd, self.test_loader)
            accs.append(acc)

        acc_avg = float(np.average(accs))
        acc_std = float(np.std(accs))

        return (acc_avg, acc_std)

    @staticmethod
    def name() -> str:
        return "accuracy"

    @staticmethod
    def optimization_type() -> str:
        return "max"

    @staticmethod
    def get_params() -> List[str]:
        return ["hd_dim", "f1", "d1", "reram_size", "kron", "frequency"]

    @staticmethod
    def ref_point() -> float:
        return 0.6
