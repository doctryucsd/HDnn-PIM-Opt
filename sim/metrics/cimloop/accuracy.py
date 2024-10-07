from __future__ import annotations

from logging import Logger
from typing import Any, Dict, List, Tuple

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
        self.input_channels: int = model_args["input_channels"]

        # training args
        self.trainer = HDTrainer(training_args)
        self.num_tests: int = training_args["num_tests"]
        self.hd_epochs: int = training_args["hd_epochs"]
        self.hd_lr: float = training_args["hd_lr"]
        self.cnn_epochs: int = training_args["cnn_epochs"]
        self.cnn_lr: float = training_args["cnn_lr"]
        self.device: str = training_args["device"]

        # hardware args
        self.noisy: bool = hardware_args["noise"]
        self.temperature: int = hardware_args["temperature"]
        self.cnn: bool = hardware_args["cnn"]

        # dataloaders
        self.train_loader: DataLoader = train_loader
        self.test_loader: DataLoader = test_loader

    def evaluate(
        self, params: Dict[str, Any], logger: Logger
    ) -> Dict[str, Tuple[float, float]]:
        # params
        hd_dim: int = params["hd_dim"]
        reram_size: int = params["reram_size"]
        frequency: int = params["frequency"]
        binarize_type: bool = False
        out_channels_1: int = params["out_channels_1"]
        kernel_size_1: int = params["kernel_size_1"]
        stride_1: int = params["stride_1"]
        padding_1: int = params["padding_1"]
        dilation_1: int = params["dilation_1"]
        out_channels_2: int = params["out_channels_2"]
        kernel_size_2: int = params["kernel_size_2"]
        stride_2: int = params["stride_2"]
        padding_2: int = params["padding_2"]
        dilation_2: int = params["dilation_2"]
        inner_dim: int = params["inner_dim"]

        # construct hd
        hd_factory = HDFactory(
            self.input_size,
            hd_dim,
            self.num_classes,
            binarize_type,
            self.device,
            logger,
        )

        # construct cnn and encoder
        if self.cnn:
            hd_factory.set_cnn(
                self.input_channels,
                out_channels_1,
                kernel_size_1,
                stride_1,
                padding_1,
                dilation_1,
                out_channels_2,
                kernel_size_2,
                stride_2,
                padding_2,
                dilation_2,
                inner_dim,
                self.cnn_epochs,
                self.cnn_lr,
                self.device,
                self.train_loader,
            )
        else:
            kron: bool = params["kron"]
            if kron:
                f1: int = params["f1"]
                d1: int = params["d1"]
                hd_factory.set_kronecker(d1, f1)

        hd_factory.bernoulli()
        hd_factory.binarize(binarize_type)
        hd_factory.init_buffer(self.train_loader)
        hd_factory.retrain(self.train_loader, self.hd_epochs, self.hd_lr)

        # finish training
        if self.noisy:
            if not self.cnn:
                if not kron:
                    hd_factory.noisy_encoder(reram_size, frequency, self.temperature)
            hd_factory.noisy_inference(reram_size, frequency, self.temperature)
        hd = hd_factory.create().to(self.device)

        # testing
        (acc_avg, acc_std) = self._inference(hd)

        logger.info(f"Accuracy: {acc_avg:.2f} ± {acc_std:.2f}")

        # HACK: weight is 10
        WEIGHT = 1.0
        ret_avg = acc_avg * WEIGHT
        ret_std = acc_std * WEIGHT
        metrics = {"accuracy": (ret_avg, ret_std)}

        self.hd_factory = hd_factory

        return metrics

    @property
    def hd_model_non_noisy(self) -> HD:
        assert hasattr(self, "hd_factory"), "evaluate should be called first"
        return self.hd_factory.create_neurosim()

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
        return 0.0
