from __future__ import annotations

from logging import Logger

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .cnn import CNN, CNNFactory
from .encoders import EncoderFactory
from .hd import HD
from .hd_inferences import HDInferenceFactory


class HDFactory:
    def __init__(
        self,
        input_size: int,
        hd_dim: int,
        num_classes: int,
        binarize_type: bool,
        device: str | int,
        logger: Logger,
    ):
        # parameters
        self.binary: bool = False
        self.binarize_type = binarize_type
        self.num_classes = num_classes
        self.device = device
        self.logger = logger

        # components
        self.encoder_factory = EncoderFactory(input_size, hd_dim, device)
        self.hd_inference_factory = HDInferenceFactory(hd_dim, self.num_classes, device)

        # default cnn
        self.cnn: CNN | None = None

    def set_cnn(
        self,
        in_channels_1: int,
        out_channels_1: int,
        kernel_size_1: int,
        stride_1: int,
        padding_1: int,
        dilation_1: int,
        out_channels_2: int,
        kernel_size_2: int,
        stride_2: int,
        padding_2,
        dilation_2: int,
        inner_dim: int,
        epochs: int,
        lr: float,
        device: str | int,
        train_loader: DataLoader,
    ):
        self.cnn_factory = CNNFactory(
            in_channels_1,
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
            device,
        )
        trial_image = next(iter(train_loader))[0][0].to(device)
        output_dim: int = self.cnn_factory.get_output_dim(trial_image[None, :])
        self.logger.info(f"Output dimension of CNN: {output_dim}")
        self.cnn_factory.train(
            train_loader, self.num_classes, inner_dim, epochs, lr, True
        )
        self.cnn = self.cnn_factory.create()
        self.encoder_factory = EncoderFactory(
            output_dim, self.hd_inference_factory.hd_dim, device
        )

    def set_kronecker(self, d1: int, f1: int):
        self.encoder_factory.set_kronecker(d1, f1)

    def clone_param(self, hd: HD):
        if hd.cnn is not None:
            self.cnn_factory.clone_param(hd.cnn)
            self.cnn = self.cnn_factory.create()
        self.encoder_factory.clone_param(hd.encoder)
        self.hd_inference_factory.clone_param(hd.hd_inference)

    def bernoulli(self):
        self.encoder_factory.bernoulli()

    def binarize(self, binarize_type: bool):
        self.binary = True
        self.binarize_type = binarize_type
        self.encoder_factory.binarize(binarize_type)
        self.hd_inference_factory.binarize(binarize_type)

    def noisy_inference(self, reram_size: int, frequency: int, temperature: int):
        self.hd_inference_factory.noisy(reram_size, frequency, temperature)

    def noisy_encoder(self, reram_size: int, frequency: int, temperature: int):
        self.encoder_factory.noisy(reram_size, frequency, temperature)

    @torch.no_grad()
    def init_buffer(self, train_loader: DataLoader) -> None:
        self._encoder = self.encoder_factory.create().to(self.device)
        for x_train, y_train in train_loader:
            x_train_device: Tensor = x_train.to(self.device)
            y_train_device: Tensor = y_train.to(self.device)
            if self.cnn is not None:
                x_train_extracted: Tensor = self.cnn(x_train_device)
            else:
                x_train_extracted = x_train_device
            self._init_buffer(x_train_extracted, y_train_device)

        if self.binary:
            self.hd_inference_factory.binarize(self.binarize_type)

    @torch.no_grad()
    def retrain(self, train_loader: DataLoader, epochs: int, lr: float) -> None:
        assert hasattr(self, "_encoder"), "init_buffer should be called first"
        for _ in tqdm(range(epochs), desc="HD retraining"):
            for x_train, y_train in train_loader:
                # go through cnn
                x_train_device: Tensor = x_train.to(self.device)
                y_train_device: Tensor = y_train.to(self.device)
                if self.cnn is not None:
                    x_train_extracted: Tensor = self.cnn(x_train_device)
                else:
                    x_train_extracted = x_train_device
                self._retrain(x_train_extracted, y_train_device, lr)

        if self.binary:
            self.hd_inference_factory.binarize(self.binarize_type)

    def create(self):
        hd_inference = self.hd_inference_factory.create(self.binarize_type)
        encoder = self.encoder_factory.create()
        return HD(self.cnn, encoder, hd_inference, self.binary, self.binarize_type)

    def create_neurosim(self):
        cnn = self.cnn_factory.create_neurosim() if self.cnn is not None else None
        hd_inference = self.hd_inference_factory.create_neurosim(self.binarize_type)
        encoder = self.encoder_factory.create_neurosim()
        return HD(cnn, encoder, hd_inference, self.binary, self.binarize_type)

    @torch.no_grad()
    def _init_buffer(self, x_train: Tensor, y_train: Tensor):
        encoded_x_train: Tensor = self._encoder(x_train)
        self.hd_inference_factory.init_buffer(encoded_x_train, y_train)

    @torch.no_grad()
    def _retrain(self, x_train: Tensor, y_train: Tensor, lr: float) -> None:
        encoded_x_train: Tensor = self._encoder(x_train)
        self.hd_inference_factory.retrain(encoded_x_train, y_train, lr)
