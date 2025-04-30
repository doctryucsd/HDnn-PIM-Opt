from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from neurosim.Inference_pytorch.modules import QLinear

from ..noisy_layers import noisy_layer_factory
from .encoder import Encoder
from .kronecker import Kronecker
from .utils import bernoulli_1_neg1


class EncoderFactory:
    def __init__(self, input_size: int, hd_dim: int, device: str | int):
        """
        Initialization: one matrix with random values.
        """
        # parameters
        self.input_size: int = input_size
        self.hd_dim: int = hd_dim
        self.device: str | int = device

        # flags
        self.binary: bool = False
        self.binarize_type: bool | None = None

        # components
        self.matrices: List[Tensor] = [
            torch.rand(
                size=(input_size, hd_dim), dtype=torch.float32, device=self.device
            ),
        ]

    def set_kronecker(self, d1: int, f1: int) -> None:
        # fix me
        assert (
            self.hd_dim % d1 == 0
        ), f"hd_dim: {self.hd_dim} should be divisible by d1: {d1}"
        assert (
            self.input_size % f1 == 0
        ), f"input_size: {self.input_size} should be divisible by f1: {f1}"

        d2 = self.hd_dim // d1
        f2 = self.input_size // f1
        self.matrices = [
            torch.rand(size=(f1, d1), dtype=torch.float32, device=self.device),
            torch.rand(size=(f2, d2), dtype=torch.float32, device=self.device),
        ]
        self.input_size = f1

    def bernoulli(self) -> None:
        """
        Initialize matrices with Bernoulli distribution.
        """
        matrices_shape = [tuple(matrix.shape) for matrix in self.matrices]
        new_matrices = [
            bernoulli_1_neg1(matrix_shape).to(self.device)
            for matrix_shape in matrices_shape
        ]
        self.matrices = new_matrices

    def binarize(self, binarize_type: bool) -> None:
        self.binary = True
        self.binarize_type = binarize_type

    def set_matrices(self, matrices: List[Tensor]) -> None:
        assert len(matrices) == len(
            self.matrices
        ), f"len(matrices): {len(matrices)} should be equal to len(self.matrices): {len(self.matrices)}"
        for new_matrix, old_matrix in zip(matrices, self.matrices):
            assert (
                new_matrix.shape == old_matrix.shape
            ), f"new_matrix.shape: {new_matrix.shape} should be equal to old_matrix.shape: {old_matrix.shape}"

        self.matrices = [matrix.clone() for matrix in matrices]

    def clone_param(self, encoder: Encoder) -> None:
        matrices: List[Tensor] = []
        for _, layer in encoder.named_children():
            assert isinstance(
                layer, nn.Linear
            ), f"layer should be nn.Linear but got {type(layer)}"
            matrices.append(layer.weight.data.clone())
        self.set_matrices(matrices)

    def noisy(self, reram_size: int, frequency: int, temperature: int) -> None:
        layers_list: List[nn.Module] = []
        for matrix in self.matrices:
            shape = tuple(matrix.shape)
            layer = noisy_layer_factory(
                shape[0], shape[1], matrix.T.clone(), reram_size, frequency, temperature
            )
            layers_list.append(layer)

        self.layers = nn.Sequential(*layers_list)

    def create(self):
        if not hasattr(self, "layers"):
            self.layers = self._get_default_layer()
        return Encoder(self.input_size, self.layers, self.binary, self.binarize_type)

    def create_neurosim(self):
        return Encoder(
            self.input_size, self._get_neurosim_layer(), self.binary, self.binarize_type
        )

    def _get_default_layer(self) -> nn.Module:
        if len(self.matrices) == 1:
            shape = tuple(self.matrices[0].shape)
            layer = nn.Linear(shape[0], shape[1], bias=False)
            layer.weight.data = self.matrices[0].T.clone()
            return layer
        elif len(self.matrices) == 2:
            layers_list: List[nn.Linear] = []
            for matrix in self.matrices:
                shape = tuple(matrix.shape)
                layer = nn.Linear(shape[0], shape[1], bias=False)
                layer.weight.data = matrix.T.clone()
                layers_list.append(layer)
            return Kronecker(layers_list[0], layers_list[1])
        else:
            raise NotImplementedError

    def _get_neurosim_layer(self) -> nn.Module:
        if len(self.matrices) == 1:
            shape = tuple(self.matrices[0].shape)
            # layer = nn.Linear(shape[0], shape[1], bias=False)
            layer = QLinear(shape[0], shape[1], name="Encoder")
            layer.weight.data = self.matrices[0].T.clone()
            return layer
        elif len(self.matrices) == 2:
            layers_list: List[nn.Linear] = []
            for i, matrix in enumerate(self.matrices):
                shape = tuple(matrix.shape)
                # layer = nn.Linear(shape[0], shape[1], bias=False)
                layer = QLinear(shape[0], shape[1], name=f"Encoder{i}")
                layer.weight.data = matrix.T.clone()
                layers_list.append(layer)
            return Kronecker(layers_list[0], layers_list[1])
        else:
            raise NotImplementedError
