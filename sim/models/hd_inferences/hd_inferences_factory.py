from __future__ import annotations

import torch
from torch import Tensor, nn

from neurosim.Inference_pytorch.modules import QLinear

from ..noisy_layers import noisy_layer_factory
from ..utils import assert_linear_layer_matrix, binarize_factory
from .hd_inference import HDInference


class HDInferenceFactory:
    def __init__(self, hd_dim: int, num_classes: int, device: str):
        # parameters
        self.hd_dim: int = hd_dim
        self.num_classes: int = num_classes
        self.binary: bool = False

        # components
        self.class_hvs = torch.zeros(
            num_classes, hd_dim, dtype=torch.float32, device=device
        )

    @torch.no_grad()
    def init_buffer(self, x_train: Tensor, y_train: Tensor) -> None:
        for i in range(self.num_classes):
            idx = y_train == i
            self.class_hvs[i] += torch.sum(x_train[idx], dim=0)

    @torch.no_grad()
    def retrain(
        self,
        x_train: Tensor,
        y_train: Tensor,
        lr: float,
    ) -> None:
        """
        Update the class hyper-vectors based on the wrong predictions.
        """

        shuffle_idx = torch.randperm(x_train.size()[0])
        shuffle_x_train = x_train[shuffle_idx]
        shuffle_y_train = y_train[shuffle_idx]

        for i in range(shuffle_x_train.size()[0]):
            pred = self._predict(shuffle_x_train[i])

            if pred != shuffle_y_train[i]:
                self.class_hvs[pred] -= lr * shuffle_x_train[i]
                self.class_hvs[shuffle_y_train[i]] += lr * shuffle_x_train[i]

        # cnt_pos: int = 0
        # cnt_neg: int = 0

        # pred = self._predict(x_train)
        # for i in range(self.num_classes):
        #     # update wrong predictions
        #     wrong_pred = y_train != pred
        #     pos_idx = (y_train == i) * wrong_pred
        #     neg_idx = (pred == i) * wrong_pred

        #     # count wrong predictions
        #     cnt_pos += int(torch.sum(pos_idx))
        #     cnt_neg += int(torch.sum(neg_idx))

        #     # update
        #     self.class_hvs[i] += x_train[pos_idx].sum(dim=0) * lr
        #     self.class_hvs[i] -= x_train[neg_idx].sum(dim=0) * lr

        # assert cnt_pos == cnt_neg, f"cnt_pos: {cnt_pos}, cnt_neg: {cnt_neg}"

    def binarize(self, binarize_type: bool) -> None:
        self.binarize_type = binarize_type
        self.binarize_func = binarize_factory(binarize_type)
        self.class_hvs = self.binarize_func(self.class_hvs)
        self.binary = True

    def set_class_hvs(self, class_hvs: Tensor) -> None:
        assert_linear_layer_matrix(class_hvs, self.hd_dim, self.num_classes)
        self.class_hvs = class_hvs.clone()

    def clone_param(self, hd_inference: HDInference) -> None:
        layer = hd_inference.am
        assert isinstance(layer, nn.Linear), f"Expected nn.Linear but got {type(layer)}"
        self.set_class_hvs(layer.weight.data)

    def noisy(self, reram_size: int, frequency: int, temperature: int) -> None:
        self.am = noisy_layer_factory(
            self.hd_dim,
            self.num_classes,
            self.class_hvs.clone(),
            reram_size,
            frequency,
            temperature,
        )

    def create(self, binarize_type: bool) -> HDInference:
        if not hasattr(self, "am"):
            self.am = self._get_default_layer()

        return HDInference(self.am, self.binary, binarize_type)

    def create_neurosim(self, binarize_type: bool) -> HDInference:
        return HDInference(self._get_neurosim_layer(), self.binary, binarize_type)

    def _get_default_layer(self) -> nn.Linear:
        am = nn.Linear(self.hd_dim, self.num_classes, bias=False)
        am.weight.data = self.class_hvs.clone()
        return am

    def _get_neurosim_layer(self) -> QLinear:
        am = QLinear(self.hd_dim, self.num_classes, name="HDInference")
        am.weight.data = self.class_hvs.clone()
        return am

    def _predict(self, x: Tensor) -> Tensor:
        assert self.binarize_type is not None, "should call binarize first"
        binarize_func = binarize_factory(self.binarize_type)
        binarized_class_hvs = binarize_func(self.class_hvs)
        if self.binarize_type:
            num_one_one: Tensor = x @ self.class_hvs.T

            flipped_x = 1 - x
            num_zero_one: Tensor = flipped_x @ binarized_class_hvs.T
            num_x_zero = torch.sum(torch.isclose(x, torch.tensor(0.0)), dim=-1)
            num_zero_zero = num_x_zero.unsqueeze(-1) - num_zero_one

            sim = num_one_one + num_zero_zero
        else:
            sim = x @ binarized_class_hvs.T

        # return torch.argmax(sim, 1)
        return torch.argmax(sim)
