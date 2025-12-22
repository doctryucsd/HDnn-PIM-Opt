from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .cnn import CNN, CNN1D
from .encoders import Encoder
from .hd_inferences import HDInference


class HD(nn.Module):
    def __init__(
        self,
        cnn: CNN | CNN1D | None,
        encoder: Encoder,
        hd_inference: HDInference,
        binary: bool,
        binarize_type: bool,
    ) -> None:
        super().__init__()

        # parameters
        self.binary: bool = binary
        self.binarize_type: bool = binarize_type

        # modules
        self.cnn = cnn
        self.encoder = encoder
        self.hd_inference = hd_inference

    def forward(self, x: Tensor) -> Tensor:
        _, pred = self._forward(x)
        return pred

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        hv = self.feature_encode(x)

        # hd inference
        sim_score: Tensor = self.hd_inference(hv)
        assert (
            len(sim_score.shape) == 2
        ), f"sim_score should be 2D but got {sim_score.shape}"

        # prediction
        pred = torch.argmax(sim_score, dim=-1)

        return (hv, pred)

    def feature_encode(self, x: Tensor) -> Tensor:
        # cnn inference
        if self.cnn is not None:
            hd_in: Tensor = self.cnn(x)
        else:
            hd_in = x

        # encoding
        hv: Tensor = self.encoder(hd_in)
        return hv
