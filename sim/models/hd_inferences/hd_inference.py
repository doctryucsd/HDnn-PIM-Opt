from __future__ import annotations

import torch
from torch import Tensor, nn

from ..utils import assert_1_0, assert_1_neg1


class HDInference(nn.Module):
    def __init__(self, am: nn.Linear, binary: bool, binarize_type: bool):
        super().__init__()
        # parameters
        self.binary: bool = binary

        # modules
        self.am = am

        # get forward function
        if binarize_type:
            assert_1_0(self.am.weight.data)
            self._forward = self._hamming_distance
        else:
            assert_1_neg1(self.am.weight.data)
            self._forward = self._cosine_similarity

    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)

    def _hamming_distance(self, x: Tensor) -> Tensor:
        if self.binary:
            assert_1_0(x)

        num_one_one: Tensor = self.am(x)

        flipped_x = 1 - x
        num_zero_one: Tensor = self.am(flipped_x)
        num_x_zero = torch.sum(torch.isclose(x, torch.tensor(0.0)), dim=-1)
        num_zero_zero = num_x_zero.unsqueeze(-1) - num_zero_one

        return num_one_one + num_zero_zero

    def _cosine_similarity(self, x: Tensor) -> Tensor:
        if self.binary:
            assert_1_neg1(x)
        return self.am(x)
