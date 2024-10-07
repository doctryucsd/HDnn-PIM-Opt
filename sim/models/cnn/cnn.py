from __future__ import annotations

import torch
from torch import Tensor, nn


class CNN(nn.Module):
    def __init__(self, cnn1: nn.Conv2d, cnn2: nn.Conv2d):
        super().__init__()

        assert (
            cnn1.out_channels == cnn2.in_channels
        ), f"cnn1.out_channels: {cnn1.out_channels} != cnn2.in_channels: {cnn2.in_channels}"

        self.cnn1 = cnn1
        self.cnn2 = cnn2

    def forward(self, x: Tensor) -> Tensor:
        out1: Tensor = self.cnn1(x)
        relu_out: Tensor = torch.relu(out1)
        out2: Tensor = self.cnn2(relu_out)
        pool_out: Tensor = torch.max_pool2d(out2, 2)

        ret = torch.flatten(pool_out, 1)
        return ret
