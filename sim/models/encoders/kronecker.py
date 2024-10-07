from __future__ import annotations

import torch
from torch import Tensor, nn


class Kronecker(nn.Module):
    def __init__(self, layer1: nn.Linear, layer2: nn.Linear):
        super().__init__()

        self.f1 = layer1.in_features
        self.f2 = layer2.in_features

        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x: Tensor):
        reshaped_x = x.reshape(x.shape[0], -1, self.f1)
        mid: Tensor = self.layer1(reshaped_x)
        reshaped_mid = mid.transpose(-2, -1)
        ret: Tensor = self.layer2(reshaped_mid)

        return torch.flatten(ret, start_dim=1)
