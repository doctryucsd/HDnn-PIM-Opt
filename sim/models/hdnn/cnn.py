from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# from torchvision import models


class CNN(nn.Module):
    def __init__(self, input_channel: int) -> None:
        super().__init__()

        # HACK: parameters
        kernel_size: int = 5
        num_filter: int = 16
        output_channel: int = 128

        self.model0 = nn.Sequential(
            nn.Conv2d(input_channel, num_filter, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.model1 = nn.Sequential(
            nn.Conv2d(num_filter, output_channel, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # self.model = nn.Sequential(*list(model.features.children())[:5])

    def forward(self, x: Tensor) -> Tensor:
        model0_out: Tensor = self.model0(x)
        model1_out: Tensor = self.model1(model0_out)
        out: Tensor = torch.flatten(model1_out, 1)
        return out
