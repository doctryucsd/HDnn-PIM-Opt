from __future__ import annotations

from torch import Tensor, nn

from .cnn import CNN


class PreTrain(nn.Module):
    def __init__(self, cnn: CNN, num_classes: int, cnn_output_dim: int) -> None:
        super().__init__()

        self.cnn = cnn

        self.fc = nn.Linear(cnn_output_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        cnn_out: Tensor = self.cnn(x)
        out: Tensor = self.fc(cnn_out)
        return out
