from __future__ import annotations

from torch import Tensor, nn

from .cnn import CNN, CNN1D


class CNNTrain(nn.Module):
    def __init__(
        self, cnn: CNN | CNN1D, cnn_output_dim: int, num_classes: int, inner_dim: int
    ) -> None:
        super().__init__()

        self.cnn = cnn
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        cnn_out: Tensor = self.cnn(x)
        logit: Tensor = self.fc(cnn_out)
        ret: Tensor = nn.functional.log_softmax(logit, dim=1)
        return ret
