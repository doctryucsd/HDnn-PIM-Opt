from __future__ import annotations

from torch import Tensor, nn

from ..hd import HD
from .cnn import CNN


class HDNN(nn.Module):
    def __init__(self, cnn: CNN, hd: HD) -> None:
        super().__init__()

        self.cnn = cnn
        self.hd = hd

    def forward(self, x: Tensor, noise: bool) -> Tensor:
        nn_output: Tensor = self.cnn(x)
        ret: Tensor = self.hd(nn_output, noise)
        return ret

    def hd_init_buffer(self, x_train: Tensor, y_train: Tensor) -> None:
        hd_x_train: Tensor = self.cnn(x_train)
        self.hd.init_buffer(hd_x_train, y_train)

    def hd_retrain(self, x_train: Tensor, y_train: Tensor, lr: float) -> None:
        hd_x_train: Tensor = self.cnn(x_train)
        self.hd.retrain(hd_x_train, y_train, lr)
