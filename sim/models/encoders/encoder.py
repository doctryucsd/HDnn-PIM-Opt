from __future__ import annotations

from torch import Tensor, nn

from ..utils import binarize_factory


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        layers: nn.Module,
        binary: bool,
        binarize_type: bool | None,
    ):
        super().__init__()

        # parameters
        self.input_size = input_size

        # modules
        self.layers = layers

        # binarize
        self.binary = binary
        if self.binary:
            assert binarize_type is not None, "binarize_type should be provided"
            self.binarize = binarize_factory(binarize_type)

    def forward(self, x: Tensor) -> Tensor:
        reshaped_x = x.view(x.shape[0], -1, self.input_size)
        if reshaped_x.shape[1] == 1:
            processed_x = reshaped_x.squeeze(1)
        else:
            processed_x = reshaped_x

        encoded_x: Tensor = self.layers(processed_x)
        flattened_x = encoded_x.flatten(1)
        if self.binary:
            ret = self.binarize(flattened_x)
        return ret
