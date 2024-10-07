from __future__ import annotations

import torch
from torch import Tensor, nn


def flip_with_prob(x: Tensor, p: float) -> Tensor:
    assert 0 <= p <= 1, f"Probability must be in [0, 1] but got {p}"

    flip_mask = torch.rand(x.shape) < p
    ret = x.clone()
    ret[flip_mask] = x[flip_mask] * -1

    # HACK: assume negative value is -1
    assert (
        torch.isclose(ret, Tensor(0.0)) | torch.isclose(ret, Tensor(1.0))
    ).all(), f"ret={ret}"

    return ret


class Flip(nn.Linear):
    def __init__(self, hd_dim: int, num_classes: int, prob: float, matrix: Tensor):
        super().__init__(hd_dim, num_classes, bias=False)

        # parameters
        self.prob = prob

        self.weight.data = matrix

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = super().__call__(x)
        return flip_with_prob(out, self.prob)
