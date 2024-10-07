from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import Tensor


def bernoulli_1_0(shape: Tuple[int, ...]):
    return torch.bernoulli(0.5 * torch.ones(shape, dtype=torch.float32))


def bernoulli_1_neg1(shape: Tuple[int, ...]):
    return torch.bernoulli(0.5 * torch.ones(shape, dtype=torch.float32)) * 2 - 1


def bernoulli_factory(binarize_type: bool) -> Callable[[Tuple[int, ...]], Tensor]:
    if binarize_type:
        return bernoulli_1_0
    else:
        return bernoulli_1_neg1
