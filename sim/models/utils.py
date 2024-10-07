from __future__ import annotations

import copy

import torch
from torch import Tensor


def assert_linear_layer_matrix(matrix: Tensor, input_size: int, output_size: int):
    """
    Asserts that the matrix is a 2-d array with shape (output_size, input_size), so that the matrix can be used as a linear layer.
    """
    assert (
        len(matrix.shape) == 2
    ), f"matrix should be a 2-d array but got shape: {matrix.shape}"
    assert (
        matrix.shape[1] == input_size and matrix.shape[0] == output_size
    ), f"(input_size, output_size): ({input_size}, {output_size}) not compatible with matrix.T.shape: {matrix.T.shape}"


def assert_1_0(x: Tensor):
    assert (
        torch.isclose(x, torch.tensor(0.0)) | torch.isclose(x, torch.tensor(1.0))
    ).all(), f"Expected all elements to be either 0 or 1 but got {x}"


def assert_1_neg1(x: Tensor):
    assert (
        torch.isclose(x, torch.tensor(1.0)) | torch.isclose(x, torch.tensor(-1.0))
    ).all(), f"Expected all elements to be either 1 or -1 but got {x}"


def binarize_1_0(x: Tensor):
    assert len(x.shape) == 2, f"Input tensor must be 2D but got {x.shape}"
    return torch.where(x > 0, 1.0, 0.0)


def binarize_1_neg1(x: Tensor):
    assert len(x.shape) == 2, f"Input tensor must be 2D but got {x.shape}"
    return torch.where(x > 0, 1.0, -1.0)


def binarize_1_0_soft(x: Tensor):
    assert len(x.shape) == 2, f"Input tensor must be 2D but got {x.shape}"
    return torch.sigmoid(x)


def binarize_1_neg1_soft(x: Tensor):
    assert len(x.shape) == 2, f"Input tensor must be 2D but got {x.shape}"
    return torch.tanh(x)


def binarize_factory(binarize_type: bool, soft: bool = False):
    """
    Returns a binarize function based on the binarize type.

    binarize_type: True for 1/0, False for 1/-1
    """
    if binarize_type:
        if soft:
            return binarize_1_0_soft
        else:
            return binarize_1_0
    else:
        if soft:
            return binarize_1_neg1_soft
        else:
            return binarize_1_neg1


def quantize_feature(x: Tensor, Q: int, scale: float = 0.5, zero: int = 0):
    return torch.fake_quantize_per_tensor_affine(
        copy.copy(x),
        scale=scale,
        zero_point=zero,
        quant_min=int(-Q / 2),
        quant_max=int(Q / 2 - 1),
    )
