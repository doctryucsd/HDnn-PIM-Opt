from __future__ import annotations

from torch import Tensor, nn

from pytorx.python.torx.module.layer import crxb_Linear

from ..utils import assert_linear_layer_matrix


def init_pytorx_linear_layer(
    input_size: int,
    output_size: int,
    tens: Tensor,
    reram_size: int,
    frequency: float,
    temperature: int,
):
    params = nn.parameter.Parameter(tens, requires_grad=False)
    ir_drop: bool = False
    torch_device = tens.device
    gmax: float = 1 / 3.03e3
    gmin: float = 1 / 3.03e6
    gwire: float = 1
    gload: float = 0.25
    quantize: int = 8

    return crxb_Linear(
        in_features=input_size,
        out_features=output_size,
        ir_drop=ir_drop,
        device=torch_device,
        gmax=gmax,
        gmin=gmin,
        gwire=gwire,
        gload=gload,
        bias=False,
        params=params,
        temp=temperature,
        freq=frequency,
        quantize=quantize,
        crxb_size=reram_size,
    )


def noisy_layer_factory(
    input_size: int,
    output_size: int,
    matrix: Tensor,
    reram_size: int,
    frequency: int,
    temperature: int,
    noise_type: str = "pytorx",
):
    assert_linear_layer_matrix(matrix, input_size, output_size)
    if noise_type == "pytorx":
        return init_pytorx_linear_layer(
            input_size, output_size, matrix, reram_size, frequency, temperature
        )
    elif noise_type == "flip":
        from .flip import Flip

        return Flip(input_size, output_size, 0.1, matrix)
    else:
        raise ValueError(f"Invalid noise type: {noise_type}")
