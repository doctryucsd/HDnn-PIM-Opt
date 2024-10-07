from typing import Callable, List

import torch


# Define the constraint function
def sum_constraint(x: torch.Tensor) -> torch.Tensor:
    """
    Constraint function that requires the sum of two objectives to be less than 10.

    :param x: Tensor of dimension (sample_shape, batch_shape, q, m)
              where m = 2 (two objectives).
    :return: Tensor of dimension (sample_shape, batch_shape, q)
             with negative values indicating feasibility.
    """
    # Assume the tensor x has the shape (sample_shape, batch_shape, q, m)
    # where m = 2 (two objectives).
    return 10 - torch.sum(x, dim=-1)
