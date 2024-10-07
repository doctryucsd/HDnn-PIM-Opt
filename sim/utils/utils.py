from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from botorch.utils.multi_objective import is_non_dominated
from torch import Tensor
from torch.utils.data import DataLoader


def calculate_acc(pred: Tensor, label: Tensor):
    correct = (pred == label).float()
    accuracy = correct.sum() / len(correct)
    return float(accuracy)


def plot_xy(
    x: Sequence,
    y: Sequence,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
    texts: List = [],
):
    assert len(x) == len(y), f"len(x)={len(x)} != len(y)={len(y)}"

    indices = list(range(len(x)))

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot with colors based on original indices
    scatter = ax.scatter(x, y, c=indices, cmap="viridis")

    # Add a colorbar with selected indices
    num_ticks = 5  # Number of ticks to display on the colorbar
    tick_indices = [
        int(i) for i in [len(indices) * j / (num_ticks - 1) for j in range(num_ticks)]
    ]
    tick_indices.append(len(indices) - 1)  # Add the final index
    tick_indices = sorted(set(tick_indices))  # Remove duplicates and sort
    cbar = plt.colorbar(scatter, ax=ax, ticks=tick_indices)
    cbar.ax.set_yticklabels(tick_indices)  # type: ignore
    cbar.set_label("Iteration", rotation=270, labelpad=20)

    # Set axis labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.savefig(save_path)
    plt.close()


def plot_pareto_set(
    obj1: Sequence,
    obj2: Sequence,
    pareto1: Sequence,
    pareto2: Sequence,
    posterior_pareto1: Sequence,
    posterior_pareto2: Sequence,
    obj1_label: str,
    obj2_label: str,
    title: str,
    save_path: str,
):
    assert len(obj1) == len(obj2), f"len(x)={len(obj1)} != len(y)={len(obj2)}"
    plt.scatter(obj1, obj2, c="b")
    plt.scatter(pareto1, pareto2, c="r")

    # Sort posterior based on metric1
    sorted_pairs = sorted(
        zip(posterior_pareto1, posterior_pareto2), key=lambda pair: pair[0]
    )
    sorted_posterior_pareto1, sorted_posterior_pareto2 = zip(*sorted_pairs)
    plt.plot(sorted_posterior_pareto1, sorted_posterior_pareto2, c="g")

    plt.legend(["Dominated", "Non Dominated", "Posterior Pareto"])
    plt.xlabel(obj1_label)
    plt.ylabel(obj2_label)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def generate_arithmetic_sequence(min_val: float, max_val: float, num_elements: int):
    # Calculate the common difference
    d = (max_val - min_val) / (num_elements - 1)

    # Generate the sequence
    sequence = [min_val + i * d for i in range(num_elements)]

    return sequence


def get_image_shape(dataloader: DataLoader) -> Tuple[int, int, int]:
    # Take the first batch
    data = next(iter(dataloader))
    # Check if the DataLoader returns a tuple (images, labels) or just images
    if isinstance(data, tuple):
        images, _ = data
    elif isinstance(data, list):
        assert len(data) == 2, f"len(data)={len(data)}"
        images = data[0]
    else:
        images = data

    # Print the shape of the images in the first batch and exit the loop
    shape = images.shape
    if len(shape) == 4:
        _, c, h, w = shape
        return c, h, w
    elif len(shape) == 3:
        _, h, w = shape
        return 1, h, w
    elif len(shape) == 2:
        _, w = shape
        return 1, 1, w
    else:
        raise ValueError(f"Invalid shape: {shape}")


def get_num_classes(dataloader: DataLoader) -> int:
    # Iterate over the DataLoader
    data = next(iter(dataloader))
    # Check if the DataLoader returns a tuple (images, labels) or just images
    if isinstance(data, tuple):
        _, labels = data
    elif isinstance(data, list):
        assert len(data) == 2, f"len(data)={len(data)}"
        labels = data[1]
    else:
        raise ValueError("The DataLoader must include (images, labels)")

    # Return the number of unique classes
    return len(labels.unique())


def dim_after_conv_or_pool(size: int, filter_dim: int, stride: int, padding: int = 0):
    return ((size - filter_dim + 2 * padding) // stride) + 1


def get_cnn_out_dim(h: int, w: int) -> int:
    # HACK: parameters
    kernel_size: int = 5
    output_channel: int = 128

    stride_conv = 1  # Typically 1 for convolutional layers
    stride_pool = 2  # Typically equal to pool size for max pooling layers
    pool_size = 2

    # HACK: model architecture
    # Calculate size after first conv+pool
    height_after_model0 = dim_after_conv_or_pool(h, kernel_size, stride_conv)
    width_after_model0 = dim_after_conv_or_pool(w, kernel_size, stride_conv)
    height_after_pool0 = dim_after_conv_or_pool(
        height_after_model0, pool_size, stride_pool
    )
    width_after_pool0 = dim_after_conv_or_pool(
        width_after_model0, pool_size, stride_pool
    )

    # Calculate size after second conv+pool
    height_after_model1 = dim_after_conv_or_pool(
        height_after_pool0, kernel_size, stride_conv
    )
    width_after_model1 = dim_after_conv_or_pool(
        width_after_pool0, kernel_size, stride_conv
    )
    height_after_pool1 = dim_after_conv_or_pool(
        height_after_model1, pool_size, stride_pool
    )
    width_after_pool1 = dim_after_conv_or_pool(
        width_after_model1, pool_size, stride_pool
    )

    # Calculate total features after flattening
    total_features = height_after_pool1 * width_after_pool1 * output_channel

    return total_features


def get_params_from_loader(dataloader: DataLoader):
    channels, h, w = get_image_shape(dataloader)
    num_classes = get_num_classes(dataloader)
    # cnn_output_dim = get_cnn_out_dim(h, w)
    input_size = h * w * channels
    return num_classes, input_size, channels


def assert_binary(tensor: Tensor):
    """
    Asserts that the tensor contains only 0s and 1s.
    """
    assert torch.all(
        torch.isclose(tensor, torch.zeros_like(tensor))
        | torch.isclose(tensor, torch.ones_like(tensor))
    ), "Only 1 and 0 in the query!"


def get_pareto_set(
    metrics: List[Dict[str, Tuple[float, float]]], metrics_type: List[Tuple[str, str]]
):
    y_list: List[List[float]] = []
    for metric in metrics:
        pareto_point: List[float] = []
        for metric_name, metric_type in metrics_type:
            if metric_type == "min":
                pareto_point.append(-metric[metric_name][0])
            elif metric_type == "max":
                pareto_point.append(metric[metric_name][0])
            else:
                raise ValueError(f"Unknown metric_type: {metric_type}")
        y_list.append(pareto_point)
    y: Tensor = Tensor(y_list)
    output = is_non_dominated(y)
    assert len(output.shape) == 1, "output should be 1D"
    ret = []
    for idx, is_in_pareto in enumerate(output):
        if is_in_pareto:
            ret.append(metrics[idx])
    return ret


def dump_metrics(metrics: Dict[str, List[float]], filename: str) -> None:
    with open(f"{filename}.json", "w") as f:
        json.dump(metrics, f)
