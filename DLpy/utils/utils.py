from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray


def calculate_fan_in_fan_out(tensor: NDArray[Any]) -> Tuple[int, int]:
    """
    Calculate fan-in and fan-out of a tensor.

    Args:
        tensor: Input tensor

    Returns:
        Tuple of (fan_in, fan_out)

    Note:
        For linear layers:
            fan_in is input dimensions,
            fan_out is output dimensions
        For conv layers:
            fan_in is (channels_in * kernel_size),
            fan_out is (channels_out * kernel_size)
    """
    dimensions = tensor.shape

    if len(dimensions) == 2:  # Linear layer
        fan_in, fan_out = dimensions[1], dimensions[0]

    elif len(dimensions) > 2:  # Convolution layers
        receptive_field_size = np.prod(dimensions[2:])  # kernel size
        fan_in = dimensions[1] * receptive_field_size  # channels_in * kernel_size
        fan_out = dimensions[0] * receptive_field_size  # channels_out * kernel_size

    else:
        raise ValueError(
            f"tensor.shape should have at least 2 dimensions, got {dimensions}"
        )

    return fan_in, fan_out
