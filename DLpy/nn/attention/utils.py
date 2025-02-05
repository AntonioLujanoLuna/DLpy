from typing import Any

import numpy as np
from numpy.typing import NDArray


def get_angles(pos: NDArray[Any], i: NDArray[Any], d_model: int) -> NDArray[Any]:
    """
    Calculate the angles for positional encoding.

    Args:
        pos: Array of positions [0, 1, 2, ...]
        i: Array of dimension indices [0, 2, 4, ...] for even indices
        d_model: The model dimension

    Returns:
        Array of angles for positional encoding
    """
    # Compute angle rates
    angle_rates = 1 / np.power(10000, (2 * i) / np.float32(d_model))

    # Return position-dependent angles
    return pos[:, np.newaxis] * angle_rates[np.newaxis, :]


# TODO Add: Attention score computation utilities, mask generation utilities
