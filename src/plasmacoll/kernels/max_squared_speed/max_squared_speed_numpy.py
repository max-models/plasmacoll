"""The ``max_squared_speed`` kernel with NumPy array operations."""

import numpy as np


def max_squared_speed(vx, vy, vz) -> float:
    """Return ``max(vx^2 + vy^2 + vz^2)`` (0 for no markers)."""
    if vx.shape[0] == 0:
        return 0.0
    return float(np.max(vx * vx + vy * vy + vz * vz))
