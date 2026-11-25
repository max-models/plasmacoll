"""The largest squared speed of markers stored as three component rows.

The Pyccel version (CPU) of the ``max_squared_speed`` kernel;
``max_squared_speed_numpy.py`` is the NumPy version with the same arguments.
"""

# mypy: ignore-errors


def max_squared_speed(vx: "float[:]", vy: "float[:]", vz: "float[:]") -> float:
    """Return ``max(vx^2 + vy^2 + vz^2)`` (0 for no markers)."""
    largest = 0.0
    for i in range(vx.shape[0]):
        square = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]
        if square > largest:
            largest = square
    return largest
