"""The largest squared speed of markers stored as three component rows.

The Pyccel version (CPU) of the ``max_squared_speed`` kernel;
``max_squared_speed_numpy.py`` is the NumPy version with the same arguments.

With OpenMP the markers are split over the threads in contiguous chunks, each finding
the largest of its own (Pyccel has no ``reduction(max: ...)``), and the largest of the
chunks is returned: the Monte Carlo collisions take this bound every step, so it is a
pass at the speed of the threads, not of one core.
"""

# mypy: ignore-errors

from numpy import zeros
from pyccel.stdlib.internal.openmp import omp_get_max_threads, omp_get_thread_num

#: Fewest markers worth splitting over the threads.
MIN_PARALLEL_MARKERS = 20000


def _chunk_max(
    vx: "float[:]", vy: "float[:]", vz: "float[:]", first: int, last: int
) -> float:
    """Return the largest squared speed of the markers ``first .. last - 1`` (0 for none).

    A conditional expression, not ``max`` (a call in Pyccel's C), and no branch.
    """
    largest = 0.0
    for i in range(first, last):
        square = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]
        largest = square if square > largest else largest  # noqa: FURB136
    return largest


def max_squared_speed(vx: "float[:]", vy: "float[:]", vz: "float[:]") -> float:
    """Return ``max(vx^2 + vy^2 + vz^2)`` (0 for no markers)."""
    num_markers = vx.shape[0]
    num_threads = int(omp_get_max_threads())
    if num_markers < MIN_PARALLEL_MARKERS:
        num_threads = 1
    chunk = (num_markers + num_threads - 1) // num_threads
    partial = zeros(num_threads)
    # fmt: off
    #$ omp parallel num_threads(num_threads) private(thread, first, last)
    # fmt: on
    thread = int(omp_get_thread_num())
    first = min(thread * chunk, num_markers)
    last = min(first + chunk, num_markers)
    partial[thread] = _chunk_max(vx, vy, vz, first, last)
    # fmt: off
    #$ omp end parallel
    # fmt: on
    largest = 0.0
    for thread in range(num_threads):
        largest = partial[thread] if partial[thread] > largest else largest  # noqa: FURB136
    return largest
