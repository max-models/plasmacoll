"""The largest squared speed of markers stored as three component rows.

The Pyccel version (CPU) of the ``max_squared_speed`` kernel;
``max_squared_speed_numpy.py`` is the NumPy version with the same arguments.

With OpenMP the markers are split into ``CHUNKS`` contiguous chunks, shared over the
threads, each chunk finding the largest of its own (Pyccel has no ``reduction(max:
...)``), and the largest of the chunks is returned: the Monte Carlo collisions take this
bound every step, so it is a pass at the speed of the threads, not of one core. No
import from ``pyccel``, so the module also imports where Pyccel is missing (Pyodide).
"""

# mypy: ignore-errors

from numpy import zeros

#: Fewest markers worth splitting over the threads.
MIN_PARALLEL_MARKERS = 20000
#: Chunks of a split: a few per thread on any machine.
CHUNKS = 64


def _chunk_max(
    vx: "float[:]", vy: "float[:]", vz: "float[:]", chunk: int, size: int
) -> float:
    """Return the largest squared speed of chunk ``chunk`` of ``size`` markers (0 for none).

    A conditional expression, not ``max`` (a call in Pyccel's C), and no branch.
    """
    num_markers = vx.shape[0]
    first = chunk * size
    last = first + size
    if last > num_markers:
        last = num_markers
    largest = 0.0
    for i in range(first, last):
        square = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]
        largest = square if square > largest else largest  # noqa: FURB136
    return largest


def max_squared_speed(vx: "float[:]", vy: "float[:]", vz: "float[:]") -> float:
    """Return ``max(vx^2 + vy^2 + vz^2)`` (0 for no markers)."""
    num_markers = vx.shape[0]
    num_chunks = CHUNKS
    if num_markers < MIN_PARALLEL_MARKERS:
        num_chunks = 1
    size = (num_markers + num_chunks - 1) // num_chunks
    partial = zeros(num_chunks)
    # fmt: off
    #$ omp parallel for schedule(static)
    # fmt: on
    for chunk in range(num_chunks):
        partial[chunk] = _chunk_max(vx, vy, vz, chunk, size)
    largest = 0.0
    for chunk in range(num_chunks):
        largest = partial[chunk] if partial[chunk] > largest else largest  # noqa: FURB136
    return largest
