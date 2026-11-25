"""The max_squared_speed kernel: the largest squared speed of component rows.

Pyccel and NumPy versions with the same arguments (no CUDA version; on the GPU
`plasmacoll.mcc` reduces with array operations). `plasmacoll.mcc` calls it for the
bound of the Monte Carlo collisions.
"""

from plasmacoll.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__, missing_cuda="fallback")
