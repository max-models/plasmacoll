"""The mcc_collide kernel of the Monte Carlo collisions with neutral backgrounds.

Pyccel and NumPy versions with the same arguments (no CUDA version: on the GPU
`plasmacoll.mcc.MonteCarloCollisions.collide` keeps its array operations).
`plasmacoll.mcc` calls it for the common case, see
`plasmacoll.mcc.MonteCarloCollisions.kernel_plan`.
"""

from plasmacoll.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__, missing_cuda="fallback")
