"""The coulomb_collide_inter kernel of the binary Coulomb collisions.

Pyccel, NumPy and CUDA versions with the same arguments. ``kernel(...)`` runs the CUDA
version when the arrays are CuPy arrays, and otherwise the Pyccel version, or the NumPy
version if Pyccel is unavailable; one CUDA thread per cell (the first array is the
per-cell coefficient). `plasmacoll.coulomb` calls it.
"""

from plasmacoll.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
