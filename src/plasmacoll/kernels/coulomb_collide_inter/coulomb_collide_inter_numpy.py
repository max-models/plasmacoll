"""The ``coulomb_collide_inter`` kernel without Pyccel: the Pyccel source run as Python.

The pairing and the scattering are sequential per cell, with random numbers drawn
outside the kernel, so this version is the Pyccel one interpreted: identical results,
slow (use it only on small cases and for checks).
"""

from plasmacoll.kernels.coulomb_collide_inter.coulomb_collide_inter_pyccel import (
    coulomb_collide_inter,
)

__all__ = ["coulomb_collide_inter"]
