"""The ``mcc_collide`` kernel without Pyccel: the Pyccel source run as Python.

Identical results, slow (use it only on small cases and for checks).
"""

from plasmacoll.kernels.mcc_collide.mcc_collide_pyccel import mcc_collide

__all__ = ["mcc_collide"]
