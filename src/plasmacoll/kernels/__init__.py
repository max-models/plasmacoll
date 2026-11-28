"""plasmacoll's compiled kernels, one folder per kernel.

Every kernel ``<name>`` lives in its own folder, with versions that take the same arguments::

    plasmacoll/kernels/<name>/__init__.py        kernel = KERNEL_PACKAGE.from_folder(__name__)
    plasmacoll/kernels/<name>/<name>_pyccel.py   CPU, compiled with Pyccel on first use
    plasmacoll/kernels/<name>/<name>_numpy.py    CPU, when Pyccel is unavailable
    plasmacoll/kernels/<name>/<name>_cuda.cu     GPU, compiled with NVRTC

``kernel(...)`` runs the CUDA version when the arrays are CuPy arrays and the compiled
Pyccel version otherwise (see `cunumpy.kernels.KernelPackage`). The Pyccel builds are
kept under ``PLASMACOLL_KERNEL_CACHE`` (default ``~/.cache/plasmacoll/kernels``;
``PLASMACOLL_OPENMP=0`` builds serial). Importing a kernel needs neither Pyccel nor CuPy.

The kernels are those of the binary Coulomb collisions (`plasmacoll.coulomb`):
``coulomb_collide_intra`` and ``coulomb_collide_inter``, with the pair scattering of
their CUDA versions in ``coulomb_pair.cuh``.
"""

import cunumpy as xp

#: The build cache and launch options of plasmacoll's kernels.
KERNEL_PACKAGE = xp.kernels.KernelPackage("plasmacoll")
