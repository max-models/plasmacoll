"""Null-collision Monte Carlo collisions of charged particles with neutral gases.

pymcc collides markers (positions, velocities and weights in NumPy or CuPy
arrays, through :mod:`cunumpy`) with prescribed Maxwellian neutral backgrounds:
elastic and backscatter collisions, excitation, ionization, attachment,
detachment and charge transfer, with cross sections from tables, LXCat files
or functions. It is the collision step of a particle-in-cell code, or with
:class:`ZeroDReactor` a homogeneous reactor for testing cross-section sets.

Example:
    >>> import pymcc
    >>> from pymcc.constants import ELECTRON_MASS, ATOMIC_MASS
    >>> mcc = pymcc.MonteCarloCollisions(
    ...     species_masses={"e": ELECTRON_MASS},
    ...     backgrounds=[pymcc.NeutralBackground("Ar", 1e21, 300.0, 39.95 * ATOMIC_MASS)],
    ...     processes={"e": [pymcc.CollisionProcess(
    ...         "elastic", "Ar", pymcc.CrossSection.constant(1e-19))]},
    ...     seed=1,
    ... )
"""

from pymcc.background import DensityProfile, NeutralBackground
from pymcc.cross_sections import (
    CrossSection,
    LXCatProcess,
    find_lxcat_process,
    read_lxcat,
)
from pymcc.hydrogen import proton_hydrogen_processes
from pymcc.markers import MarkerSet, ParticleArrays
from pymcc.mcc import (
    MCCDiagnostics,
    MCCResult,
    MonteCarloCollisions,
    NewMarkers,
    make_rng,
)
from pymcc.process import CollisionProcess
from pymcc.reactor import ReactorHistory, ReactorState, ZeroDReactor
from pymcc.vibrational import (
    VibrationalDistribution,
    boltzmann_level_populations,
    estimate_electronegativity,
)

__version__ = "0.1.0"  # x-release-please-version

__all__ = [
    "CollisionProcess",
    "CrossSection",
    "DensityProfile",
    "LXCatProcess",
    "MCCDiagnostics",
    "MCCResult",
    "MarkerSet",
    "MonteCarloCollisions",
    "NeutralBackground",
    "NewMarkers",
    "ParticleArrays",
    "ReactorHistory",
    "ReactorState",
    "VibrationalDistribution",
    "ZeroDReactor",
    "boltzmann_level_populations",
    "estimate_electronegativity",
    "find_lxcat_process",
    "make_rng",
    "proton_hydrogen_processes",
    "read_lxcat",
]
