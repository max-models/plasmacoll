"""Null-collision Monte Carlo collisions of charged particles with neutral gases.

plasmacoll collides markers (positions, velocities and weights in NumPy or CuPy
arrays, through :mod:`cunumpy`) with prescribed Maxwellian neutral backgrounds:
elastic and backscatter collisions, excitation, ionization, attachment,
detachment and charge transfer, with cross sections from tables, LXCat files
or functions. It is the collision step of a particle-in-cell code, or with
:class:`ZeroDReactor` a homogeneous reactor for testing cross-section sets.

Example:
    >>> import plasmacoll
    >>> from plasmacoll.constants import ELECTRON_MASS, ATOMIC_MASS
    >>> mcc = plasmacoll.MonteCarloCollisions(
    ...     species_masses={"e": ELECTRON_MASS},
    ...     backgrounds=[plasmacoll.NeutralBackground("Ar", 1e21, 300.0, 39.95 * ATOMIC_MASS)],
    ...     processes={"e": [plasmacoll.CollisionProcess(
    ...         "elastic", "Ar", plasmacoll.CrossSection.constant(1e-19))]},
    ...     seed=1,
    ... )
"""

from plasmacoll.background import DensityProfile, NeutralBackground
from plasmacoll.cross_sections import (
    CrossSection,
    LXCatProcess,
    find_lxcat_process,
    read_lxcat,
)
from plasmacoll.hydrogen import proton_hydrogen_processes
from plasmacoll.markers import MarkerSet, ParticleArrays
from plasmacoll.mcc import (
    MCCDiagnostics,
    MCCResult,
    MonteCarloCollisions,
    NewMarkers,
    make_rng,
)
from plasmacoll.process import CollisionProcess
from plasmacoll.reactor import ReactorHistory, ReactorState, ZeroDReactor
from plasmacoll.vibrational import (
    VibrationalDistribution,
    boltzmann_level_populations,
    estimate_electronegativity,
)

__version__ = "0.1.0"

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
