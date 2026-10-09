"""Null-collision Monte Carlo collisions of charged particles with neutral gases.

plasmacoll collides markers (positions, velocities and weights in NumPy or CuPy
arrays, through :mod:`cunumpy`) with prescribed Maxwellian neutral backgrounds:
elastic and backscatter collisions, excitation, ionization, attachment,
detachment, charge transfer and dissociation, with cross sections from tables,
LXCat files, fits or functions. :class:`ChargedCollisions` adds Coulomb
collisions and reactions between charged particles. It is the collision step of
a particle-in-cell code, or with :class:`ZeroDReactor` a homogeneous reactor
(optionally in electric and magnetic fields) for testing cross-section sets.

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

from plasmacoll.background import (
    DensityProfile,
    FunctionProfile,
    NeutralBackground,
    Profile,
)
from plasmacoll.charged import (
    ChargedCollisions,
    ChargedDiagnostics,
    ChargedReaction,
    coulomb_logarithm,
)
from plasmacoll.cross_sections import (
    CrossSection,
    LXCatProcess,
    elastic_from_lxcat,
    find_lxcat_process,
    read_lxcat,
)
from plasmacoll.fits import (
    cross_section_from_log_polynomial,
    double_log_polynomial,
    log_polynomial,
)
from plasmacoll.hydrogen import proton_hydrogen_processes
from plasmacoll.markers import MarkerSet, ParticleArrays
from plasmacoll.mcc import (
    BackgroundTransfer,
    MCCDiagnostics,
    MCCResult,
    MonteCarloCollisions,
    NewMarkers,
    TransferEvents,
    make_rng,
    scattering_cosine,
)
from plasmacoll.population import merge_markers, split_markers
from plasmacoll.process import CollisionProcess
from plasmacoll.reactor import ReactorHistory, ReactorState, ZeroDReactor
from plasmacoll.vibrational import (
    VibrationalDistribution,
    boltzmann_level_populations,
    estimate_electronegativity,
)

__version__ = "0.1.0"

__all__ = [
    "BackgroundTransfer",
    "ChargedCollisions",
    "ChargedDiagnostics",
    "ChargedReaction",
    "CollisionProcess",
    "CrossSection",
    "DensityProfile",
    "FunctionProfile",
    "LXCatProcess",
    "MCCDiagnostics",
    "MCCResult",
    "MarkerSet",
    "MonteCarloCollisions",
    "NeutralBackground",
    "NewMarkers",
    "ParticleArrays",
    "Profile",
    "ReactorHistory",
    "ReactorState",
    "TransferEvents",
    "VibrationalDistribution",
    "ZeroDReactor",
    "boltzmann_level_populations",
    "coulomb_logarithm",
    "cross_section_from_log_polynomial",
    "double_log_polynomial",
    "elastic_from_lxcat",
    "estimate_electronegativity",
    "find_lxcat_process",
    "log_polynomial",
    "make_rng",
    "merge_markers",
    "proton_hydrogen_processes",
    "read_lxcat",
    "scattering_cosine",
    "split_markers",
]
