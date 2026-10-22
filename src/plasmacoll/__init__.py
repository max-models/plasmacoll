"""Null-collision Monte Carlo collisions of charged particles with neutral gases.

plasmacoll collides markers (positions, velocities and weights in NumPy or CuPy
arrays, through :mod:`cunumpy`) with prescribed Maxwellian neutral backgrounds:
elastic and backscatter collisions, excitation, ionization, attachment,
detachment, charge transfer and dissociation, with cross sections from tables,
LXCat files or functions. It is the collision step of a particle-in-cell code,
or with :class:`ZeroDReactor` a homogeneous reactor (optionally in an electric
field) for testing cross-section sets. :func:`solve_eedf` solves the two-term
Boltzmann equation for the same processes, for rates and transport
coefficients consistent with them.

Example:
    >>> import plasmacoll
    >>> from plasmacoll.constants import ELECTRON_MASS, ATOMIC_MASS, kelvin_to_ev
    >>> mcc = plasmacoll.MonteCarloCollisions(
    ...     species_masses={"e": ELECTRON_MASS},
    ...     backgrounds=[plasmacoll.NeutralBackground(
    ...         "Ar", 1e21, kelvin_to_ev(300.0), 39.95 * ATOMIC_MASS)],
    ...     processes={"e": [plasmacoll.CollisionProcess(
    ...         "elastic", "Ar", plasmacoll.CrossSection.constant(1e-19))]},
    ...     seed=1,
    ... )
"""

import scope_profiler as sp

from plasmacoll._profiling import PROFILE_TAGS
from plasmacoll.background import (
    DensityProfile,
    DynamicProfile,
    FunctionProfile,
    NeutralBackground,
    Profile,
)
from plasmacoll.boltzmann import (
    BoltzmannSolution,
    SwarmTable,
    solve_eedf,
    swarm_table,
)
from plasmacoll.charge_states import ChargeStateTransitions, charge_state_fractions
from plasmacoll.cross_sections import (
    CrossSection,
    LXCatProcess,
    elastic_from_lxcat,
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
    scattering_cosine,
)
from plasmacoll.population import merge_markers, split_markers
from plasmacoll.process import CollisionProcess
from plasmacoll.rates import RateTable, rate_coefficient
from plasmacoll.reactor import ReactorHistory, ReactorState, ZeroDReactor
from plasmacoll.secondaries import RUDD_H2, RuddIonization, RuddShell
from plasmacoll.vibrational import (
    VibrationalDistribution,
    boltzmann_level_populations,
    estimate_electronegativity,
)

__version__ = "0.3.0"

# The decorated functions pick the tag up, so registering after the imports is in time.
sp.register_opt_in_tag(*PROFILE_TAGS)

__all__ = [
    "RUDD_H2",
    "BoltzmannSolution",
    "ChargeStateTransitions",
    "CollisionProcess",
    "CrossSection",
    "DensityProfile",
    "DynamicProfile",
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
    "RateTable",
    "ReactorHistory",
    "ReactorState",
    "RuddIonization",
    "RuddShell",
    "SwarmTable",
    "VibrationalDistribution",
    "ZeroDReactor",
    "boltzmann_level_populations",
    "elastic_from_lxcat",
    "estimate_electronegativity",
    "find_lxcat_process",
    "charge_state_fractions",
    "make_rng",
    "merge_markers",
    "proton_hydrogen_processes",
    "rate_coefficient",
    "read_lxcat",
    "scattering_cosine",
    "solve_eedf",
    "split_markers",
    "swarm_table",
]
