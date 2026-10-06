"""Shared helpers of the unit tests; the cross sections are synthetic, not physical data."""

from __future__ import annotations

import math

import cunumpy as xp

from pymcc import (
    CollisionProcess,
    CrossSection,
    MonteCarloCollisions,
    NeutralBackground,
)
from pymcc.constants import ATOMIC_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE, PROTON_MASS

M_E = ELECTRON_MASS
M_P = PROTON_MASS
M_HE = 4.002602 * ATOMIC_MASS
HEAVY = 1.0e10 * M_E  # effectively infinite mass: the neutral does not recoil


def constant(sigma: float, threshold: float = 0.0) -> CrossSection:
    return CrossSection.constant(sigma, threshold=threshold)


def energy_ev(velocities, mass: float):
    return 0.5 * mass * xp.sum(velocities**2, axis=1) / ELEMENTARY_CHARGE


def speed_for(energy: float, mass: float) -> float:
    return math.sqrt(2.0 * energy * ELEMENTARY_CHARGE / mass)


def beam(num: int, energy: float, mass: float):
    velocities = xp.zeros((num, 3))
    velocities[:, 0] = speed_for(energy, mass)
    return velocities


def operator(
    processes: list[CollisionProcess],
    masses: dict[str, float],
    neutral_mass: float = HEAVY,
    temperature: float = 0.0,
    density: float = 1.0e20,
    incident: str = "e",
    bound_safety: float = 1.0,
    seed: int = 1234,
) -> MonteCarloCollisions:
    return MonteCarloCollisions(
        species_masses=masses,
        backgrounds=[NeutralBackground("gas", density, temperature, neutral_mass)],
        processes={incident: processes},
        seed=seed,
        bound_safety=bound_safety,
    )
