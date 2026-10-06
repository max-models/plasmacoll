"""Neutral gas backgrounds: a Maxwellian at rest, uniform or with a density profile."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cunumpy as xp

from pymcc._types import Array
from pymcc.constants import BOLTZMANN

__all__ = ["DensityProfile", "NeutralBackground"]


@dataclass(frozen=True)
class DensityProfile:
    """A piecewise-linear relative neutral density along one coordinate axis.

    ``factors`` (dimensionless, >= 0) scale the background's reference density
    at the tabulated ``positions`` (m, strictly increasing) of coordinate
    ``axis``. Outside the table the end values are used.

    Attributes:
        axis: The column of the marker positions the profile depends on.
        positions: Table positions in m.
        factors: Density factors at ``positions``.
    """

    axis: int
    positions: Array
    factors: Array

    def __post_init__(self) -> None:
        """Validate and normalize the table."""
        positions = xp.asarray(self.positions, dtype=float)
        factors = xp.asarray(self.factors, dtype=float)
        if positions.ndim != 1 or positions.shape != factors.shape:
            raise ValueError("positions and factors must be 1D arrays of equal length")
        if positions.size < 2:
            raise ValueError("a density profile needs at least two points")
        if bool(xp.any(xp.diff(positions) <= 0.0)):
            raise ValueError("profile positions must be strictly increasing")
        if bool(xp.any(factors < 0.0)):
            raise ValueError("profile factors must be >= 0")
        if self.axis < 0:
            raise ValueError("profile axis must be >= 0")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "factors", factors)

    @property
    def max_factor(self) -> float:
        """Return the largest factor, which bounds the density."""
        return float(xp.max(self.factors))

    def __call__(self, positions: Array) -> Array:
        """Return the density factor at marker ``positions`` (shape ``(N, ndim)``)."""
        return xp.interp(positions[:, self.axis], self.positions, self.factors)


@dataclass(frozen=True)
class NeutralBackground:
    """A stationary Maxwellian neutral gas.

    The density is ``density`` everywhere, or ``density * profile(x)`` with a
    :class:`DensityProfile`; ``density`` is then the reference density the
    profile's factors scale.

    Attributes:
        name: The name processes refer to the background by.
        density: Number density in m^-3.
        temperature: Temperature in K.
        mass: Mass of one neutral particle in kg.
        profile: Optional spatial variation of the density.
    """

    name: str
    density: float
    temperature: float
    mass: float
    profile: DensityProfile | None = None

    def __post_init__(self) -> None:
        """Validate the background parameters."""
        if self.density < 0.0:
            raise ValueError("background density must be >= 0")
        if self.temperature < 0.0:
            raise ValueError("background temperature must be >= 0")
        if self.mass <= 0.0:
            raise ValueError("background mass must be > 0")

    @property
    def max_density_factor(self) -> float:
        """Return the largest factor ``density`` is scaled by (1 if uniform)."""
        return 1.0 if self.profile is None else self.profile.max_factor

    def density_factor(self, positions: Array) -> Array | float:
        """Return the factor ``density`` is scaled by at ``positions``."""
        return 1.0 if self.profile is None else self.profile(positions)

    @property
    def thermal_speed(self) -> float:
        """Return the 1D thermal speed ``sqrt(k T / M)`` in m/s."""
        return math.sqrt(BOLTZMANN * self.temperature / self.mass)

    def sample_velocities(self, rng: Any, num: int) -> Array:
        """Sample ``num`` neutral velocities (3 components) from the Maxwellian.

        Args:
            rng: A NumPy or CuPy ``Generator``.
            num: The number of velocities.
        """
        return rng.standard_normal((num, 3)) * self.thermal_speed
