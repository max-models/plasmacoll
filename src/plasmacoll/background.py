"""Neutral gas backgrounds: drifting Maxwellians, uniform or with spatial profiles.

The density and the temperature of a :class:`NeutralBackground` can vary in
space through a profile: any object with a ``max_factor`` and a call that
returns the factor at marker positions (the :class:`Profile` protocol), such as
the piecewise-linear :class:`DensityProfile` along one axis or a
:class:`FunctionProfile` of all coordinates. To change a background in time
(gas heating, depletion), replace it with
:meth:`~plasmacoll.mcc.MonteCarloCollisions.set_background`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import cunumpy as xp

from plasmacoll._types import Array
from plasmacoll.constants import BOLTZMANN

__all__ = ["DensityProfile", "FunctionProfile", "NeutralBackground", "Profile"]


@runtime_checkable
class Profile(Protocol):
    """A dimensionless factor that varies in space, bounded by ``max_factor``."""

    @property
    def max_factor(self) -> float:
        """Return the largest factor the profile reaches."""
        ...

    def __call__(self, positions: Array) -> Array:
        """Return the factor at marker ``positions`` (shape ``(N, ndim)``)."""
        ...


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
class FunctionProfile:
    """A factor given by a function of all position components.

    ``function`` takes the marker positions, shape ``(N, ndim)``, as arrays of
    the active backend and returns ``N`` factors (>= 0) that never exceed
    ``max_factor``. The collision operator uses ``max_factor`` as the bound of
    the null-collision method and counts the markers where it is exceeded in
    :attr:`~plasmacoll.mcc.MCCDiagnostics.bound_violations`.

    Attributes:
        function: The factor at the given positions.
        max_factor: The largest factor ``function`` returns.
    """

    function: Callable[[Array], Array]
    max_factor: float

    def __post_init__(self) -> None:
        """Validate the bound."""
        if self.max_factor < 0.0:
            raise ValueError("max_factor must be >= 0")

    def __call__(self, positions: Array) -> Array:
        """Return the factor at marker ``positions`` (shape ``(N, ndim)``)."""
        return xp.asarray(self.function(positions), dtype=float)


@dataclass(frozen=True)
class NeutralBackground:
    """A drifting Maxwellian neutral gas.

    The density is ``density`` everywhere, or ``density * profile(x)`` with a
    :class:`Profile`; ``density`` is then the reference density the profile's
    factors scale. The temperature is likewise ``temperature`` or
    ``temperature * temperature_profile(x)``. The gas flows with the mean
    velocity ``drift``.

    Attributes:
        name: The name processes refer to the background by.
        density: Number density in m^-3.
        temperature: Temperature in K.
        mass: Mass of one neutral particle in kg.
        profile: Optional spatial variation of the density.
        temperature_profile: Optional spatial variation of the temperature.
        drift: Mean (flow) velocity of the gas in m/s.
    """

    name: str
    density: float
    temperature: float
    mass: float
    profile: Profile | None = None
    temperature_profile: Profile | None = None
    drift: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        """Validate the background parameters."""
        if self.density < 0.0:
            raise ValueError("background density must be >= 0")
        if self.temperature < 0.0:
            raise ValueError("background temperature must be >= 0")
        if self.mass <= 0.0:
            raise ValueError("background mass must be > 0")
        drift = tuple(float(component) for component in self.drift)
        if len(drift) != 3:
            raise ValueError("drift must have three components")
        object.__setattr__(self, "drift", drift)

    @property
    def max_density_factor(self) -> float:
        """Return the largest factor ``density`` is scaled by (1 if uniform)."""
        return 1.0 if self.profile is None else self.profile.max_factor

    def density_factor(self, positions: Array) -> Array | float:
        """Return the factor ``density`` is scaled by at ``positions``."""
        return 1.0 if self.profile is None else self.profile(positions)

    @property
    def thermal_speed(self) -> float:
        """Return the 1D thermal speed ``sqrt(k T / M)`` in m/s at ``temperature``."""
        return math.sqrt(BOLTZMANN * self.temperature / self.mass)

    @property
    def max_thermal_speed(self) -> float:
        """Return the largest 1D thermal speed the temperature profile reaches."""
        factor = (
            1.0
            if self.temperature_profile is None
            else self.temperature_profile.max_factor
        )
        return self.thermal_speed * math.sqrt(factor)

    @property
    def drift_speed(self) -> float:
        """Return the flow speed in m/s."""
        return math.sqrt(sum(component**2 for component in self.drift))

    def sample_velocities(
        self, rng: Any, num: int, positions: Array | None = None
    ) -> Array:
        """Sample ``num`` neutral velocities (3 components) from the Maxwellian.

        Args:
            rng: A NumPy or CuPy ``Generator``.
            num: The number of velocities.
            positions: The positions, shape ``(num, ndim)``, at which the
                temperature profile is evaluated; needed only with one.
        """
        velocities = rng.standard_normal((num, 3)) * self.thermal_speed
        if self.temperature_profile is not None:
            if positions is None:
                raise ValueError("a temperature profile needs the marker positions")
            factor = xp.clip(self.temperature_profile(positions), 0.0, None)
            velocities = velocities * xp.sqrt(factor)[:, None]
        if self.drift_speed > 0.0:
            velocities = velocities + xp.asarray(self.drift, dtype=float)
        return velocities
