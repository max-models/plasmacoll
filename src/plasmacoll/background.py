"""Neutral gas backgrounds: drifting Maxwellians, uniform or with spatial profiles.

The density and the temperature of a `NeutralBackground` can vary in
space through a profile: any object with a ``max_factor`` and a call that
returns the factor at marker positions (the `Profile` protocol), such as
the piecewise-linear `DensityProfile` along one axis or a
`FunctionProfile` of all coordinates. To change a background in time
(gas heating, depletion), replace it with
`plasmacoll.mcc.MonteCarloCollisions.set_background`.

A `DynamicProfile` changes between collision steps without that: its
function and its bound are set anew before every step, e.g. from the density of
a plasma species on a grid, which makes that species a collision target (the
electrons of ``e + H- -> H + 2e``). The collision operator scales the
frequency bound of such a background by the current ``max_factor`` instead of
tabulating it again.

A gas that the collisions deplete is the same thing with the density of the
gas itself on the grid: the collision operator reports the neutrals each step
consumed (`plasmacoll.mcc.MCCResult.consumed`, per background, with
the positions and weights of the events), the gas model subtracts them from
its density and refills, and the `DynamicProfile` reads that density.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import cunumpy as xp
import numpy as np
from methodcard import ConvergenceProblem, method

from plasmacoll._types import Array
from plasmacoll.constants import ELEMENTARY_CHARGE
from plasmacoll.units import convert_units, to_array, unit_field

__all__ = [
    "DensityProfile",
    "DynamicProfile",
    "FunctionProfile",
    "NeutralBackground",
    "Profile",
]


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


def _density_profile_problem() -> ConvergenceProblem:
    """A smooth profile sampled with spacing ``h``, against the function between."""
    points = np.linspace(0.0, 2.0, 401)[:, None]
    exact = 1.0 + 0.5 * np.sin(points[:, 0])

    def error(step: float) -> float:
        nodes = np.linspace(0.0, 2.0, round(2.0 / step) + 1)
        profile = DensityProfile(0, nodes, 1.0 + 0.5 * np.sin(nodes))
        return float(np.max(np.abs(xp.to_numpy(profile(xp.asarray(points))) - exact)))

    return ConvergenceProblem(error, steps=[0.2, 0.1, 0.05, 0.025])


@method(
    equations={
        "factor": r"f(\mathbf x) = f_i + \frac{f_{i+1} - f_i}{x_{i+1} - x_i}\,(x_a - x_i), \qquad x_i \le x_a \le x_{i+1}",
        "density": r"n(\mathbf x) = n_b\,f(\mathbf x), \qquad F_b = \max_i f_i",
    },
    order=2,
    assumptions=(
        r"$x_a$ the position component `axis`; the end values outside the table",
        r"$F_b$ bounds the density for the null-collision method",
    ),
    problem=_density_profile_problem,
)
@dataclass(frozen=True)
class DensityProfile:
    """A piecewise-linear relative neutral density along one coordinate axis.

    ``factors`` (dimensionless, >= 0) scale the background's reference density
    at the tabulated ``positions`` (m, strictly increasing) of coordinate
    ``axis``. Outside the table the end values are used. ``positions`` may be
    an astropy Quantity (``[0, 5] * u.cm``), a string (``"[0, 5] cm"``) or an
    array in m.

    Attributes:
        axis: The column of the marker positions the profile depends on.
        positions: Table positions in m.
        factors: Density factors at ``positions``.
    """

    axis: int
    positions: Array
    factors: Array

    def __post_init__(self) -> None:
        """Convert the positions to m, validate and normalize the table."""
        positions = xp.asarray(to_array(self.positions, "m", "positions"), dtype=float)
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
    `plasmacoll.mcc.MCCDiagnostics.bound_violations`.

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


class DynamicProfile:
    """A factor that is set anew before every collision step.

    ``function`` takes the marker positions, shape ``(N, ndim)``, and returns
    ``N`` factors (>= 0) up to ``max_factor``. `update` replaces either,
    e.g. with the interpolated density of a plasma species and its largest
    value on the grid. The collision operator reads ``max_factor`` when it
    collides, so a background with this profile needs no
    `plasmacoll.mcc.MonteCarloCollisions.set_background`; until the
    first update the factor is zero and nothing collides with it.

    Args:
        function: The factor at the given positions; zero everywhere if None.
        max_factor: The largest factor ``function`` returns.
    """

    #: Marks the profile for the collision operator, which keeps its bound apart.
    dynamic = True

    def __init__(
        self,
        function: Callable[[Array], Array] | None = None,
        max_factor: float = 0.0,
    ) -> None:
        """Start with ``function`` and its bound ``max_factor``."""
        self._function = function
        self._max_factor = 0.0
        self.update(function, max_factor)

    def update(
        self,
        function: Callable[[Array], Array] | None = None,
        max_factor: float | None = None,
    ) -> None:
        """Replace the function and/or the bound (None keeps the current one)."""
        if function is not None:
            self._function = function
        if max_factor is not None:
            if max_factor < 0.0 or not math.isfinite(max_factor):
                raise ValueError("max_factor must be finite and >= 0")
            self._max_factor = float(max_factor)

    @property
    def max_factor(self) -> float:
        """Return the current bound of the factor."""
        return self._max_factor

    def __call__(self, positions: Array) -> Array:
        """Return the factor at marker ``positions`` (shape ``(N, ndim)``)."""
        if self._function is None:
            return xp.zeros(positions.shape[0])
        return xp.asarray(self._function(positions), dtype=float)

    def __repr__(self) -> str:
        """Show the current bound."""
        return f"DynamicProfile(max_factor={self._max_factor:g})"


def is_dynamic(profile: object) -> bool:
    """Return whether ``profile`` is set anew between steps (a `DynamicProfile`)."""
    return bool(getattr(profile, "dynamic", False))


@dataclass(frozen=True)
class NeutralBackground:
    """A drifting Maxwellian neutral gas.

    The density is ``density`` everywhere, or ``density * profile(x)`` with a
    `Profile`; ``density`` is then the reference density the profile's
    factors scale. The temperature is likewise ``temperature`` or
    ``temperature * temperature_profile(x)``. The gas flows with the mean
    velocity ``drift``.

    The physical settings take astropy Quantities (``300 * u.K``), strings
    (``"40 u"``) or plain numbers in the units below; the temperature may be
    given in K, eV or J and is kept as ``kT`` in eV.

    Attributes:
        name: The name processes refer to the background by.
        density: Number density in m^-3.
        temperature: Temperature ``kT`` in eV.
        mass: Mass of one neutral particle in kg.
        profile: Optional spatial variation of the density.
        temperature_profile: Optional spatial variation of the temperature.
        drift: Mean (flow) velocity of the gas in m/s.
    """

    name: str
    density: float = unit_field("m^-3")
    temperature: float = unit_field("eV")
    mass: float = unit_field("kg")
    profile: Profile | None = None
    temperature_profile: Profile | None = None
    drift: tuple[float, float, float] = unit_field("m/s", (0.0, 0.0, 0.0))

    def __post_init__(self) -> None:
        """Convert the quantities and validate the background parameters."""
        convert_units(self)
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
        return math.sqrt(ELEMENTARY_CHARGE * self.temperature / self.mass)

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

    @method(
        equations={
            "velocity": r"\mathbf w = \mathbf u_b + \sqrt{\frac{e\,T_b\,f_{T,b}(\mathbf x)}{M}}\;\boldsymbol\xi, \qquad \boldsymbol\xi \sim \mathcal N(0, 1)^3",
        },
        assumptions=(
            r"a drifting Maxwellian: $\mathbf u_b$ the drift, $T_b$ the temperature in eV, $f_{T,b}$ the temperature profile clipped at zero (1 without one)",
        ),
    )
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
