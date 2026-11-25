"""Yields of particles that hit a surface: secondary emission, reflection, conversion.

A yield is the mean number of particles a surface sends back per incident one, as a
function of the incident kinetic energy (eV) and of the cosine of the angle between
the incident velocity and the surface normal. The same form serves the secondary
electrons released by ions or electrons, the reflected or backscattered fraction of a
species, and the conversion of one species into another at a surface (H⁻ from
H⁺ on a caesiated grid). A particle code draws the products of each impact from
it; see picnic's ``surface_processes``.

- `ConstantYield`: one number, whatever the energy and angle (the Townsend
  ``gamma`` of a discharge).
- `TabulatedYield`: a table of yields against energy, linear in between and constant
  beyond its ends (measured or fitted curves, such as Phelps and Petrović's
  ion-induced secondary emission of argon).
- `VaughanYield`: Vaughan's empirical formula for the secondary emission by
  electrons, with its dependence on the angle of incidence.

`surface_yield` builds one from a configuration-file entry.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from methodcard import Rule, method

from plasmacoll._types import Array
from plasmacoll.units import convert_units, to_array, unit_field

__all__ = [
    "ConstantYield",
    "SurfaceYield",
    "TabulatedYield",
    "VaughanYield",
    "surface_yield",
]


class SurfaceYield(Protocol):
    """The mean number of particles a surface returns per incident particle."""

    def to_dict(self) -> dict:
        """Return the configuration-file form (`surface_yield` reads it)."""
        ...

    def __call__(
        self, energy: Array | float, cos_incidence: Array | float = 1.0
    ) -> np.ndarray:
        """Return the yield at ``energy`` (eV) and the incidence cosines."""
        ...


def _energies(
    energy: Array | float, cos_incidence: Array | float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the energies and cosines as float arrays of one shape."""
    energy = np.asarray(energy, dtype=float)
    cosine = np.asarray(cos_incidence, dtype=float)
    energy, cosine = np.broadcast_arrays(energy, cosine)
    return energy, cosine


@dataclass(frozen=True)
class ConstantYield:
    """The same yield at every energy and angle.

    Attributes:
        value: Particles returned per incident one (>= 0).
    """

    value: float

    def __post_init__(self) -> None:
        """Check the yield."""
        if not float(self.value) >= 0.0:
            raise ValueError(f"a yield must be >= 0, got {self.value}")
        object.__setattr__(self, "value", float(self.value))

    def to_dict(self) -> dict:
        """Return the configuration-file form."""
        return {"model": "constant", "value": self.value}

    def __call__(
        self, energy: Array | float, cos_incidence: Array | float = 1.0
    ) -> np.ndarray:
        """Return `value` in the shape of ``energy`` and ``cos_incidence``."""
        energy, _ = _energies(energy, cos_incidence)
        return np.full(energy.shape, self.value)


@dataclass(frozen=True)
class TabulatedYield:
    """A yield tabulated against the incident energy, independent of the angle.

    Linear interpolation between the points; below the first and above the last
    point the yield of the end point holds (set an end point to 0 for a threshold).

    Attributes:
        energy: Incident energies in eV, increasing; also an astropy Quantity
            or a string such as ``"[10, 100, 1000] eV"``.
        values: The yields at ``energy`` (>= 0).
    """

    energy: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Convert the energies and check the table."""
        energy = np.asarray(to_array(self.energy, "eV", "energy"), dtype=float)
        values = np.asarray(self.values, dtype=float)
        if energy.ndim != 1 or energy.shape != values.shape or energy.size < 1:
            raise ValueError("energy and values must be 1-D and of the same length")
        if np.any(np.diff(energy) <= 0.0):
            raise ValueError("the energies of a yield table must increase")
        if np.any(values < 0.0):
            raise ValueError("a yield must be >= 0")
        object.__setattr__(self, "energy", tuple(float(e) for e in energy))
        object.__setattr__(self, "values", tuple(float(v) for v in values))

    def to_dict(self) -> dict:
        """Return the configuration-file form."""
        return {
            "model": "table",
            "energy": list(self.energy),
            "values": list(self.values),
        }

    def __call__(
        self, energy: Array | float, cos_incidence: Array | float = 1.0
    ) -> np.ndarray:
        """Return the interpolated yield at ``energy`` (eV)."""
        energy, _ = _energies(energy, cos_incidence)
        return np.asarray(np.interp(energy, self.energy, self.values))


def _vaughan_rules() -> dict:
    """The SymPy rule of `VaughanYield` below the high-energy tail (for the tests)."""
    import sympy

    e, d, em, e0 = sympy.symbols(
        "energy max_yield energy_at_max threshold", positive=True
    )
    v = (e - e0) / (em - e0)
    return {
        "yield": Rule(
            d * (v * sympy.exp(1 - v)) ** sympy.Rational(56, 100),
            implementation=lambda energy, max_yield, energy_at_max, threshold: (
                VaughanYield(max_yield, energy_at_max, threshold)(energy, 1.0)
            ),
            ranges={
                "energy": (60.0, 250.0),
                "max_yield": (0.5, 3.0),
                "energy_at_max": (300.0, 600.0),
                "threshold": (5.0, 50.0),
            },
        )
    }


@method(
    equations={
        "yield": r"\delta(E, \theta) = \delta_\mathrm{max}(\theta)\,\big(v\,e^{1 - v}\big)^{k}, \qquad v = \frac{E - E_0}{E_\mathrm{max}(\theta) - E_0}, \qquad k = \begin{cases} 0.56 & v < 1 \\ 0.25 & 1 \le v \le 3.6 \end{cases}",
        "tail": r"\delta = 1.125\,\delta_\mathrm{max}(\theta)\,v^{-0.35}, \qquad v > 3.6",
        "angle": r"\delta_\mathrm{max}(\theta) = \delta_\mathrm{max}\Big(1 + \frac{k_s\theta^2}{2\pi}\Big), \qquad E_\mathrm{max}(\theta) = E_\mathrm{max}\Big(1 + \frac{k_s\theta^2}{2\pi}\Big)",
    },
    assumptions=(
        r"$\theta$ the angle between the incident velocity and the surface normal, $k_s$ the smoothness (1 for a dull surface, 2 for a polished one, 0 without angular dependence)",
        r"no emission at or below the threshold $E_0$",
    ),
    references=(
        "[Vaughan, IEEE Trans. Electron Devices 36, 1963 (1989)](https://doi.org/10.1109/16.34278)",
        "[Vaughan, IEEE Trans. Electron Devices 40, 830 (1993)](https://doi.org/10.1109/16.202798)",
    ),
    symbolic=lambda: _vaughan_rules(),
)
@dataclass(frozen=True)
class VaughanYield:
    """Vaughan's secondary-electron yield of a surface hit by electrons.

    The yield rises from the threshold to ``max_yield`` at ``energy_at_max`` (both
    at normal incidence) and falls slowly beyond; oblique incidence raises both
    (``smoothness``). It counts the true secondaries and the backscattered
    electrons together, as the fits in the literature do.

    Attributes:
        max_yield: ``delta_max`` at normal incidence.
        energy_at_max: ``E_max`` at normal incidence, in eV; also an astropy
            Quantity or a string.
        threshold: ``E_0`` in eV, below which nothing is emitted (12.5 eV in
            Vaughan's 1993 revision).
        smoothness: ``k_s``: 1 for a dull surface, 2 for a polished one, 0 for no
            dependence on the angle.
    """

    max_yield: float
    energy_at_max: float = unit_field("eV", 400.0)
    threshold: float = unit_field("eV", 12.5)
    smoothness: float = 1.0

    def __post_init__(self) -> None:
        """Convert the energies and check them."""
        convert_units(self)
        if not self.max_yield >= 0.0:
            raise ValueError("max_yield must be >= 0")
        if not 0.0 <= self.threshold < self.energy_at_max:
            raise ValueError("need 0 <= threshold < energy_at_max")
        if not self.smoothness >= 0.0:
            raise ValueError("smoothness must be >= 0")

    def to_dict(self) -> dict:
        """Return the configuration-file form."""
        return {
            "model": "vaughan",
            "max_yield": self.max_yield,
            "energy_at_max": self.energy_at_max,
            "threshold": self.threshold,
            "smoothness": self.smoothness,
        }

    def __call__(
        self, energy: Array | float, cos_incidence: Array | float = 1.0
    ) -> np.ndarray:
        """Return the yield at ``energy`` (eV) and the incidence cosines."""
        energy, cosine = _energies(energy, cos_incidence)
        theta = np.arccos(np.clip(np.abs(cosine), 0.0, 1.0))
        angle = 1.0 + self.smoothness * theta**2 / (2.0 * np.pi)
        peak = self.max_yield * angle
        energy_at_max = self.energy_at_max * angle
        v = np.maximum(energy - self.threshold, 0.0) / (energy_at_max - self.threshold)
        exponent = np.where(v < 1.0, 0.56, 0.25)
        rising = peak * (v * np.exp(1.0 - v)) ** exponent
        tail = 1.125 * peak / np.maximum(v, 3.6) ** 0.35
        return np.where(energy <= self.threshold, 0.0, np.where(v <= 3.6, rising, tail))


_MODELS = {
    "constant": ConstantYield,
    "table": TabulatedYield,
    "vaughan": VaughanYield,
}


def surface_yield(entry: float | Mapping[str, Any] | SurfaceYield) -> SurfaceYield:
    """Return the yield of a configuration-file entry.

    Args:
        entry: A number (a `ConstantYield`), a mapping with ``model`` (``"constant"``,
            ``"table"`` or ``"vaughan"``) and the settings of that class, or a
            yield, which is returned as it is.

    Returns:
        The yield.
    """
    if isinstance(entry, (int, float)) and not isinstance(entry, bool):
        return ConstantYield(float(entry))
    if not isinstance(entry, Mapping):
        if callable(entry):
            return entry
        raise TypeError(f"cannot make a surface yield of {entry!r}")
    params = dict(entry)
    model = str(params.pop("model", "constant")).lower()
    if model not in _MODELS:
        raise ValueError(f"yield model must be one of {sorted(_MODELS)}, got {model!r}")
    return _MODELS[model](**params)
