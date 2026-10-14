"""Rate coefficients: cross sections averaged over a Maxwellian distribution.

For two species whose relative velocities are Maxwellian at temperature ``T``
(e.g. electrons on a cold gas, or ions and atoms at the same temperature), the
rate coefficient of a process is

    K(T) = <sigma g> = sqrt(8 / (pi mu)) (kT)^(-3/2) int sigma(E) E exp(-E / kT) dE,

with ``E = mu g^2 / 2`` the centre-of-mass energy and ``mu`` the reduced mass.
Cross sections tabulated against the incident particle's energy in the target's
rest frame (``energy_frame="lab"``) are looked up at ``E m / mu`` instead, with
``m`` the incident mass. :class:`RateTable` tabulates ``K(T)`` once for models
that need it at many temperatures, such as global (0D) and fluid models.
"""

from __future__ import annotations

from dataclasses import dataclass

import cunumpy as xp
import numpy as np

from plasmacoll._types import Array
from plasmacoll.constants import ELEMENTARY_CHARGE
from plasmacoll.cross_sections import CrossSection

__all__ = ["RateTable", "rate_coefficient"]

#: Energies from ``_LOW`` to ``_HIGH`` times ``kT`` carry the Maxwellian integral.
_LOW, _HIGH = 1.0e-6, 60.0


def rate_coefficient(
    cross_section: CrossSection,
    temperature: float | Array,
    reduced_mass: float,
    *,
    incident_mass: float | None = None,
    num: int = 4000,
) -> np.ndarray:
    """Return ``<sigma g>`` in m^3/s over a Maxwellian of relative speeds.

    Args:
        cross_section: The cross section against energy (eV).
        temperature: ``kT`` of the relative motion in eV, a number or an array.
        reduced_mass: ``mu`` in kg; the electron mass for electrons on a gas.
        incident_mass: For a table in the lab frame (the incident particle's
            energy in the target's rest frame), the incident mass in kg; the
            table is then looked up at ``E incident_mass / mu``. ``None`` for a
            table in the centre-of-mass frame.
        num: Integration points per temperature, logarithmically spaced, plus
            the table's own energies (where it has kinks) and its threshold.

    Returns:
        The rate coefficients, of the shape of ``temperature``.
    """
    if reduced_mass <= 0.0:
        raise ValueError("reduced_mass must be > 0")
    temperatures = np.asarray(temperature, dtype=float)
    if np.any(temperatures <= 0.0):
        raise ValueError("temperature must be > 0")
    scale = 1.0 if incident_mass is None else incident_mass / reduced_mass
    # the table's kinks and threshold in centre-of-mass energy
    table = np.asarray(xp.to_numpy(cross_section.energy), dtype=float) / scale
    threshold = cross_section.threshold / scale
    rates = np.empty(temperatures.size)
    for k, kt in enumerate(temperatures.ravel()):
        grid = np.geomspace(_LOW * kt, _HIGH * kt, num)
        inside = table[(table > grid[0]) & (table < grid[-1])]
        extra = [threshold] if grid[0] < threshold < grid[-1] else []
        energy = np.unique(np.concatenate([grid, inside, extra]))
        # the cross section may jump at its threshold: integrate from it
        energy = energy[energy >= threshold]
        if energy.size < 2:
            rates[k] = 0.0
            continue
        sigma = np.asarray(xp.to_numpy(cross_section(energy * scale)), dtype=float)
        integrand = sigma * energy * np.exp(-energy / kt)
        integral = float(np.trapezoid(integrand, energy))
        rates[k] = (
            np.sqrt(8.0 / (np.pi * reduced_mass))
            * integral
            * (kt * ELEMENTARY_CHARGE) ** -1.5
            * ELEMENTARY_CHARGE**2
        )
    return rates.reshape(temperatures.shape)


@dataclass(frozen=True)
class RateTable:
    """``K(T)`` of a cross section, tabulated once and interpolated in log-log.

    Attributes:
        temperature: The table temperatures in eV, increasing.
        rate: ``K`` in m^3/s at ``temperature``.
        label: A name for plots and messages.
    """

    temperature: np.ndarray
    rate: np.ndarray
    label: str = ""

    @classmethod
    def from_cross_section(
        cls,
        cross_section: CrossSection,
        reduced_mass: float,
        *,
        incident_mass: float | None = None,
        min_temperature: float = 0.01,
        max_temperature: float = 1000.0,
        num: int = 121,
        label: str = "",
    ) -> RateTable:
        """Tabulate :func:`rate_coefficient` on ``num`` temperatures (eV)."""
        if not 0.0 < min_temperature < max_temperature:
            raise ValueError("need 0 < min_temperature < max_temperature")
        temperature = np.geomspace(min_temperature, max_temperature, num)
        rate = rate_coefficient(
            cross_section, temperature, reduced_mass, incident_mass=incident_mass
        )
        return cls(temperature, rate, label or cross_section.label)

    @classmethod
    def arrhenius(
        cls,
        a: float,
        b: float = 0.0,
        c: float = 0.0,
        *,
        min_temperature: float = 0.01,
        max_temperature: float = 1000.0,
        num: int = 121,
        label: str = "",
    ) -> RateTable:
        """Tabulate ``K = a T^b exp(-c / T)`` (``T`` in eV, ``K`` in m^3/s)."""
        temperature = np.geomspace(min_temperature, max_temperature, num)
        rate = a * temperature**b * np.exp(-c / temperature)
        return cls(temperature, rate, label)

    def __post_init__(self) -> None:
        """Validate the table."""
        temperature = np.asarray(self.temperature, dtype=float)
        rate = np.asarray(self.rate, dtype=float)
        if temperature.ndim != 1 or temperature.shape != rate.shape:
            raise ValueError("temperature and rate must be 1D arrays of equal length")
        if temperature.size < 2 or np.any(np.diff(temperature) <= 0.0):
            raise ValueError("temperatures must be increasing, at least two")
        if np.any(temperature <= 0.0) or np.any(rate < 0.0):
            raise ValueError("temperatures must be > 0 and rates >= 0")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "rate", rate)

    def __call__(self, temperature: float | Array) -> np.ndarray:
        """Return ``K`` in m^3/s at ``temperature`` (eV), clamped to the table."""
        t = np.clip(
            np.asarray(temperature, dtype=float),
            self.temperature[0],
            self.temperature[-1],
        )
        log_t = np.log(self.temperature)
        # interpolate log K where K > 0; a zero rate stays zero
        tiny = np.finfo(float).tiny
        log_k = np.log(np.maximum(self.rate, tiny))
        value = np.exp(np.interp(np.log(t), log_t, log_k))
        return np.where(value <= tiny * 10, 0.0, value)
