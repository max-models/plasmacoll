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

For any other electron energy distribution, :func:`eedf_rate_coefficient`
integrates ``k = gamma int eps sigma(eps) F0(eps) deps`` over a tabulated
energy probability function ``F0`` (eV^-3/2, normalised by
``int eps^1/2 F0 deps = 1``, the convention of :func:`~plasmacoll.solve_eedf`),
with ``gamma = (2 e / m_e)^1/2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cunumpy as xp
import numpy as np
import scope_profiler as sp

from plasmacoll._profiling import PROFILE_TAGS
from plasmacoll._types import Array
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
from plasmacoll.cross_sections import CrossSection

__all__ = ["RateTable", "eedf_rate_coefficient", "rate_coefficient"]

#: Energies from ``_LOW`` to ``_HIGH`` times ``kT`` carry the Maxwellian integral.
_LOW, _HIGH = 1.0e-6, 60.0


@sp.profile("plasmacoll.rate_coefficient", tags=PROFILE_TAGS)
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


def _interpolate_eepf(
    energy: np.ndarray, grid: np.ndarray, f0: np.ndarray
) -> np.ndarray:
    """``F0`` at ``energy`` per row, linear in ``log F0`` where both ends are positive."""
    out = np.empty((f0.shape[0], energy.size))
    tiny = np.finfo(float).tiny
    for k, row in enumerate(f0):
        linear = np.interp(energy, grid, row, right=0.0)
        logged = np.exp(
            np.interp(energy, grid, np.log(np.maximum(row, tiny)), right=-np.inf)
        )
        positive = np.interp(energy, grid, (row > 0.0).astype(float), right=0.0) >= 1.0
        out[k] = np.where(positive, logged, linear)
    return out


@sp.profile("plasmacoll.eedf_rate_coefficient", tags=PROFILE_TAGS)
def eedf_rate_coefficient(
    cross_section: CrossSection,
    energy: Array,
    eepf: Array,
    *,
    mass: float = ELECTRON_MASS,
    normalise: bool = True,
    refine: int = 4,
) -> np.ndarray:
    """Return ``k = (2 e / m)^1/2 int eps sigma(eps) F0(eps) deps`` in m^3/s.

    ``F0`` is the energy probability function in eV^-3/2 on the grid ``energy``
    (eV). Between the grid points ``log F0`` is interpolated linearly (``F0``
    itself where it is zero), and the integral runs over the grid, the
    cross section's own table energies and its threshold, each interval cut
    into ``refine`` pieces. Beyond the last grid energy ``F0`` is taken as zero,
    so the grid must reach the tail that matters for the threshold. The
    cross section is looked up at the electron energy (the lab frame of
    electron-impact data).

    Args:
        cross_section: The cross section against energy (eV).
        energy: The energies of ``eepf`` in eV, increasing, ``>= 0``. A last
            axis of several distributions is allowed: ``eepf`` of shape
            ``(..., len(energy))``.
        eepf: ``F0`` in eV^-3/2, non-negative.
        mass: The mass of the incident particle in kg.
        normalise: Divide by ``int eps^1/2 F0 deps`` on the same quadrature, so
            that a distribution normalised only approximately gives a rate per
            particle.
        refine: Sub-intervals per interval of the merged grid.

    Returns:
        The rate coefficients, of the shape of ``eepf`` without its last axis.
    """
    grid = np.asarray(xp.to_numpy(xp.asarray(energy)), dtype=float)
    f0 = np.asarray(xp.to_numpy(xp.asarray(eepf)), dtype=float)
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("energy must be a 1D increasing grid of at least two points")
    if grid[0] < 0.0:
        raise ValueError("energy must be >= 0")
    if f0.shape[-1] != grid.size:
        raise ValueError("the last axis of eepf must match energy")
    if np.any(f0 < 0.0):
        raise ValueError("eepf must be non-negative")
    if mass <= 0.0 or refine < 1:
        raise ValueError("need mass > 0 and refine >= 1")
    table = np.asarray(xp.to_numpy(cross_section.energy), dtype=float)
    threshold = cross_section.threshold
    extra = table[(table > grid[0]) & (table < grid[-1])]
    nodes = np.unique(
        np.concatenate(
            [grid, extra, [threshold] if grid[0] < threshold < grid[-1] else []]
        )
    )
    steps = np.arange(refine) / refine
    fine = np.append(
        (nodes[:-1, None] + np.diff(nodes)[:, None] * steps).ravel(), nodes[-1]
    )
    f_fine = _interpolate_eepf(fine, grid, f0.reshape(-1, grid.size))
    sigma = np.asarray(xp.to_numpy(cross_section(fine)), dtype=float)
    # integrate from the threshold, where sigma may jump
    sigma = np.where(fine >= threshold, sigma, 0.0)
    gamma = np.sqrt(2.0 * ELEMENTARY_CHARGE / mass)
    rate = gamma * np.trapezoid(fine * sigma * f_fine, fine, axis=-1)
    if normalise:
        norm = np.trapezoid(np.sqrt(fine) * f_fine, fine, axis=-1)
        if np.any(norm <= 0.0):
            raise ValueError("eepf is zero everywhere")
        rate = rate / norm
    return rate.reshape(f0.shape[:-1])
