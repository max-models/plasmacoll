"""Evaluators of the polynomial fits of atomic and molecular data compilations.

The hydrogen and helium compilations of Janev and collaborators and the
EIRENE databases AMJUEL and HYDHEL give cross sections and rate coefficients as
polynomials in logarithms:

``H.1``, cross section against energy
    ``ln sigma = sum_n b_n (ln E)^n``, ``sigma`` in cm^2 and ``E`` in eV.
``H.2``, rate coefficient against temperature
    ``ln <sigma v> = sum_n b_n (ln T)^n``, in cm^3/s and eV.
``H.3``, rate coefficient against temperature and energy
    ``ln <sigma v> = sum_n sum_m a_nm (ln E)^n (ln T)^m``.

This module evaluates the forms; it does not ship coefficients. Copy them from
the source (and cite it) together with the fit's range of validity, outside of
which a polynomial in logarithms quickly becomes meaningless.
"""

from __future__ import annotations

from collections.abc import Sequence

import cunumpy as xp
import numpy as np

from plasmacoll._types import Array
from plasmacoll.cross_sections import CrossSection

__all__ = [
    "CM2",
    "CM3_PER_S",
    "cross_section_from_log_polynomial",
    "double_log_polynomial",
    "log_polynomial",
]

#: One cm^2 in m^2, the unit of the H.1 cross-section fits.
CM2 = 1.0e-4
#: One cm^3/s in m^3/s, the unit of the H.2 and H.3 rate-coefficient fits.
CM3_PER_S = 1.0e-6


def log_polynomial(
    coefficients: Sequence[float], x: Array | float, unit: float = 1.0
) -> Array:
    """Return ``unit * exp(sum_n b_n (ln x)^n)``, the H.1 and H.2 fit forms.

    Args:
        coefficients: ``b_0, b_1, ...``, lowest order first.
        x: The energy or temperature in eV (> 0).
        unit: The unit of the fitted quantity, e.g. :data:`CM2` or
            :data:`CM3_PER_S` to return SI values.
    """
    values = xp.asarray(x, dtype=float)
    if bool(xp.any(values <= 0.0)):
        raise ValueError("log-polynomial fits need x > 0")
    logarithm = xp.log(values)
    total = xp.zeros_like(logarithm)
    for coefficient in reversed(list(coefficients)):
        total = total * logarithm + float(coefficient)
    return unit * xp.exp(total)


def double_log_polynomial(
    coefficients: Sequence[Sequence[float]] | np.ndarray,
    energy: Array | float,
    temperature: Array | float,
    unit: float = 1.0,
) -> Array:
    """Return ``unit * exp(sum_nm a_nm (ln E)^n (ln T)^m)``, the H.3 fit form.

    Args:
        coefficients: ``a[n][m]``, with ``n`` the power of ``ln E`` and ``m``
            that of ``ln T``.
        energy: The (beam) energy in eV, > 0.
        temperature: The temperature in eV, > 0; broadcast against ``energy``.
        unit: The unit of the fitted quantity, e.g. :data:`CM3_PER_S`.
    """
    table = np.asarray(coefficients, dtype=float)
    if table.ndim != 2:
        raise ValueError("H.3 coefficients must be a 2D table a[n][m]")
    energy = xp.asarray(energy, dtype=float)
    temperature = xp.asarray(temperature, dtype=float)
    if bool(xp.any(energy <= 0.0)) or bool(xp.any(temperature <= 0.0)):
        raise ValueError("log-polynomial fits need positive energy and temperature")
    log_energy = xp.log(energy)
    log_temperature = xp.log(temperature)
    total = xp.zeros_like(log_energy * log_temperature)
    for row in reversed(table):
        inner = xp.zeros_like(log_temperature)
        for coefficient in reversed(row):
            inner = inner * log_temperature + float(coefficient)
        total = total * log_energy + inner
    return unit * xp.exp(total)


def cross_section_from_log_polynomial(
    coefficients: Sequence[float],
    energy_range: tuple[float, float],
    num: int = 200,
    threshold: float = 0.0,
    unit: float = CM2,
    label: str = "",
) -> CrossSection:
    """Tabulate an H.1 cross-section fit on its range of validity.

    The table has ``num`` logarithmically spaced points from ``energy_range[0]``
    to ``energy_range[1]`` (eV). Like every :class:`CrossSection`, the last
    value is held above the range, and the first value below it down to
    ``threshold``; pass the reaction's threshold so that the cross section
    vanishes below it.

    Args:
        coefficients: ``b_0, b_1, ...`` of ``ln sigma = sum b_n (ln E)^n``.
        energy_range: The fit's range of validity in eV.
        num: Number of table points.
        threshold: Energy in eV below which the cross section is zero.
        unit: The unit of the fit's cross section, :data:`CM2` by default.
        label: A name for plots and messages.
    """
    low, high = energy_range
    if not 0.0 < low < high:
        raise ValueError("need 0 < energy_range[0] < energy_range[1]")
    energy = np.geomspace(low, high, num)
    sigma = log_polynomial(coefficients, xp.asarray(energy), unit=unit)
    return CrossSection(
        energy=xp.asarray(energy),
        sigma=sigma,
        threshold=threshold,
        label=label or "log-polynomial fit",
    )
