"""Maxwellian rate coefficients against closed forms; the cross sections are synthetic."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plasmacoll import CrossSection, RateTable, rate_coefficient
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE

M_E = ELECTRON_MASS
M_H2 = 2.016 * ATOMIC_MASS


def mean_speed(kt: float, mass: float) -> float:
    """``sqrt(8 kT / (pi m))`` with ``kT`` in eV."""
    return math.sqrt(8.0 * kt * ELEMENTARY_CHARGE / (math.pi * mass))


def test_a_maxwell_molecule_has_the_same_rate_at_every_temperature() -> None:
    cross_section = CrossSection.maxwell_molecule(2.0e-14, M_E)
    rates = rate_coefficient(cross_section, np.array([0.1, 1.0, 10.0, 100.0]), M_E)
    assert rates == pytest.approx(2.0e-14, rel=1e-4)


def test_a_constant_cross_section_gives_sigma_times_the_mean_speed() -> None:
    sigma = 3.0e-20
    for kt in (0.05, 2.0, 50.0):
        rate = rate_coefficient(CrossSection.constant(sigma), kt, M_E)
        assert rate == pytest.approx(sigma * mean_speed(kt, M_E), rel=1e-4)


def test_a_step_cross_section_gives_the_arrhenius_form() -> None:
    """sigma0 above E0: K = sigma0 <g> (1 + E0 / kT) exp(-E0 / kT)."""
    sigma, threshold = 1.0e-20, 15.0
    cross_section = CrossSection.constant(sigma, threshold=threshold)
    for kt in (1.0, 3.0, 20.0):
        expected = (
            sigma
            * mean_speed(kt, M_E)
            * (1.0 + threshold / kt)
            * math.exp(-threshold / kt)
        )
        assert rate_coefficient(cross_section, kt, M_E) == pytest.approx(
            expected, rel=1e-4
        )
    # far below the threshold the rate vanishes, and never goes negative
    assert 0.0 <= rate_coefficient(cross_section, 0.1, M_E) < 1e-60


def test_a_lab_frame_table_is_looked_up_at_the_incident_energy() -> None:
    """A step at E0 in the lab frame is a step at E0 mu / m in the centre of mass."""
    mu = M_H2 / 2.0  # H2 on H2
    lab = CrossSection.constant(1.0e-19, threshold=4.0)
    centre_of_mass = CrossSection.constant(1.0e-19, threshold=4.0 * mu / M_H2)
    kt = 1.5
    assert rate_coefficient(lab, kt, mu, incident_mass=M_H2) == pytest.approx(
        rate_coefficient(centre_of_mass, kt, mu), rel=1e-6
    )


def test_a_rate_table_interpolates_the_rate_coefficient() -> None:
    cross_section = CrossSection.constant(1.0e-20, threshold=10.0)
    table = RateTable.from_cross_section(cross_section, M_E, max_temperature=100.0)
    for kt in (0.7, 4.3, 37.0):
        assert table(kt) == pytest.approx(
            rate_coefficient(cross_section, kt, M_E), rel=2e-3
        )
    # clamped outside the table
    assert table(1.0e4) == pytest.approx(table(100.0))
    arrhenius = RateTable.arrhenius(2.0e-14, 0.5, 12.0)
    assert arrhenius(3.0) == pytest.approx(2.0e-14 * 3.0**0.5 * math.exp(-4.0), 1e-3)


def test_bad_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="temperature"):
        rate_coefficient(CrossSection.constant(1e-20), 0.0, M_E)
    with pytest.raises(ValueError, match="reduced_mass"):
        rate_coefficient(CrossSection.constant(1e-20), 1.0, 0.0)
    with pytest.raises(ValueError, match="increasing"):
        RateTable(np.array([2.0, 1.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="equal length"):
        RateTable(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match=">= 0"):
        RateTable(np.array([1.0, 2.0]), np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="min_temperature"):
        RateTable.from_cross_section(
            CrossSection.constant(1e-20), M_E, min_temperature=2.0, max_temperature=1.0
        )
