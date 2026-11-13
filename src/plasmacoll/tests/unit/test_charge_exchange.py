"""The H+ + H(1s) charge-exchange fit of Janev and Smith (1993)."""

from __future__ import annotations

import math

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import (
    proton_hydrogen_charge_exchange,
    proton_hydrogen_charge_exchange_table,
)


def riviere(energy_per_nucleon):
    """Riviere's older fit (Nucl. Fusion 11, 363, 1971), E in eV/u, sigma in m^2."""
    e = np.asarray(energy_per_nucleon, dtype=float)
    return 0.6937e-18 * (1.0 - 0.155 * np.log10(e)) ** 2 / (1.0 + 0.1112e-14 * e**3.3)


def test_fit_by_hand_at_10_kev() -> None:
    # E = 10 keV/u: 3.2345 ln(23.588 + 2.3713) / (1 + 0.38371 + 3.8068e-6 10^3.5 + 1.1832e-10 10^5.4)
    expected = (
        3.2345
        * math.log(235.88 / 10.0 + 2.3713)
        / (1.0 + 0.38371 + 3.8068e-6 * 10.0**3.5 + 1.1832e-10 * 10.0**5.4)
    )
    assert float(proton_hydrogen_charge_exchange(1.0e4)) == pytest.approx(
        expected * 1.0e-20, rel=1e-12
    )
    assert expected == pytest.approx(7.55, abs=0.01)  # in 1e-16 cm^2


def test_fit_against_riviere_at_beam_energies() -> None:
    # two independent fits to the measurements agree within 35 % from 1 to 200 keV/u (Riviere is higher
    # around 20 keV/u)
    energy = np.geomspace(1.0e3, 2.0e5, 30)
    ratio = proton_hydrogen_charge_exchange(energy) / riviere(energy)
    assert np.all(np.abs(ratio - 1.0) < 0.35)


def test_fit_falls_with_energy() -> None:
    sigma = proton_hydrogen_charge_exchange(np.geomspace(1.0, 1.0e6, 200))
    assert np.all(np.diff(sigma) < 0.0)
    assert np.all(sigma > 0.0)


def test_table_takes_the_hydrogen_data_at_the_same_speed() -> None:
    hydrogen = proton_hydrogen_charge_exchange_table()
    deuterium = proton_hydrogen_charge_exchange_table(
        mass_number=2.0, max_energy="4 MeV"
    )
    sigma_h = np.asarray(xp.to_numpy(hydrogen(xp.asarray([3.0e4]))))
    sigma_d = np.asarray(xp.to_numpy(deuterium(xp.asarray([6.0e4]))))
    assert sigma_d == pytest.approx(sigma_h, rel=1e-2)
    assert sigma_h == pytest.approx(proton_hydrogen_charge_exchange(3.0e4), rel=1e-2)
    assert "Janev and Smith" in deuterium.label


def test_table_rejects_a_bad_range() -> None:
    with pytest.raises(ValueError, match="min_energy"):
        proton_hydrogen_charge_exchange_table(min_energy=10.0, max_energy=1.0)
