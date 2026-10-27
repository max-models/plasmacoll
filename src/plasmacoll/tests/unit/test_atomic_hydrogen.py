"""The atomic-hydrogen fits of Janev, Reiter and Samm (2003), section 2.1, and EEDF rates."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plasmacoll import (
    CrossSection,
    eedf_rate_coefficient,
    hydrogen_atom_processes,
    rate_coefficient,
)
from plasmacoll.atomic_hydrogen import (
    excitation_fit,
    gaunt_coefficients,
    ionization_energy,
    ionization_fit,
    oscillator_strength,
    transition_energy,
)
from plasmacoll.constants import ELECTRON_MASS

#: 1e-16 cm^2 in m^2
A2 = 1.0e-20


@pytest.mark.parametrize(
    ("n", "m", "exact"),
    # exact n-bundled absorption oscillator strengths of H (Wiese and Fuhr 2009)
    [
        (1, 2, 0.4162),
        (1, 3, 0.07910),
        (1, 4, 0.02899),
        (2, 3, 0.6407),
        (2, 4, 0.1193),
        (3, 4, 0.8421),
    ],
)
def test_johnson_oscillator_strengths_are_within_a_percent_of_the_exact_ones(
    n, m, exact
):
    assert oscillator_strength(n, m) == pytest.approx(exact, rel=0.01)


def test_the_gaunt_coefficients_of_the_ground_state_are_johnsons() -> None:
    assert gaunt_coefficients(1) == (1.1330, -0.4059, 0.07014)
    # the bound-free Gaunt factor of 1s at threshold, g0 + g1 + g2 = 0.797
    assert sum(gaunt_coefficients(1)) == pytest.approx(0.797, abs=1e-3)


def test_energies() -> None:
    assert ionization_energy(2) == pytest.approx(3.4)
    assert transition_energy(1, 2) == pytest.approx(10.2)
    assert transition_energy(2, 3) == pytest.approx(13.6 * (1 / 4 - 1 / 9))


@pytest.mark.parametrize("bad", [(0, 2), (2, 2), (3, 2), (1.5, 3)])
def test_bad_levels_are_refused(bad) -> None:
    with pytest.raises(ValueError):
        oscillator_strength(*bad)


def test_the_1s_2_fit_is_continuous_and_matches_its_plateau() -> None:
    fit = excitation_fit(1, 2)
    assert fit(np.array([12.0]))[0] == pytest.approx(0.5025 * A2)
    # eq. (4c) meets the plateau c = 0.5025 at 12.23 eV
    above = fit(np.array([12.23 + 1e-9]))[0]
    assert above == pytest.approx(0.5025 * A2, rel=5e-3)
    assert fit(np.array([10.0]))[0] == 0.0


def test_the_ground_state_ionization_peaks_near_the_measured_value() -> None:
    # Shah et al. (1987): about 0.6e-16 cm^2 at 50-60 eV
    energy = np.linspace(20.0, 200.0, 400)
    sigma = ionization_fit(1)(energy)
    assert 40.0 < energy[np.argmax(sigma)] < 80.0
    assert sigma.max() == pytest.approx(0.62 * A2, rel=0.05)


def test_eq14_and_johnson_agree_roughly_for_n3_and_join_at_n4() -> None:
    from plasmacoll.atomic_hydrogen import _johnson_ionization

    energy = np.array([4.0, 10.0]) * ionization_energy(3)
    assert _johnson_ionization(3)(energy) == pytest.approx(
        ionization_fit(3)(energy), rel=0.15
    )
    assert "Johnson" in ionization_fit(4).source


def test_johnson_excitation_follows_bethe_at_high_energy() -> None:
    # sigma -> 4 pi a0^2 (Ry/dE)(Ry/E) f ln(E) up to a constant in the log
    n, m = 3, 5
    fit = excitation_fit(n, m)
    delta = transition_energy(n, m)
    energy = np.array([1e3, 1e4]) * delta
    bethe_slope = (
        4
        * math.pi
        * (0.529177e-10) ** 2
        * (13.6 / delta)
        * 13.6
        * oscillator_strength(n, m)
    )
    e_sigma = fit(energy) * energy
    slope = (e_sigma[1] - e_sigma[0]) / math.log(10.0)
    assert slope == pytest.approx(bethe_slope, rel=0.02)


def test_every_pair_gives_a_positive_cross_section_above_threshold() -> None:
    for n in range(1, 8):
        for m in range(n + 1, 12):
            fit = excitation_fit(n, m)
            sigma = fit(fit.threshold * np.geomspace(1.001, 1e3, 50))
            assert np.all(sigma > 0.0), (n, m)
        fit = ionization_fit(n)
        assert np.all(fit(fit.threshold * np.geomspace(1.001, 1e3, 50)) > 0.0)


def test_processes_for_the_mcc() -> None:
    processes = hydrogen_atom_processes(max_level=4)
    assert [p.kind for p in processes] == ["excitation"] * 3 + ["ionization"]
    assert processes[-1].products == {"electron": "e", "ion": "H+"}
    assert processes[0].cross_section.threshold == pytest.approx(10.2)


def maxwellian_eepf(energy: np.ndarray, kt: float) -> np.ndarray:
    return 2.0 / math.sqrt(math.pi) * kt**-1.5 * np.exp(-energy / kt)


@pytest.mark.parametrize("kt", [1.0, 3.0, 20.0])
def test_the_eedf_rate_of_a_maxwellian_is_the_maxwellian_rate(kt) -> None:
    cross_section = ionization_fit(1).cross_section()
    energy = np.linspace(0.0, 60.0 * kt, 6001)
    k = eedf_rate_coefficient(cross_section, energy, maxwellian_eepf(energy, kt))
    assert k == pytest.approx(
        rate_coefficient(cross_section, kt, ELECTRON_MASS), rel=2e-3
    )


def test_eedf_rates_broadcast_over_distributions() -> None:
    cross_section = CrossSection.constant(1e-20)
    energy = np.linspace(0.0, 100.0, 2001)
    eepf = np.stack([maxwellian_eepf(energy, kt) for kt in (1.0, 2.0)])
    k = eedf_rate_coefficient(cross_section, energy, eepf)
    assert k.shape == (2,)
    # constant sigma: k = sigma <v> = sigma sqrt(8 kT / (pi m))
    mean = np.sqrt(
        8 * np.array([1.0, 2.0]) * 1.602176634e-19 / (math.pi * ELECTRON_MASS)
    )
    assert k == pytest.approx(1e-20 * mean, rel=2e-3)


def test_eedf_rates_check_their_input() -> None:
    cross_section = CrossSection.constant(1e-20)
    energy = np.linspace(0.0, 10.0, 11)
    with pytest.raises(ValueError):
        eedf_rate_coefficient(cross_section, energy[::-1], np.ones(11))
    with pytest.raises(ValueError):
        eedf_rate_coefficient(cross_section, energy, np.ones(10))
    with pytest.raises(ValueError):
        eedf_rate_coefficient(cross_section, energy, -np.ones(11))
    with pytest.raises(ValueError):
        eedf_rate_coefficient(cross_section, energy, np.zeros(11))
    with pytest.raises(ValueError):
        eedf_rate_coefficient(cross_section, energy - 1.0, np.ones(11))
    with pytest.raises(ValueError):
        eedf_rate_coefficient(cross_section, energy, np.ones(11), refine=0)
    # unnormalised: the raw integral
    raw = eedf_rate_coefficient(cross_section, energy, np.ones(11), normalise=False)
    assert raw > 0.0


def test_johnson_is_clipped_where_it_turns_negative() -> None:
    from plasmacoll.atomic_hydrogen import JOHNSON_MAX_LEVEL

    fit = excitation_fit(30, 31)
    sigma = fit(fit.threshold * np.geomspace(1.001, 100.0, 200))
    assert np.all(sigma >= 0.0) and np.any(sigma == 0.0)
    n = JOHNSON_MAX_LEVEL
    for m in range(n + 1, n + 4):
        fit = excitation_fit(n, m)
        assert np.all(fit(fit.threshold * np.geomspace(1.001, 1e3, 200)) > 0.0)
