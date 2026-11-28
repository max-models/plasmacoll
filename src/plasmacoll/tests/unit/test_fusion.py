"""The fusion cross sections and reactivities of Bosch and Hale (1992).

The reactivity fits are checked against the R-matrix values the paper tabulates
(its Table VIII), the cross-section fits against the reactivity fits through the
Maxwellian average (`rate_coefficient`; two independent fits of the paper, Tables
IV and VII), the D-T peak against its well-known 5.0 b at 64 keV, and the
beam-target average against its closed form for a constant cross section, its cold
and its at-rest limits, and the thermal reactivity when averaged over a Maxwellian
of beam ions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from plasmacoll import FUSION_REACTIONS, FusionReaction, rate_coefficient
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE
from plasmacoll.fusion import NUCLEAR_MASSES, beam_maxwellian_reactivity

#: Table VIII of Bosch and Hale (1992): <sigma v> [cm^3/s] at T [keV].
TABLE_VIII = {
    "T(d,n)4He": {0.2: 1.254e-26, 1.0: 6.857e-21, 2.0: 2.977e-19, 10.0: 1.136e-16},
    "3He(d,p)4He": {0.2: 1.414e-35, 1.0: 3.057e-26},
    "D(d,p)T": {0.2: 4.640e-28, 1.0: 1.017e-22},
    "D(d,n)3He": {0.2: 4.482e-28, 1.0: 9.933e-23},
}


@pytest.mark.parametrize(
    ("key", "kev", "value"),
    [(k, t, v) for k, row in TABLE_VIII.items() for t, v in row.items()],
)
def test_reactivity_against_table_viii(key: str, kev: float, value: float) -> None:
    reactivity = FUSION_REACTIONS[key].reactivity(kev * 1.0e3)
    assert reactivity == pytest.approx(value * 1.0e-6, rel=3.0e-3)


@pytest.mark.parametrize("key", list(FUSION_REACTIONS))
def test_cross_section_averages_to_the_reactivity_fit(key: str) -> None:
    # Tables IV and VII are separate fits to the R-matrix values; the paper gives the
    # S fits to about 2%, so their Maxwellian average agrees with Table VII to 3.5%
    reaction = FUSION_REACTIONS[key]
    temperature = np.array([1.0e3, 3.0e3, 10.0e3, 30.0e3, 100.0e3])
    table = reaction.cross_section(num=4001)
    averaged = rate_coefficient(table, temperature, reaction.reduced_mass)
    assert averaged == pytest.approx(reaction.reactivity(temperature), rel=0.035)


def test_dt_cross_section_peaks_at_5_barn_near_64_kev() -> None:
    dt = FUSION_REACTIONS["T(d,n)4He"]
    energy = np.linspace(40.0e3, 90.0e3, 501)
    sigma = dt.sigma(energy)
    assert energy[np.argmax(sigma)] == pytest.approx(64.0e3, rel=0.03)
    assert sigma.max() == pytest.approx(5.0e-28, rel=0.02)
    # D-3He: about 0.8 b near 250 keV
    dhe3 = FUSION_REACTIONS["3He(d,p)4He"]
    energy = np.linspace(150.0e3, 400.0e3, 501)
    assert dhe3.sigma(energy).max() == pytest.approx(0.8e-28, rel=0.05)


def test_cross_section_follows_eq_8_by_hand() -> None:
    dd = FUSION_REACTIONS["D(d,n)3He"]
    e = 50.0  # keV
    s = 5.3701e4 + e * (3.3027e2 + e * (-1.2706e-1 + e * (2.9327e-5 - e * 2.5151e-9)))
    expected = s / (e * math.exp(31.3970 / math.sqrt(e))) * 1.0e-31
    assert dd.sigma(50.0e3) == pytest.approx(expected, rel=1e-12)
    assert dd.sigma(0.0) == 0.0
    assert dd.sigma("50 keV") == pytest.approx(expected, rel=1e-12)


def test_sfactor_ranges_join_and_end() -> None:
    # the two fits of a reaction meet to within their accuracy (1% for D-T, 3.4% for
    # D-3He, where each fit is good to about 2%)
    for key, edge, rel in (
        ("T(d,n)4He", 550.0e3, 0.01),
        ("3He(d,p)4He", 900.0e3, 0.04),
    ):
        reaction = FUSION_REACTIONS[key]
        below, above = reaction.astrophysical_factor([edge * 0.9999, edge * 1.0001])
        assert above == pytest.approx(below, rel=rel)
        assert reaction.sfactor[1](edge / 1.0e3) == pytest.approx(above, rel=1e-3)
    # beyond the last range S stays at its end value (the polynomials turn negative)
    for reaction in FUSION_REACTIONS.values():
        end = reaction.sfactor[-1].valid_to
        s_end, s_far = reaction.astrophysical_factor([end, 3.0 * end])
        assert s_far == s_end > 0.0


def test_reaction_metadata() -> None:
    dd = FUSION_REACTIONS["D(d,n)3He"]
    assert isinstance(dd, FusionReaction)
    assert dd.neutron and not FUSION_REACTIONS["D(d,p)T"].neutron
    assert "Bosch and Hale" in dd.label
    # m_r c^2 of Table VII from the nuclear masses, to 0.01%
    c2_kev = 299792458.0**2 / ELEMENTARY_CHARGE / 1.0e3
    for reaction in FUSION_REACTIONS.values():
        assert reaction.reduced_mass * c2_kev == pytest.approx(
            reaction.reduced_mass_energy, rel=1e-4
        )
    dt = FUSION_REACTIONS["T(d,n)4He"]
    assert dt.center_of_mass_energy(100.0e3) == pytest.approx(
        100.0e3 * NUCLEAR_MASSES["T"] / (NUCLEAR_MASSES["T"] + NUCLEAR_MASSES["D"])
    )
    assert dt.center_of_mass_energy(100.0e3, beam="T") == pytest.approx(
        100.0e3 * NUCLEAR_MASSES["D"] / (NUCLEAR_MASSES["T"] + NUCLEAR_MASSES["D"])
    )
    with pytest.raises(ValueError, match="the beam must be"):
        dt.center_of_mass_energy(1.0e3, beam="3He")
    with pytest.raises(ValueError, match="temperature"):
        dt.reactivity(0.0)


def test_lab_frame_table() -> None:
    dd = FUSION_REACTIONS["D(d,n)3He"]
    lab = dd.cross_section(max_energy="1 MeV", num=101, beam="D")
    assert float(lab(np.asarray([100.0e3]))[0]) == pytest.approx(
        float(dd.sigma(50.0e3)), rel=1e-3
    )


@pytest.mark.parametrize("x", [0.0, 0.05, 0.5, 1.0, 2.0, 5.0, 30.0])
def test_constant_cross_section_against_closed_form(x: float) -> None:
    # <|v_b - v|> over a Maxwellian: v_t [(x + 1/(2x)) erf(x) + exp(-x^2)/sqrt(pi)]
    sigma0, target, kt = 1.0e-28, 3.0 * ATOMIC_MASS, 2.0e3
    v_t = math.sqrt(2.0 * kt * ELEMENTARY_CHARGE / target)
    if x == 0.0:
        exact = sigma0 * 2.0 * v_t / math.sqrt(math.pi)
    else:
        exact = (
            sigma0
            * v_t
            * ((x + 0.5 / x) * math.erf(x) + math.exp(-(x**2)) / math.sqrt(math.pi))
        )
    got = beam_maxwellian_reactivity(
        lambda e: np.full_like(e, sigma0), x * v_t, kt, 1.0 * ATOMIC_MASS, target
    )
    assert float(got) == pytest.approx(exact, rel=1e-8)


def test_cold_target_and_its_limit() -> None:
    dd = FUSION_REACTIONS["D(d,n)3He"]
    energy = np.array([20.0e3, 60.0e3, 100.0e3])
    speed = np.sqrt(2.0 * energy * ELEMENTARY_CHARGE / NUCLEAR_MASSES["D"])
    cold = dd.beam_reactivity(energy)
    assert cold == pytest.approx(dd.sigma(energy / 2.0) * speed, rel=1e-12)
    # a target far colder than the beam: the cold value, warmer: larger (Gamow)
    assert dd.beam_reactivity(energy, 1.0) == pytest.approx(cold, rel=1e-3)
    warm = dd.beam_reactivity(energy, 5.0e3)
    assert np.all(warm > cold)
    # broadcasting: energies by temperatures
    grid = dd.beam_reactivity(energy[:, None], np.array([0.0, 1.0e3, 5.0e3]))
    assert grid.shape == (3, 3)
    assert grid[:, 0] == pytest.approx(cold)
    assert grid[:, 2] == pytest.approx(warm)
    with pytest.raises(ValueError, match=">= 0"):
        beam_maxwellian_reactivity(dd.sigma, 1.0e6, -1.0, 1.0, 1.0)


def test_beam_average_over_a_maxwellian_is_the_thermal_reactivity() -> None:
    # beam ions themselves Maxwellian at T: the average of <sigma v>_b over their
    # energies is the thermal reactivity of the same cross section
    dt = FUSION_REACTIONS["T(d,n)4He"]
    kt = 10.0e3
    x = np.linspace(1.0e-4, 40.0, 4001)
    weight = 2.0 * np.sqrt(x / math.pi) * np.exp(-x)
    beam = dt.beam_reactivity(x * kt, kt)
    averaged = np.trapezoid(weight * beam, x)
    table = dt.cross_section(num=8001)
    thermal = float(rate_coefficient(table, kt, dt.reduced_mass))
    assert averaged == pytest.approx(thermal, rel=2e-3)
    # and the reverse roles: tritons on a deuterium target
    t_beam = dt.beam_reactivity(x * kt, kt, beam="T")
    assert np.trapezoid(weight * t_beam, x) == pytest.approx(thermal, rel=2e-3)
