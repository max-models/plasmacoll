"""The fits of Janev, Reiter and Samm (2003), Jül-4105, against the report.

Each fit is checked against its equation evaluated by hand, against the values
the report prints (in its text and on its figures), against the same fits in
HYDHEL (D. Reiter, "The data file HYDHEL", from Janev, Langer, Evans and Post
1987, https://eirene.de/Documentation/hydhel.pdf) where they are the same, and
the Maxwellian rate coefficients against HYDHEL's rate fits (H.2), which come
from the 1987 cross sections and so agree only as well as the two compilations.
"""

from __future__ import annotations

import math

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import CrossSection, VibrationalDistribution, rate_coefficient
from plasmacoll.constants import ELECTRON_MASS
from plasmacoll.janev import (
    DISSOCIATIVE_ATTACHMENT_E0,
    DISSOCIATIVE_ATTACHMENT_LEVELS,
    HMINUS_H_COEFFICIENTS,
    JANEV_FITS,
    MUTUAL_NEUTRALIZATION_COEFFICIENTS,
    dissociative_attachment,
    dissociative_attachment_fit,
    dissociative_attachment_levels,
)

#: 1e-16 cm^2 in m^2
A2 = 1.0e-20


def host(array) -> np.ndarray:
    """A table's array on the host (the tables follow the cunumpy backend)."""
    return np.asarray(xp.to_numpy(xp.asarray(array)), dtype=float)


def sigma(key: str, energy: float) -> float:
    return float(JANEV_FITS[key](energy))


def test_fits_against_their_equations_by_hand() -> None:
    # eq. (74), 0 -> 1 at 3 eV: 5.78 / dE^4 / x^2 (1 - 1/x)^(6.11 / sqrt(dE))
    x = 3.0 / 0.516
    expected = 5.78 / 0.516**4 / x**2 * (1 - 1 / x) ** (6.11 / math.sqrt(0.516))
    assert sigma("e_h2_vibrational_1", 3.0) == pytest.approx(expected * A2)
    # eq. (80), table 15, B 1Sigma_u at 50 eV
    x = 50.0 / 12.754
    bracket = 3.651e-2 - 0.8405 / x + 1.2365 / x**2 + 2.5236 * math.log(x)
    expected = 5.984 / (12.754 * x) * (1 - 1 / x) ** 0.550 * bracket
    assert sigma("e_h2_singlet_B", 50.0) == pytest.approx(expected * A2)
    # eq. (81), table 16, EF 1Sigma_g at 40 eV
    x = 40.0 / 13.13
    expected = 5.984 * 0.8322 / (13.13 * x) * (1 - 1 / x) ** 2.71
    assert sigma("e_h2_singlet_EF", 40.0) == pytest.approx(expected * A2)
    # eq. (95), table 22, b 3Sigma_u at 15 eV
    x = 15.0 / 7.93
    expected = 11.16 / x**3 * (1 - x**-2.33) ** 3.78
    assert sigma("e_h2_triplet_b", 15.0) == pytest.approx(expected * A2)
    # eq. (106) at 70 eV, C_0 = 2.05 dE
    x = 70.0 / 15.42
    expected = 1.828 / x * (1 - x**-0.92) ** 2.19 * math.log(2.05 * 15.42 * x)
    assert sigma("e_h2_ionization", 70.0) == pytest.approx(expected * A2)
    # eqs. (107a) + (107b) at 100 eV
    xg, xu = 100.0 / 18.15, 100.0 / 30.6
    expected = (
        0.02905 / xg**1.25 * (1 - xg**-2.78) ** 1.886
        + 0.5927 / xu**1.20 * (1 - xu**-1.22) ** 3.375
    )
    assert sigma("e_h2_dissociative_ionization", 100.0) == pytest.approx(expected * A2)
    # eq. (58) at 10 eV, in 1e-13 cm^2
    expected = (
        2.06
        / 10.0
        * math.log(math.e + 2.335e-4 * 10.0)
        * math.exp(-13.75 / 10.0**0.868)
        * (1 - (0.754 / 10.0) ** 2)
    )
    assert sigma("e_hminus_detachment", 10.0) == pytest.approx(expected * 1.0e-17)
    # eqs. (63) and (71) at E = 1 eV, where ln E = 0: sigma = exp(a_0) cm^2
    assert sigma("mutual_neutralization", 1.0) == pytest.approx(
        (math.exp(-3.49880888e1) + math.exp(-3.11479336e1)) * 1.0e-4
    )
    assert sigma("hminus_h_associative_detachment", 1.0) == pytest.approx(
        math.exp(-3.44152907e1) * 1.0e-4
    )
    assert sigma("hminus_h_detachment", 1.0) == pytest.approx(
        math.exp(-3.61799082e1) * 1.0e-4
    )


def test_values_printed_in_the_report() -> None:
    # section 3.1.2: sigma_det(H-) is about 2.3e-15 cm^2 at 65 eV
    assert sigma("e_hminus_detachment", 65.0) == pytest.approx(2.3e-19, rel=0.05)
    # section 4.2.2: the triplet excitations peak between 14 and 16 eV
    energy = np.linspace(8.0, 40.0, 3201)
    for key in ("e_h2_triplet_a", "e_h2_triplet_b", "e_h2_triplet_c"):
        assert 14.0 <= energy[np.argmax(JANEV_FITS[key](energy))] <= 16.0
    # figure 20: the 0 -> 1 vibrational excitation peaks near 5e-17 cm^2 at ~3 eV
    energy = np.geomspace(0.6, 20.0, 2001)
    values = JANEV_FITS["e_h2_vibrational_1"](energy)
    assert values.max() == pytest.approx(5.0e-21, rel=0.05)
    assert 2.0 < energy[np.argmax(values)] < 3.5
    # figure 26: the non-dissociative ionization peaks near 1.1e-16 cm^2 at 50-100 eV
    energy = np.geomspace(16.0, 1000.0, 2001)
    values = JANEV_FITS["e_h2_ionization"](energy)
    assert values.max() == pytest.approx(1.1e-20, rel=0.05)
    assert 50.0 < energy[np.argmax(values)] < 100.0
    # figure 22: B and C peak near 3.5e-17 cm^2 at about 50 eV
    energy = np.geomspace(13.0, 1000.0, 2001)
    for key in ("e_h2_singlet_B", "e_h2_singlet_C"):
        values = JANEV_FITS[key](energy)
        assert values.max() == pytest.approx(3.5e-21, rel=0.1)
        assert 40.0 < energy[np.argmax(values)] < 60.0


#: HYDHEL H.1 coefficients a_0..a_8 (ln sigma[cm^2] against ln E) of reactions
#: 7.2.2 and 7.2.3 (H+ + H- -> H(n=2, 3) + H) and 7.3.2a and 7.3.2b (H- + H).
HYDHEL = {
    "7.2.2": (-3.498808881328e01, 2.152450508307e-01, -2.356286641829e-02,
              5.494715530438e-02, 5.379328883390e-03, -6.055070213687e-03,
              9.991683287907e-04, -6.636255640130e-05, 1.612283845094e-06),
    "7.2.3": (-3.114793359560e01, -7.730205271965e-01, 5.492043775127e-02,
              -2.733249841523e-03, -1.228312883866e-03, 4.350498278844e-04,
              -6.216595007254e-05, 4.120468065072e-06, -1.039784996359e-07),
    "7.3.2a": (-3.617990822381e01, 1.166151722958e00, -1.419286024586e-01,
               -1.111959587916e-02, -1.725059947629e-03, 1.590403561378e-03,
               -2.531961443012e-04, 1.669782349801e-05, -4.097257966950e-07),
    "7.3.2b": (-3.441529065653e01, -3.393482086017e-01, 5.665917047817e-02,
               -9.051504593339e-03, 7.660604175354e-04, -4.271264623381e-05,
               -1.572737493346e-07, 2.576076771707e-07, -1.200719193800e-08),
}  # fmt: skip


def test_polynomial_coefficients_are_those_of_hydhel() -> None:
    # tables 10 and 11 print the HYDHEL fits rounded to nine digits
    pairs = [
        (MUTUAL_NEUTRALIZATION_COEFFICIENTS[2], HYDHEL["7.2.2"]),
        (MUTUAL_NEUTRALIZATION_COEFFICIENTS[3], HYDHEL["7.2.3"]),
        (HMINUS_H_COEFFICIENTS["detachment"], HYDHEL["7.3.2a"]),
        (HMINUS_H_COEFFICIENTS["associative_detachment"], HYDHEL["7.3.2b"]),
    ]
    for ours, hydhel in pairs:
        assert ours == pytest.approx(hydhel, rel=1.0e-8)


def test_maxima_printed_in_hydhel() -> None:
    # HYDHEL lists each fit's maximum (smax, cm^2); 7.1.1 is the 1987 fit of
    # e + H- detachment, a different expression from eq. (58)
    energy = np.geomspace(0.1, 2.0e4, 20001)
    assert JANEV_FITS["mutual_neutralization"].function(energy).max() == (
        pytest.approx(2.25e-17, rel=0.01)  # 7.2.3 at 0.1 eV (n = 3 dominates)
    )
    n2 = (
        JANEV_FITS["mutual_neutralization"].function(energy)
        - np.exp(sum(a * np.log(energy) ** j for j, a in enumerate(HYDHEL["7.2.3"])))
        * 1.0e-4
    )
    assert n2.max() == pytest.approx(1.38e-18, rel=0.01)  # 7.2.2
    assert JANEV_FITS["hminus_h_detachment"].function(energy).max() == (
        pytest.approx(1.54e-19, rel=0.02)  # 7.3.2a
    )
    assert JANEV_FITS["hminus_h_associative_detachment"].function(energy).max() == (
        pytest.approx(3.84e-19, rel=0.01)  # 7.3.2b
    )
    energy = np.geomspace(0.76, 1.0e4, 20001)
    assert JANEV_FITS["e_hminus_detachment"](energy).max() == pytest.approx(
        3.75e-19,
        rel=0.02,  # 7.1.1
    )


@pytest.mark.parametrize("key", sorted(JANEV_FITS))
def test_thresholds_and_positivity(key: str) -> None:
    fit = JANEV_FITS[key]
    table = fit.cross_section()
    values = host(table.sigma)
    assert np.all(np.isfinite(values)) and np.all(values >= 0.0)
    assert table.threshold == fit.threshold
    assert float(table.energy[0]) == max(fit.threshold, fit.valid_from)
    above = host(table.energy) > fit.threshold
    assert np.all(values[above][:-1] > 0.0)
    if fit.threshold > 0.0:
        assert float(fit(fit.threshold)) == 0.0
        assert float(table(fit.threshold * 0.999)) == 0.0
    # the table follows the fit between its points (to 0.1 % of the maximum in
    # the steep rise above the threshold)
    energy = np.sqrt(host(table.energy)[1:] * host(table.energy)[:-1])
    assert host(table(energy)) == pytest.approx(
        fit(energy), rel=0.01, abs=1.0e-3 * values.max()
    )
    assert fit.label.startswith(fit.reaction) and fit.source in fit.label


def test_polynomial_fits_hold_their_end_values() -> None:
    fit = JANEV_FITS["mutual_neutralization"]
    assert (fit.valid_from, fit.valid_to) == (0.1, 2.0e4)
    table = fit.cross_section()
    for energy in (0.01, 0.05):
        assert float(fit(energy)) == pytest.approx(float(fit(0.1)))
        assert float(table(energy)) == pytest.approx(float(fit(0.1)))
    assert float(fit(1.0e5)) == pytest.approx(float(fit(2.0e4)))


def _hydhel_rate(b: tuple[float, ...], kt: float) -> float:
    """HYDHEL H.2: ln <sigma v>[cm^3/s] = sum_n b_n (ln kT)^n, in m^3/s."""
    return math.exp(sum(c * math.log(kt) ** n for n, c in enumerate(b))) * 1.0e-6


#: HYDHEL H.2 coefficients b_0..b_8, and the agreement of the 2003 cross sections
#: with them (the compilations differ: e.g. the 2003 vibrational excitation follows
#: experiment, larger than the 1987 theory).
HYDHEL_RATES = {
    "e_h2_ionization": (  # 2.2.9
        (-3.568640293666e01, 1.733468989961e01, -7.767469363538e00,
         2.211579405415e00, -4.169840174384e-01, 5.088289820867e-02,
         -3.832737518325e-03, 1.612863120371e-04, -2.893391904431e-06),
        (2.0, 5.0, 20.0, 100.0), 0.12,
    ),
    "e_hminus_detachment": (  # 7.1.1
        (-1.801849334273e01, 2.360852208681e00, -2.827443061704e-01,
         1.623316639567e-02, -3.365012031363e-02, 1.178329782711e-02,
         -1.656194699504e-03, 1.068275202678e-04, -2.631285809207e-06),
        (1.0, 2.0, 5.0, 20.0, 100.0), 0.3,
    ),
    "e_h2_vibrational_1": (  # 2.2.1a
        (-2.017212494454e01, 9.563952280637e-01, -6.931474225637e-01,
         1.673671529631e-01, -3.228762898855e-02, 5.838603222226e-03,
         -8.580542691302e-04, 7.456311623168e-05, -2.666978300757e-06),
        (0.5, 1.0, 2.0, 5.0, 20.0), 0.45,
    ),
    "e_h2_singlet_B": (  # 2.2.2
        (-3.081902926338e01, 1.038866780735e01, -4.259768348687e00,
         1.181228673120e00, -2.277513907465e-01, 2.900576728856e-02,
         -2.287591474628e-03, 1.004346442778e-04, -1.869930069131e-06),
        (5.0, 20.0, 100.0), 0.4,
    ),
    "e_h2_singlet_C": (  # 2.2.3
        (-3.348199796300e01, 1.371702271009e01, -5.922607900694e00,
         1.709719148860e00, -3.505232830275e-01, 4.834376067841e-02,
         -4.131406425550e-03, 1.948388368131e-04, -3.854278715563e-06),
        (5.0, 20.0, 100.0), 0.4,
    ),
}  # fmt: skip


@pytest.mark.parametrize("key", sorted(HYDHEL_RATES))
def test_rate_coefficients_against_hydhel(key: str) -> None:
    coefficients, temperatures, tolerance = HYDHEL_RATES[key]
    table = JANEV_FITS[key].cross_section()
    for kt in temperatures:
        rate = float(rate_coefficient(table, kt, ELECTRON_MASS))
        assert rate == pytest.approx(_hydhel_rate(coefficients, kt), rel=tolerance)


def test_dissociative_attachment_levels_against_table_27() -> None:
    assert len(DISSOCIATIVE_ATTACHMENT_LEVELS) == 15
    assert DISSOCIATIVE_ATTACHMENT_LEVELS[0] == (3.72, 3.22e-5)
    assert DISSOCIATIVE_ATTACHMENT_LEVELS[9] == (0.113, 13.9)
    assert DISSOCIATIVE_ATTACHMENT_LEVELS[14] == (-0.736, 3.35)
    tables = dissociative_attachment_levels()
    for level, (threshold, peak) in enumerate(DISSOCIATIVE_ATTACHMENT_LEVELS):
        fit = dissociative_attachment_fit(level)
        # the peak value at the onset |E_th,v|, zero below, e-fold over E_0
        assert fit.threshold == abs(threshold)
        assert float(tables[level](abs(threshold))) == pytest.approx(peak * A2)
        assert float(tables[level](abs(threshold) * 0.99)) == 0.0
        assert float(
            tables[level](abs(threshold) + DISSOCIATIVE_ATTACHMENT_E0)
        ) == pytest.approx(peak * A2 / math.e, rel=1e-3)
    # figure 27: v = 9 starts at about 1.4e-15 cm^2, v = 0 at about 3e-21 cm^2
    assert float(tables[9](0.113)) == pytest.approx(1.39e-19)
    with pytest.raises(ValueError, match="level"):
        dissociative_attachment_fit(15)


def _eq_125_shape(peak: float, threshold: float, kt: float) -> float:
    """Eq. (125) without its prefactor: the Maxwellian average of eq. (124)."""
    ratio = 1.0 + kt / DISSOCIATIVE_ATTACHMENT_E0
    return (
        peak
        * math.sqrt(kt)
        / ratio
        * math.exp(-threshold / kt)
        * (threshold / kt + 1.0 / ratio)
    )


@pytest.mark.parametrize("level", [2, 5, 7, 9])
def test_dissociative_attachment_rate_against_eq_125(level: int) -> None:
    threshold, peak = DISSOCIATIVE_ATTACHMENT_LEVELS[level]
    table = dissociative_attachment_levels()[level]
    # exact average sqrt(8 kT / (pi m_e)) sigma_v ... of the cross section (eq. (125)
    # prints 1.972e-8 for the prefactor, 2.95 times this one; see the docstring)
    prefactor = math.sqrt(8.0 * 1.602176634e-19 / (math.pi * ELECTRON_MASS))
    for kt in (0.3, 1.0, 3.0, 10.0):
        rate = float(rate_coefficient(table, kt, ELECTRON_MASS))
        exact = prefactor * A2 * _eq_125_shape(peak, threshold, kt)
        assert rate == pytest.approx(exact, rel=0.01)


def test_dissociative_attachment_over_a_vibrational_population() -> None:
    distribution = VibrationalDistribution.h2_ground_state_from_kelvin(5000.0)
    mixture = dissociative_attachment(distribution)
    weights = host(distribution.populations)
    tables = dissociative_attachment_levels()
    for energy in (0.2, 0.5, 1.0, 2.0, 4.0):
        expected = sum(
            w * float(t(energy)) for w, t in zip(weights, tables, strict=True)
        )
        assert float(mixture(energy)) == pytest.approx(expected, rel=1e-6)
    assert "T_vib" in mixture.label and "table 27" in mixture.label
    # a hotter population attaches far more: the rate rises with T_vib
    hot = dissociative_attachment(
        VibrationalDistribution.h2_ground_state_from_kelvin(8000.0)
    )
    assert float(rate_coefficient(hot, 1.0, ELECTRON_MASS)) > 10.0 * float(
        rate_coefficient(mixture, 1.0, ELECTRON_MASS)
    )
    with pytest.raises(ValueError, match="levels"):
        dissociative_attachment(VibrationalDistribution((0.0, 0.5), 0.3))
    assert isinstance(mixture, CrossSection)
