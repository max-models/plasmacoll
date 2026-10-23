"""The elastic e + H and e + H2 fits of Park et al. (2022) against independent data.

The reference values are printed in their sources:

- England, Elford and Crompton, Aust. J. Phys. 41, 573 (1988), table 5: the
  swarm-derived e + H2 momentum-transfer cross section (in A^2), reliable below
  5 eV (Brunger et al. 1991, citing Elford 1990).
- Brunger, Buckman, Newman and Alle, J. Phys. B 24, 1435 (1991), table 3: the
  integral elastic and momentum-transfer cross sections of H2 from beam
  measurements, 1 to 5 eV, with errors of about 20 %.
- The CCC database of e + H scattering (https://atom.curtin.edu.au/CCC-WWW/,
  file 1S.1S of e-H_XSEC_LS.zip, in a0^2), which the paper's integral fit for
  H reproduces.
- Schwartz, Phys. Rev. 124, 1468 (1961): the e + H singlet and triplet
  scattering lengths a_s = 5.965 a0 and a_t = 1.7686 a0.
"""

from __future__ import annotations

import math

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import (
    CollisionProcess,
    NeutralBackground,
    solve_eedf,
)
from plasmacoll.constants import ATOMIC_MASS, kelvin_to_ev
from plasmacoll.hydrogen_elastic import ELASTIC_FITS, PARK_2022, ElasticFit
from plasmacoll.janev import JANEV_FITS

#: 1 A^2 and 1 a0^2 in m^2
A2 = 1.0e-20
A0_2 = (0.529177210903e-10) ** 2

H = ELASTIC_FITS["H"]
H2 = ELASTIC_FITS["H2"]

#: England et al. (1988), table 5: (E [eV], sigma_m [A^2]) up to 5 eV.
ENGLAND_1988 = (
    (0.02, 7.95),
    (0.03, 8.45),
    (0.04, 8.91),
    (0.05, 9.22),
    (0.07, 9.79),
    (0.10, 10.44),
    (0.15, 11.33),
    (0.20, 11.93),
    (0.30, 12.92),
    (0.50, 14.61),
    (0.70, 16.20),
    (1.0, 17.30),
    (1.5, 18.14),
    (2.0, 17.89),
    (3.0, 16.90),
    (4.0, 14.71),
    (5.0, 12.92),
)
#: Brunger et al. (1991), table 3: (E [eV], sigma_el, sigma_m [A^2]).
BRUNGER_1991 = (
    (1.0, 11.68, 14.59),
    (1.25, 13.03, 16.84),
    (1.5, 13.85, 16.41),
    (2.0, 14.56, 16.51),
    (2.5, 15.10, 17.11),
    (3.0, 15.52, 16.33),
    (5.0, 13.90, 11.45),
)
#: CCC database, e + H 1S -> 1S: (E [eV], sigma [a0^2]).
CCC_H_ELASTIC = (
    (0.0, 145.0),
    (0.1, 139.7),
    (1.0, 82.21),
    (2.0, 60.90),
    (5.0, 38.07),
    (10.0, 22.74),
    (20.6, 10.04),
    (48.6, 3.403),
    (104.6, 1.322),
    (200.6, 0.606),
    (520.6, 0.201),
    (968.6, 0.103),
)


def host(array) -> np.ndarray:
    return np.asarray(xp.to_numpy(xp.asarray(array)), dtype=float)


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def test_the_fits_against_their_equations_by_hand() -> None:
    # eq. (26) with table II for H2 at 10 eV
    b = (5.36772e-18, -5.01754e-18, 2.09077e-19, 1.36876, 0.959997, 1.16409)
    expected = (
        (b[0] + b[1] / 11.0 + b[2] / 101.0)
        * (10.0 ** b[3] + math.exp(-b[4] * 10.0)) ** b[5]
        / (11.0**3 + 4.08348 * 11.0**2 + 4.23339 * 11.0)
    )
    assert float(H2.integral(10.0)) == pytest.approx(expected)
    # eqs. (19), (20) with table I for H at 2 eV
    assert float(H.forward_weight(2.0)) == pytest.approx(
        0.0502 + 0.9190 * math.tanh(0.0799 * (2.0 - 0.9521))
    )
    eta_f, eta_b = H.screening(2.0)
    assert float(eta_f) == pytest.approx(3.29 / (2.0 + 10.91) + 0.008)
    assert float(eta_b) == pytest.approx(0.936 / (2.0 + 0.0177) + 1.251)
    # H2's backward screening is constant
    assert host(H2.screening([0.1, 100.0])[1]) == pytest.approx([0.729, 0.729])


@pytest.mark.parametrize("fit", [H, H2], ids=["H", "H2"])
@pytest.mark.parametrize("energy", [0.01, 1.0, 10.0, 100.0])
def test_closed_form_momentum_transfer_integrates_the_angular_distribution(
    fit: ElasticFit, energy: float
) -> None:
    mu = np.linspace(-1.0, 1.0, 400_001)
    distribution = fit.angular_distribution(energy, mu)
    assert 2.0 * math.pi * trapezoid(distribution, mu) == pytest.approx(1.0, rel=1e-6)
    ratio = 2.0 * math.pi * trapezoid((1.0 - mu) * distribution, mu)
    assert float(fit.momentum_transfer(energy)) == pytest.approx(
        ratio * float(fit.integral(energy)), rel=1e-6
    )


def test_h2_momentum_transfer_against_the_swarm_cross_section() -> None:
    # England et al. (1988) below 5 eV, where the swarm analysis is unique
    ratios = [float(H2.momentum_transfer(e)) / (s * A2) for e, s in ENGLAND_1988]
    assert min(ratios) > 0.95
    assert max(ratios) < 1.101
    # below the CCC range the fit exceeds the swarm value: 6.36 A^2 at 0 eV
    assert float(H2.momentum_transfer(0.0)) / (6.36 * A2) == pytest.approx(
        1.23, abs=0.01
    )


def test_h2_against_the_beam_measurements() -> None:
    for energy, elastic, momentum in BRUNGER_1991:
        # within their 20 % errors, the fits 3 to 15 % above
        assert 1.0 < float(H2.integral(energy)) / (elastic * A2) < 1.15
        assert 1.0 < float(H2.momentum_transfer(energy)) / (momentum * A2) < 1.15


def test_h_integral_cross_section_against_the_ccc_database() -> None:
    for energy, sigma in CCC_H_ELASTIC:
        assert float(H.integral(energy)) == pytest.approx(sigma * A0_2, rel=0.03)


def test_h_momentum_transfer_at_zero_energy_against_the_scattering_lengths() -> None:
    # s-wave limit: sigma_m = sigma = pi (a_s^2 / 4 * 4 + 3 a_t^2) (1/4 singlet, 3/4 triplet)
    expected = math.pi * (5.965**2 + 3.0 * 1.7686**2) * A0_2
    assert float(H.momentum_transfer(0.0)) / expected == pytest.approx(1.054, abs=0.005)


def first_born_momentum_transfer_h(energy: float) -> float:
    """sigma_m of e + H(1s) in the first Born approximation, m^2 (no exchange).

    With f(q) = 2 (8 + q^2) / (4 + q^2)^2 a0 and u = (q a0)^2,
    sigma_m = 2 pi / k^4 int_0^{4 k^2} u (8 + u)^2 / (4 + u)^4 du, whose
    antiderivative in v = 4 + u is ln v - 4 / v + 8 / v^2 + 64 / (3 v^3).
    """
    k2 = energy / 13.605693122994  # (k a0)^2

    def antiderivative(v: float) -> float:
        return math.log(v) - 4.0 / v + 8.0 / v**2 + 64.0 / (3.0 * v**3)

    integral = antiderivative(4.0 + 4.0 * k2) - antiderivative(4.0)
    return 2.0 * math.pi / k2**2 * integral * A0_2


def test_h_momentum_transfer_above_the_fits_range_is_too_large() -> None:
    # documented: at 1 keV, where the first Born integral cross section agrees
    # with CCC to 1 %, the fit's momentum transfer is 2.2 times the Born one
    born_integral = math.pi * (7 * (1e3 / 13.6057) ** 2 + 18 * 1e3 / 13.6057 + 12)
    born_integral /= 3 * (1e3 / 13.6057 + 1) ** 3
    assert born_integral * A0_2 == pytest.approx(0.1021 * A0_2, rel=0.01)
    ratio = float(H.momentum_transfer(1e3)) / first_born_momentum_transfer_h(1e3)
    assert ratio == pytest.approx(2.16, abs=0.02)
    # H2 stays within 15 % of twice the H Born value (independent atoms) there
    assert float(H2.momentum_transfer(1e3)) / (
        2.0 * first_born_momentum_transfer_h(1e3)
    ) == pytest.approx(1.0, abs=0.15)


@pytest.mark.parametrize("fit", [H, H2], ids=["H", "H2"])
def test_tables(fit: ElasticFit) -> None:
    table = fit.cross_section()
    energy = host(table.energy)
    sigma = host(table.sigma)
    assert energy[0] == 0.0
    assert energy[-1] == pytest.approx(1.0e4)
    assert table.threshold == 0.0
    assert "Park et al. (2022)" in table.label
    assert sigma == pytest.approx(host(fit.momentum_transfer(energy)))
    assert np.all(sigma > 0.0)
    # the momentum transfer falls monotonically above 5 eV
    assert np.all(np.diff(sigma[energy > 5.0]) < 0.0)
    integral = fit.cross_section("integral", max_energy=100.0, num=11)
    assert host(integral.sigma) == pytest.approx(
        host(fit.integral(host(integral.energy)))
    )
    assert host(integral.energy)[-1] == pytest.approx(100.0)
    with pytest.raises(ValueError, match="kind"):
        fit.cross_section("viscosity")
    assert fit.reaction in fit.label


def test_reference() -> None:
    assert "Plasma Sources Sci. Technol. 31, 065013 (2022)" in PARK_2022


def test_a_hydrogen_set_in_the_boltzmann_solver() -> None:
    """The H2 momentum transfer with the Janev e + H2 set solves to a sane swarm."""
    gas = NeutralBackground("H2", 1.0e22, kelvin_to_ev(300.0), 2.016 * ATOMIC_MASS)
    processes = [
        CollisionProcess(
            "elastic", "H2", H2.cross_section(), name="elastic", energy_frame="lab"
        )
    ]
    for key, fit in JANEV_FITS.items():
        if not key.startswith("e_h2"):
            continue
        kind, products = "excitation", {}
        if key == "e_h2_ionization":
            kind, products = "ionization", {"electron": "e", "ion": "H2+"}
        elif key == "e_h2_dissociative_ionization":
            kind, products = "dissociative_ionization", {"electron": "e", "ion": "H+"}
        processes.append(
            CollisionProcess(
                kind,
                "H2",
                fit.cross_section(),
                name=key,
                energy_frame="lab",
                products=products,
            )
        )
    solution = solve_eedf(processes, [gas], 100.0)
    assert 1.0 < solution.mean_energy < 10.0
    assert 1.0e23 < solution.mobility < 1.0e25
    assert solution.townsend_coefficient > 0.0
