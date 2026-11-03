"""Electron collisions with hydrogen atoms H(n), from Janev, Reiter and Samm (2003).

The analytic fits of section 2.1 of R. K. Janev, D. Reiter and U. Samm,
"Collision Processes in Low-Temperature Hydrogen Plasmas", Jül-4105 (2003),
https://eirene.de/Documentation/report_4105.pdf, for every principal quantum
number ``n`` (the states are resolved in ``n`` only, the ``l`` sub-states
summed over upper and averaged over lower states):

====================================  =================================================
process                               report
====================================  =================================================
e + H(1s) -> e + H(n=2)               eq. (4), table 1 (the ``1s -> n = 2`` column)
e + H(1s) -> e + H(n=3, 4, 5)         eq. (5), table 2
e + H(n=2) -> e + H(n=3)              eq. (5), parameters in section 2.1.1 B
e + H(n) -> e + H(m), other           eqs. (6), (9) and (10), Johnson (1972)
e + H(n) -> 2e + H+, n = 1, 2, 3      eq. (14), table 3
e + H(n) -> 2e + H+, n >= 4           eqs. (15) and (16), Johnson (1972)
====================================  =================================================

The absorption oscillator strengths ``f_nm`` of Johnson, eqs. (11) to (13),
are in `oscillator_strength`. Eq. (13a) of the report prints
``g2(1) = 0.0714``; Johnson's value is 0.07014 (L. C. Johnson, Astrophys. J.
174, 227 (1972)), which gives ``f_12 = 0.4161`` against the exact 0.4162,
so it is used here. The threshold of ``n -> m`` is
``13.6 (1/n^2 - 1/m^2)`` eV and of the ionization of ``n`` ``13.6 / n^2`` eV,
as in the report (eqs. 7, 10 and 14). The report gives the accuracy of the
fits as 10 to 20% from the ground state and 20 to 50% between excited states
near threshold.

Johnson's excitation formula turns negative for ``n -> n + 1`` from ``n = 25``
on, over a range of energies a few times the threshold (``B - A ln(2 n^2 / y)``
outweighs the logarithmic term); there the cross section is set to zero, so
use these fits for ``n <= 24`` (`JOHNSON_MAX_LEVEL`).

Above that, `vriens_smeets_excitation_rate` and
`vriens_smeets_ionization_rate` give the Maxwellian rate coefficients of
L. Vriens and A. H. M. Smeets, Phys. Rev. A 22, 940 (1980), eqs. (17) and (8),
fitted to classical and Born results for any ``n`` (stated accuracy about 20%
for ``n >= 3``): rate coefficients, not cross sections, so for a Maxwellian
electron distribution only.
"""

from __future__ import annotations

import math

import numpy as np
from methodcard import Rule, method

from plasmacoll.janev import HydrogenFit

__all__ = [
    "HYDROGEN_RYDBERG",
    "JOHNSON_MAX_LEVEL",
    "excitation_fit",
    "gaunt_coefficients",
    "hydrogen_atom_processes",
    "ionization_energy",
    "ionization_fit",
    "oscillator_strength",
    "transition_energy",
    "vriens_smeets_excitation_rate",
    "vriens_smeets_ionization_rate",
]

#: The Rydberg energy of the report's thresholds [eV].
HYDROGEN_RYDBERG = 13.6

#: The highest level for which Johnson's excitation formula stays positive.
JOHNSON_MAX_LEVEL = 24

#: 1e-16 cm^2 and 1 cm^2 in m^2.
_A2 = 1.0e-20
_CM2 = 1.0e-4
#: 1 cm^3 in m^3.
_CM3 = 1.0e-6
#: sigma_0 of eqs. (4c) and (5) [1e-16 cm^2 eV].
_SIGMA0 = 5.984

#: Eq. (4), table 1, 1s -> n = 2: a, b, c and A_0..A_5.
_EXCITATION_1_2 = (
    0.228, 0.1865, 0.5025,
    (4.4979, 1.4182, -20.877, 49.735, -46.249, 17.442),
)  # fmt: skip
#: Eq. (5): (alpha, A_0..A_4) for 1s -> n = 3, 4, 5 (table 2) and 2 -> 3 (sec. 2.1.1 B).
_EXCITATION_EQ5: dict[tuple[int, int], tuple[float, tuple[float, ...]]] = {
    (1, 3): (0.38277, (0.75448, 0.42956, -0.58288, 1.0693, 0.0)),
    (1, 4): (0.41844, (0.24300, 0.24846, 0.19701, 0.0, 0.0)),
    (1, 5): (0.45929, (0.11508, 0.13092, 0.23581, 0.0, 0.0)),
    (2, 3): (1.3196, (38.906, 5.2373, 119.25, -595.39, 816.71)),
}
#: Table 2 prints the thresholds of 1s -> n = 3, 4, 5 rounded; eq. (5) uses them.
_EXCITATION_EQ5_THRESHOLD = {(1, 3): 12.09, (1, 4): 12.75, (1, 5): 13.06}
#: Eq. (14), table 3: A_0..A_5 of the ionization of n = 1, 2, 3.
_IONIZATION_EQ14: dict[int, tuple[float, ...]] = {
    1: (0.18450, -0.032226, -0.034539, 1.4003, -2.8115, 2.2986),
    2: (0.14784, 0.0080871, -0.062270, 1.9414, -2.1980, 0.95894),
    3: (0.058463, -0.051272, 0.85310, -0.57014, 0.76684, 0.0),
}


def _check_level(n: int) -> None:
    if int(n) != n or n < 1:
        raise ValueError(
            f"the principal quantum number must be an integer >= 1, not {n}"
        )


def gaunt_coefficients(n: int) -> tuple[float, float, float]:
    """Return Johnson's Gaunt-factor coefficients ``g0, g1, g2`` of level ``n``.

    Eqs. (13a) to (13e) of the report, with Johnson's ``g2(1) = 0.07014``.
    """
    _check_level(n)
    if n == 1:
        return 1.1330, -0.4059, 0.07014
    if n == 2:
        return 1.0785, -0.2319, 0.02947
    g0 = 0.9935 + 0.2328 / n - 0.1296 / n**2
    g1 = -(0.6282 - 0.5598 / n + 0.5299 / n**2) / n
    g2 = (0.3887 - 1.181 / n + 1.470 / n**2) / n**2
    return g0, g1, g2


def ionization_energy(n: int) -> float:
    """Return ``13.6 / n^2``, the ionization energy of H(n) in eV."""
    _check_level(n)
    return HYDROGEN_RYDBERG / n**2


def _check_pair(n: int, m: int) -> None:
    _check_level(n)
    _check_level(m)
    if m <= n:
        raise ValueError(f"need m > n, not n={n}, m={m}")


def transition_energy(n: int, m: int) -> float:
    """Return ``13.6 (1/n^2 - 1/m^2)``, the energy of ``n -> m`` in eV."""
    _check_pair(n, m)
    return HYDROGEN_RYDBERG * (1.0 / n**2 - 1.0 / m**2)


@method(
    equations={
        "oscillator_strength": r"f_{nm} = \frac{32}{3\sqrt3\,\pi}\,\frac{n}{m^3 y^3}\Big(g_0 + \frac{g_1}{y} + \frac{g_2}{y^2}\Big), \qquad y = 1 - \Big(\frac{n}{m}\Big)^2",
        "gaunt": r"g_0 = 0.9935 + \frac{0.2328}{n} - \frac{0.1296}{n^2}, \quad g_1 = -\frac{1}{n}\Big(0.6282 - \frac{0.5598}{n} + \frac{0.5299}{n^2}\Big), \quad g_2 = \frac{1}{n^2}\Big(0.3887 - \frac{1.181}{n} + \frac{1.470}{n^2}\Big) \quad (n \ge 3)",
        "ground_states": r"(g_0, g_1, g_2) = (1.1330, -0.4059, 0.07014)\ (n = 1), \qquad (1.0785, -0.2319, 0.02947)\ (n = 2)",
    },
    assumptions=(
        r"within about 1 % of the exact values; Johnson's $g_2(1) = 0.07014$ instead of the report's 0.0714 gives $f_{12} = 0.4161$ against the exact 0.4162",
    ),
    references=(
        "Johnson, Astrophys. J. 174, 227 (1972)",
        "[Janev, Reiter and Samm, Jül-4105 (2003)](https://eirene.de/Documentation/report_4105.pdf), section 2.1, eqs. (11)-(13)",
    ),
)
def oscillator_strength(n: int, m: int) -> float:
    """Return the absorption oscillator strength ``f_nm`` of H, ``m > n``.

    Johnson's eqs. (11) and (12): ``f = 32 / (3 sqrt(3) pi) n / m^3 / y^3 g``
    with ``y = 1 - (n/m)^2`` and ``g = g0 + g1 / y + g2 / y^2``, within about
    1% of the exact values.
    """
    _check_pair(n, m)
    y = 1.0 - (n / m) ** 2
    g0, g1, g2 = gaunt_coefficients(n)
    gaunt = g0 + g1 / y + g2 / y**2
    return 32.0 / (3.0 * math.sqrt(3.0) * math.pi) * n / m**3 / y**3 * gaunt


def _excitation_1_2(energy: np.ndarray) -> np.ndarray:
    """Eq. (4): linear to 11.56 eV, constant to 12.23 eV, then the Bethe-like fit."""
    a, b, c, coefficients = _EXCITATION_1_2
    delta_e = 10.2
    x = energy / delta_e
    series = coefficients[0] * np.log(x) + sum(
        coefficients[j] / x ** (j - 1) for j in range(1, 6)
    )
    high = _SIGMA0 / (delta_e * x) * series
    sigma = np.where(
        energy <= 11.56, a + b * (energy - delta_e), np.where(energy <= 12.23, c, high)
    )
    return _A2 * sigma


def _eq5(delta_e: float, alpha: float, a: tuple[float, ...]):
    def fit(energy: np.ndarray) -> np.ndarray:
        x = energy / delta_e
        series = a[0] * np.log(x) + sum(a[j] / x ** (j - 1) for j in range(1, 5))
        return _A2 * _SIGMA0 / (delta_e * x) * (1.0 - 1.0 / x) ** alpha * series

    return fit


def _b_coefficient(n: int) -> float:
    """``b_n`` of eqs. (7b) and (10c); ``-0.603`` for the ground state."""
    if n == 1:
        return -0.603
    return (4.0 - 18.63 / n + 36.24 / n**2 - 28.09 / n**3) / n


def _r_coefficient(n: int) -> float:
    """``r_n`` of eqs. (7a) and (10a); ``0.45`` for the ground state."""
    return 0.45 if n == 1 else 1.94 / n**1.57


def _johnson_excitation(n: int, m: int):
    """Eq. (9) (eq. (6) for ``n = 1``) with eqs. (10)."""
    delta_e = transition_energy(n, m)
    y = 1.0 - (n / m) ** 2
    big_a = 2.0 * n**2 * oscillator_strength(n, m) / y
    big_b = (
        4.0 * n**4 / (m**3 * y**2) * (1.0 + 4.0 / (3.0 * y) + _b_coefficient(n) / y**2)
    )
    r = _r_coefficient(n)

    def fit(energy: np.ndarray) -> np.ndarray:
        x = energy / delta_e
        bracket = big_a * (np.log(x) + 0.5 / x) + (
            big_b - big_a * math.log(2.0 * n**2 / y)
        ) * (1.0 - 1.0 / x)
        sigma = _A2 * 1.76 * n**2 / (y * x) * (1.0 - np.exp(-r * y * x)) * bracket
        return np.maximum(sigma, 0.0)

    return fit


def _excitation_rules() -> dict:
    """The SymPy rules of `excitation_fit` for 1 -> 2, 1 -> 3 and 3 -> 5 (for the tests)."""
    import sympy

    e = sympy.symbols("energy", positive=True)
    f, q = sympy.Float, sympy.Rational
    sigma0 = f(_SIGMA0)
    a = (4.4979, 1.4182, -20.877, 49.735, -46.249, 17.442)
    x = e / f(10.2)
    eq4 = (
        sigma0
        / (f(10.2) * x)
        * (f(a[0]) * sympy.log(x) + sum(f(a[j]) / x ** (j - 1) for j in range(1, 6)))
    )
    a = (0.75448, 0.42956, -0.58288, 1.0693)
    x = e / f(12.09)
    eq5 = (
        sigma0
        / (f(12.09) * x)
        * (1 - 1 / x) ** f(0.38277)
        * (f(a[0]) * sympy.log(x) + sum(f(a[j]) / x ** (j - 1) for j in range(1, 4)))
    )
    # Johnson, 3 -> 5
    n, m = 3, 5
    y = 1 - q(n, m) ** 2
    g0 = f(0.9935) + f(0.2328) / n - f(0.1296) / n**2
    g1 = -(f(0.6282) - f(0.5598) / n + f(0.5299) / n**2) / n
    g2 = (f(0.3887) - f(1.181) / n + f(1.470) / n**2) / n**2
    oscillator = 32 / (3 * sympy.sqrt(3) * sympy.pi) * n / (m**3 * y**3)
    oscillator *= g0 + g1 / y + g2 / y**2
    big_a = 2 * n**2 * oscillator / y
    b_n = (4 - f(18.63) / n + f(36.24) / n**2 - f(28.09) / n**3) / n
    big_b = 4 * n**4 / (m**3 * y**2) * (1 + 4 / (3 * y) + b_n / y**2)
    r = f(1.94) / n ** f(1.57)
    x = e / (f(13.6) * (q(1, n**2) - q(1, m**2)))
    bracket = big_a * (sympy.log(x) + 1 / (2 * x)) + (
        big_b - big_a * sympy.log(2 * n**2 / y)
    ) * (1 - 1 / x)
    johnson = f(1.76) * n**2 / (y * x) * (1 - sympy.exp(-r * y * x)) * bracket

    def rule(pair: tuple[int, int], expr, low: float, high: float) -> Rule:
        return Rule(
            f(1.0e-20) * expr,
            implementation=lambda energy: excitation_fit(*pair)(energy),
            ranges={"energy": (low, high)},
        )

    return {
        "eq4": rule((1, 2), eq4, 13.0, 1.0e3),
        "eq5": rule((1, 3), eq5, 14.0, 1.0e3),
        "johnson": rule((n, m), johnson, 1.0, 1.0e3),
    }


@method(
    equations={
        "eq4": r"\sigma_{1\to2} = \begin{cases} a + b(E - \Delta E) & E \le 11.56\ \mathrm{eV} \\ c & E \le 12.23\ \mathrm{eV} \\ \dfrac{\sigma_0}{\Delta E\,x}\Big[A_0\ln x + \sum_{j=1}^{5} A_j x^{1-j}\Big] & \text{above} \end{cases}, \qquad \sigma_0 = 5.984, \ x = \frac{E}{\Delta E}",
        "eq5": r"\sigma = \frac{\sigma_0}{\Delta E\,x}\Big(1 - \frac1x\Big)^\alpha\Big[A_0\ln x + \sum_{j=1}^{4} A_j x^{1-j}\Big] \qquad (1 \to 3, 4, 5;\ 2 \to 3)",
        "johnson": r"\sigma = \frac{1.76\,n^2}{y\,x}\big(1 - e^{-r_n y x}\big)\Big[A\Big(\ln x + \frac{1}{2x}\Big) + \Big(B - A\ln\frac{2n^2}{y}\Big)\Big(1 - \frac1x\Big)\Big], \qquad A = \frac{2n^2 f_{nm}}{y}, \quad B = \frac{4n^4}{m^3 y^2}\Big(1 + \frac{4}{3y} + \frac{b_n}{y^2}\Big)",
        "coefficients": r"b_n = \frac{1}{n}\Big(4 - \frac{18.63}{n} + \frac{36.24}{n^2} - \frac{28.09}{n^3}\Big),\ b_1 = -0.603, \qquad r_n = 1.94\,n^{-1.57},\ r_1 = 0.45",
        "threshold": r"\Delta E = 13.6\Big(\frac{1}{n^2} - \frac{1}{m^2}\Big)\ \mathrm{eV}",
    },
    assumptions=(
        r"$\sigma$ in $10^{-16}$ cm$^2$, $E$ in eV, lab frame; the $l$ sub-states summed over upper and averaged over lower states",
        r"eq. (4) for $1 \to 2$, eq. (5) with table 2 for $1 \to 3, 4, 5$ (their rounded thresholds) and section 2.1.1 B for $2 \to 3$, Johnson's eqs. (6), (9), (10) otherwise",
        r"Johnson's form clipped at zero: it turns negative for $n \to n + 1$ from $n = 25$ (`JOHNSON_MAX_LEVEL` = 24)",
        r"stated accuracy 10 to 20 % from the ground state, 20 to 50 % between excited states near threshold",
    ),
    references=(
        "[Janev, Reiter and Samm, Jül-4105 (2003)](https://eirene.de/Documentation/report_4105.pdf), section 2.1",
        "Johnson, Astrophys. J. 174, 227 (1972)",
    ),
    components=(
        "plasmacoll.atomic_hydrogen.oscillator_strength",
        "plasmacoll.janev.HydrogenFit",
    ),
    symbolic=lambda: _excitation_rules(),
)
def excitation_fit(n: int, m: int) -> HydrogenFit:
    """Return the fit of ``e + H(n) -> e + H(m)``, ``m > n``, in the lab frame.

    See the module docstring for the equation used for each pair.
    """
    _check_pair(n, m)
    reaction = f"e + H(n={n}) -> e + H(n={m})"
    if (n, m) == (1, 2):
        return HydrogenFit(
            "e_h_excitation_1_2", reaction, "eq. (4), table 1", 10.2, "lab",
            _excitation_1_2,
        )  # fmt: skip
    if (n, m) in _EXCITATION_EQ5:
        threshold = _EXCITATION_EQ5_THRESHOLD.get((n, m), transition_energy(n, m))
        alpha, a = _EXCITATION_EQ5[(n, m)]
        source = "eq. (5), table 2" if n == 1 else "eq. (5), section 2.1.1 B"
        return HydrogenFit(
            f"e_h_excitation_{n}_{m}", reaction, source, threshold, "lab",
            _eq5(threshold, alpha, a),
        )  # fmt: skip
    source = "eq. (6), Johnson (1972)" if n == 1 else "eqs. (9), (10), Johnson (1972)"
    return HydrogenFit(
        f"e_h_excitation_{n}_{m}", reaction, source, transition_energy(n, m), "lab",
        _johnson_excitation(n, m),
    )  # fmt: skip


def _eq14(threshold: float, a: tuple[float, ...]):
    def fit(energy: np.ndarray) -> np.ndarray:
        t = 1.0 - threshold / energy
        series = a[0] * np.log(energy / threshold) + sum(
            a[j] * t**j for j in range(1, 6)
        )
        return _CM2 * 1.0e-13 / (threshold * energy) * series

    return fit


def _johnson_ionization(n: int):
    """Eq. (15) with eqs. (16)."""
    g = gaunt_coefficients(n)
    big_a = (
        32.0
        * n
        / (3.0 * math.sqrt(3.0) * math.pi)
        * sum(g[j] / (j + 3) for j in range(3))
    )
    big_b = 2.0 / 3.0 * n**2 * (5.0 + _b_coefficient(n))
    r = _r_coefficient(n)
    threshold = ionization_energy(n)

    def fit(energy: np.ndarray) -> np.ndarray:
        x = energy / threshold
        bracket = (
            big_a * np.log(x)
            + (big_b - big_a * math.log(2.0 * n**2)) * (1.0 - 1.0 / x) ** 2
        )
        return _A2 * 1.76 * n**2 / x * (1.0 - np.exp(-r * x)) * bracket

    return fit


def _ionization_rules() -> dict:
    """The SymPy rules of `ionization_fit` for n = 1 and 4 (for the tests)."""
    import sympy

    e = sympy.symbols("energy", positive=True)
    f = sympy.Float
    a = (0.18450, -0.032226, -0.034539, 1.4003, -2.8115, 2.2986)
    i1 = f(13.6)
    t = 1 - i1 / e
    eq14 = f(1.0e-17) / (i1 * e)
    eq14 *= f(a[0]) * sympy.log(e / i1) + sum(f(a[j]) * t**j for j in range(1, 6))
    n = 4
    g = (
        f(0.9935) + f(0.2328) / n - f(0.1296) / n**2,
        -(f(0.6282) - f(0.5598) / n + f(0.5299) / n**2) / n,
        (f(0.3887) - f(1.181) / n + f(1.470) / n**2) / n**2,
    )
    big_a = (
        32 * n / (3 * sympy.sqrt(3) * sympy.pi) * sum(g[j] / (j + 3) for j in range(3))
    )
    b_n = (4 - f(18.63) / n + f(36.24) / n**2 - f(28.09) / n**3) / n
    big_b = sympy.Rational(2, 3) * n**2 * (5 + b_n)
    x = e / (f(13.6) / n**2)
    bracket = (
        big_a * sympy.log(x) + (big_b - big_a * sympy.log(2 * n**2)) * (1 - 1 / x) ** 2
    )
    johnson = (
        f(1.0e-20) * f(1.76) * n**2 / x * (1 - sympy.exp(-f(1.94) / n ** f(1.57) * x))
    )
    johnson *= bracket
    return {
        "eq14": Rule(
            eq14,
            implementation=lambda energy: ionization_fit(1)(energy),
            ranges={"energy": (20.0, 1.0e3)},
        ),
        "johnson": Rule(
            johnson,
            implementation=lambda energy: ionization_fit(4)(energy),
            ranges={"energy": (2.0, 1.0e3)},
        ),
    }


@method(
    equations={
        "eq14": r"\sigma = \frac{10^{-13}}{I E}\Big[A_0\ln\frac{E}{I} + \sum_{j=1}^{5} A_j\Big(1 - \frac{I}{E}\Big)^j\Big]\ \mathrm{cm}^2 \qquad (n \le 3,\ \text{table 3})",
        "johnson": r"\sigma = \frac{1.76\,n^2}{x}\big(1 - e^{-r_n x}\big)\Big[A\ln x + \big(B - A\ln 2n^2\big)\Big(1 - \frac1x\Big)^2\Big], \qquad A = \frac{32n}{3\sqrt3\,\pi}\sum_{j=0}^{2}\frac{g_j}{j + 3}, \quad B = \frac23 n^2(5 + b_n), \quad x = \frac{E}{I}",
        "threshold": r"I = \frac{13.6}{n^2}\ \mathrm{eV}",
    },
    assumptions=(
        r"Johnson's form in $10^{-16}$ cm$^2$ for $n \ge 4$; $b_n$, $r_n$, $g_j$ as for the excitation; lab frame",
    ),
    references=(
        "[Janev, Reiter and Samm, Jül-4105 (2003)](https://eirene.de/Documentation/report_4105.pdf), section 2.1, eqs. (14)-(16)",
        "Johnson, Astrophys. J. 174, 227 (1972)",
    ),
    components=("plasmacoll.janev.HydrogenFit",),
    symbolic=lambda: _ionization_rules(),
)
def ionization_fit(n: int) -> HydrogenFit:
    """Return the fit of ``e + H(n) -> 2e + H+`` in the lab frame.

    Eq. (14) with table 3 for ``n <= 3``, Johnson's eq. (15) above.
    """
    _check_level(n)
    threshold = ionization_energy(n)
    reaction = f"e + H(n={n}) -> 2e + H+"
    if n in _IONIZATION_EQ14:
        return HydrogenFit(
            f"e_h_ionization_{n}", reaction, "eq. (14), table 3", threshold, "lab",
            _eq14(threshold, _IONIZATION_EQ14[n]),
        )  # fmt: skip
    return HydrogenFit(
        f"e_h_ionization_{n}", reaction, "eqs. (15), (16), Johnson (1972)",
        threshold, "lab", _johnson_ionization(n),
    )  # fmt: skip


def _vs_b(n: int) -> float:
    """``b_p`` of Vriens and Smeets, eq. (11); ``-0.603`` for ``n = 1`` as Johnson's."""
    if n == 1:
        return -0.603
    return 1.4 * math.log(n) / n - 0.7 / n - 0.51 / n**2 + 1.16 / n**3 - 0.55 / n**4


@method(
    equations={
        "rate": r"K = \frac{1.6\cdot10^{-7}\sqrt T}{T + \Gamma}\,e^{-\Delta E/T}\Big(A\ln\Big(\frac{0.3\,T}{R_y} + \Delta\Big) + B\Big)\ \mathrm{cm^3/s}",
        "coefficients": r"A = \frac{2R_y}{\Delta E}f_{nm}, \quad B = \frac{4R_y^2}{m^3}\Big(\frac{1}{\Delta E^2} + \frac43\frac{E_n}{\Delta E^3} + b_n\frac{E_n^2}{\Delta E^4}\Big), \quad \Delta = e^{-B/A} + \frac{0.06\,s^2}{m n^2}",
        "gamma": r"\Gamma = R_y\ln\Big(1 + \frac{n^3 T}{R_y}\Big)\frac{3 + 11 s^2/n^2}{6 + 1.6\,m s + 0.3/s^2 + 0.8\,m^{1.5}s^{-0.5}|s - 0.6|}, \qquad s = m - n, \quad E_n = \frac{R_y}{n^2}",
        "b_n": r"b_n = \frac{1.4\ln n}{n} - \frac{0.7}{n} - \frac{0.51}{n^2} + \frac{1.16}{n^3} - \frac{0.55}{n^4}, \qquad b_1 = -0.603",
    },
    assumptions=(
        r"a Maxwellian electron distribution of temperature $T$ (eV); any $n$, stated accuracy about 20 % for $n \ge 3$; clipped at zero",
    ),
    references=(
        "[Vriens and Smeets, Phys. Rev. A 22, 940 (1980)](https://doi.org/10.1103/PhysRevA.22.940), eq. (17)",
    ),
    components=("plasmacoll.atomic_hydrogen.oscillator_strength",),
)
def vriens_smeets_excitation_rate(
    n: int, m: int, temperature: np.ndarray | float
) -> np.ndarray:
    """Return the Maxwellian rate coefficient of ``e + H(n) -> e + H(m)`` in m^3/s.

    Vriens and Smeets (1980), eq. (17), with ``f_nm`` of
    `oscillator_strength`:

    ``K = 1.6e-7 sqrt(T) / (T + Gamma) exp(-dE/T) (A ln(0.3 T/Ry + Delta) + B)``
    cm^3/s, ``A = 2 Ry/dE f_nm``, ``B = 4 Ry^2/m^3 (1/dE^2 + 4/3 E_n/dE^3 + b_n E_n^2/dE^4)``,
    ``Delta = exp(-B/A) + 0.06 s^2/(m n^2)`` and ``Gamma = Ry ln(1 + n^3 T/Ry)
    (3 + 11 s^2/n^2) / (6 + 1.6 m s + 0.3/s^2 + 0.8 m^1.5/s^0.5 |s - 0.6|)``,
    ``s = m - n``, ``E_n = Ry/n^2``.

    Args:
        n: The lower level.
        m: The upper level, ``m > n``.
        temperature: ``kT_e`` in eV.
    """
    _check_pair(n, m)
    t = np.asarray(temperature, dtype=float)
    ry = HYDROGEN_RYDBERG
    delta_e = transition_energy(n, m)
    energy_n = ry / n**2
    s = m - n
    a = 2.0 * ry / delta_e * oscillator_strength(n, m)
    b = (
        4.0
        * ry**2
        / m**3
        * (
            1.0 / delta_e**2
            + 4.0 / 3.0 * energy_n / delta_e**3
            + _vs_b(n) * energy_n**2 / delta_e**4
        )
    )
    delta = math.exp(-b / a) + 0.06 * s**2 / (m * n**2)
    gamma = (
        ry
        * np.log(1.0 + n**3 * t / ry)
        * (3.0 + 11.0 * s**2 / n**2)
        / (6.0 + 1.6 * m * s + 0.3 / s**2 + 0.8 * m**1.5 / s**0.5 * abs(s - 0.6))
    )
    k = (
        1.6e-7
        * np.sqrt(t)
        / (t + gamma)
        * np.exp(-delta_e / t)
        * (a * np.log(0.3 * t / ry + delta) + b)
    )
    return _CM3 * np.maximum(k, 0.0)


def _vs_ionization_rules() -> dict:
    """The SymPy rule of `vriens_smeets_ionization_rate` for n = 2 (for the tests)."""
    import sympy

    t = sympy.symbols("temperature", positive=True)
    f = sympy.Float
    eps = f(13.6) / 4 / t
    rule = f(1.0e-6) * f(9.56e-6) * t ** f(-1.5) * sympy.exp(-eps)
    rule /= eps ** f(2.33) + f(4.38) * eps ** f(1.72) + f(1.32) * eps
    return {
        "rate": Rule(
            rule,
            implementation=lambda temperature: vriens_smeets_ionization_rate(
                2, temperature
            ),
            ranges={"temperature": (0.5, 50.0)},
        )
    }


@method(
    equations={
        "rate": r"K = \frac{9.56\cdot10^{-6}\,T^{-1.5}\,e^{-\epsilon_n}}{\epsilon_n^{2.33} + 4.38\,\epsilon_n^{1.72} + 1.32\,\epsilon_n}\ \mathrm{cm^3/s}, \qquad \epsilon_n = \frac{E_n}{T}, \quad E_n = \frac{13.6}{n^2}\ \mathrm{eV}",
    },
    assumptions=(
        r"a Maxwellian electron distribution of temperature $T$ (eV); any $n$",
    ),
    references=(
        "[Vriens and Smeets, Phys. Rev. A 22, 940 (1980)](https://doi.org/10.1103/PhysRevA.22.940), eq. (8)",
    ),
    symbolic=lambda: _vs_ionization_rules(),
)
def vriens_smeets_ionization_rate(
    n: int, temperature: np.ndarray | float
) -> np.ndarray:
    """Return the Maxwellian rate coefficient of ``e + H(n) -> 2e + H+`` in m^3/s.

    Vriens and Smeets (1980), eq. (8): ``K = 9.56e-6 T^-1.5 exp(-e_n) /
    (e_n^2.33 + 4.38 e_n^1.72 + 1.32 e_n)`` cm^3/s with ``e_n = E_n / T``.
    """
    _check_level(n)
    t = np.asarray(temperature, dtype=float)
    eps = ionization_energy(n) / t
    return (
        _CM3
        * 9.56e-6
        * t**-1.5
        * np.exp(-eps)
        / (eps**2.33 + 4.38 * eps**1.72 + 1.32 * eps)
    )


def hydrogen_atom_processes(
    background: str = "H",
    max_level: int = 5,
    ion: str = "H+",
    max_energy: float = 1.0e4,
) -> list:
    """Return the electron excitation and ionization of ground-state H for the MCC.

    Excitation ``1 -> m`` for ``m = 2..max_level`` and ionization, as
    `plasmacoll.process.CollisionProcess` objects on the background
    ``background``. The elastic e + H cross section is in
    `plasmacoll.hydrogen_elastic`.

    Args:
        background: The name of the H background.
        max_level: The highest upper level of the excitations.
        ion: The species of the ion of the ionization.
        max_energy: The last tabulated energy in eV.
    """
    from plasmacoll.process import CollisionProcess

    processes = [
        CollisionProcess(
            "excitation",
            background,
            excitation_fit(1, m).cross_section(max_energy),
            name=f"excitation:{background}(n={m})",
            energy_frame="lab",
        )
        for m in range(2, max_level + 1)
    ]
    processes.append(
        CollisionProcess(
            "ionization",
            background,
            ionization_fit(1).cross_section(max_energy),
            name=f"ionization:{background}",
            energy_frame="lab",
            products={"electron": "e", "ion": ion},
        )
    )
    return processes
