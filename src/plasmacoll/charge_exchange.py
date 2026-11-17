r"""Charge exchange of fast hydrogen ions with hydrogen atoms in the ground state.

A proton (or deuteron, triton) that meets a hydrogen atom H(1s) can capture its
electron, H$^+$ + H(1s) $\to$ H + H$^+$: the resonant charge exchange that turns
fast ions in a fusion plasma into fast neutrals (their loss and the signal of
neutral particle analysers) and that couples ions and neutrals in the edge.
The total capture cross section is the analytic fit of Janev and Smith,

$$
\sigma_\mathrm{cx}(E) = 10^{-16}\,\mathrm{cm^2}\;
\frac{a_1 \ln\!\left(a_2 / E + a_6\right)}{1 + a_3 E + a_4 E^{3.5} + a_5 E^{5.4}},
$$

with $E$ the collision energy per nucleon in keV/u (the projectile's energy per
atomic mass unit with the atom at rest, so a function of the relative speed only)
and $a_1 \dots a_6 = 3.2345,\ 235.88,\ 0.038371,\ 3.8068\cdot10^{-6},\
1.1832\cdot10^{-10},\ 2.3713$ (R. K. Janev and J. J. Smith, Atomic and
Plasma-Material Interaction Data for Fusion 4, IAEA, Vienna, 1993; the
recommended H$^+$ + H(1s) electron-capture data of the IAEA ALADDIN database).
It sums the capture into all final states of the atom. The isotopes take the
hydrogen data at the same speed: the electron capture depends on the relative
velocity, the isotope effect is small above a few eV/u.
"""

from __future__ import annotations

import numpy as np
from methodcard import Rule, method

from plasmacoll._types import Array
from plasmacoll.cross_sections import CrossSection
from plasmacoll.units import to_unit

__all__ = [
    "JANEV_SMITH_1993",
    "PROTON_HYDROGEN_CX_COEFFICIENTS",
    "proton_hydrogen_charge_exchange",
    "proton_hydrogen_charge_exchange_table",
]

#: The source of the fit.
JANEV_SMITH_1993 = (
    "R. K. Janev and J. J. Smith, Atomic and Plasma-Material Interaction Data "
    "for Fusion 4 (IAEA, Vienna, 1993)"
)
#: $a_1 \dots a_6$ of the fit (E in keV/u, sigma in 1e-16 cm^2).
PROTON_HYDROGEN_CX_COEFFICIENTS = (
    3.2345,
    235.88,
    0.038371,
    3.8068e-6,
    1.1832e-10,
    2.3713,
)


def _charge_exchange_rules() -> dict:
    """The SymPy rule of `proton_hydrogen_charge_exchange` (for the tests)."""
    import sympy

    energy = sympy.symbols("energy_per_nucleon", positive=True)
    a1, a2, a3, a4, a5, a6 = (sympy.Float(a) for a in PROTON_HYDROGEN_CX_COEFFICIENTS)
    e = energy / 1000
    rule = (
        sympy.Float(1.0e-20)
        * a1
        * sympy.log(a2 / e + a6)
        / (
            1
            + a3 * e
            + a4 * e ** sympy.Rational(7, 2)
            + a5 * e ** sympy.Rational(27, 5)
        )
    )
    return {
        "capture": Rule(rule, ranges={"energy_per_nucleon": (1.0, 1.0e6)}, rtol=1e-9)
    }


@method(
    equations={
        "capture": r"\sigma_\mathrm{cx} = 10^{-16}\,\mathrm{cm^2}\;\frac{a_1 \ln(a_2 / E + a_6)}{1 + a_3 E + a_4 E^{3.5} + a_5 E^{5.4}}, \qquad E\ \text{in keV/u}",
    },
    assumptions=(
        r"H$^+$ + H(1s), capture into all final states; the atom at rest, $E$ the projectile's energy per atomic mass unit",
        r"D$^+$ and T$^+$ on H, D or T at the same relative speed (no isotope effect)",
    ),
    references=(
        "R. K. Janev and J. J. Smith, Atomic and Plasma-Material Interaction Data for Fusion 4 (IAEA, Vienna, 1993)",
    ),
    symbolic=_charge_exchange_rules,
)
def proton_hydrogen_charge_exchange(energy_per_nucleon: Array | float) -> np.ndarray:
    """Return the H+ + H(1s) charge-exchange cross section [m^2].

    Args:
        energy_per_nucleon: Collision energy per atomic mass unit of the
            projectile, the atom at rest [eV/u] (positive).
    """
    a1, a2, a3, a4, a5, a6 = PROTON_HYDROGEN_CX_COEFFICIENTS
    e = np.asarray(energy_per_nucleon, dtype=float) / 1000.0
    return (
        1.0e-20 * a1 * np.log(a2 / e + a6) / (1.0 + a3 * e + a4 * e**3.5 + a5 * e**5.4)
    )


@method(
    equations={
        "same_speed": r"\sigma(E) = \sigma_\mathrm{cx}(E / A), \qquad E_i = \mathrm{geomspace}(E_{\min}, E_{\max}, n)",
    },
    assumptions=(r"the end values below and above the table (`CrossSection`)",),
    components=(
        proton_hydrogen_charge_exchange,
        "plasmacoll.cross_sections.CrossSection",
    ),
)
def proton_hydrogen_charge_exchange_table(
    mass_number: float = 1.0,
    min_energy: float = 1.0,
    max_energy: float = 2.0e6,
    num: int = 241,
) -> CrossSection:
    """Tabulate `proton_hydrogen_charge_exchange` against the projectile's lab energy.

    Args:
        mass_number: The projectile's mass in u (1 for H+, 2 for D+); it takes
            the hydrogen data at the same speed.
        min_energy: The smallest projectile energy of the table [eV] (or a
            Quantity).
        max_energy: The largest [eV] (or a Quantity); beyond the table its end
            values hold.
        num: The number of logarithmically spaced points.
    """
    low = to_unit(min_energy, "eV", "min_energy")
    high = to_unit(max_energy, "eV", "max_energy")
    if not 0.0 < low < high:
        raise ValueError("need 0 < min_energy < max_energy")
    return CrossSection.from_function(
        lambda energy: proton_hydrogen_charge_exchange(energy / mass_number),
        np.geomspace(low, high, num),
        label=f"H+ + H(1s) -> H + H+: Janev and Smith (1993), mass {mass_number:g} u",
    )
