"""Electron-impact ionization of sputtered metal atoms and ions from the Lotz formula.

Cross-section sets for metals are scarce: LXCat has e + Cu in the SIGLO
database (digitized from Tkachev, Fedenev and Yakovlenko, Laser Phys. 17, 775
(2007)), which `plasmacoll.cross_sections.read_lxcat` reads like any
other set, and nothing for Ti. Where there is no set, the empirical formula of
W. Lotz, Z. Phys. 206, 205 (1967) and Z. Phys. 216, 241 (1968) gives the
ionization cross section as a sum over the subshells ``i`` of the target::

    sigma(E) = sum_i a_i q_i ln(E/P_i) / (E P_i) [1 - b_i exp(-c_i (E/P_i - 1))]

for ``E >= P_i``, with ``q_i`` electrons of binding energy ``P_i`` (eV). With
``b_i = c_i = 0`` and ``a_i = 4.5e-14 cm^2 eV^2`` it is Lotz's simplified
formula, which needs only the binding energies.

`metal_ionization_cross_section` applies the simplified formula to the
outermost subshell of Cu, Ti and their singly charged ions, at the ionization
energy of the NIST Atomic Spectra Database. That leaves out the deeper
subshells (the 3d electrons of Cu, the 3d and 3p electrons of Ti), which add to
the cross section above their own binding energies: it is a lower estimate,
and `lotz_cross_section` takes any subshells for which binding energies
are at hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from methodcard import Rule, method

from plasmacoll._types import Array
from plasmacoll.constants import ATOMIC_MASS
from plasmacoll.cross_sections import CrossSection
from plasmacoll.units import convert_units, to_unit, unit_field

__all__ = [
    "LOTZ_CONSTANT",
    "METAL_MASSES",
    "METAL_OUTER_SUBSHELLS",
    "LotzSubshell",
    "lotz_cross_section",
    "lotz_table",
    "metal_ionization_cross_section",
]

#: Lotz's ``a = 4.5e-14 cm^2 eV^2`` of the simplified formula, in m^2 eV^2.
LOTZ_CONSTANT = 4.5e-18

#: Atomic masses in kg (standard atomic weights in u).
METAL_MASSES = {
    "Cu": 63.546 * ATOMIC_MASS,
    "Ti": 47.867 * ATOMIC_MASS,
}

#: ``(element, charge) -> (subshell, electrons, ionization energy in eV)`` of the
#: ground configurations and ionization energies of the NIST Atomic Spectra
#: Database: Cu I 3d10 4s, Cu II 3d10, Ti I 3d2 4s2, Ti II 3d2 4s.
METAL_OUTER_SUBSHELLS: dict[tuple[str, int], tuple[str, int, float]] = {
    ("Cu", 0): ("4s", 1, 7.726380),
    ("Cu", 1): ("3d", 10, 20.29239),
    ("Ti", 0): ("4s", 2, 6.828120),
    ("Ti", 1): ("4s", 1, 13.5755),
}


@dataclass(frozen=True)
class LotzSubshell:
    """One subshell of the Lotz formula.

    ``binding_energy`` and ``a`` take astropy Quantities, strings or plain
    numbers in the units below.

    Attributes:
        binding_energy: ``P_i`` in eV, the threshold of the subshell.
        electrons: ``q_i``, the number of electrons in the subshell.
        a: ``a_i`` in m^2 eV^2 (Lotz's ``4.5e-14 cm^2 eV^2`` by default).
        b: ``b_i``, the near-threshold reduction; 0 in the simplified formula.
        c: ``c_i``, its energy scale; 0 in the simplified formula.
    """

    binding_energy: float
    electrons: float
    a: float = unit_field("m^2 eV^2", LOTZ_CONSTANT)
    b: float = 0.0
    c: float = 0.0

    def __post_init__(self) -> None:
        """Convert the quantities and validate the parameters."""
        convert_units(self)
        energy = to_unit(self.binding_energy, "eV", "binding_energy")
        object.__setattr__(self, "binding_energy", energy)
        if self.binding_energy <= 0.0:
            raise ValueError("binding_energy must be > 0")
        if self.electrons <= 0.0:
            raise ValueError("electrons must be > 0")
        if self.a < 0.0 or not 0.0 <= self.b <= 1.0 or self.c < 0.0:
            raise ValueError("need a >= 0, 0 <= b <= 1 and c >= 0")


def _lotz_rules() -> dict:
    """The SymPy rule of `lotz_cross_section` for one subshell (for the tests)."""
    import sympy

    e, p, q, b, c = sympy.symbols("energy binding_energy electrons b c", positive=True)
    rule = sympy.Float(LOTZ_CONSTANT) * q * sympy.log(e / p) / (e * p)
    rule *= 1 - b * sympy.exp(-c * (e / p - 1))
    return {
        "lotz": Rule(
            rule,
            implementation=lambda energy, binding_energy, electrons, b, c: (
                lotz_cross_section(
                    energy, [LotzSubshell(binding_energy, electrons, b=b, c=c)]
                )[0]
            ),
            ranges={
                "energy": (20.0, 1.0e3),
                "binding_energy": (5.0, 15.0),
                "electrons": (1.0, 4.0),
                "b": (0.0, 1.0),
                "c": (0.0, 2.0),
            },
        )
    }


@method(
    equations={
        "lotz": r"\sigma(E) = \sum_i a_i q_i\,\frac{\ln(E/P_i)}{E P_i}\Big[1 - b_i\,e^{-c_i(E/P_i - 1)}\Big], \qquad E \ge P_i",
    },
    assumptions=(
        r"$q_i$ electrons of binding energy $P_i$ (eV) per subshell; $E$ in eV, $\sigma$ in m$^2$",
        r"simplified: $b_i = c_i = 0$, $a_i = 4.5\cdot10^{-14}$ cm$^2$ eV$^2$",
    ),
    references=(
        "[Lotz, Z. Phys. 206, 205 (1967)](https://doi.org/10.1007/BF01325928)",
        "[Lotz, Z. Phys. 216, 241 (1968)](https://doi.org/10.1007/BF01392963)",
    ),
    symbolic=lambda: _lotz_rules(),
)
def lotz_cross_section(
    energy: Array | float, subshells: Sequence[LotzSubshell]
) -> np.ndarray:
    """Return the Lotz ionization cross section in m^2 at ``energy`` (eV).

    Args:
        energy: Electron energies in eV.
        subshells: The subshells of the target.
    """
    energy = np.atleast_1d(np.asarray(energy, dtype=float))
    sigma = np.zeros_like(energy)
    for shell in subshells:
        ratio = energy / shell.binding_energy
        above = ratio >= 1.0
        r = ratio[above]
        sigma[above] += (
            shell.a
            * shell.electrons
            * np.log(r)
            / (energy[above] * shell.binding_energy)
            * (1.0 - shell.b * np.exp(-shell.c * (r - 1.0)))
        )
    return sigma


def lotz_table(
    subshells: Sequence[LotzSubshell],
    max_energy: float = 1.0e4,
    num: int = 400,
    label: str = "",
) -> CrossSection:
    """Tabulate `lotz_cross_section` from the lowest binding energy up.

    The table has ``num`` logarithmically spaced points from the threshold to
    ``max_energy`` (eV), plus a point at every other binding energy; the
    threshold of the table is the lowest binding energy.

    Args:
        subshells: The subshells of the target.
        max_energy: Last table energy in eV (or a Quantity).
        num: Number of logarithmically spaced points.
        label: A name for plots and messages.
    """
    max_energy = to_unit(max_energy, "eV", "max_energy")
    if not subshells:
        raise ValueError("subshells must not be empty")
    threshold = min(shell.binding_energy for shell in subshells)
    if max_energy <= threshold:
        raise ValueError("max_energy must exceed the lowest binding energy")
    energy = np.unique(
        np.concatenate(
            [
                np.geomspace(threshold, max_energy, num),
                [s.binding_energy for s in subshells if s.binding_energy < max_energy],
            ]
        )
    )
    return CrossSection.from_function(
        lambda e: lotz_cross_section(e, subshells),
        energy,
        threshold=threshold,
        label=label or "Lotz ionization",
    )


@method(
    equations={
        "simplified": r"\sigma(E) = a\,q\,\frac{\ln(E/P)}{E P}, \qquad a = 4.5\cdot10^{-14}\ \mathrm{cm^2\,eV^2}",
        "peak": r"E_\mathrm{peak} = e_\mathrm{Euler}\,P, \qquad \sigma_{\max} = \frac{a\,q}{e_\mathrm{Euler} P^2}",
    },
    assumptions=(
        r"the outermost subshell only (Cu 4s$^1$, Cu$^+$ 3d$^{10}$, Ti 4s$^2$, Ti$^+$ 4s$^1$) at the NIST ionization energies: a lower estimate, the deeper subshells left out",
        r"400 logarithmic points from $P$ to 10 keV, the last value beyond",
    ),
    references=(
        "[Lotz, Z. Phys. 206, 205 (1967)](https://doi.org/10.1007/BF01325928)",
        "[Lotz, Z. Phys. 216, 241 (1968)](https://doi.org/10.1007/BF01392963)",
        "NIST Atomic Spectra Database (ionization energies)",
    ),
    components=("plasmacoll.metals.lotz_cross_section",),
)
def metal_ionization_cross_section(
    element: str,
    charge: int = 0,
    max_energy: float = 1.0e4,
    num: int = 400,
) -> CrossSection:
    """Return the simplified Lotz cross section of the outermost subshell of a metal.

    The ionization ``M^(charge)+ + e -> M^(charge+1)+ + 2e`` of the elements
    and charges in `METAL_OUTER_SUBSHELLS`, a lower estimate (see the
    module docstring).

    Args:
        element: ``"Cu"`` or ``"Ti"``.
        charge: 0 for the atom, 1 for the singly charged ion.
        max_energy: Last table energy in eV (or a Quantity).
        num: Number of table points.

    Raises:
        KeyError: If the element and charge are not in the table.
    """
    key = (element, charge)
    if key not in METAL_OUTER_SUBSHELLS:
        raise KeyError(
            f"No outer subshell for {element} with charge {charge}; "
            f"known: {sorted(METAL_OUTER_SUBSHELLS)}"
        )
    subshell, electrons, energy = METAL_OUTER_SUBSHELLS[key]
    name = element + ("+" * charge)
    return lotz_table(
        [LotzSubshell(energy, electrons)],
        max_energy=max_energy,
        num=num,
        label=f"e + {name} ionization (Lotz, {subshell}^{electrons})",
    )
