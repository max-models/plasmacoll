"""Caesium in a hydrogen plasma: e + Cs ionization and the Cs and Cs+ collisions with H2.

Caesium evaporated into a negative-ion source drifts through the hydrogen
plasma, where electrons ionize it, ``e + Cs -> Cs+ + 2e``, and it scatters on
the H2 gas. The ionizing electrons are a background with a density ``n_e(x)``
and a temperature ``T_e(x)`` (`electron_background`), not markers: the
ionization is a ``stripping`` process of the Cs marker on that background
(`caesium_ionization_process`), which turns the marker into a Cs+ of the
same weight and does not track the new electron. The null-collision method
samples the electron velocities from the Maxwellian of the local
temperature, so the rate is ``n_e <sigma v>(T_e)`` at every marker.

The data, and how good they are:

=================  ======================================================  ===========
process            cross section                                           status
=================  ======================================================  ===========
e + Cs -> Cs+      Lotz, Z. Phys. 216, 241 (1968), 6s shell at the NIST    fit
                   ionization energy 3.893906 eV
Cs + H2 elastic    hard spheres, Cs van der Waals radius 3.43 A (Mantina    placeholder
                   et al., J. Phys. Chem. A 113, 5806 (2009)), H2 kinetic
                   diameter 2.89 A (Breck, Zeolite Molecular Sieves, 1974)
Cs+ + H2 elastic   Langevin capture, H2 polarizability 0.8045 A^3 (CRC      model
                   Handbook)
=================  ======================================================  ===========

The Lotz cross section of the 6s electron peaks at about 1.1e-19 m^2 near
10.6 eV. It leaves out the 5p shell and its autoionization, which add to the
cross section above about 12 eV, so it is a lower estimate there; a measured
set (none in LXCat today) should replace it.
The Cs + H2 cross section is the geometric one; CsFlow3D (Gutser et al., Plasma
Phys. Control. Fusion 53, 105014 (2011)) and Mimo et al. (2017) cite their own
data, which are not at hand. Charge transfer Cs+ + H2 -> Cs + H2+ is
endothermic by about 11.5 eV and left out, as is the ionization of Cs+ (23.2 eV).
"""

from __future__ import annotations

import math
from typing import Any

from methodcard import Rule, method

from plasmacoll.background import NeutralBackground, Profile
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE
from plasmacoll.cross_sections import CrossSection
from plasmacoll.mcc import MonteCarloCollisions
from plasmacoll.metals import LotzSubshell, lotz_table
from plasmacoll.process import CollisionProcess

__all__ = [
    "CAESIUM_IONIZATION_ENERGY",
    "CAESIUM_MASS",
    "CS_H2_HARD_SPHERE_CROSS_SECTION",
    "H2_MASS",
    "H2_POLARIZABILITY",
    "caesium_collisions",
    "caesium_h2_elastic_process",
    "caesium_ion_h2_elastic_process",
    "caesium_ionization_cross_section",
    "caesium_ionization_process",
    "electron_background",
    "langevin_rate_coefficient",
]

#: Mass of 133Cs in kg (standard atomic weight 132.90545196 u).
CAESIUM_MASS = 132.90545196 * ATOMIC_MASS
#: Mass of H2 in kg.
H2_MASS = 2.01588 * ATOMIC_MASS
#: First ionization energy of Cs in eV (NIST Atomic Spectra Database).
CAESIUM_IONIZATION_ENERGY = 3.893906
#: Polarizability volume of H2 in m^3 (0.8045 A^3, CRC Handbook).
H2_POLARIZABILITY = 0.8045e-30
#: Hard-sphere cross section ``pi d^2`` of Cs on H2 in m^2, with ``d`` the mean
#: of the Cs van der Waals diameter (6.86 A) and the H2 kinetic diameter (2.89 A).
CS_H2_HARD_SPHERE_CROSS_SECTION = math.pi * (0.5 * (6.86e-10 + 2.89e-10)) ** 2
_EPSILON_0 = 8.8541878128e-12


@method(
    equations={
        "lotz": r"\sigma(E) = a\,\frac{\ln(E/P)}{E P}, \qquad P = 3.893906\ \mathrm{eV}, \quad a = 4.5\cdot10^{-14}\ \mathrm{cm^2\,eV^2}",
        "peak": r"E_\mathrm{peak} = e_\mathrm{Euler}\,P \approx 10.6\ \mathrm{eV}, \qquad \sigma_{\max} \approx 1.1\cdot10^{-19}\ \mathrm{m}^2",
    },
    assumptions=(
        r"the 6s electron only: the 5p shell and its autoionization are left out, a lower estimate above about 12 eV",
    ),
    references=(
        "[Lotz, Z. Phys. 216, 241 (1968)](https://doi.org/10.1007/BF01392963)",
        "NIST Atomic Spectra Database (ionization energy)",
    ),
    components=("plasmacoll.metals.lotz_cross_section",),
)
def caesium_ionization_cross_section(
    max_energy: float = 1.0e4, num: int = 400
) -> CrossSection:
    """Return the Lotz cross section of ``e + Cs -> Cs+ + 2e`` (6s shell) in m^2.

    Args:
        max_energy: Last table energy in eV.
        num: Number of table points.
    """
    return lotz_table(
        [LotzSubshell(CAESIUM_IONIZATION_ENERGY, 1)],
        max_energy=max_energy,
        num=num,
        label="e + Cs ionization (Lotz, 6s)",
    )


def electron_background(
    density: float,
    temperature: float,
    density_profile: Profile | None = None,
    temperature_profile: Profile | None = None,
    name: str = "e",
) -> NeutralBackground:
    """Return the ionizing electrons as a background, ``n_e(x)`` and ``T_e(x)``.

    Args:
        density: Reference electron density in m^-3.
        temperature: Reference electron temperature in eV.
        density_profile: Factor of ``density`` against position.
        temperature_profile: Factor of ``temperature`` against position.
        name: The background's name.
    """
    return NeutralBackground(
        name,
        density,
        temperature,
        ELECTRON_MASS,
        profile=density_profile,
        temperature_profile=temperature_profile,
    )


def caesium_ionization_process(
    background: str = "e", ion: str = "Cs+", max_energy: float = 1.0e4
) -> CollisionProcess:
    """Return the ionization of a Cs marker by the electron background.

    The marker becomes ``ion`` with its weight and velocity; the electron is
    not tracked.

    Args:
        background: Name of the electron background.
        ion: The product species.
        max_energy: Last table energy of the cross section in eV.
    """
    return CollisionProcess(
        "stripping",
        background,
        caesium_ionization_cross_section(max_energy),
        name="ionization:Cs",
        products={"projectile": ion},
        charge_change=1,
    )


@method(
    equations={
        "hard_spheres": r"\sigma = \pi d^2, \qquad d = \frac{d_\mathrm{Cs} + d_\mathrm{H_2}}{2} = \frac{6.86 + 2.89}{2}\ \text{\AA}",
    },
    assumptions=(
        r"a placeholder: the geometric cross section, isotropic; $d_\mathrm{Cs}$ twice the van der Waals radius 3.43 Å, $d_\mathrm{H_2}$ the kinetic diameter",
    ),
    references=(
        "[Mantina et al., J. Phys. Chem. A 113, 5806 (2009)](https://doi.org/10.1021/jp8111556)",
        "Breck, Zeolite Molecular Sieves (1974)",
    ),
)
def caesium_h2_elastic_process(background: str = "H2") -> CollisionProcess:
    """Return isotropic hard-sphere scattering of Cs on H2 (a placeholder).

    Args:
        background: Name of the H2 background.
    """
    return CollisionProcess(
        "elastic",
        background,
        CrossSection.constant(
            CS_H2_HARD_SPHERE_CROSS_SECTION, label="Cs + H2 hard spheres"
        ),
        name="elastic:Cs-H2",
    )


def _langevin_rules() -> dict:
    """The SymPy rule of `langevin_rate_coefficient` (for the tests)."""
    import sympy

    alpha, mu = sympy.symbols("polarizability reduced_mass", positive=True)
    e, eps0 = sympy.Float(ELEMENTARY_CHARGE), sympy.Float(_EPSILON_0)
    rule = 2 * sympy.pi * sympy.sqrt(alpha * e**2 / (4 * sympy.pi * eps0 * mu))
    ranges = {"polarizability": (1.0e-31, 1.0e-29), "reduced_mass": (1.0e-27, 1.0e-25)}
    return {"langevin": Rule(rule, ranges=ranges)}


@method(
    equations={
        "langevin": r"K_L = \sigma_L g = 2\pi\sqrt{\frac{\alpha e^2}{4\pi\varepsilon_0\mu}}",
        "cross_section": r"\sigma_L(g) = \frac{2\pi}{g}\sqrt{\frac{\alpha e^2}{4\pi\varepsilon_0\mu}}",
    },
    assumptions=(
        r"$\alpha$ the polarizability volume of the neutral (m$^3$), $\mu$ the reduced mass: capture in the polarization potential, independent of the speed",
    ),
    symbolic=lambda: _langevin_rules(),
)
def langevin_rate_coefficient(polarizability: float, reduced_mass: float) -> float:
    """Return the Langevin capture rate ``2 pi sqrt(alpha e^2 / (4 pi eps0 mu))`` (m^3/s).

    Args:
        polarizability: Polarizability volume of the neutral in m^3.
        reduced_mass: Reduced mass of ion and neutral in kg.
    """
    if polarizability <= 0.0 or reduced_mass <= 0.0:
        raise ValueError("polarizability and reduced_mass must be > 0")
    return (
        2.0
        * math.pi
        * math.sqrt(
            polarizability
            * ELEMENTARY_CHARGE**2
            / (4.0 * math.pi * _EPSILON_0 * reduced_mass)
        )
    )


@method(
    equations={
        "cross_section": r"\sigma(E) = \frac{K_L}{g}, \qquad K_L = 2\pi\sqrt{\frac{\alpha_\mathrm{H_2} e^2}{4\pi\varepsilon_0\mu}} \approx 1.49\cdot10^{-15}\ \mathrm{m^3/s}",
    },
    assumptions=(
        r"Langevin capture with isotropic scattering, $\alpha_\mathrm{H_2} = 0.8045$ Å$^3$ (CRC Handbook), entered as a Maxwell-molecule cross section",
        r"Cs$^+$ + H$_2$ charge transfer (endothermic by about 11.5 eV) and the ionization of Cs$^+$ (23.2 eV) left out",
    ),
    components=(
        "plasmacoll.caesium.langevin_rate_coefficient",
        "plasmacoll.cross_sections.CrossSection.maxwell_molecule",
    ),
)
def caesium_ion_h2_elastic_process(background: str = "H2") -> CollisionProcess:
    """Return Langevin capture of Cs+ on H2 with isotropic scattering.

    Args:
        background: Name of the H2 background.
    """
    reduced_mass = CAESIUM_MASS * H2_MASS / (CAESIUM_MASS + H2_MASS)
    return CollisionProcess(
        "elastic",
        background,
        CrossSection.maxwell_molecule(
            langevin_rate_coefficient(H2_POLARIZABILITY, reduced_mass),
            reduced_mass,
            label="Cs+ + H2 Langevin",
        ),
        name="elastic:Cs+-H2",
    )


@method(
    equations={
        "ionization": r"\nu_\mathrm{iz}(\mathbf x) = n_e(\mathbf x)\,\langle\sigma v\rangle\big(T_e(\mathbf x)\big)",
    },
    assumptions=(
        r"the electrons a Maxwellian background with profiles $n_e(\mathbf x)$, $T_e(\mathbf x)$, not markers; the ionization a `stripping` of the Cs marker into Cs$^+$ of the same weight, the new electron not tracked",
        r"Cs scatters on H$_2$ as hard spheres, Cs$^+$ by Langevin capture",
    ),
    components=(
        "plasmacoll.caesium.caesium_ionization_cross_section",
        "plasmacoll.caesium.caesium_h2_elastic_process",
        "plasmacoll.caesium.caesium_ion_h2_elastic_process",
    ),
)
def caesium_collisions(
    electrons: NeutralBackground,
    gas: NeutralBackground,
    seed: int | None = None,
    **kwargs: Any,
) -> MonteCarloCollisions:
    """Return the collisions of ``Cs`` and ``Cs+`` markers in a hydrogen plasma.

    ``Cs`` is ionized by ``electrons`` (see `electron_background`) and
    scatters on ``gas``; ``Cs+`` scatters on ``gas``.

    Args:
        electrons: The electron background.
        gas: The H2 background.
        seed: Seed of the random stream.
        **kwargs: Passed on to `plasmacoll.mcc.MonteCarloCollisions`.
    """
    return MonteCarloCollisions(
        {"Cs": CAESIUM_MASS, "Cs+": CAESIUM_MASS - ELECTRON_MASS},
        [electrons, gas],
        {
            "Cs": (
                caesium_ionization_process(electrons.name),
                caesium_h2_elastic_process(gas.name),
            ),
            "Cs+": (caesium_ion_h2_elastic_process(gas.name),),
        },
        seed=seed,
        **kwargs,
    )
