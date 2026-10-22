"""Secondary electrons of ionization by fast ions: Rudd's singly differential cross section.

Rudd's semi-empirical model (Rudd, Phys. Rev. A 38, 6129, 1988; Rudd, Kim,
Madison and Gay, Rev. Mod. Phys. 64, 441, 1992, Eqs. (37)-(48)) gives the
cross section for ejecting an electron of energy ``W`` from a target by a fast
ion, per unit ``W``, as a sum over the target's shells ``i`` of binding energy
``I`` and occupation ``N``:

    dsigma/dW = (S / I) (F1 + F2 w) (1 + w)^-3 / (1 + exp[alpha (w - w_c) / v]),

with ``w = W / I``, ``v = sqrt(T / I)``, ``T = (m_e / M) T_0`` the energy of an
electron at the projectile's speed (``T_0`` the projectile's energy, ``M`` its
mass), ``S = 4 pi a_0^2 N (R / I)^2``, the cutoff ``w_c = 4 v^2 - 2 v - R / (4 I)``
and ``F1 = L1 + H1``, ``F2 = L2 H2 / (L2 + H2)``, where

    L1 = C1 v^D1 / (1 + E1 v^(D1 + 4)),   H1 = A1 ln(1 + v^2) / (v^2 + B1 / v^2),
    L2 = C2 v^D2,                          H2 = A2 / v^2 + B2 / v^4.

The ten parameters ``A1 ... D2, alpha`` are fitted per target (Table V of the
1992 review). The model depends on the projectile only through its speed, so a
deuteron (or a fast deuterium atom, to the extent the model holds for it) at
``T_0`` is a proton at ``T_0 m_p / m_d``. Integrated over ``W`` it gives the
total ionization cross section (:meth:`RuddIonization.total`) and the mean
energy of the secondary electrons (:meth:`RuddIonization.mean_energy`), the
source terms of a plasma that a beam ionizes.

:data:`RUDD_H2` holds the H2 parameters of the review (Table V, binding energy
from Table I(a)); it reproduces its Table VII of recommended H+ + H2 cross
sections to 1%.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cunumpy as xp
import numpy as np

from plasmacoll._types import Array
from plasmacoll.constants import ELECTRON_MASS, PROTON_MASS
from plasmacoll.cross_sections import CrossSection

__all__ = ["RUDD_H2", "RuddIonization", "RuddShell"]

#: Bohr radius in m (CODATA 2018).
BOHR_RADIUS = 5.29177210903e-11
#: Rydberg energy in eV (CODATA 2018).
RYDBERG_ENERGY = 13.605693122994
#: Integration points over ``ln(1 + w)`` per shell.
_POINTS = 4001
#: The integrals stop this many cutoff widths ``v / alpha`` above the cutoff.
_TAIL = 60.0


@dataclass(frozen=True)
class RuddShell:
    """One shell of a target in Rudd's model.

    Attributes:
        binding_energy: ``I`` in eV.
        electrons: ``N``, the electrons in the shell.
        a1: ``A1``.
        b1: ``B1``.
        c1: ``C1``.
        d1: ``D1``.
        e1: ``E1``.
        a2: ``A2``.
        b2: ``B2``.
        c2: ``C2``.
        d2: ``D2``.
        alpha: ``alpha``, the steepness of the cutoff.
    """

    binding_energy: float
    electrons: float
    a1: float
    b1: float
    c1: float
    d1: float
    e1: float
    a2: float
    b2: float
    c2: float
    d2: float
    alpha: float

    def __post_init__(self) -> None:
        """Validate the shell."""
        if self.binding_energy <= 0.0 or self.electrons <= 0.0:
            raise ValueError("a shell needs binding_energy > 0 and electrons > 0")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be > 0")

    def reduced_speed(self, electron_energy: float) -> float:
        """``v = sqrt(T / I)`` for ``T`` (eV) the energy of an electron at the projectile's speed."""
        return float(np.sqrt(electron_energy / self.binding_energy))

    def factors(self, v: float) -> tuple[float, float]:
        """``F1`` and ``F2`` at the reduced speed ``v`` (Eqs. (43)-(48))."""
        v2 = v * v
        low1 = self.c1 * v**self.d1 / (1.0 + self.e1 * v ** (self.d1 + 4.0))
        high1 = self.a1 * np.log1p(v2) / (v2 + self.b1 / v2)
        low2 = self.c2 * v**self.d2
        high2 = self.a2 / v2 + self.b2 / v2**2
        return float(low1 + high1), float(low2 * high2 / (low2 + high2))

    def cutoff(self, v: float) -> float:
        """``w_c = 4 v^2 - 2 v - R / (4 I)`` (Eq. (42))."""
        return 4.0 * v * v - 2.0 * v - RYDBERG_ENERGY / (4.0 * self.binding_energy)

    def differential(self, secondary_energy: np.ndarray, electron_energy: float):
        """``dsigma/dW`` in m^2/eV of this shell (Eq. (41))."""
        v = self.reduced_speed(electron_energy)
        f1, f2 = self.factors(v)
        w = np.asarray(secondary_energy, dtype=float) / self.binding_energy
        s = (
            4.0
            * np.pi
            * BOHR_RADIUS**2
            * self.electrons
            * (RYDBERG_ENERGY / self.binding_energy) ** 2
        )
        # 1 / (1 + exp(x)) without overflow
        x = self.alpha * (w - self.cutoff(v)) / v
        cutoff = 0.5 * (1.0 - np.tanh(0.5 * x))
        return s / self.binding_energy * (f1 + f2 * w) / (1.0 + w) ** 3 * cutoff

    def moments(self, electron_energy: float) -> tuple[float, float]:
        """``int dsigma/dW dW`` (m^2) and ``int W dsigma/dW dW`` (m^2 eV) of this shell.

        Integrated by Simpson's rule in ``s = ln(1 + w)``, where the
        ``(1 + w)^-3`` fall-off is smooth, up to ``_TAIL`` widths of the cutoff
        above it.
        """
        v = self.reduced_speed(electron_energy)
        top = max(self.cutoff(v), 0.0) + _TAIL * v / self.alpha
        s = np.linspace(0.0, np.log1p(top), _POINTS)
        w = np.expm1(s)
        energy = w * self.binding_energy
        # dW = I (1 + w) ds
        integrand = self.differential(energy, electron_energy) * self.binding_energy
        integrand = integrand * (1.0 + w)
        weights = np.ones(_POINTS)
        weights[1:-1:2], weights[2:-1:2] = 4.0, 2.0
        weights *= (s[1] - s[0]) / 3.0
        return float(weights @ integrand), float(weights @ (integrand * energy))


@dataclass(frozen=True)
class RuddIonization:
    """Ionization of a target by a fast ion, after Rudd; see the module docstring.

    Attributes:
        shells: The target's shells.
        projectile_mass: ``M`` in kg; the proton mass by default.
        label: A name for plots and messages.
    """

    shells: Sequence[RuddShell]
    projectile_mass: float = PROTON_MASS
    label: str = ""

    def __post_init__(self) -> None:
        """Validate the target and the projectile."""
        object.__setattr__(self, "shells", tuple(self.shells))
        if not self.shells:
            raise ValueError("a target needs at least one shell")
        if self.projectile_mass <= 0.0:
            raise ValueError("projectile_mass must be > 0")

    def with_projectile(self, mass: float) -> RuddIonization:
        """The same target ionized by a projectile of ``mass`` (kg)."""
        return RuddIonization(self.shells, mass, self.label)

    def electron_energy(self, projectile_energy: float) -> float:
        """``T = (m_e / M) T_0`` in eV, the energy of an electron at the projectile's speed."""
        if projectile_energy <= 0.0:
            raise ValueError("projectile_energy must be > 0")
        return ELECTRON_MASS / self.projectile_mass * float(projectile_energy)

    def differential(
        self, secondary_energy: float | Array, projectile_energy: float
    ) -> np.ndarray:
        """``dsigma/dW`` in m^2/eV at the secondary energies ``W`` (eV).

        Args:
            secondary_energy: ``W`` in eV, a number or an array.
            projectile_energy: ``T_0``, the projectile's kinetic energy in eV.
        """
        energy = np.asarray(xp.to_numpy(xp.asarray(secondary_energy)), dtype=float)
        if np.any(energy < 0.0):
            raise ValueError("secondary_energy must be >= 0")
        t = self.electron_energy(projectile_energy)
        return sum(
            (shell.differential(energy, t) for shell in self.shells),
            np.zeros(energy.shape),
        )

    def total(self, projectile_energy: float) -> float:
        """The ionization cross section ``int dsigma/dW dW`` in m^2."""
        t = self.electron_energy(projectile_energy)
        return sum(shell.moments(t)[0] for shell in self.shells)

    def mean_energy(self, projectile_energy: float) -> float:
        """The mean energy of the secondary electrons in eV."""
        t = self.electron_energy(projectile_energy)
        moments = [shell.moments(t) for shell in self.shells]
        return sum(m[1] for m in moments) / sum(m[0] for m in moments)

    def cross_section(self, projectile_energy: Array | Sequence[float]) -> CrossSection:
        """The total cross section tabulated at the projectile energies (eV, lab frame)."""
        grid = np.asarray(xp.to_numpy(xp.asarray(projectile_energy)), dtype=float)
        return CrossSection.from_function(
            lambda e: np.array([self.total(float(x)) for x in e]),
            grid,
            label=self.label or "Rudd ionization",
        )


#: H2 (Rudd et al. 1992, Table V; ``I`` and ``N`` from Table I(a)), by protons.
RUDD_H2 = RuddIonization(
    [
        RuddShell(
            binding_energy=15.43,
            electrons=2,
            a1=0.96,
            b1=2.6,
            c1=0.38,
            d1=0.23,
            e1=2.2,
            a2=1.04,
            b2=5.9,
            c2=1.15,
            d2=0.20,
            alpha=0.87,
        )
    ],
    label="Rudd H2",
)
