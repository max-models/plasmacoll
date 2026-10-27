"""Elastic electron scattering on H and H2 from the CCC fits of Park et al. (2022).

R. M. Park, W. Kupets, M. C. Zammit, J. Colgan, C. J. Fontes, B. S. Scheiner,
E. Timmermans, X.-Z. Tang, L. H. Scarlett, D. V. Fursa, I. Bray and N. A. Garland,
"Anisotropic angular scattering models of elastic electron-neutral collisions
for Monte Carlo plasma simulations", Plasma Sources Sci. Technol. 31, 065013
(2022), doi:10.1088/1361-6595/ac781f; accepted manuscript openly available as
LANL report LA-UR-21-32313 (https://www.osti.gov/servlets/purl/2318946).

The paper fits the convergent close-coupling (CCC) calculations of the Curtin
and Los Alamos groups for elastic electron scattering on H (Bray and Stelbovics,
Phys. Rev. A 46, 6995 (1992)) and on H2 in its ground state (Zammit et al.) with
closed forms, with the coefficients of its tables I and II:

- the integral elastic cross section ``sigma(E)``, eq. (26);
- the angular distribution, a forward and a backward screened-Coulomb peak,
  eq. (17), with the weight ``C(E)`` of eq. (19) and the screening parameters
  ``eta_F(E)``, ``eta_B(E)`` of eq. (20);
- from these, the momentum-transfer cross section
  ``sigma_m = 2 pi sigma [C Phi_F + (1 - C) Phi_B]``, eqs. (28)-(30).

========  ===================  ===========================================
key       reaction             CCC data the fits reproduce
========  ===================  ===========================================
``H``     e + H(1s) -> e + H   0.001 to 500 eV (sigma within 6 %)
``H2``    e + H2 -> e + H2     0.1 to 250 eV (sigma within 6 %)
========  ===================  ===========================================

The momentum-transfer cross section is the one a collision operator with
isotropic elastic scattering, or a two-term Boltzmann solver, needs
(:meth:`ElasticFit.cross_section`); the integral one goes with an anisotropic
angular distribution. Energies are the electron's energy with the target at
rest (``energy_frame="lab"``), in eV; cross sections are in m^2. The fits also
hold for D, T and the isotopologues of H2.

How good they are (checked in the tests):

- e + H2: the momentum-transfer cross section agrees with the swarm-derived one
  of England, Elford and Crompton, Aust. J. Phys. 41, 573 (1988), table 5,
  within 10 % from 0.02 to 5 eV, the range in which that one is reliable
  (Brunger et al. 1991), and with the beam measurements of Brunger, Buckman,
  Newman and Alle, J. Phys. B 24, 1435 (1991), table 3, within their 20 % from
  1 to 5 eV (the fit lies 3 to 15 % above them). Below 0.02 eV it exceeds the
  swarm value, by 23 % at zero energy, outside the CCC range.
- e + H: the integral cross section reproduces the CCC database
  (https://atom.curtin.edu.au/CCC-WWW/) within 3 % from 0 to 1000 eV, and the
  momentum-transfer one at zero energy is 5 % above
  ``pi (a_s^2 + 3 a_t^2)`` with the scattering lengths of Schwartz, Phys. Rev.
  124, 1468 (1961).
- Above a few hundred eV the angular fit has too much large-angle scattering
  and the momentum-transfer cross section falls too slowly: the paper finds it
  too large for He by up to a factor 2 above 400 eV, and for H it is 2.2 times
  the first Born approximation at 1 keV (where the Born integral cross section
  agrees with CCC to 1 %); for H2 it stays within 15 % of twice that H value.
  At these energies inelastic collisions dominate the momentum loss of
  electrons.

Beyond the last table point (10 keV by default) the last value holds, as for
any :class:`~plasmacoll.CrossSection`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from plasmacoll._types import Array
from plasmacoll.cross_sections import CrossSection

__all__ = [
    "ELASTIC_FITS",
    "PARK_2022",
    "ElasticFit",
]

#: The paper the fits come from.
PARK_2022 = (
    "R. M. Park, W. Kupets, M. C. Zammit, J. Colgan, C. J. Fontes, B. S. Scheiner, "
    "E. Timmermans, X.-Z. Tang, L. H. Scarlett, D. V. Fursa, I. Bray and "
    "N. A. Garland, Anisotropic angular scattering models of elastic "
    "electron-neutral collisions for Monte Carlo plasma simulations, Plasma "
    "Sources Sci. Technol. 31, 065013 (2022)"
)
_SHORT = "Park et al. (2022), CCC fit"

#: The kinds of cross section :meth:`ElasticFit.cross_section` tabulates.
_KINDS = ("momentum_transfer", "integral")


@dataclass(frozen=True)
class ElasticFit:
    """The elastic scattering fits of one target, tables I and II of the paper.

    Attributes:
        target: ``"H"`` or ``"H2"``.
        reaction: The reaction, e.g. ``"e + H2 -> e + H2"``.
        alpha: ``alpha_1..alpha_4`` of the weight ``C(E)``, eq. (19).
        eta_forward: ``lambda_1..lambda_3`` of ``eta_F(E)``, eq. (20).
        eta_backward: ``lambda_1..lambda_3`` of ``eta_B(E)``, eq. (20).
        beta: ``beta_0..beta_7`` of the integral cross section, eq. (26), in m^2.
        valid_from: Lowest energy in eV of the CCC data the fits reproduce.
        valid_to: Highest energy in eV of the CCC data the fits reproduce.
    """

    target: str
    reaction: str
    alpha: tuple[float, float, float, float]
    eta_forward: tuple[float, float, float]
    eta_backward: tuple[float, float, float]
    beta: tuple[float, float, float, float, float, float, float, float]
    valid_from: float
    valid_to: float

    @property
    def label(self) -> str:
        """The reaction and its source, for plots and messages."""
        return f"{self.reaction}: {_SHORT}"

    def integral(self, energy: Array | float) -> np.ndarray:
        """The integral elastic cross section in m^2 at ``energy`` (eV), eq. (26)."""
        e = np.asarray(energy, dtype=float)
        b0, b1, b2, b3, b4, b5, b6, b7 = self.beta
        prefactor = b0 + b1 / (e + 1.0) + b2 / (e**2 + 1.0)
        rise = (e**b3 + np.exp(-b4 * e)) ** b5
        denominator = (e + 1.0) ** 3 + b6 * (e + 1.0) ** 2 + b7 * (e + 1.0)
        return prefactor * rise / denominator

    def forward_weight(self, energy: Array | float) -> np.ndarray:
        """The weight ``C(E)`` of the forward peak, eq. (19)."""
        a1, a2, a3, a4 = self.alpha
        return a1 + a2 * np.tanh(a3 * (np.asarray(energy, dtype=float) - a4))

    def screening(self, energy: Array | float) -> tuple[np.ndarray, np.ndarray]:
        """The screening parameters ``(eta_F(E), eta_B(E))``, eq. (20)."""
        e = np.asarray(energy, dtype=float)

        def eta(lam: tuple[float, float, float]) -> np.ndarray:
            if lam[0] == 0.0:  # constant (H2's backward peak, lambda_2 = 0 too)
                return np.full_like(e, lam[2])
            return lam[0] / (e + lam[1]) + lam[2]

        return eta(self.eta_forward), eta(self.eta_backward)

    def angular_distribution(
        self, energy: Array | float, cos_chi: Array | float
    ) -> np.ndarray:
        """The normalized angular distribution ``I(E, chi)`` in 1/sr, eq. (17).

        ``2 pi int I d(cos chi) = 1``; the differential cross section is
        ``sigma(E) I(E, chi)``.
        """
        mu = np.asarray(cos_chi, dtype=float)
        weight = self.forward_weight(energy)
        eta_f, eta_b = self.screening(energy)
        forward = eta_f * (eta_f + 1.0) / (math.pi * (2.0 * eta_f + 1.0 - mu) ** 2)
        backward = eta_b * (eta_b + 1.0) / (math.pi * (2.0 * eta_b + 1.0 + mu) ** 2)
        return weight * forward + (1.0 - weight) * backward

    def momentum_transfer(self, energy: Array | float) -> np.ndarray:
        """The momentum-transfer cross section in m^2 at ``energy`` (eV), eqs. (28)-(30).

        ``sigma_m = sigma int (1 - cos chi) I dOmega``, in closed form.
        """
        weight = self.forward_weight(energy)
        eta_f, eta_b = self.screening(energy)
        phi_f = (
            eta_f
            * (eta_f + 1.0)
            / math.pi
            * (eta_f / (eta_f + 1.0) - 1.0 + np.log((eta_f + 1.0) / eta_f))
        )
        phi_b = (
            eta_b
            * (eta_b + 1.0)
            / math.pi
            * ((eta_b + 1.0) / eta_b - 1.0 + np.log(eta_b / (eta_b + 1.0)))
        )
        return (
            2.0
            * math.pi
            * self.integral(energy)
            * (weight * phi_f + (1.0 - weight) * phi_b)
        )

    def cross_section(
        self,
        kind: str = "momentum_transfer",
        max_energy: float = 1.0e4,
        num: int = 241,
    ) -> CrossSection:
        """Tabulate a cross section for the collision operator.

        The table has the point 0 eV and ``num`` points spaced logarithmically
        from 1 meV to ``max_energy`` (eV); the last value holds beyond.

        Args:
            kind: ``"momentum_transfer"`` for an ``elastic`` process with
                isotropic scattering and for :func:`~plasmacoll.solve_eedf`, or
                ``"integral"`` for one with an anisotropic angular distribution.
            max_energy: The last table energy in eV.
            num: Number of logarithmically spaced points.

        Raises:
            ValueError: For another ``kind``.
        """
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, not {kind!r}")
        function = (
            self.momentum_transfer if kind == "momentum_transfer" else self.integral
        )
        energy = np.concatenate([[0.0], np.geomspace(1.0e-3, max_energy, num)])
        name = (
            "momentum transfer" if kind == "momentum_transfer" else "integral elastic"
        )
        return CrossSection.from_function(
            function, energy, label=f"{self.reaction} ({name}): {_SHORT}"
        )


#: The fits by target (tables I and II of the paper).
ELASTIC_FITS: dict[str, ElasticFit] = {
    "H": ElasticFit(
        target="H",
        reaction="e + H -> e + H",
        alpha=(0.0502, 0.9190, 0.0799, 0.9521),
        eta_forward=(3.29, 10.91, 0.008),
        eta_backward=(0.936, 0.0177, 1.251),
        beta=(
            4.58797e-19,
            1.54909e-17,
            5.14811e-19,
            8.92535e-01,
            5.47418e-01,
            2.15397,
            6.27850,
            3.27804e01,
        ),
        valid_from=1.0e-3,
        valid_to=500.0,
    ),
    "H2": ElasticFit(
        target="H2",
        reaction="e + H2 -> e + H2",
        alpha=(0.409, 0.5737, 0.2665, 3.912),
        eta_forward=(2.81, 4.33, 0.006),
        eta_backward=(0.0, 0.0, 0.729),
        beta=(
            5.36772e-18,
            -5.01754e-18,
            2.09077e-19,
            1.36876,
            9.59997e-01,
            1.16409,
            4.08348,
            4.23339,
        ),
        valid_from=0.1,
        valid_to=250.0,
    ),
}
