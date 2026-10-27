"""Hydrogen cross sections from the analytic fits of Janev, Reiter and Samm (2003).

R. K. Janev, D. Reiter and U. Samm, "Collision Processes in Low-Temperature
Hydrogen Plasmas", Forschungszentrum Jülich report Jül-4105 (2003), openly
available at https://eirene.de/Documentation/report_4105.pdf. The report gives
its cross sections as analytic fits to critically assessed data; this module
implements the fits for the electron collisions with H2 and the collisions of
H- that a negative-ion source needs, with the equation and table numbers of the
report:

====================================  =========================================  ===================
key                                   reaction                                   report
====================================  =========================================  ===================
``e_h2_vibrational_1``, ``_2``        e + H2(v=0) -> e + H2(v=1, 2)              eq. (74)
``e_h2_singlet_B`` ... ``_Dprime``    e + H2(X, v=0) -> e + H2(B, B', B'', C,    eq. (80), table 15
                                      D, D')
``e_h2_singlet_EF`` ... ``_I``        e + H2(X, v=0) -> e + H2(EF, HH, GK, I)    eq. (81), table 16
``e_h2_triplet_a`` ... ``_d``         e + H2(X, v=0) -> e + H2(a, b, c, e, d)    eq. (95), table 22
``e_h2_ionization``                   e + H2 -> 2e + H2+                         eq. (106)
``e_h2_dissociative_ionization``      e + H2 -> 2e + H+ + H                      eqs. (107a), (107b)
``e_h2_dissociative_attachment_v``    e + H2(v) -> H- + H, v = 0..14             eq. (124), table 27
``e_hminus_detachment``               e + H- -> 2e + H                           eq. (58)
``mutual_neutralization``             H+ + H- -> H(n=2, 3) + H                   eq. (63), table 10
``hminus_h_associative_detachment``   H- + H -> H2 + e                           eq. (71), table 11
``hminus_h_detachment``               H- + H -> 2H + e                           eq. (71), table 11
====================================  =========================================  ===================

Energies are in eV and cross sections in m^2 (the report's are in cm^2).
Electron collisions with H2 are fitted against the electron's energy with the
molecule at rest (``energy_frame="lab"``), the collisions of H- against the
centre-of-mass energy (``energy_frame="center_of_mass"``; for e + H- the two
agree to O(m_e/M)). The report gives the H- polynomial fits (eqs. (63), (71))
against E_CM and as valid from 0.1 eV to 20 keV; the same coefficients are in
HYDHEL (from Janev, Langer, Evans and Post 1987) with the mass of 1 u on their
energy scale, i.e. against E = m_p g^2 / 2 = 2 E_CM. This module follows the
2003 report, so their energy is E_CM.

Each fit's :meth:`HydrogenFit.cross_section` tabulates it for the collision
operator: beyond the fit's range of validity the end values hold. Elastic
collisions are outside the report's scope (its section 1), so the elastic
momentum-transfer cross sections of e + H2 and e + H are not here.

The electron excitation of H2 starts from H2(X, v=0), as the report's fits do.
The dissociative attachment depends strongly on the vibrational level:
:func:`dissociative_attachment` weights the levels with a
:class:`~plasmacoll.vibrational.VibrationalDistribution`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from plasmacoll._types import Array
from plasmacoll.cross_sections import CrossSection
from plasmacoll.vibrational import VibrationalDistribution

__all__ = [
    "DISSOCIATIVE_ATTACHMENT_E0",
    "DISSOCIATIVE_ATTACHMENT_LEVELS",
    "HMINUS_H_COEFFICIENTS",
    "H_MINUS_BINDING_ENERGY",
    "JANEV_2003",
    "JANEV_FITS",
    "MUTUAL_NEUTRALIZATION_COEFFICIENTS",
    "HydrogenFit",
    "dissociative_attachment",
    "dissociative_attachment_fit",
    "dissociative_attachment_levels",
]

#: The report the fits come from.
JANEV_2003 = (
    "R. K. Janev, D. Reiter and U. Samm, Collision Processes in Low-Temperature "
    "Hydrogen Plasmas, Forschungszentrum Jülich report Jül-4105 (2003)"
)
_SHORT = "Janev, Reiter and Samm (2003)"

#: 1e-16 cm^2 and 1 cm^2 in m^2.
_A2 = 1.0e-20
_CM2 = 1.0e-4

#: Binding energy of the outer electron of H- [eV] (section 3.1.1).
H_MINUS_BINDING_ENERGY = 0.754

#: Dissociative attachment to H2(X, v): (E_th,v [eV], sigma_v^(0) [1e-16 cm^2]) for
#: v = 0..14, table 27. E_th,v < 0 for v >= 10 (exothermic attachment).
DISSOCIATIVE_ATTACHMENT_LEVELS: tuple[tuple[float, float], ...] = (
    (3.72, 3.22e-5),
    (3.21, 5.18e-4),
    (2.72, 4.16e-3),
    (2.26, 2.20e-2),
    (1.83, 1.22e-1),
    (1.43, 4.53e-1),
    (1.36, 1.51),
    (0.713, 4.48),
    (0.397, 10.1),
    (0.113, 13.9),
    (-0.139, 11.8),
    (-0.354, 8.87),
    (-0.529, 7.11),
    (-0.659, 5.00),
    (-0.736, 3.35),
)
#: The decay energy E_0 [eV] of eq. (124).
DISSOCIATIVE_ATTACHMENT_E0 = 0.45


@dataclass(frozen=True)
class HydrogenFit:
    """One analytic cross-section fit of the report.

    Attributes:
        key: The name of the fit in :data:`JANEV_FITS`.
        reaction: The reaction, e.g. ``"e + H2(X, v=0) -> e + H2(B)"``.
        source: Equation and table of the report, e.g. ``"eq. (80), table 15"``.
        threshold: Energy in eV below which the cross section is zero.
        energy_frame: ``"lab"`` (incident energy, target at rest) or
            ``"center_of_mass"``, as for :class:`~plasmacoll.CollisionProcess`.
        function: The fit, cross section in m^2 against energy in eV, for
            energies above the threshold.
        valid_from: Lowest energy in eV of the fit's validity (0: the threshold).
        valid_to: Highest energy in eV of the fit's validity.
    """

    key: str
    reaction: str
    source: str
    threshold: float
    energy_frame: str
    function: Callable[[np.ndarray], np.ndarray]
    valid_from: float = 0.0
    valid_to: float = math.inf

    @property
    def label(self) -> str:
        """The reaction and its source, for plots and messages."""
        return f"{self.reaction}: {_SHORT}, {self.source}"

    def cross_section(self, max_energy: float = 1.0e4, num: int = 241) -> CrossSection:
        """Tabulate the fit up to ``max_energy`` (eV) for the collision operator.

        The table starts at the threshold (or at ``valid_from``) and is spaced
        logarithmically in the energy above the threshold, which resolves the
        rise of the cross section there. Below ``valid_from`` and above the last
        point the end values hold (see :class:`~plasmacoll.CrossSection`).
        """
        upper = min(max_energy, self.valid_to)
        if self.threshold <= 0.0 or self.valid_from > self.threshold:
            energy = np.geomspace(self.valid_from or 1.0e-3, upper, num)
            sigma = self.function(energy)
        else:
            offset = np.geomspace(1.0e-6 * self.threshold, upper - self.threshold, num)
            energy = np.concatenate([[self.threshold], self.threshold + offset])
            sigma = self(energy)
        return CrossSection(
            energy=energy, sigma=sigma, threshold=self.threshold, label=self.label
        )

    def __call__(self, energy: Array | float) -> np.ndarray:
        """Return the fit in m^2 at ``energy`` (eV), zero at and below the threshold."""
        energy = np.asarray(energy, dtype=float)
        above = energy > self.threshold
        # evaluate above the threshold only (the fits are singular or complex below),
        # with the end values beyond the range of validity, as the table has them
        lower = max(self.valid_from, self.threshold * (1.0 + 1.0e-12), 1.0e-12)
        safe = np.clip(np.where(above, energy, lower), lower, self.valid_to)
        return np.where(above, self.function(safe), 0.0)


# --------------------------------------------------------------------------- #
# The fit functions, in the report's units (cm^2), converted to m^2.
# --------------------------------------------------------------------------- #
def _vibrational(delta_e: float, xi: float) -> Callable[[np.ndarray], np.ndarray]:
    """Eq. (74): 5.78 xi / dE^4 / x^2 (1 - 1/x)^(6.11/sqrt(dE)) [1e-16 cm^2]."""

    def fit(energy: np.ndarray) -> np.ndarray:
        x = energy / delta_e
        power = 6.11 / math.sqrt(delta_e)
        return _A2 * 5.78 * xi / delta_e**4 / x**2 * (1.0 - 1.0 / x) ** power

    return fit


def _singlet_u(
    delta_e: float, alpha: float, a: Sequence[float]
) -> Callable[[np.ndarray], np.ndarray]:
    """Eq. (80): dipole-allowed X -> N 1Lambda_u, table 15."""

    def fit(energy: np.ndarray) -> np.ndarray:
        x = energy / delta_e
        bracket = a[0] + a[1] / x + a[2] / x**2 + a[3] * np.log(x)
        return _A2 * 5.984 / (delta_e * x) * (1.0 - 1.0 / x) ** alpha * bracket

    return fit


def _singlet_g(
    delta_e: float, alpha: float, a1: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Eq. (81): symmetry-forbidden X -> N 1Lambda_g, table 16."""

    def fit(energy: np.ndarray) -> np.ndarray:
        x = energy / delta_e
        return _A2 * 5.984 * a1 / (delta_e * x) * (1.0 - 1.0 / x) ** alpha

    return fit


def _triplet(
    delta_e: float, a: float, beta: float, gamma: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Eq. (95): X -> N 3Lambda, A / x^3 (1 - 1/x^beta)^gamma [1e-16 cm^2], table 22."""

    def fit(energy: np.ndarray) -> np.ndarray:
        x = energy / delta_e
        return _A2 * a / x**3 * (1.0 - x**-beta) ** gamma

    return fit


def _ionization(energy: np.ndarray) -> np.ndarray:
    """Eq. (106): e + H2 -> 2e + H2+, with C_0 = 2.05 dE and dE = 15.42 eV."""
    delta_e = 15.42
    x = energy / delta_e
    return _A2 * 1.828 / x * (1.0 - x**-0.92) ** 2.19 * np.log(2.05 * delta_e * x)


def _dissociative_ionization(energy: np.ndarray) -> np.ndarray:
    """Eqs. (107a) + (107b): via H2+(2Sigma_g, dE = 18.15 eV) and H2+(2Sigma_u, 30.6 eV)."""
    xg = energy / 18.15
    sigma_g = np.where(
        xg > 1.0,
        0.02905 / xg**1.25 * (1.0 - np.maximum(xg, 1.0) ** -2.78) ** 1.886,
        0.0,
    )
    xu = energy / 30.6
    sigma_u = np.where(
        xu > 1.0,
        0.5927 / xu**1.20 * (1.0 - np.maximum(xu, 1.0) ** -1.22) ** 3.375,
        0.0,
    )
    return _A2 * (sigma_g + sigma_u)


def _hminus_detachment(energy: np.ndarray) -> np.ndarray:
    """Eq. (58): e + H- -> 2e + H [1e-13 cm^2]."""
    return (
        1.0e-13
        * _CM2
        * 2.06
        / energy
        * np.log(math.e + 2.335e-4 * energy)
        * np.exp(-13.75 / energy**0.868)
        * (1.0 - (H_MINUS_BINDING_ENERGY / energy) ** 2)
    )


def _log_polynomial(*coefficients: float) -> Callable[[np.ndarray], np.ndarray]:
    """Eqs. (63) and (71): ln sigma[cm^2] = sum_j a_j (ln E)^j."""

    def fit(energy: np.ndarray) -> np.ndarray:
        log_e = np.log(energy)
        return _CM2 * np.exp(sum(a * log_e**j for j, a in enumerate(coefficients)))

    return fit


#: Table 10: the coefficients a_0..a_8 of eq. (63) for the mutual neutralization
#: H+ + H- -> H(n) + H into n = 2 and n = 3.
MUTUAL_NEUTRALIZATION_COEFFICIENTS: dict[int, tuple[float, ...]] = {
    2: (
        -3.49880888e1,
        2.15245051e-1,
        -2.35628664e-2,
        5.49471553e-2,
        5.37932888e-3,
        -6.05507021e-3,
        9.99168329e-4,
        -6.63625564e-5,
        1.61228385e-6,
    ),
    3: (
        -3.11479336e1,
        -7.73020527e-1,
        5.49204378e-2,
        -2.73324984e-3,
        -1.22831288e-3,
        4.35049828e-4,
        -6.21659501e-5,
        4.12046807e-6,
        -1.039784996e-7,
    ),
}
#: Table 11: the coefficients a_0..a_8 of eq. (71) for H- + H -> H + H + e
#: (``"detachment"``) and H- + H -> H2 + e (``"associative_detachment"``).
HMINUS_H_COEFFICIENTS: dict[str, tuple[float, ...]] = {
    "detachment": (
        -3.61799082e1,
        1.16615172,
        -1.41928602e-1,
        -1.11195959e-2,
        -1.72505995e-3,
        1.59040356e-3,
        -2.53196144e-4,
        1.66978235e-5,
        -4.09725797e-7,
    ),
    "associative_detachment": (
        -3.44152907e1,
        -3.39348209e-1,
        5.66591705e-2,
        -9.05150459e-3,
        7.66060418e-4,
        -4.27126462e-5,
        -1.57273749e-7,
        2.57607677e-7,
        -1.20071919e-8,
    ),
}


def _mutual_neutralization(energy: np.ndarray) -> np.ndarray:
    """Eq. (63): the sum of the n = 2 and n = 3 channels."""
    n2, n3 = (_log_polynomial(*MUTUAL_NEUTRALIZATION_COEFFICIENTS[n]) for n in (2, 3))
    return n2(energy) + n3(energy)


#: Table 15: (dE [eV], alpha, A1..A4) of the dipole-allowed singlet excitations.
_SINGLET_U = {
    "B": ("B 1Sigma_u", 12.754, 0.550, (3.651e-2, -0.8405, 1.2365, 2.5236)),
    "Bprime": ("B' 1Sigma_u", 14.85, 0.550, (6.827e-3, -0.1572, 0.23122, 0.47191)),
    "Bdoubleprime": (
        "B'' 1Sigma_u",
        15.47,
        0.550,
        (2.446e-3, -5.631e-2, 6.2846e-2, 0.16908),
    ),
    "C": ("C 1Pi_u", 13.29, 0.552, (3.653e-2, -0.8398, 1.2368, 2.8740)),
    "D": ("D 1Pi_u", 14.996, 0.552, (8.913e-3, -0.2049, 0.30178, 0.70126)),
    "Dprime": ("D' 1Pi_u", 15.555, 0.552, (3.872e-3, -8.902e-2, 0.13110, 0.30464)),
}
#: Table 16: (dE [eV], alpha, A1) of the symmetry-forbidden singlet excitations
#: (alpha and A1 of GK are the report's estimates).
_SINGLET_G = {
    "EF": ("EF 1Sigma_g", 13.13, 2.71, 0.8322),
    "HH": ("HHbar 1Sigma_g", 14.98, 2.71, 2.913e-2),
    "GK": ("GK 1Sigma_g", 14.816, 2.75, 1.43e-2),
    "I": ("I 1Pi_g", 14.824, 2.80, 5.409e-2),
}
#: Table 22: (dE [eV], A, beta, gamma) of the triplet excitations.
_TRIPLET = {
    "a": ("a 3Sigma_g", 11.72, 0.544, 4.5, 1.55),
    "b": ("b 3Sigma_u", 7.93, 11.16, 2.33, 3.78),
    "c": ("c 3Pi_u", 11.72, 1.43, 5.5, 1.65),
    "e": ("e 3Sigma_u", 13.0, 0.190, 4.5, 1.60),
    "d": ("d 3Pi_u", 13.6, 0.375, 5.5, 1.75),
}


def _fits() -> dict[str, HydrogenFit]:
    fits = [
        HydrogenFit(
            "e_h2_vibrational_1", "e + H2(v=0) -> e + H2(v=1)", "eq. (74)",
            0.516, "lab", _vibrational(0.516, 1.0),
        ),
        HydrogenFit(
            "e_h2_vibrational_2", "e + H2(v=0) -> e + H2(v=2)", "eq. (74)",
            1.003, "lab", _vibrational(1.003, 0.628),
        ),
    ]  # fmt: skip
    for key, (state, delta_e, alpha, a) in _SINGLET_U.items():
        fits.append(
            HydrogenFit(
                f"e_h2_singlet_{key}", f"e + H2(X, v=0) -> e + H2({state})",
                "eq. (80), table 15", delta_e, "lab", _singlet_u(delta_e, alpha, a),
            )
        )  # fmt: skip
    for key, (state, delta_e, alpha, a1) in _SINGLET_G.items():
        fits.append(
            HydrogenFit(
                f"e_h2_singlet_{key}", f"e + H2(X, v=0) -> e + H2({state})",
                "eq. (81), table 16", delta_e, "lab", _singlet_g(delta_e, alpha, a1),
            )
        )  # fmt: skip
    for key, (state, delta_e, a, beta, gamma) in _TRIPLET.items():
        products = "H(1s) + H(1s)" if key == "b" else f"H2({state})"
        fits.append(
            HydrogenFit(
                f"e_h2_triplet_{key}", f"e + H2(X, v=0) -> e + {products}",
                "eq. (95), table 22", delta_e, "lab", _triplet(delta_e, a, beta, gamma),
            )
        )  # fmt: skip
    fits += [
        HydrogenFit(
            "e_h2_ionization", "e + H2 -> 2e + H2+", "eq. (106)",
            15.42, "lab", _ionization,
        ),
        HydrogenFit(
            "e_h2_dissociative_ionization", "e + H2 -> 2e + H+ + H",
            "eqs. (107a) and (107b)", 18.15, "lab", _dissociative_ionization,
        ),
        HydrogenFit(
            "e_hminus_detachment", "e + H- -> 2e + H", "eq. (58)",
            H_MINUS_BINDING_ENERGY, "center_of_mass", _hminus_detachment,
        ),
        HydrogenFit(
            "mutual_neutralization", "H+ + H- -> H(n=2, 3) + H",
            "eq. (63), table 10", 0.0, "center_of_mass", _mutual_neutralization,
            valid_from=0.1, valid_to=2.0e4,
        ),
        HydrogenFit(
            "hminus_h_associative_detachment", "H- + H -> H2 + e",
            "eq. (71), table 11", 0.0, "center_of_mass",
            _log_polynomial(*HMINUS_H_COEFFICIENTS["associative_detachment"]),
            valid_from=0.1, valid_to=2.0e4,
        ),
        HydrogenFit(
            "hminus_h_detachment", "H- + H -> H + H + e", "eq. (71), table 11",
            0.0, "center_of_mass",
            _log_polynomial(*HMINUS_H_COEFFICIENTS["detachment"]),
            valid_from=0.1, valid_to=2.0e4,
        ),
    ]  # fmt: skip
    return {fit.key: fit for fit in fits}


#: Every fit of the module by key (see the module docstring); the dissociative
#: attachment levels are :func:`dissociative_attachment_fit`.
JANEV_FITS: dict[str, HydrogenFit] = _fits()


def dissociative_attachment_fit(level: int) -> HydrogenFit:
    """The dissociative attachment e + H2(X, v) -> H- + H of one level, eq. (124).

    sigma_v(E) = sigma_v^(0) exp(-(E - |E_th,v|) / E_0) above |E_th,v|, with
    E_0 = 0.45 eV and the thresholds and peak values of table 27. The report
    writes the fit, and its rate coefficient (eq. (125)), with |E_th,v| also for
    the exothermic levels v >= 10, whose E_th,v < 0; it is used as printed.

    Eq. (125) has the temperature dependence of the Maxwellian average of this
    fit, but its prefactor, 1.972e-8 cm^3/s per 1e-16 cm^2 and eV^(1/2), is
    2.95 times sqrt(8 e / (pi m_e)) = 0.669e-8 of that average; compute rate
    coefficients from the cross section (:func:`~plasmacoll.rate_coefficient`).

    Args:
        level: The vibrational level v, 0 to 14.
    """
    if not 0 <= level < len(DISSOCIATIVE_ATTACHMENT_LEVELS):
        raise ValueError(
            f"level must be in [0, {len(DISSOCIATIVE_ATTACHMENT_LEVELS) - 1}]"
        )
    threshold_v, peak = DISSOCIATIVE_ATTACHMENT_LEVELS[level]
    onset = abs(threshold_v)

    def fit(energy: np.ndarray) -> np.ndarray:
        return _A2 * peak * np.exp(-(energy - onset) / DISSOCIATIVE_ATTACHMENT_E0)

    return HydrogenFit(
        f"e_h2_dissociative_attachment_{level}",
        f"e + H2(X, v={level}) -> H- + H",
        "eq. (124), table 27",
        onset,
        "lab",
        fit,
    )


def dissociative_attachment_levels(max_energy: float = 30.0) -> list[CrossSection]:
    """The tabulated dissociative attachment cross sections of v = 0..14.

    The table of each level starts at the peak value at its onset |E_th,v|
    (zero below it) and ends at ``max_energy`` (eV), where the exponential has
    fallen by many orders of magnitude.
    """
    tables = []
    for level in range(len(DISSOCIATIVE_ATTACHMENT_LEVELS)):
        fit = dissociative_attachment_fit(level)
        offset = np.geomspace(1.0e-5, max_energy - fit.threshold, 161)
        energy = np.concatenate([[fit.threshold], fit.threshold + offset])
        sigma = fit.function(energy)
        tables.append(
            CrossSection(
                energy=energy, sigma=sigma, threshold=fit.threshold, label=fit.label
            )
        )
    return tables


def dissociative_attachment(
    distribution: VibrationalDistribution, label: str = ""
) -> CrossSection:
    """The dissociative attachment to H2 over a vibrational population, eq. (124).

    ``sum_v n_v sigma_v(E)`` with the populations of ``distribution`` (15 levels,
    v = 0..14, e.g. ``VibrationalDistribution.h2_ground_state_from_kelvin(5000)``),
    as one cross section per H2 molecule.

    Raises:
        ValueError: If the distribution does not have the 15 levels of H2(X).
    """
    if distribution.num_levels != len(DISSOCIATIVE_ATTACHMENT_LEVELS):
        raise ValueError(
            f"the distribution has {distribution.num_levels} levels, the fits "
            f"{len(DISSOCIATIVE_ATTACHMENT_LEVELS)} (v = 0..14)"
        )
    return CrossSection.vibrational_mixture(
        dissociative_attachment_levels(),
        distribution.populations,
        label=label
        or (
            f"e + H2(v) -> H- + H at T_vib = {distribution.temperature_ev:.3g} eV: "
            f"{_SHORT}, eq. (124), table 27"
        ),
    )
