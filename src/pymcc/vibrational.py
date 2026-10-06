"""Prescribed non-equilibrium vibrational populations for H2 negative-ion sources.

Volume production of H- proceeds overwhelmingly through dissociative
attachment (DA) of low-energy electrons to *vibrationally excited* hydrogen,

    e + H2(v)  ->  H- + H,

whose cross section rises by orders of magnitude between v = 0 and the
highest bound levels (Bardsley & Wadehra, Phys. Rev. A 20, 1398 (1979); Wadehra,
Phys. Rev. A 29, 106 (1984)). A collision model that uses a single,
ground-state DA cross section for all of H2 therefore gets the volume
production rate wrong by orders of magnitude.

This module supplies the cheap half of a fix: a *prescribed* (not
self-consistently solved) non-equilibrium vibrational population, built from
the real H2 vibrational energy ladder and a single vibrational temperature.
This is standard practice for characterizing negative-ion-source-relevant
plasmas from a measured or assumed T_vib -- see Krištof et al., "Diagnostics
of low pressure hydrogen discharge created in a 13.56 MHz RF plasma reactor",
arXiv:1911.03319 (2019), eq. (5), and references therein (Kalache et al. 2004;
Kimura and Kasugai 2010).

It does **not** solve the vibrational kinetics (electron-impact excitation,
V-V/V-T relaxation, wall association) self-consistently. It also does **not**
ship level-resolved DA cross sections (Wadehra 1984; Fabrikant; Celiberto):
:meth:`pymcc.cross_sections.CrossSection.vibrational_mixture` takes them as
input from the user.
"""

from __future__ import annotations

from dataclasses import dataclass

import cunumpy as xp

from pymcc._types import Array
from pymcc.constants import kelvin_to_ev

__all__ = [
    "H2_GROUND_STATE_VIBRATIONAL_ENERGY_EV",
    "VibrationalDistribution",
    "boltzmann_level_populations",
    "estimate_electronegativity",
]

# H2(X 1-Sigma-g+) vibrational term energies E(v), v = 0..14, in eV, measured
# from the potential minimum. Fantz, U. and Wünderlich, D., "Franck-Condon
# factors, transition probabilities and radiative lifetimes for hydrogen
# molecules and their isotopomeres", At. Data Nucl. Data Tables 92, 853 (2006)
# (IAEA report INDC(NDS)-457), Table 4.1. v = 14 is the last bound level.
#
# The populations computed from this table are invariant to a constant shift
# of all E(v) (see boltzmann_level_populations), so using the absolute term
# value rather than E(v) - E(0) makes no difference to the result.
#: H2(X) vibrational term energies E(v), v = 0..14, in eV (Fantz and Wünderlich 2006).
H2_GROUND_STATE_VIBRATIONAL_ENERGY_EV: tuple[float, ...] = (
    0.27504,
    0.79104,
    1.27809,
    1.73664,
    2.16685,
    2.56873,
    2.94214,
    3.28667,
    3.60148,
    3.88512,
    4.13553,
    4.34985,
    4.52401,
    4.65343,
    4.72986,
)


def boltzmann_level_populations(
    energies_ev: Array,
    temperature_ev: float,
) -> Array:
    """Return normalized Boltzmann populations over a ladder of levels.

    ``n_i = exp(-E_i / T) / sum_j exp(-E_j / T)``, with ``E`` in eV and ``T``
    in eV. The result is unchanged by adding a constant to every energy.
    """
    energies = xp.asarray(energies_ev, dtype=float)
    if energies.ndim != 1 or energies.size == 0:
        raise ValueError("energies_ev must be a non-empty 1D array")
    if temperature_ev <= 0.0:
        raise ValueError("temperature_ev must be > 0")
    shifted = energies - xp.min(energies)
    weights = xp.exp(-shifted / temperature_ev)
    return weights / xp.sum(weights)


@dataclass(frozen=True)
class VibrationalDistribution:
    """A single-temperature Boltzmann population over a vibrational ladder.

    This is a *prescribed* distribution: ``temperature_ev`` is an input (a
    measured or assumed vibrational temperature), not the output of a solved
    kinetics balance. Real low-temperature-plasma H2 populations deviate from
    a single Boltzmann distribution at high v (electron-impact excitation via
    the triplet states populates high v faster than V-V/V-T relaxation
    thermalizes it), so this systematically *underestimates* the high-v tail
    that drives dissociative attachment. It is still the right first
    approximation: Krištof et al. (2019) note that negative-ion concentration
    is negligible for T_vib below about 3000 K and rises exponentially above
    it (citing Kalache et al. 2004), so getting the order of magnitude of the
    high-v population right matters far more than its detailed shape.
    """

    energies_ev: tuple[float, ...]
    temperature_ev: float

    def __post_init__(self) -> None:
        """Validate the distribution's parameters."""
        if len(self.energies_ev) == 0:
            raise ValueError("energies_ev must not be empty")
        if self.temperature_ev <= 0.0:
            raise ValueError("temperature_ev must be > 0")

    @classmethod
    def h2_ground_state(cls, temperature_ev: float) -> VibrationalDistribution:
        """Build the distribution over the real H2(X) vibrational ladder."""
        return cls(
            energies_ev=H2_GROUND_STATE_VIBRATIONAL_ENERGY_EV,
            temperature_ev=temperature_ev,
        )

    @classmethod
    def h2_ground_state_from_kelvin(
        cls, temperature_k: float
    ) -> VibrationalDistribution:
        """Build the H2(X) distribution from a vibrational temperature in K."""
        if temperature_k <= 0.0:
            raise ValueError("temperature_k must be > 0")
        return cls.h2_ground_state(kelvin_to_ev(temperature_k))

    @property
    def num_levels(self) -> int:
        """Return the number of vibrational levels in the ladder."""
        return len(self.energies_ev)

    @property
    def populations(self) -> Array:
        """Return the normalized population of each level, ``n_v / n_total``."""
        return boltzmann_level_populations(
            xp.asarray(self.energies_ev), self.temperature_ev
        )

    def fraction_at_least(self, v_min: int) -> float:
        """Return the population fraction in levels ``v >= v_min``.

        This is ``[H2(v>=v_min)] / [H2]``, the quantity Kalache et al. (2004)
        and Kimura and Kasugai (2010) use to characterize dissociative
        attachment: DA is negligible below v ~ 4 and dominates above it.
        """
        if not 0 <= v_min < self.num_levels:
            raise ValueError(f"v_min must be in [0, {self.num_levels - 1}]")
        return float(xp.sum(self.populations[v_min:]))

    def mean_level(self) -> float:
        """Return the population-weighted mean vibrational quantum number."""
        levels = xp.arange(self.num_levels, dtype=float)
        return float(xp.sum(levels * self.populations))


def estimate_electronegativity(
    distribution: VibrationalDistribution,
    hydrogen_atom_density: float,
    molecule_density: float,
    v_min: int = 4,
    rate_ratio: float = 10.0,
) -> float:
    """Estimate ``[H-]/[e]`` from the vibrationally excited H2 fraction.

    Implements Kimura and Kasugai's (2010) balance between dissociative
    attachment production and detachment-by-H destruction of H-, as given by
    Krištof et al. (2019), eq. (11):

        alpha = rate_ratio * [H2(v>=v_min)] / [H]

    ``rate_ratio`` is the ratio of the DA rate coefficient to the detachment
    rate coefficient, estimated at around 10 at maximum (Zorat et al. 2000;
    Janev et al. 1987; Graham 1995, as summarized by Krištof et al. 2019).
    This is an order-of-magnitude estimate, not a validated absolute rate --
    treat the result the same way.
    """
    if hydrogen_atom_density <= 0.0:
        raise ValueError("hydrogen_atom_density must be > 0")
    if molecule_density < 0.0:
        raise ValueError("molecule_density must be >= 0")
    if rate_ratio < 0.0:
        raise ValueError("rate_ratio must be >= 0")
    excited_h2_density = distribution.fraction_at_least(v_min) * molecule_density
    return rate_ratio * excited_h2_density / hydrogen_atom_density
