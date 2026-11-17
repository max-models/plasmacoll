r"""Fusion cross sections and reactivities of Bosch and Hale (1992).

H.-S. Bosch and G. M. Hale, "Improved formulas for fusion cross-sections and
thermal reactivities", Nucl. Fusion 32, 611 (1992). The paper fits R-matrix
evaluations of the four reactions of the light fusion fuels:

=================  ====================  =======  ===============  ==============
key                reaction              Q [MeV]  cross section    reactivity
=================  ====================  =======  ===============  ==============
``"D(d,n)3He"``    D + D -> n + 3He      3.269    0.5-4900 keV     0.2-100 keV
``"D(d,p)T"``      D + D -> p + T        4.033    0.5-5000 keV     0.2-100 keV
``"T(d,n)4He"``    T + D -> n + 4He      17.589   0.5-4700 keV     0.2-100 keV
``"3He(d,p)4He"``  3He + D -> p + 4He    18.353   0.3-4800 keV     0.5-190 keV
=================  ====================  =======  ===============  ==============

(the ranges of validity: of the cross section in the centre-of-mass energy, of
the reactivity in the temperature). The cross section is a Gamow factor times
the astrophysical factor $S(E)$, a Padé approximant whose parameters are the
paper's Table IV (two energy ranges for T(d,n) and 3He(d,p)); the Maxwellian
reactivity is a fit of the same kind in the temperature, Table VII. The paper
gives both to within about 0.3% of the R-matrix values in their ranges.

For a fast ion in a thermal plasma, `FusionReaction.beam_reactivity` averages
$\sigma v_\mathrm{rel}$ over a Maxwellian target at rest
(`beam_maxwellian_reactivity`, the integral over the relative speed of
Mikkelsen, Nucl. Fusion 29, 1113, 1989); it needs the cross section only, and
a cold target gives $\sigma(E_\mathrm{cm})\,v_b$.

Energies and temperatures are in eV (the paper's in keV), cross sections in
m^2 (the paper's in mb), reactivities in m^3/s (cm^3/s); every argument also
takes an astropy Quantity or a string. The masses are those of the nuclei
(CODATA 2018), i.e. of the fully ionized beam ion. Outside the ranges of
validity the formulas are evaluated as they stand, an extrapolation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scope_profiler as sp
from methodcard import Rule, method

from plasmacoll._profiling import PROFILE_TAGS
from plasmacoll._types import Array
from plasmacoll.constants import ATOMIC_MASS, ELEMENTARY_CHARGE
from plasmacoll.cross_sections import CrossSection
from plasmacoll.units import to_array, to_unit

__all__ = [
    "BOSCH_HALE_1992",
    "FUSION_REACTIONS",
    "NUCLEAR_MASSES",
    "FusionReaction",
    "SFactorFit",
    "beam_maxwellian_reactivity",
]

#: The paper the parameters come from.
BOSCH_HALE_1992 = (
    "H.-S. Bosch and G. M. Hale, Improved formulas for fusion cross-sections and "
    "thermal reactivities, Nucl. Fusion 32, 611 (1992)"
)
#: The masses of the nuclei in kg (CODATA 2018: deuteron, triton, helion).
NUCLEAR_MASSES = {
    "D": 2.013553212745 * ATOMIC_MASS,
    "T": 3.01550071621 * ATOMIC_MASS,
    "3He": 3.014932247175 * ATOMIC_MASS,
}
#: 1 mb in m^2, 1 cm^3/s in m^3/s, 1 keV in eV.
_MB = 1.0e-31
_CM3 = 1.0e-6
_KEV = 1.0e3


@dataclass(frozen=True)
class SFactorFit:
    """The astrophysical factor of one reaction in one energy range (Table IV).

    ``S(E) = (A1 + E (A2 + E (A3 + E (A4 + E A5)))) / (1 + E (B1 + E (B2 + E (B3 + E B4))))``
    in keV mb, with the centre-of-mass energy ``E`` in keV.

    Attributes:
        valid_from: Lowest centre-of-mass energy of the fit [eV].
        valid_to: Highest centre-of-mass energy of the fit [eV].
        a: ``A1`` ... ``A5``.
        b: ``B1`` ... ``B4``.
    """

    valid_from: float
    valid_to: float
    a: tuple[float, float, float, float, float]
    b: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __call__(self, energy_kev: Array) -> np.ndarray:
        """Return ``S`` in keV mb at ``energy_kev`` (keV, centre of mass)."""
        e = np.asarray(energy_kev, dtype=float)
        a, b = self.a, self.b
        numerator = a[0] + e * (a[1] + e * (a[2] + e * (a[3] + e * a[4])))
        denominator = 1.0 + e * (b[0] + e * (b[1] + e * (b[2] + e * b[3])))
        return numerator / denominator


def _speed_grid(num: int) -> tuple[np.ndarray, np.ndarray]:
    """Points ``s`` in [0, 1] and their trapezoidal weights."""
    s = np.linspace(0.0, 1.0, num)
    weights = np.full(num, 1.0 / (num - 1))
    weights[0] = weights[-1] = 0.5 / (num - 1)
    return s, weights


#: Thermal speeds on each side of the beam speed that carry the integral.
_WIDTH = 10.0


def _at_rest(
    sigma: Callable[[np.ndarray], np.ndarray],
    kt: np.ndarray,
    v_t: np.ndarray,
    reduced_mass: float,
    num: int = 801,
) -> np.ndarray:
    r"""The limit $v_b \to 0$: $\frac{4}{\sqrt\pi v_t^3}\int_0^\infty \sigma u^3 e^{-u^2/v_t^2} du$."""
    s, weights = _speed_grid(num)
    scale = np.where(v_t > 0.0, v_t, 1.0)[..., None]
    u = _WIDTH * scale * s
    energy = 0.5 * reduced_mass * u**2 / ELEMENTARY_CHARGE
    integral = np.sum(weights * sigma(energy) * u**3 * np.exp(-((u / scale) ** 2)), -1)
    return np.where(
        kt > 0.0,
        4.0 * _WIDTH * integral / (math.sqrt(math.pi) * scale[..., 0] ** 2),
        0.0,
    )


@method(
    equations={
        "average": r"\langle\sigma v\rangle_b = \frac{1}{\sqrt\pi\,v_t\,v_b}\int_0^\infty \sigma\!\Big(\tfrac12 m_r u^2\Big)\,u^2 \left[e^{-(u - v_b)^2/v_t^2} - e^{-(u + v_b)^2/v_t^2}\right] du, \qquad v_t = \sqrt{\frac{2kT}{m_t}}",
        "cold": r"\langle\sigma v\rangle_b = \sigma\!\Big(\tfrac12 m_r v_b^2\Big)\,v_b \quad (T = 0)",
        "trapezoid": r"u \in \big[\max(0,\ v_b - 10 v_t),\ v_b + 10 v_t\big], \quad \text{trapezoidal rule on } N \text{ equal steps}",
    },
    assumptions=(
        r"a fast ion of speed $v_b$ in an isotropic Maxwellian target at rest, $f_t \propto e^{-v^2/v_t^2}$; $u = |\mathbf{v}_b - \mathbf{v}_t|$ the relative speed, $\sigma$ against the centre-of-mass energy $\tfrac12 m_r u^2$",
        r"the angular integral done exactly; the Gaussian tails beyond $10\,v_t$ carry $e^{-100}$",
        r"the Gamow peak of a thermal target lies within $10\,v_t$ for $kT \gtrsim 0.1$ keV",
    ),
    references=(
        "[Mikkelsen, Nucl. Fusion 29, 1113 (1989)](https://doi.org/10.1088/0029-5515/29/7/004)",
    ),
)
def beam_maxwellian_reactivity(
    sigma: Callable[[np.ndarray], np.ndarray],
    beam_speed: Array,
    temperature: Array,
    reduced_mass: float,
    target_mass: float,
    *,
    num: int = 801,
) -> np.ndarray:
    r"""Return $\langle\sigma v_\mathrm{rel}\rangle$ of fast ions in a Maxwellian target [m^3/s].

    Args:
        sigma: Cross section in m^2 against the centre-of-mass energy in eV.
        beam_speed: Speed of the fast ion in the target's rest frame [m/s].
        temperature: ``kT`` of the target [eV], broadcast against ``beam_speed``;
            zero for a cold target.
        reduced_mass: Reduced mass of the pair [kg].
        target_mass: Mass of a target particle [kg].
        num: Points of the trapezoidal rule in the relative speed.

    Returns:
        The reactivity, of the broadcast shape of ``beam_speed`` and ``temperature``.
    """
    v_b, kt = np.broadcast_arrays(
        np.asarray(beam_speed, dtype=float), np.asarray(temperature, dtype=float)
    )
    if np.any(v_b < 0.0) or np.any(kt < 0.0):
        raise ValueError("beam_speed and temperature must be >= 0")
    v_t = np.sqrt(2.0 * kt * ELEMENTARY_CHARGE / target_mass)[..., None]
    v = v_b[..., None]
    s, weights = _speed_grid(num)
    low = np.maximum(v - _WIDTH * v_t, 0.0)
    high = v + _WIDTH * v_t
    u = low + (high - low) * s
    energy = 0.5 * reduced_mass * u**2 / ELEMENTARY_CHARGE
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(v_t > 0.0, v_t, 1.0)
        bracket = np.exp(-(((u - v) / scale) ** 2)) - np.exp(-(((u + v) / scale) ** 2))
        integral = np.sum(weights * sigma(energy) * u**2 * bracket, axis=-1)
        warm = (
            integral * (high - low)[..., 0] / (math.sqrt(math.pi) * v_t[..., 0] * v_b)
        )
    cold_energy = 0.5 * reduced_mass * v_b**2 / ELEMENTARY_CHARGE
    cold = sigma(cold_energy) * v_b
    warm = np.where(v_b > 0.0, warm, _at_rest(sigma, kt, v_t[..., 0], reduced_mass))
    return np.where(kt > 0.0, warm, cold)


def _fusion_rules() -> dict:
    """The SymPy rules of the cross section and the reactivity, of T(d,n)4He (all terms)."""
    import sympy

    e, t = sympy.symbols("energy temperature", positive=True)
    f = sympy.Float
    reaction = FUSION_REACTIONS["T(d,n)4He"]
    fit = reaction.sfactor[0]
    ek = e / f(_KEV)
    a, b = [f(x) for x in fit.a], [f(x) for x in fit.b]
    s_factor = (a[0] + ek * (a[1] + ek * (a[2] + ek * (a[3] + ek * a[4])))) / (
        1 + ek * (b[0] + ek * (b[1] + ek * (b[2] + ek * b[3])))
    )
    bg = f(reaction.gamow)
    sigma = f(_MB) * s_factor / (ek * sympy.exp(bg / sympy.sqrt(ek)))
    c = [f(x) for x in reaction.reactivity_coefficients]
    tk = t / f(_KEV)
    theta = tk / (
        1
        - tk
        * (c[1] + tk * (c[3] + tk * c[5]))
        / (1 + tk * (c[2] + tk * (c[4] + tk * c[6])))
    )
    xi = (bg**2 / (4 * theta)) ** sympy.Rational(1, 3)
    reactivity = (
        f(_CM3)
        * c[0]
        * theta
        * sympy.sqrt(xi / (f(reaction.reduced_mass_energy) * tk**3))
        * sympy.exp(-3 * xi)
    )
    return {
        "cross_section": Rule(
            sigma,
            implementation=lambda energy: reaction.sigma(energy),
            ranges={"energy": (fit.valid_from, fit.valid_to)},
        ),
        "reactivity": Rule(
            reactivity,
            implementation=lambda temperature: reaction.reactivity(temperature),
            ranges={"temperature": reaction.reactivity_range},
        ),
    }


@method(
    equations={
        "cross_section": r"\sigma(E) = \frac{S(E)}{E\,\exp\!\big(B_G/\sqrt{E}\big)}, \qquad S(E) = \frac{A_1 + E(A_2 + E(A_3 + E(A_4 + E A_5)))}{1 + E(B_1 + E(B_2 + E(B_3 + E B_4)))}, \qquad \text{eqs. (8), (9), Table IV}",
        "gamow": r"B_G = \pi\alpha Z_1 Z_2 \sqrt{2 m_r c^2}",
        "reactivity": r"\langle\sigma v\rangle = C_1\,\theta\,\sqrt{\frac{\xi}{m_r c^2\,T^3}}\,e^{-3\xi}, \quad \theta = \frac{T}{1 - \frac{T(C_2 + T(C_4 + T C_6))}{1 + T(C_3 + T(C_5 + T C_7))}}, \quad \xi = \Big(\frac{B_G^2}{4\theta}\Big)^{1/3}, \qquad \text{eqs. (12)-(14), Table VII}",
    },
    assumptions=(
        r"$E$ the centre-of-mass energy and $T$ the temperature in keV, $\sigma$ in mb and $\langle\sigma v\rangle$ in cm$^3$/s, returned in eV, m$^2$ and m$^3$/s",
        r"the reactivity of two Maxwellians at the same temperature $T$ (thermonuclear), without polarisation or plasma screening",
        r"within the ranges of validity of Tables IV and VII to about 0.3%; below them the formulas extrapolated, above the last range $S$ held at its end value",
    ),
    references=(
        "[Bosch and Hale, Nucl. Fusion 32, 611 (1992)](https://doi.org/10.1088/0029-5515/32/4/I07)",
    ),
    symbolic=lambda: _fusion_rules(),
)
@dataclass(frozen=True)
class FusionReaction:
    """One fusion reaction ``target(projectile, light)heavy`` of Bosch and Hale.

    Attributes:
        key: The reaction in nuclear notation, e.g. ``"D(d,n)3He"``.
        target: The target nucleus, a key of `NUCLEAR_MASSES`.
        projectile: The projectile nucleus (``"D"`` for all four reactions).
        products: The light and the heavy product, e.g. ``("n", "3He")``.
        q_value: Energy released [eV].
        gamow: The Gamow constant ``B_G`` [keV^1/2].
        reduced_mass_energy: ``m_r c^2`` of Table VII [keV].
        sfactor: The fits of the astrophysical factor, by increasing energy.
        reactivity_coefficients: ``C1`` ... ``C7`` of Table VII.
        reactivity_range: Temperatures of the reactivity fit [eV].
    """

    key: str
    target: str
    projectile: str
    products: tuple[str, str]
    q_value: float
    gamow: float
    reduced_mass_energy: float
    sfactor: tuple[SFactorFit, ...]
    reactivity_coefficients: tuple[float, float, float, float, float, float, float]
    reactivity_range: tuple[float, float]

    @property
    def neutron(self) -> bool:
        """Whether one product is a neutron."""
        return "n" in self.products

    @property
    def masses(self) -> tuple[float, float]:
        """The masses of the target and the projectile [kg]."""
        return NUCLEAR_MASSES[self.target], NUCLEAR_MASSES[self.projectile]

    @property
    def reduced_mass(self) -> float:
        """The reduced mass of the pair [kg]."""
        m_t, m_p = self.masses
        return m_t * m_p / (m_t + m_p)

    @property
    def label(self) -> str:
        """The reaction and its source, for plots and messages."""
        return f"{self.key}: Bosch and Hale (1992)"

    def astrophysical_factor(self, energy: Array | float) -> np.ndarray:
        """Return ``S(E)`` [keV mb, the paper's unit] at the centre-of-mass energy (eV).

        Each fit of `sfactor` is used in its range; below the first range the
        first fit (which tends to ``A1``), above the last the value at its end
        (the Padé approximants turn negative beyond their ranges).
        """
        e = np.asarray(to_array(energy, "eV", "energy"), dtype=float)
        e_kev = np.minimum(e, self.sfactor[-1].valid_to) / _KEV
        result = self.sfactor[0](e_kev)
        for fit in self.sfactor[1:]:
            result = np.where(e_kev * _KEV > fit.valid_from, fit(e_kev), result)
        return result

    def sigma(self, energy: Array | float) -> np.ndarray:
        """Return the cross section [m^2] at the centre-of-mass energy (eV); zero at 0."""
        e = np.asarray(to_array(energy, "eV", "energy"), dtype=float)
        e_kev = np.where(e > 0.0, e, 1.0) / _KEV
        value = (
            _MB
            * self.astrophysical_factor(e_kev * _KEV)
            * np.exp(-self.gamow / np.sqrt(e_kev))
            / e_kev
        )
        return np.where(e > 0.0, value, 0.0)

    def reactivity(self, temperature: Array | float) -> np.ndarray:
        """Return the thermal reactivity <sigma v> [m^3/s] at ``temperature`` (eV, or K)."""
        t = np.asarray(to_array(temperature, "eV", "temperature"), dtype=float)
        if np.any(t <= 0.0):
            raise ValueError("temperature must be > 0")
        t = t / _KEV
        c = self.reactivity_coefficients
        theta = t / (
            1.0
            - t
            * (c[1] + t * (c[3] + t * c[5]))
            / (1.0 + t * (c[2] + t * (c[4] + t * c[6])))
        )
        xi = (self.gamow**2 / (4.0 * theta)) ** (1.0 / 3.0)
        return (
            _CM3
            * c[0]
            * theta
            * np.sqrt(xi / (self.reduced_mass_energy * t**3))
            * np.exp(-3.0 * xi)
        )

    def _beam_mass(self, beam: str | None) -> float:
        """The mass of the reactant ``beam`` (the projectile if None)."""
        if beam is None:
            beam = self.projectile
        if beam not in (self.target, self.projectile):
            raise ValueError(
                f"{self.key}: the beam must be {self.target!r} or {self.projectile!r}, "
                f"got {beam!r}"
            )
        return NUCLEAR_MASSES[beam]

    def center_of_mass_energy(
        self, beam_energy: Array | float, beam: str | None = None
    ) -> np.ndarray:
        """Return the centre-of-mass energy [eV] of a ``beam`` ion on a target at rest.

        Args:
            beam_energy: The beam ion's energy in the target's frame [eV].
            beam: Which reactant is the beam (`target` or `projectile`); the
                projectile if None.
        """
        m_beam = self._beam_mass(beam)
        e = np.asarray(to_array(beam_energy, "eV", "beam_energy"), dtype=float)
        return e * self.reduced_mass / m_beam

    def cross_section(
        self,
        max_energy: float | None = None,
        num: int = 2001,
        *,
        beam: str | None = None,
    ) -> CrossSection:
        """Tabulate the cross section for `plasmacoll.rate_coefficient` and the collisions.

        Logarithmic in the energy from 100 eV to ``max_energy`` (eV; the end of
        the last fit if None), in the centre-of-mass frame, or with ``beam``
        given against that reactant's energy with the other at rest (lab frame).
        Beyond the table its last value holds (`plasmacoll.CrossSection`).
        """
        upper = (
            self.sfactor[-1].valid_to
            if max_energy is None
            else to_unit(max_energy, "eV", "max_energy")
        )
        energy = np.geomspace(100.0, upper, num)
        scale = 1.0 if beam is None else self.reduced_mass / self._beam_mass(beam)
        return CrossSection(
            energy=energy, sigma=self.sigma(energy * scale), label=self.label
        )

    @method(
        equations={
            "beam": r"\langle\sigma v\rangle_b(E_b, T) = \langle\sigma v\rangle_b\Big(v_b = \sqrt{2E_b/m_b},\ T,\ m_r,\ m_t\Big), \qquad m_t = m_1 + m_2 - m_b",
        },
        assumptions=(
            r"the target Maxwellian at rest, of the other reactant's mass; the beam ion mono-energetic",
        ),
        components=("plasmacoll.fusion.beam_maxwellian_reactivity",),
    )
    @sp.profile("plasmacoll.beam_reactivity", tags=PROFILE_TAGS)
    def beam_reactivity(
        self,
        beam_energy: Array | float,
        temperature: Array | float = 0.0,
        *,
        beam: str | None = None,
        num: int = 801,
    ) -> np.ndarray:
        r"""Return $\langle\sigma v_\mathrm{rel}\rangle$ [m^3/s] of beam ions in a Maxwellian target.

        Args:
            beam_energy: Energy of the beam ion in the target's frame [eV].
            temperature: ``kT`` of the target [eV], broadcast against
                ``beam_energy``; zero for a target at rest.
            beam: Which reactant is the beam (`target` or `projectile`, e.g.
                ``"T"`` for tritons on deuterium); the projectile if None.
            num: Points of the integral over the relative speed.
        """
        m_beam = self._beam_mass(beam)
        m_target = sum(self.masses) - m_beam
        e = np.asarray(to_array(beam_energy, "eV", "beam_energy"), dtype=float)
        t = np.asarray(to_array(temperature, "eV", "temperature"), dtype=float)
        speed = np.sqrt(2.0 * np.maximum(e, 0.0) * ELEMENTARY_CHARGE / m_beam)
        return beam_maxwellian_reactivity(
            self.sigma, speed, t, self.reduced_mass, m_target, num=num
        )


#: The reactions of Bosch and Hale (1992) by key, with Tables IV and VII.
FUSION_REACTIONS: dict[str, FusionReaction] = {
    "T(d,n)4He": FusionReaction(
        key="T(d,n)4He",
        target="T",
        projectile="D",
        products=("n", "4He"),
        q_value=17.589e6,
        gamow=34.3827,
        reduced_mass_energy=1124656.0,
        sfactor=(
            SFactorFit(
                0.5e3,
                550.0e3,
                (6.927e4, 7.454e8, 2.050e6, 5.2002e4, 0.0),
                (6.38e1, -9.95e-1, 6.981e-5, 1.728e-4),
            ),
            SFactorFit(
                550.0e3,
                4700.0e3,
                (-1.4714e6, 0.0, 0.0, 0.0, 0.0),
                (-8.4127e-3, 4.7983e-6, -1.0748e-9, 8.5184e-14),
            ),
        ),
        reactivity_coefficients=(
            1.17302e-9,
            1.51361e-2,
            7.51886e-2,
            4.60643e-3,
            1.35000e-2,
            -1.06750e-4,
            1.36600e-5,
        ),
        reactivity_range=(0.2e3, 100.0e3),
    ),
    "3He(d,p)4He": FusionReaction(
        key="3He(d,p)4He",
        target="3He",
        projectile="D",
        products=("p", "4He"),
        q_value=18.353e6,
        gamow=68.7508,
        reduced_mass_energy=1124572.0,
        sfactor=(
            SFactorFit(
                0.3e3,
                900.0e3,
                (5.7501e6, 2.5226e3, 4.5566e1, 0.0, 0.0),
                (-3.1995e-3, -8.5530e-6, 5.9014e-8, 0.0),
            ),
            SFactorFit(
                900.0e3,
                4800.0e3,
                (-8.3993e5, 0.0, 0.0, 0.0, 0.0),
                (-2.6830e-3, 1.1633e-6, -2.1332e-10, 1.4250e-14),
            ),
        ),
        reactivity_coefficients=(
            5.51036e-10,
            6.41918e-3,
            -2.02896e-3,
            -1.91080e-5,
            1.35776e-4,
            0.0,
            0.0,
        ),
        reactivity_range=(0.5e3, 190.0e3),
    ),
    "D(d,p)T": FusionReaction(
        key="D(d,p)T",
        target="D",
        projectile="D",
        products=("p", "T"),
        q_value=4.033e6,
        gamow=31.3970,
        reduced_mass_energy=937814.0,
        sfactor=(
            SFactorFit(
                0.5e3,
                5000.0e3,
                (5.5576e4, 2.1054e2, -3.2638e-2, 1.4987e-6, 1.8181e-10),
            ),
        ),
        reactivity_coefficients=(
            5.65718e-12,
            3.41267e-3,
            1.99167e-3,
            0.0,
            1.05060e-5,
            0.0,
            0.0,
        ),
        reactivity_range=(0.2e3, 100.0e3),
    ),
    "D(d,n)3He": FusionReaction(
        key="D(d,n)3He",
        target="D",
        projectile="D",
        products=("n", "3He"),
        q_value=3.269e6,
        gamow=31.3970,
        reduced_mass_energy=937814.0,
        sfactor=(
            SFactorFit(
                0.5e3,
                4900.0e3,
                (5.3701e4, 3.3027e2, -1.2706e-1, 2.9327e-5, -2.5151e-9),
            ),
        ),
        reactivity_coefficients=(
            5.43360e-12,
            5.85778e-3,
            7.68222e-3,
            0.0,
            -2.96400e-6,
            0.0,
            0.0,
        ),
        reactivity_range=(0.2e3, 100.0e3),
    ),
}
