"""The definition of one collision process of a charged species with a neutral gas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from methodcard import method

from plasmacoll.cross_sections import CrossSection

__all__ = [
    "ENERGY_FRAMES",
    "ENERGY_SHARING",
    "OPTIONAL_PRODUCT_ROLES",
    "PROCESS_KINDS",
    "PRODUCT_ROLES",
    "SCATTERING_MODELS",
    "CollisionProcess",
]

#: The collision kinds `plasmacoll.mcc.MonteCarloCollisions` implements.
PROCESS_KINDS = (
    "elastic",
    "backscatter",
    "excitation",
    "ionization",
    "attachment",
    "detachment",
    "charge_transfer",
    "dissociation",
    "dissociative_attachment",
    "dissociative_ionization",
    "stripping",
    "beam_ionization",
)
#: The product species each kind needs, by role.
PRODUCT_ROLES: dict[str, tuple[str, ...]] = {
    "elastic": (),
    "backscatter": (),
    "excitation": (),
    "ionization": ("electron", "ion"),
    "attachment": ("negative_ion",),
    "detachment": ("electron",),
    "charge_transfer": (),
    "dissociation": (),
    "dissociative_attachment": ("negative_ion",),
    "dissociative_ionization": ("electron", "ion"),
    "stripping": ("projectile",),
    "beam_ionization": ("ion",),
}
#: The product species a kind may have. ``neutral`` is the fast neutral of a
#: charge exchange and ``fragment`` the neutral fragment of a dissociation; both
#: are dropped (not tracked) unless a species is given for them. ``electron``
#: and ``ion`` of ``stripping`` are the released electrons and the target ion,
#: ``electron`` of ``beam_ionization`` the electron ejected from the target.
OPTIONAL_PRODUCT_ROLES: dict[str, tuple[str, ...]] = {
    "backscatter": ("neutral",),
    "charge_transfer": ("ion", "neutral"),
    "dissociation": ("fragment",),
    "dissociative_attachment": ("fragment",),
    "dissociative_ionization": ("fragment",),
    "stripping": ("electron", "ion"),
    "beam_ionization": ("electron",),
}
#: The frames whose kinetic energy the cross section is looked up at.
ENERGY_FRAMES = ("center_of_mass", "lab")
#: How ionization shares the residual energy between the two electrons.
ENERGY_SHARING = ("equal", "uniform", "opal")
#: The angular distributions of elastic, excitation and ionization scattering.
SCATTERING_MODELS = ("isotropic", "vahedi_surendra", "okhrimovskyy")

_NO_ENERGY_LOSS = (
    "elastic",
    "backscatter",
    "attachment",
    "charge_transfer",
    "dissociative_attachment",
    "stripping",
    "beam_ionization",
)
_IONIZATION_KINDS = ("ionization", "dissociative_ionization")
#: The kinds that remove one particle of the background per collision: the
#: ionized, attached-to or dissociated neutral. Charge transfer swaps a neutral
#: for an ion and leaves the count; stripping and detachment act on the incident.
_CONSUMING_KINDS = (
    "ionization",
    "dissociative_ionization",
    "beam_ionization",
    "attachment",
    "dissociative_attachment",
    "dissociation",
)


@method(
    equations={
        "centre_of_mass": r"\mathbf V = \frac{m\mathbf v + M\mathbf w}{m + M}, \qquad \mathbf g = \mathbf v - \mathbf w, \qquad \mathbf v = \mathbf V + \frac{M}{m + M}\,\mathbf g, \qquad \mu = \frac{mM}{m + M}",
        "scattering": r"g'^2 = \max\Big(g^2 - \frac{2e\,\Delta E}{\mu},\ 0\Big), \qquad \mathbf v' = \mathbf V + \frac{M}{m + M}\,g'\,\hat{\boldsymbol\Omega}(\chi, \phi)",
        "backscatter": r"\mathbf v' = \mathbf V - \frac{M}{m + M}\,\mathbf g \qquad (= \mathbf w \text{ for } m = M)",
        "electron_impact_ionization": r"E_r = \max\big(\tfrac12 m_e g^2 - e\,\Delta E,\ 0\big), \qquad \mathbf v_p = \mathbf w + \sqrt{\frac{2(1 - f)E_r}{m_e}}\,\hat{\boldsymbol\Omega}_p, \qquad \mathbf v_s = \mathbf w + \sqrt{\frac{2fE_r}{m_e}}\,\hat{\boldsymbol\Omega}_s",
        "energy_sharing": r"f = \tfrac12 \ (\text{equal}), \qquad f = R \ (\text{uniform}), \qquad f = \frac{E_s}{E_r}, \ E_s = w\tan\Big(R\arctan\frac{E_r}{2w}\Big) \ (\text{Opal}, \ p(E_s) \propto \frac{1}{1 + (E_s/w)^2} \text{ on } [0, E_r/2])",
        "ion_impact_ionization": r"E_r = \max\big(\tfrac12\mu g^2 - e\,\Delta E,\ 0\big), \quad \mathbf g' = \sqrt{\frac{2(1 - f)E_r}{\mu}}\,\hat{\mathbf g}, \quad \mathbf v' = \mathbf V + \frac{M}{m + M}\mathbf g', \quad \mathbf v_{B^+} = \mathbf V - \frac{m}{m + M}\mathbf g', \quad \mathbf v_e = \mathbf V + \sqrt{\frac{2fE_r}{m_e}}\,\hat{\boldsymbol\Omega}_\mathrm{iso}",
        "detachment": r"E_e = \max\big(\tfrac12\mu g^2 - e\,\Delta E,\ 0\big), \qquad \mathbf v_e = \mathbf V + \sqrt{\frac{2E_e}{m_e}}\,\hat{\boldsymbol\Omega}_\mathrm{iso}",
        "stripping": r"\mathbf v_\mathrm{projectile} = \mathbf v, \qquad \mathbf v_e = \mathbf v \ \text{(one marker of weight } q\,W), \qquad \mathbf v_{B^+} = \mathbf w",
        "beam_ionization": r"\mathbf v' = \mathbf v, \quad \mathbf v_{B^+} = \mathbf w, \quad \mathbf v_e = \mathbf w + \sqrt{\frac{2E}{m_e}}\,\hat{\boldsymbol\Omega}_\mathrm{iso}, \quad E = w\tan\Big(R\arctan\frac{E_{\max}}{w}\Big), \quad E_{\max} = 2m_e g^2",
        "energy_loss": r"\Delta E = \begin{cases} 0 & \text{elastic, backscatter, (dissociative) attachment, charge transfer, stripping, beam ionization} \\ \texttt{energy\_loss}, \text{ else the table threshold} & \text{the other kinds} \end{cases}",
    },
    assumptions=(
        r"the kinematics `MonteCarloCollisions.collide` and `collide_now` apply, in the centre-of-mass frame of the marker ($m$, $\mathbf v$, weight $W$) and a neutral ($M$, $\mathbf w$) sampled from its background",
        r"$\hat{\mathbf g} = \mathbf g / g$ ($\hat{\mathbf z}$ where $g = 0$); $\hat{\boldsymbol\Omega}$ isotropic, or at the angle $\chi$ of the `scattering` model about $\hat{\mathbf g}$ with a uniform azimuth $\phi$; $R$ uniform in $[0, 1)$",
        r"elastic, excitation and dissociation conserve energy and momentum exactly; dissociation may create a `fragment` at $\mathbf w$",
        r"electron-impact ionization (the incident species is the product electron): exact in the neutral's frame, the ion recoil $O(m_e/M)$ dropped; ion and fragment start at $\mathbf w$",
        r"ion-impact ionization: energy and momentum up to the electron's momentum, $O(\sqrt{m_e/M})$; the ions keep $\hat{\mathbf g}$, the electron is isotropic, `scattering` is not used",
        r"attachment and dissociative attachment remove the electron and create the negative ion (and fragment) at $\mathbf w$; charge transfer removes the incident, with an optional ion at $\mathbf w$ and neutral at $\mathbf v$",
        r"detachment: the neutral fragments are not tracked, the electron takes the whole residual energy",
        r"stripping: momentum exact when $m = m_\mathrm{projectile} + q\,m_e$ ($q$ = `charge_change`), binding energies neglected; beam ionization drops the target's and the electron's recoil",
        r"products take the position and weight of the incident marker; a collision of the ionizing, attaching or dissociating kinds consumes one neutral",
    ),
    references=(
        "[Vahedi and Surendra, Comput. Phys. Commun. 87, 179 (1995)](https://doi.org/10.1016/0010-4655(94)00171-W)",
        "[Opal, Peterson and Beaty, J. Chem. Phys. 55, 4100 (1971)](https://doi.org/10.1063/1.1676707)",
    ),
    components=("plasmacoll.mcc.scattering_cosine",),
)
@dataclass(frozen=True)
class CollisionProcess:
    """One collision process of an incident species with a neutral background.

    ``energy_frame`` selects the energy used to look up the cross section: the
    centre-of-mass energy ``mu g^2 / 2`` or the incident particle's energy in
    the neutral's rest frame ``m g^2 / 2`` (the convention of electron
    cross-section compilations; both agree to O(m_e/M) for electrons), where
    ``g`` is the relative speed and ``mu`` the reduced mass.

    ``scattering`` selects the angular distribution of elastic, excitation,
    dissociation and ionization collisions: ``"isotropic"``, or for electrons
    the screened-Coulomb forms of Vahedi and Surendra, Comput. Phys. Commun.
    87, 179 (1995), and Okhrimovskyy et al., Phys. Rev. E 65, 037402 (2002),
    which become forward-peaked above a few eV. With isotropic scattering the
    elastic cross section must be the momentum-transfer cross section; with an
    anisotropic model it must be the integral elastic cross section.

    Attributes:
        kind: One of `PROCESS_KINDS`.
        background: The name of the `plasmacoll.background.NeutralBackground`.
        cross_section: The cross section against energy.
        name: A unique name for diagnostics; ``"<kind>:<background>"`` if empty.
        energy_loss: Energy in eV lost by excitation, ionization and detachment;
            the threshold of the cross section if None.
        energy_frame: One of `ENERGY_FRAMES`.
        products: Product species by role, see `PRODUCT_ROLES`, e.g.
            ``{"electron": "e", "ion": "Ar+"}`` for ionization.
        energy_sharing: One of `ENERGY_SHARING`, for ionization.
            ``"opal"`` samples the secondary electron's energy from the
            Opal-Peterson-Beaty distribution ``1 / (1 + (E_s / w)^2)``.
        sharing_energy: The parameter ``w`` in eV of ``"opal"`` sharing, of
            order the ionization energy (e.g. 10.0 for argon, 8.3 for H2); for
            ``beam_ionization`` the width of the ejected electron's energy
            distribution (required with an ``electron`` product).
        scattering: One of `SCATTERING_MODELS`.
        charge_change: For ``stripping``, the change of the projectile's charge
            in e: 1 for single stripping (``H- -> H``, ``H -> H+``), 2 for
            double stripping (``H- -> H+``), -1 for electron capture
            (``H+ -> H``). A positive change releases that many electrons.
        upper_state: For ``excitation``, the background (or state, see
            `plasmacoll.boltzmann.solve_eedf`) of the excited molecules. If
            given, the Boltzmann solver adds the inverse (superelastic)
            collisions with them, whose cross section follows from this one by
            detailed balance; `plasmacoll.mcc.MonteCarloCollisions` does not
            take such processes.
        statistical_weight_ratio: The ratio ``g_upper / g_lower`` of the
            statistical weights of the excited and the initial state, for
            ``upper_state``.
    """

    kind: str
    background: str
    cross_section: CrossSection
    name: str = ""
    energy_loss: float | None = None
    energy_frame: str = "center_of_mass"
    products: Mapping[str, str] = field(default_factory=dict)
    energy_sharing: str = "equal"
    sharing_energy: float | None = None
    scattering: str = "isotropic"
    charge_change: int = 1
    upper_state: str | None = None
    statistical_weight_ratio: float = 1.0

    def __post_init__(self) -> None:
        """Validate the process definition."""
        if self.kind not in PROCESS_KINDS:
            raise ValueError(
                f"Unknown process kind {self.kind!r}; expected one of {PROCESS_KINDS}"
            )
        if self.energy_frame not in ENERGY_FRAMES:
            raise ValueError(f"energy_frame must be one of {ENERGY_FRAMES}")
        if self.energy_sharing not in ENERGY_SHARING:
            raise ValueError(f"energy_sharing must be one of {ENERGY_SHARING}")
        if self.energy_sharing == "opal" and not (
            self.sharing_energy is not None and self.sharing_energy > 0.0
        ):
            raise ValueError('energy_sharing="opal" needs sharing_energy > 0')
        if self.scattering not in SCATTERING_MODELS:
            raise ValueError(f"scattering must be one of {SCATTERING_MODELS}")
        missing = set(PRODUCT_ROLES[self.kind]) - set(self.products)
        if missing:
            raise ValueError(
                f"{self.kind} process needs product species for {sorted(missing)}"
            )
        unknown = (
            set(self.products)
            - set(PRODUCT_ROLES[self.kind])
            - set(OPTIONAL_PRODUCT_ROLES.get(self.kind, ()))
        )
        if unknown:
            raise ValueError(
                f"{self.kind} process has no product role(s) {sorted(unknown)}"
            )
        if self.kind == "stripping":
            if int(self.charge_change) != self.charge_change or not self.charge_change:
                raise ValueError("stripping needs a non-zero integer charge_change")
            if self.charge_change < 0 and "electron" in self.products:
                raise ValueError(
                    "electron capture (charge_change < 0) releases no electron"
                )
            object.__setattr__(self, "charge_change", int(self.charge_change))
        if (
            self.kind == "beam_ionization"
            and "electron" in self.products
            and not (self.sharing_energy and self.sharing_energy > 0.0)
        ):
            raise ValueError(
                "beam_ionization with an electron product needs sharing_energy > 0"
            )
        if self.energy_loss is not None and self.energy_loss < 0.0:
            raise ValueError("energy_loss must be >= 0")
        if self.upper_state is not None and self.kind != "excitation":
            raise ValueError(
                "upper_state (superelastic collisions) needs an excitation"
            )
        if not self.statistical_weight_ratio > 0.0:
            raise ValueError("statistical_weight_ratio must be > 0")
        object.__setattr__(self, "products", dict(self.products))
        if not self.name:
            object.__setattr__(self, "name", f"{self.kind}:{self.background}")

    @property
    def is_ionization(self) -> bool:
        """Return whether the process creates an electron-ion pair."""
        return self.kind in _IONIZATION_KINDS

    @property
    def consumes_neutral(self) -> bool:
        """Return whether each collision removes one particle of the background.

        True for the ionizing, attaching and dissociating kinds, whose target
        neutral becomes an ion, a negative ion or fragments; these are the
        collisions that deplete a gas. Charge transfer turns a neutral into an
        ion and an ion into a neutral, so it leaves the count unchanged.
        """
        return self.kind in _CONSUMING_KINDS

    @property
    def loss(self) -> float:
        """Return the energy lost in the collision in eV."""
        if self.kind in _NO_ENERGY_LOSS:
            return 0.0
        if self.energy_loss is not None:
            return self.energy_loss
        return self.cross_section.threshold
