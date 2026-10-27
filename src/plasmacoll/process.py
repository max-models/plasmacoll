"""The definition of one collision process of a charged species with a neutral gas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

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

#: The collision kinds :class:`~plasmacoll.mcc.MonteCarloCollisions` implements.
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
}
#: The product species a kind may have. ``neutral`` is the fast neutral of a
#: charge exchange and ``fragment`` the neutral fragment of a dissociation; both
#: are dropped (not tracked) unless a species is given for them. ``electron``
#: and ``ion`` of ``stripping`` are the released electrons and the target ion.
OPTIONAL_PRODUCT_ROLES: dict[str, tuple[str, ...]] = {
    "backscatter": ("neutral",),
    "charge_transfer": ("ion", "neutral"),
    "dissociation": ("fragment",),
    "dissociative_attachment": ("fragment",),
    "dissociative_ionization": ("fragment",),
    "stripping": ("electron", "ion"),
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
)
_IONIZATION_KINDS = ("ionization", "dissociative_ionization")


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
        kind: One of :data:`PROCESS_KINDS`.
        background: The name of the :class:`~plasmacoll.background.NeutralBackground`.
        cross_section: The cross section against energy.
        name: A unique name for diagnostics; ``"<kind>:<background>"`` if empty.
        energy_loss: Energy in eV lost by excitation, ionization and detachment;
            the threshold of the cross section if None.
        energy_frame: One of :data:`ENERGY_FRAMES`.
        products: Product species by role, see :data:`PRODUCT_ROLES`, e.g.
            ``{"electron": "e", "ion": "Ar+"}`` for ionization.
        energy_sharing: One of :data:`ENERGY_SHARING`, for ionization.
            ``"opal"`` samples the secondary electron's energy from the
            Opal-Peterson-Beaty distribution ``1 / (1 + (E_s / w)^2)``.
        sharing_energy: The parameter ``w`` in eV of ``"opal"`` sharing, of
            order the ionization energy (e.g. 10.0 for argon, 8.3 for H2).
        scattering: One of :data:`SCATTERING_MODELS`.
        charge_change: For ``stripping``, the change of the projectile's charge
            in e: 1 for single stripping (``H- -> H``, ``H -> H+``), 2 for
            double stripping (``H- -> H+``), -1 for electron capture
            (``H+ -> H``). A positive change releases that many electrons.
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
        if self.energy_loss is not None and self.energy_loss < 0.0:
            raise ValueError("energy_loss must be >= 0")
        object.__setattr__(self, "products", dict(self.products))
        if not self.name:
            object.__setattr__(self, "name", f"{self.kind}:{self.background}")

    @property
    def is_ionization(self) -> bool:
        """Return whether the process creates an electron-ion pair."""
        return self.kind in _IONIZATION_KINDS

    @property
    def loss(self) -> float:
        """Return the energy lost in the collision in eV."""
        if self.kind in _NO_ENERGY_LOSS:
            return 0.0
        if self.energy_loss is not None:
            return self.energy_loss
        return self.cross_section.threshold
