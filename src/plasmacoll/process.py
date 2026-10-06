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
}
#: The product species a kind may have.
OPTIONAL_PRODUCT_ROLES: dict[str, tuple[str, ...]] = {"charge_transfer": ("ion",)}
#: The frames whose kinetic energy the cross section is looked up at.
ENERGY_FRAMES = ("center_of_mass", "lab")
#: How ionization shares the residual energy between the two electrons.
ENERGY_SHARING = ("equal", "uniform")

_NO_ENERGY_LOSS = ("elastic", "backscatter", "attachment", "charge_transfer")


@dataclass(frozen=True)
class CollisionProcess:
    """One collision process of an incident species with a neutral background.

    ``energy_frame`` selects the energy used to look up the cross section: the
    centre-of-mass energy ``mu g^2 / 2`` or the incident particle's energy in
    the neutral's rest frame ``m g^2 / 2`` (the convention of electron
    cross-section compilations; both agree to O(m_e/M) for electrons), where
    ``g`` is the relative speed and ``mu`` the reduced mass.

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
    """

    kind: str
    background: str
    cross_section: CrossSection
    name: str = ""
    energy_loss: float | None = None
    energy_frame: str = "center_of_mass"
    products: Mapping[str, str] = field(default_factory=dict)
    energy_sharing: str = "equal"

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
        if self.energy_loss is not None and self.energy_loss < 0.0:
            raise ValueError("energy_loss must be >= 0")
        object.__setattr__(self, "products", dict(self.products))
        if not self.name:
            object.__setattr__(self, "name", f"{self.kind}:{self.background}")

    @property
    def loss(self) -> float:
        """Return the energy lost in the collision in eV."""
        if self.kind in _NO_ENERGY_LOSS:
            return 0.0
        if self.energy_loss is not None:
            return self.energy_loss
        return self.cross_section.threshold
