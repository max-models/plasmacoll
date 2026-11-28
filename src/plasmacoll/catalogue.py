"""Representative cross sections of low-temperature plasma collisions.

A starter catalogue for quick estimates: one entry per process (electron-neutral,
ion-neutral, recombination, detachment), with a reference cross section at a reference
energy and a coarse curve for plots. The numbers are representative values from the
sources named in each entry, not evaluated data sets: use `plasmacoll.argon`,
`plasmacoll.read_lxcat` and the other process sets for physical modelling.

Example:
    >>> from plasmacoll.catalogue import get_collision_catalogue_entry
    >>> entry = get_collision_catalogue_entry("e_Ar_elastic")
    >>> entry.reference_cross_section
    2e-19
"""

from __future__ import annotations

from dataclasses import dataclass

from plasmacoll.cross_sections import CrossSection

__all__ = [
    "LOW_TEMPERATURE_PLASMA_COLLISION_CATALOGUE",
    "CollisionCatalogueEntry",
    "get_collision_catalogue",
    "get_collision_catalogue_entry",
    "list_collision_catalogue_names",
]


@dataclass(frozen=True)
class CollisionCatalogueEntry:
    """A representative collision process of the starter catalogue.

    Attributes:
        key: The name of the entry, e.g. ``"e_Ar_elastic"``.
        incident_species: The incident particle, e.g. ``"e-"``.
        background_species: The target, e.g. ``"Ar"``.
        category: ``"electron-neutral"``, ``"ion-neutral"``, ...
        process: The kind of process, e.g. ``"elastic"``.
        description: One sentence on the process.
        reference_cross_section: A representative cross section in m^2.
        reference_energy: The energy of ``reference_cross_section`` in eV.
        threshold_energy: The threshold in eV.
        source: Where the numbers come from.
        notes: Remarks on the value.
        energy_grid: Energies of the curve in eV.
        cross_section_curve: Cross sections of the curve in m^2.
    """

    key: str
    incident_species: str
    background_species: str
    category: str
    process: str
    description: str
    reference_cross_section: float
    reference_energy: float
    threshold_energy: float = 0.0
    source: str = ""
    notes: str = ""
    energy_grid: tuple[float, ...] = ()
    cross_section_curve: tuple[float, ...] = ()

    def curve_points(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return the energies (eV) and cross sections (m^2) of the curve.

        Raises:
            ValueError: If the two have different lengths.
        """
        if len(self.energy_grid) != len(self.cross_section_curve):
            raise ValueError(
                f"Collision catalogue entry {self.key!r} has mismatched curve data",
            )
        return self.energy_grid, self.cross_section_curve

    def cross_section(self) -> CrossSection:
        """Return the curve as a `CrossSection` (linear in between, with the threshold)."""
        energy, sigma = self.curve_points()
        return CrossSection(
            energy=list(energy),
            sigma=list(sigma),
            threshold=self.threshold_energy,
            label=f"{self.key} (representative)",
        )


LOW_TEMPERATURE_PLASMA_COLLISION_CATALOGUE: tuple[CollisionCatalogueEntry, ...] = (
    CollisionCatalogueEntry(
        key="e_Ar_elastic",
        incident_species="e-",
        background_species="Ar",
        category="electron-neutral",
        process="elastic",
        description="Electron-argon elastic momentum-transfer collision.",
        reference_cross_section=2.0e-19,
        reference_energy=1.0,
        source="LXCat Biagi/Phelps argon sets; Pitchford et al. (2017)",
        notes="Representative momentum-transfer cross section near 1 eV.",
        energy_grid=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
        cross_section_curve=(
            5.0e-19,
            3.2e-19,
            2.0e-19,
            1.4e-19,
            9.0e-20,
            6.5e-20,
            4.0e-20,
        ),
    ),
    CollisionCatalogueEntry(
        key="e_Ar_excitation",
        incident_species="e-",
        background_species="Ar",
        category="electron-neutral",
        process="excitation",
        description="Electron-impact excitation of argon to the 11.55 eV manifold.",
        reference_cross_section=3.0e-20,
        reference_energy=20.0,
        threshold_energy=11.55,
        source="LXCat argon excitation data; Lieberman & Lichtenberg (2005)",
        notes="Representative lumped excitation channel for low-temperature argon plasmas.",
        energy_grid=(11.55, 12.0, 15.0, 20.0, 30.0, 50.0, 100.0),
        cross_section_curve=(
            1.0e-23,
            5.0e-22,
            1.5e-20,
            3.0e-20,
            2.6e-20,
            1.8e-20,
            9.0e-21,
        ),
    ),
    CollisionCatalogueEntry(
        key="e_Ar_ionisation",
        incident_species="e-",
        background_species="Ar",
        category="electron-neutral",
        process="ionisation",
        description="Electron-impact ionisation of argon.",
        reference_cross_section=2.8e-20,
        reference_energy=100.0,
        threshold_energy=15.76,
        source="LXCat/Phelps argon ionisation; Kim & Rudd (1994)",
        notes="Representative total ionisation cross section near its broad maximum.",
        energy_grid=(15.76, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0),
        cross_section_curve=(
            1.0e-23,
            3.0e-21,
            1.0e-20,
            2.1e-20,
            2.8e-20,
            2.3e-20,
            1.4e-20,
        ),
    ),
    CollisionCatalogueEntry(
        key="e_O2_attachment",
        incident_species="e-",
        background_species="O2",
        category="electron-neutral",
        process="attachment",
        description="Low-energy electron attachment to molecular oxygen.",
        reference_cross_section=5.0e-20,
        reference_energy=0.2,
        source="LXCat oxygen attachment data; Christophorou & Olthoff (2004)",
        notes="Representative attachment resonance for electronegative oxygen plasmas.",
        energy_grid=(0.03, 0.08, 0.2, 0.5, 1.0, 2.0, 4.0),
        cross_section_curve=(
            8.0e-21,
            2.5e-20,
            5.0e-20,
            3.8e-20,
            1.8e-20,
            6.0e-21,
            1.0e-21,
        ),
    ),
    CollisionCatalogueEntry(
        key="e_O2_dissociative_attachment",
        incident_species="e-",
        background_species="O2",
        category="electron-neutral",
        process="dissociative-attachment",
        description="Dissociative electron attachment to O2 producing O⁻.",
        reference_cross_section=3.0e-20,
        reference_energy=6.5,
        source="LXCat oxygen dissociative attachment data; Rapp & Briglia (1965)",
        notes="Representative higher-energy attachment resonance in oxygen discharges.",
        energy_grid=(4.0, 5.0, 6.5, 8.0, 10.0, 15.0, 25.0),
        cross_section_curve=(
            2.0e-21,
            1.2e-20,
            3.0e-20,
            2.4e-20,
            1.3e-20,
            4.0e-21,
            8.0e-22,
        ),
    ),
    CollisionCatalogueEntry(
        key="Arp_Ar_elastic",
        incident_species="Ar+",
        background_species="Ar",
        category="ion-neutral",
        process="elastic",
        description="Argon ion momentum-transfer collision with neutral argon.",
        reference_cross_section=1.0e-18,
        reference_energy=1.0,
        source="Phelps (1994); LXCat argon ion-neutral sets",
        notes="Representative low-energy momentum-transfer cross section.",
        energy_grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
        cross_section_curve=(
            2.0e-18,
            1.6e-18,
            1.3e-18,
            1.0e-18,
            7.0e-19,
            4.8e-19,
            3.0e-19,
        ),
    ),
    CollisionCatalogueEntry(
        key="Arp_Ar_charge_exchange",
        incident_species="Ar+",
        background_species="Ar",
        category="ion-neutral",
        process="charge-exchange",
        description="Resonant charge exchange between Ar+ and Ar.",
        reference_cross_section=2.0e-18,
        reference_energy=1.0,
        source="Phelps (1994); LXCat argon ion-neutral sets",
        notes="Representative resonant charge-exchange cross section near 1 eV.",
        energy_grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
        cross_section_curve=(
            3.0e-18,
            2.8e-18,
            2.4e-18,
            2.0e-18,
            1.5e-18,
            1.1e-18,
            7.5e-19,
        ),
    ),
    CollisionCatalogueEntry(
        key="Hp_H_charge_exchange",
        incident_species="H+",
        background_species="H",
        category="ion-neutral",
        process="charge-exchange",
        description="Resonant charge exchange between H+ and H.",
        reference_cross_section=6.0e-19,
        reference_energy=1.0,
        source="Janev et al. (1987)",
        notes="Representative hydrogen resonant charge-exchange value for edge/source plasmas.",
        energy_grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
        cross_section_curve=(
            9.0e-19,
            8.0e-19,
            7.0e-19,
            6.0e-19,
            5.0e-19,
            3.6e-19,
            2.4e-19,
        ),
    ),
    CollisionCatalogueEntry(
        key="e_Arp_radiative_recombination",
        incident_species="e-",
        background_species="Ar+",
        category="electron-ion",
        process="radiative-recombination",
        description="Radiative recombination between an electron and Ar+ producing Ar*.",
        reference_cross_section=1.0e-21,
        reference_energy=1.0,
        source="OPEN-ADAS/Seaton radiative recombination fits",
        notes="Representative placeholder cross section for catalogue completeness.",
        energy_grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
        cross_section_curve=(
            8.0e-21,
            4.0e-21,
            2.0e-21,
            1.0e-21,
            4.0e-22,
            1.5e-22,
            6.0e-23,
        ),
    ),
    CollisionCatalogueEntry(
        key="Hm_H_detachment",
        incident_species="H-",
        background_species="H",
        category="negative-ion-neutral",
        process="detachment",
        description="Collisional detachment of H- by atomic hydrogen.",
        reference_cross_section=3.0e-19,
        reference_energy=1.0,
        source="Janev et al. (1987); Bacal et al. (2021)",
        notes="Representative H- loss process in low-temperature hydrogen plasmas.",
        energy_grid=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
        cross_section_curve=(
            9.0e-20,
            1.5e-19,
            2.4e-19,
            3.0e-19,
            3.4e-19,
            3.0e-19,
            2.2e-19,
        ),
    ),
)

_CATALOGUE_BY_KEY = {
    entry.key: entry for entry in LOW_TEMPERATURE_PLASMA_COLLISION_CATALOGUE
}


def get_collision_catalogue() -> tuple[CollisionCatalogueEntry, ...]:
    """Return all entries of the catalogue."""
    return LOW_TEMPERATURE_PLASMA_COLLISION_CATALOGUE


def list_collision_catalogue_names() -> list[str]:
    """Return the keys of the catalogue."""
    return list(_CATALOGUE_BY_KEY)


def get_collision_catalogue_entry(name: str) -> CollisionCatalogueEntry:
    """Return the entry ``name`` of the catalogue.

    Args:
        name: The key of the entry.

    Raises:
        KeyError: If there is no such entry; the message lists the keys.
    """
    try:
        return _CATALOGUE_BY_KEY[name]
    except KeyError as exc:
        available = ", ".join(list_collision_catalogue_names())
        raise KeyError(
            f"Unknown collision catalogue entry {name!r}. Available entries: {available}",
        ) from exc
