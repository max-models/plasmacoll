"""Tests for the built-in low-temperature plasma collision catalogue."""

from __future__ import annotations

import pytest

from plasmacoll import CrossSection
from plasmacoll.catalogue import (
    CollisionCatalogueEntry,
    get_collision_catalogue,
    get_collision_catalogue_entry,
    list_collision_catalogue_names,
)


def test_catalogue_contains_typical_low_temperature_plasma_processes() -> None:
    """The starter catalogue includes representative plasma collision classes."""
    names = list_collision_catalogue_names()

    assert "e_Ar_elastic" in names
    assert "e_Ar_ionisation" in names
    assert "e_O2_attachment" in names
    assert "Arp_Ar_charge_exchange" in names
    assert "Hm_H_detachment" in names


def test_catalogue_entries_expose_reference_metadata() -> None:
    """Each catalogue entry exposes lookup metadata and a reference cross section."""
    entry = get_collision_catalogue_entry("Hp_H_charge_exchange")

    assert isinstance(entry, CollisionCatalogueEntry)
    assert entry.incident_species == "H+"
    assert entry.background_species == "H"
    assert entry.category == "ion-neutral"
    assert entry.process == "charge-exchange"
    assert entry.reference_cross_section > 0.0
    assert entry.reference_energy > 0.0
    assert "Janev" in entry.source


def test_catalogue_lookup_reports_available_names_for_unknown_entries() -> None:
    """Unknown keys raise a helpful message that lists the built-in options."""
    with pytest.raises(KeyError, match="Unknown collision catalogue entry"):
        get_collision_catalogue_entry("not-a-real-collision")

    assert len(get_collision_catalogue()) == len(list_collision_catalogue_names())


def test_catalogue_entries_include_plotting_curves() -> None:
    """Representative plotting curves are aligned and strictly positive."""
    for entry in get_collision_catalogue():
        energies, cross_sections = entry.curve_points()
        assert len(energies) == len(cross_sections)
        assert len(energies) >= 2
        assert all(energy > 0.0 for energy in energies)
        assert all(cross_section > 0.0 for cross_section in cross_sections)
        assert list(energies) == sorted(energies)


def test_catalogue_entries_give_cross_sections() -> None:
    """Each curve is a `CrossSection` through its points, zero below the threshold."""
    for entry in get_collision_catalogue():
        cross_section = entry.cross_section()
        assert isinstance(cross_section, CrossSection)
        energies, sigmas = entry.curve_points()
        for energy, sigma in zip(energies, sigmas, strict=True):
            if energy >= entry.threshold_energy:
                assert float(cross_section(energy)) == pytest.approx(sigma, rel=1e-12)
    ionisation = get_collision_catalogue_entry("e_Ar_ionisation").cross_section()
    assert float(ionisation(1.0)) == 0.0


def test_mismatched_curves_are_refused() -> None:
    """An entry whose energies and cross sections differ in length raises."""
    entry = CollisionCatalogueEntry(
        "x", "e-", "Ar", "electron-neutral", "elastic", "", 1.0, 1.0,
        energy_grid=(1.0, 2.0), cross_section_curve=(1.0,),
    )  # fmt: skip
    with pytest.raises(ValueError, match="mismatched"):
        entry.curve_points()
