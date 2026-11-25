"""Fabricated placeholder H2 dissociative-attachment cross sections.

These exist only to exercise the vibrational-mixture pipeline (loading,
weighting, colliding, creating particles) before real level-resolved data is
sourced. They are not physical data and always warn.
"""

from __future__ import annotations

import warnings

import cunumpy as xp

from plasmacoll.cross_sections import CrossSection
from plasmacoll.vibrational import (
    H2_GROUND_STATE_VIBRATIONAL_ENERGY_EV,
    VibrationalDistribution,
)


class FakeCrossSectionDataWarning(UserWarning):
    """Warn that fabricated, non-physical cross-section data is in use.

    Applications should never filter it: a simulation result built on fake
    cross sections must not be mistaken for a physical one, including in a
    batch run where warnings would otherwise scroll past.
    """


def _warn_fake_cross_section_data() -> None:
    """Emit `FakeCrossSectionDataWarning` for the caller's caller."""
    warnings.warn(
        "fabricated cross-section data is in use for pipeline testing only "
        "-- it is not physical and must not be used for a result that will "
        "be reported or published.",
        FakeCrossSectionDataWarning,
        stacklevel=3,
    )


def _fake_h2_da_level_cross_sections_unwarned(num_levels: int) -> list[CrossSection]:
    """Build the fake level cross sections without warning; see the public function."""
    energy = xp.linspace(0.0, 8.0, 81)
    levels = []
    for v in range(num_levels):
        peak_energy = max(0.3, 3.0 - 0.18 * v)
        peak_sigma = 1.0e-23 * 10.0 ** (5.0 * v / (num_levels - 1))
        width = 1.0
        sigma = peak_sigma * xp.exp(-0.5 * ((energy - peak_energy) / width) ** 2)
        levels.append(
            CrossSection(
                energy=energy,
                sigma=sigma,
                label=f"FAKE placeholder DA cross section, H2(v={v}) -- not real data",
            )
        )
    return levels


def fake_h2_da_level_cross_sections(
    num_levels: int = len(H2_GROUND_STATE_VIBRATIONAL_ENERGY_EV),
) -> list[CrossSection]:
    """Return fabricated, illustrative per-level H2(v) DA cross sections.

    **These numbers are not physical data.** No verifiable, non-paywalled
    source for level-resolved e + H2(v) -> H- + H cross sections (Wadehra,
    Phys. Rev. A 29, 106 (1984); Fabrikant; Celiberto) was available to check
    against when this was written, and inventing precise-looking numbers
    would be worse than the gap it fills: it exists only to exercise the
    vibrational-mixture pipeline (loading, weighting, feeding a
    `plasmacoll.CollisionProcess`, running a step) end to
    end before real data is on hand.  It reproduces exactly one qualitative,
    well-established feature of the real physics -- the peak cross section
    rising by several orders of magnitude from v = 0 to the highest levels
    (Bardsley and Wadehra 1979; Wadehra 1984) -- and nothing else: not the
    resonance energies, not the linewidths, not the absolute magnitudes.

    Each level's curve is a single Gaussian in energy, peak position
    ``max(0.3, 3.0 - 0.18 * v)`` eV and peak height ``1e-23 * 10**(5 v / 14)``
    m^2 (v = 0 gives 1e-23 m^2, v = 14 gives 1e-18 m^2), width 1 eV -- an
    arbitrary, unfitted functional form chosen only to look like a resonance
    curve.  Replace this with real data via
    `plasmacoll.CrossSection.from_table` or
    `plasmacoll.read_lxcat` as soon as it is
    available; nothing else in the pipeline needs to change.
    """
    result = _fake_h2_da_level_cross_sections_unwarned(num_levels)
    _warn_fake_cross_section_data()
    return result


def fake_h2_da_cross_section(temperature_k: float) -> CrossSection:
    """Return a fabricated, population-weighted H2 DA cross section.

    Combines `fake_h2_da_level_cross_sections` with
    `VibrationalDistribution.h2_ground_state_from_kelvin` through
    `plasmacoll.CrossSection.vibrational_mixture`.
    **Not physical data** -- see `fake_h2_da_level_cross_sections`.
    """
    distribution = VibrationalDistribution.h2_ground_state_from_kelvin(temperature_k)
    levels = _fake_h2_da_level_cross_sections_unwarned(distribution.num_levels)
    _warn_fake_cross_section_data()
    return CrossSection.vibrational_mixture(
        levels,
        distribution.populations,
        label=f"FAKE placeholder H2 DA mixture at T_vib={temperature_k:.0f} K "
        "-- not real data",
    )
