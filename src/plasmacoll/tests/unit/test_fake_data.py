"""Tests for the fabricated placeholder H2 DA cross sections."""

from __future__ import annotations

import cunumpy as xp
import pytest

from plasmacoll import CollisionProcess, MonteCarloCollisions, NeutralBackground
from plasmacoll.constants import ELECTRON_MASS, kelvin_to_ev
from plasmacoll.fake_data import (
    FakeCrossSectionDataWarning,
    fake_h2_da_cross_section,
    fake_h2_da_level_cross_sections,
)
from plasmacoll.vibrational import H2_GROUND_STATE_VIBRATIONAL_ENERGY_EV


# --------------------------------------------------------------------- #
# Fake (fabricated) cross sections: pipeline-testing only.
# --------------------------------------------------------------------- #
def test_fake_level_cross_sections_warn_and_are_labelled() -> None:
    """Using the fake data warns loudly and every label says so."""
    with pytest.warns(FakeCrossSectionDataWarning, match="fabricated"):
        levels = fake_h2_da_level_cross_sections()

    assert len(levels) == len(H2_GROUND_STATE_VIBRATIONAL_ENERGY_EV)
    for level in levels:
        assert "FAKE" in level.label
        assert "not real data" in level.label


M_E = ELECTRON_MASS
M_H2 = 3.34e-27


def test_fake_level_cross_sections_warns_exactly_once_per_call() -> None:
    """One call, one warning -- no duplicate noise from internal reuse."""
    with pytest.warns(FakeCrossSectionDataWarning) as record:
        fake_h2_da_level_cross_sections()
    assert len(record) == 1


def test_fake_cross_section_warns_exactly_once_per_call() -> None:
    """The population-weighted convenience wrapper also warns exactly once."""
    with pytest.warns(FakeCrossSectionDataWarning) as record:
        fake_h2_da_cross_section(3000.0)
    assert len(record) == 1


def test_fake_level_cross_sections_peak_increases_with_v() -> None:
    """The one deliberately real qualitative feature: peak sigma rises with v.

    This is the only property of the fake data that mirrors real physics
    (Bardsley and Wadehra 1979; Wadehra 1984) -- everything else about it is
    arbitrary, so this is the only thing worth asserting on its shape.
    """
    with pytest.warns(FakeCrossSectionDataWarning):
        levels = fake_h2_da_level_cross_sections()

    peaks = [float(xp.max(level.sigma)) for level in levels]
    assert all(a < b for a, b in zip(peaks, peaks[1:], strict=False))
    assert peaks[-1] / peaks[0] > 1.0e4  # "orders of magnitude", as advertised


def test_fake_cross_section_is_population_weighted() -> None:
    """The mixture follows the vibrational population, not a fixed level.

    A colder T_vib puts fewer markers in high v, where the fake cross
    sections peak; the mixture's value there should fall accordingly.
    """
    with pytest.warns(FakeCrossSectionDataWarning):
        cold = fake_h2_da_cross_section(1500.0)
    with pytest.warns(FakeCrossSectionDataWarning):
        hot = fake_h2_da_cross_section(8000.0)

    high_v_peak_energy = 0.3  # matches level v=14's peak in the fake data
    assert float(hot(high_v_peak_energy)) > float(cold(high_v_peak_energy))


def test_fake_cross_section_feeds_a_working_collision_process() -> None:
    """End to end: fake data plugs into MonteCarloCollisions and runs a step.

    This checks the *pipeline* (loading, weighting, colliding, removing an
    electron, creating an H-) works, not that the resulting rate is physical.
    """
    with pytest.warns(FakeCrossSectionDataWarning):
        da_cross_section = fake_h2_da_cross_section(6000.0)

    operator = MonteCarloCollisions(
        species_masses={"e": M_E, "H-": M_H2},
        backgrounds=[NeutralBackground("H2", 1.0e21, kelvin_to_ev(300.0), M_H2)],
        processes={
            "e": [
                CollisionProcess(
                    "attachment",
                    "H2",
                    da_cross_section,
                    name="e attachment (FAKE data)",
                    products={"negative_ion": "H-"},
                )
            ]
        },
        seed=0,
    )

    num = 2000
    velocities = xp.zeros((num, 3))
    velocities[:, 0] = 1.0e5  # a slow beam, well inside the fake curve's support
    result = operator.collide(
        "e", xp.zeros((num, 1)), velocities, xp.ones(num), dt=1.0e-6
    )

    assert result.diagnostics.counts.get("e attachment (FAKE data)", 0) > 0
    assert "H-" in result.created
    ((positions, product_velocities, weights),) = result.created["H-"]
    assert product_velocities.shape[1] == 3
    assert weights.shape[0] == positions.shape[0]
    assert bool(xp.any(result.removed))  # attached electrons are consumed
