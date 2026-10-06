"""Tests for the prescribed H2 vibrational-population model."""

from __future__ import annotations

import math

import cunumpy as xp
import pytest

from plasmacoll import (
    CollisionProcess,
    CrossSection,
    MonteCarloCollisions,
    NeutralBackground,
    ParticleArrays,
)
from plasmacoll.constants import kelvin_to_ev
from plasmacoll.vibrational import (
    H2_GROUND_STATE_VIBRATIONAL_ENERGY_EV,
    VibrationalDistribution,
    boltzmann_level_populations,
    estimate_electronegativity,
)

from ._helpers import M_E, M_P

M_H2 = 2 * M_P


def test_h2_energy_table_matches_known_dissociation_energy() -> None:
    """The tabulated ladder reproduces H2's well-known D0 ~ 4.478 eV.

    E(v=14) is the well depth minus the (unbound) energy above the last bound
    level; D0 = E(vmax) - E(0) should land close to the textbook value as a
    sanity check on the transcribed table (Fantz and Wünderlich, At. Data
    Nucl. Data Tables 92, 853 (2006), Table 4.1).
    """
    energies = H2_GROUND_STATE_VIBRATIONAL_ENERGY_EV
    assert len(energies) == 15
    assert tuple(sorted(energies)) == energies  # monotonically increasing
    d0 = energies[-1] - energies[0]
    assert d0 == pytest.approx(4.455, abs=0.05)


def test_boltzmann_populations_two_level_ratio() -> None:
    """A two-level system reduces to the textbook ratio exp(-dE/T)."""
    populations = boltzmann_level_populations(xp.asarray([0.0, 1.0]), 0.5)
    ratio = float(populations[1] / populations[0])
    assert ratio == pytest.approx(math.exp(-2.0))
    assert float(xp.sum(populations)) == pytest.approx(1.0)


def test_boltzmann_populations_invariant_to_energy_offset() -> None:
    """Shifting every level by a constant leaves the populations unchanged."""
    energies = xp.asarray([0.0, 0.3, 0.9, 2.0])
    baseline = boltzmann_level_populations(energies, 0.4)
    shifted = boltzmann_level_populations(energies + 5.0, 0.4)
    assert bool(xp.allclose(baseline, shifted))


def test_boltzmann_populations_limits() -> None:
    """High T approaches uniform; low T collapses onto the ground level."""
    energies = xp.asarray([0.0, 1.0, 2.0])
    hot = boltzmann_level_populations(energies, 1.0e4)
    assert bool(xp.allclose(hot, 1.0 / 3.0, atol=1e-3))

    cold = boltzmann_level_populations(energies, 1.0e-3)
    assert float(cold[0]) == pytest.approx(1.0, abs=1e-10)
    assert float(xp.sum(cold[1:])) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize(
    ("bad_energies", "temperature", "message"),
    [([], 1.0, "non-empty"), ([0.0, 1.0], 0.0, "> 0"), ([0.0, 1.0], -1.0, "> 0")],
)
def test_boltzmann_populations_rejects_invalid_input(
    bad_energies: list[float], temperature: float, message: str
) -> None:
    """Empty ladders and non-positive temperatures are rejected."""
    with pytest.raises(ValueError, match=message):
        boltzmann_level_populations(xp.asarray(bad_energies), temperature)


def test_h2_ground_state_from_kelvin_matches_from_ev() -> None:
    """The kelvin constructor is exactly the eV constructor after conversion."""
    temperature_k = 3000.0
    from_kelvin = VibrationalDistribution.h2_ground_state_from_kelvin(temperature_k)
    from_ev = VibrationalDistribution.h2_ground_state(kelvin_to_ev(temperature_k))
    assert bool(xp.allclose(from_kelvin.populations, from_ev.populations))


def test_higher_vibrational_temperature_increases_high_v_population() -> None:
    """The v>=4 fraction rises monotonically with T_vib, as DA production must.

    Krištof et al. (2019), citing Kalache et al. (2004), report negligible
    negative-ion concentration below T_vib ~ 3000 K and an exponential rise
    above it; this checks the same qualitative, monotonic trend.
    """
    fractions = [
        VibrationalDistribution.h2_ground_state_from_kelvin(t).fraction_at_least(4)
        for t in (1500.0, 2000.0, 3000.0, 4000.0, 6000.0, 8000.0)
    ]
    assert all(a < b for a, b in zip(fractions, fractions[1:], strict=False))
    # Negligible well below the reported ~3000 K onset, and no longer
    # negligible well above it.
    assert fractions[0] < 1.0e-6
    assert fractions[-1] > 0.01


def test_mean_level_increases_with_temperature() -> None:
    """The population-weighted mean vibrational level rises with T_vib."""
    cold = VibrationalDistribution.h2_ground_state_from_kelvin(1000.0)
    hot = VibrationalDistribution.h2_ground_state_from_kelvin(8000.0)
    assert cold.mean_level() < hot.mean_level()
    assert cold.mean_level() >= 0.0


def test_fraction_at_least_rejects_out_of_range_level() -> None:
    """v_min outside the tabulated ladder is a configuration error."""
    distribution = VibrationalDistribution.h2_ground_state(0.3)
    with pytest.raises(ValueError, match="v_min"):
        distribution.fraction_at_least(distribution.num_levels)
    with pytest.raises(ValueError, match="v_min"):
        distribution.fraction_at_least(-1)


def test_estimate_electronegativity_matches_definition() -> None:
    """alpha = rate_ratio * [H2(v>=v_min)] / [H], eq. (11) of Krištof et al."""
    distribution = VibrationalDistribution.h2_ground_state_from_kelvin(3500.0)
    hydrogen_density = 1.0e19
    molecule_density = 2.0e20
    rate_ratio = 8.0

    alpha = estimate_electronegativity(
        distribution,
        hydrogen_atom_density=hydrogen_density,
        molecule_density=molecule_density,
        rate_ratio=rate_ratio,
    )

    expected = (
        rate_ratio
        * distribution.fraction_at_least(4)
        * molecule_density
        / hydrogen_density
    )
    assert alpha == pytest.approx(expected)


def test_estimate_electronegativity_rejects_invalid_densities() -> None:
    """Non-physical densities are rejected rather than silently misused."""
    distribution = VibrationalDistribution.h2_ground_state(0.3)
    with pytest.raises(ValueError, match="hydrogen_atom_density"):
        estimate_electronegativity(distribution, 0.0, 1.0e20)
    with pytest.raises(ValueError, match="molecule_density"):
        estimate_electronegativity(distribution, 1.0e19, -1.0)


def test_vibrational_mixture_matches_manual_weighted_sum() -> None:
    """The mixture cross section is the population-weighted sum of the levels."""
    v0 = CrossSection(energy=xp.asarray([0.0, 10.0]), sigma=xp.asarray([1.0, 1.0]))
    v1 = CrossSection(energy=xp.asarray([0.0, 10.0]), sigma=xp.asarray([100.0, 100.0]))
    v2 = CrossSection(
        energy=xp.asarray([0.0, 10.0]), sigma=xp.asarray([1000.0, 1000.0])
    )
    weights = [0.97, 0.02, 0.01]

    mixture = CrossSection.vibrational_mixture([v0, v1, v2], weights)

    expected = sum(
        w * level(3.0) for w, level in zip(weights, [v0, v1, v2], strict=True)
    )
    assert float(mixture(3.0)) == pytest.approx(float(expected))


def test_vibrational_mixture_respects_per_level_thresholds() -> None:
    """A level's own threshold still applies inside the mixture."""
    below_threshold = CrossSection(
        energy=xp.asarray([0.0, 10.0]), sigma=xp.asarray([1.0, 1.0]), threshold=5.0
    )
    always_on = CrossSection(
        energy=xp.asarray([0.0, 10.0]), sigma=xp.asarray([1.0, 1.0])
    )

    mixture = CrossSection.vibrational_mixture([below_threshold, always_on], [0.5, 0.5])

    assert float(mixture(2.0)) == pytest.approx(0.5, abs=1e-6)  # only always_on
    assert float(mixture(6.0)) == pytest.approx(1.0, abs=1e-6)  # both contribute
    # A grid point pinned right at the threshold keeps the step sharp instead
    # of blurring it into a ramp spanning the levels' own (0, 10) endpoints.
    assert float(mixture(4.999)) == pytest.approx(0.5, abs=1e-4)
    assert float(mixture(5.0)) == pytest.approx(1.0, abs=1e-6)


def test_vibrational_mixture_normalizes_unnormalized_weights() -> None:
    """Populations need not already sum to 1."""
    v0 = CrossSection(energy=xp.asarray([0.0, 10.0]), sigma=xp.asarray([2.0, 2.0]))
    v1 = CrossSection(energy=xp.asarray([0.0, 10.0]), sigma=xp.asarray([4.0, 4.0]))

    normalized = CrossSection.vibrational_mixture([v0, v1], [1.0, 1.0])
    unnormalized = CrossSection.vibrational_mixture([v0, v1], [10.0, 10.0])

    assert float(normalized(5.0)) == pytest.approx(float(unnormalized(5.0)))
    assert float(normalized(5.0)) == pytest.approx(3.0)  # equal weights -> mean


def test_vibrational_mixture_with_real_h2_populations() -> None:
    """The mixture utility composes end to end with VibrationalDistribution."""
    distribution = VibrationalDistribution.h2_ground_state_from_kelvin(5000.0)
    levels = [
        CrossSection(
            energy=xp.asarray([0.0, 20.0]),
            sigma=xp.asarray([float(v) * 1e-21, float(v) * 1e-21]),
        )
        for v in range(distribution.num_levels)
    ]

    mixture = CrossSection.vibrational_mixture(levels, distribution.populations)

    assert float(mixture(10.0)) == pytest.approx(
        sum(float(w) * v * 1e-21 for v, w in enumerate(distribution.populations))
    )


def test_vibrational_mixture_rejects_mismatched_lengths() -> None:
    """A weight per level is mandatory."""
    v0 = CrossSection(energy=xp.asarray([0.0, 10.0]), sigma=xp.asarray([1.0, 1.0]))
    with pytest.raises(ValueError, match="equal length"):
        CrossSection.vibrational_mixture([v0], [0.5, 0.5])


def test_vibrational_mixture_rejects_empty_input() -> None:
    """At least one level is required."""
    with pytest.raises(ValueError, match="not be empty"):
        CrossSection.vibrational_mixture([], [])


def test_vibrational_mixture_rejects_non_positive_total_weight() -> None:
    """All-zero (or negative) weights cannot be normalized."""
    v0 = CrossSection(energy=xp.asarray([0.0, 10.0]), sigma=xp.asarray([1.0, 1.0]))
    with pytest.raises(ValueError, match="positive"):
        CrossSection.vibrational_mixture([v0], [0.0])


def test_temperature_must_be_positive() -> None:
    with pytest.raises(ValueError, match="temperature_k"):
        VibrationalDistribution.h2_ground_state_from_kelvin(0.0)
    with pytest.raises(ValueError, match="temperature_ev"):
        VibrationalDistribution.h2_ground_state(-1.0)
    with pytest.raises(ValueError, match="energies_ev"):
        VibrationalDistribution(energies_ev=(), temperature_ev=1.0)


def test_estimate_electronegativity_rejects_negative_rate_ratio() -> None:
    distribution = VibrationalDistribution.h2_ground_state(0.3)
    with pytest.raises(ValueError, match="rate_ratio"):
        estimate_electronegativity(distribution, 1.0e19, 1.0e20, rate_ratio=-1.0)


def test_vibrational_mixture_rejects_negative_populations() -> None:
    v0 = CrossSection(energy=[0.0, 10.0], sigma=[1.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        CrossSection.vibrational_mixture([v0, v0], [1.0, -0.5])


def test_mixture_attachment_runs_through_the_operator() -> None:
    """A level-resolved (synthetic) attachment set collides as one process."""
    distribution = VibrationalDistribution.h2_ground_state_from_kelvin(6000.0)
    levels = [
        CrossSection.constant(1.0e-24 * 2.0**v, label=f"v={v}")
        for v in range(distribution.num_levels)
    ]
    mixture = CrossSection.vibrational_mixture(levels, distribution.populations)
    mcc = MonteCarloCollisions(
        species_masses={"e": M_E, "H-": M_H2},
        backgrounds=[NeutralBackground("H2", 1.0e22, 300.0, M_H2)],
        processes={
            "e": [
                CollisionProcess(
                    "attachment",
                    "H2",
                    mixture,
                    name="da",
                    products={"negative_ion": "H-"},
                )
            ]
        },
        seed=0,
    )
    electrons = ParticleArrays.maxwellian(5000, M_E, 2.0e4, seed=1)
    negative_ions = ParticleArrays.empty()
    diagnostics = mcc.collide_species({"e": electrons, "H-": negative_ions}, dt=1.0e-7)
    assert diagnostics["e"].counts["da"] == len(negative_ions) > 0
    assert len(electrons) + len(negative_ions) == 5000
