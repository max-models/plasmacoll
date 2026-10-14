"""Backgrounds with a DynamicProfile: plasma species as collision targets.

The cross sections are synthetic, not physical data.
"""

from __future__ import annotations

import math

import cunumpy as xp
import pytest

from plasmacoll import (
    CollisionProcess,
    DynamicProfile,
    MonteCarloCollisions,
    NeutralBackground,
)
from plasmacoll.tests.unit._helpers import (
    HEAVY,
    M_E,
    M_P,
    beam,
    constant,
    speed_for,
)


def _target(name: str, profile: DynamicProfile, mass: float = HEAVY, temperature=0.0):
    # reference density 1 m^-3: the profile's factor is the density itself
    return NeutralBackground(name, 1.0, temperature, mass, profile=profile)


def _ones(num: int):
    return xp.ones(num)


def _line(num: int, length: float = 1.0):
    """Positions spread along x in [0, length), shape (num, 1)."""
    return (xp.arange(num, dtype=float) * (length / num))[:, None]


def test_the_profile_starts_empty_and_takes_updates() -> None:
    profile = DynamicProfile()
    assert profile.max_factor == 0.0
    assert bool(xp.all(profile(_line(3)) == 0.0))
    profile.update(lambda positions: xp.full(positions.shape[0], 2.0), 2.0)
    assert profile.max_factor == 2.0
    assert bool(xp.all(profile(_line(3)) == 2.0))
    profile.update(max_factor=5.0)  # the function is kept
    assert profile.max_factor == 5.0 and bool(xp.all(profile(_line(2)) == 2.0))
    with pytest.raises(ValueError):
        profile.update(max_factor=-1.0)
    with pytest.raises(ValueError):
        profile.update(max_factor=math.inf)


def test_the_bound_scales_with_the_current_density_without_retabulating() -> None:
    sigma, energy = 1.0e-19, 10.0
    profile = DynamicProfile()
    process = CollisionProcess("elastic", "plasma", constant(sigma))
    dynamic = MonteCarloCollisions(
        species_masses={"e": M_E},
        backgrounds=[_target("plasma", profile)],
        processes={"e": [process]},
        seed=1,
    )
    speed = speed_for(energy, M_E)
    assert dynamic.frequency_bound("e", speed) == 0.0  # no plasma yet
    for density in (1.0e18, 3.0e19):
        profile.update(lambda p, n=density: xp.full(p.shape[0], n), density)
        static = MonteCarloCollisions(
            species_masses={"e": M_E},
            backgrounds=[NeutralBackground("plasma", density, 0.0, HEAVY)],
            processes={"e": [process]},
            seed=1,
        )
        assert dynamic.frequency_bound("e", speed) == pytest.approx(
            static.frequency_bound("e", speed), rel=1e-12
        )


def test_nothing_collides_with_an_empty_target() -> None:
    mcc = MonteCarloCollisions(
        species_masses={"e": M_E},
        backgrounds=[_target("plasma", DynamicProfile())],
        processes={"e": [CollisionProcess("elastic", "plasma", constant(1e-19))]},
        seed=2,
    )
    num = 1000
    result = mcc.collide("e", _line(num), beam(num, 10.0, M_E), _ones(num), dt=1e-6)
    assert result.diagnostics.real_collisions == 0
    assert result.diagnostics.candidates == 0


def test_the_collision_rate_follows_the_target_density_where_each_marker_is() -> None:
    """A target twice as dense on the right half collides twice as often there."""
    sigma, energy, low = 1.0e-19, 10.0, 1.0e18

    def density(positions):
        return xp.where(positions[:, 0] < 0.5, low, 2.0 * low)

    profile = DynamicProfile(density, 2.0 * low)
    mcc = MonteCarloCollisions(
        species_masses={"e": M_E},
        backgrounds=[_target("plasma", profile)],
        processes={"e": [CollisionProcess("elastic", "plasma", constant(sigma))]},
        seed=3,
    )
    num = 400_000
    speed = speed_for(energy, M_E)
    dt = 0.05 / (2.0 * low * sigma * speed)
    velocities = beam(num, energy, M_E)
    result = mcc.collide("e", _line(num), velocities, _ones(num), dt)

    assert result.diagnostics.bound_violations == 0
    # elastic on an infinitely heavy target rotates the velocity: collided
    # markers left the x axis
    collided = xp.abs(result.velocities[:, 1]) + xp.abs(result.velocities[:, 2]) > 0
    left = float(xp.mean(collided[: num // 2]))
    right = float(xp.mean(collided[num // 2 :]))
    nu_max = mcc.frequency_bound("e", speed)
    for fraction, n in ((left, low), (right, 2.0 * low)):
        expected = (1.0 - math.exp(-nu_max * dt)) * n * sigma * speed / nu_max
        tolerance = 5.0 * math.sqrt(expected * (1.0 - expected) / (num // 2))
        assert fraction == pytest.approx(expected, abs=tolerance)


def test_electron_detachment_removes_h_minus_and_emits_electrons() -> None:
    """e + H- -> H + 2e: the electrons are the target, one electron per event is created."""
    electrons = DynamicProfile(lambda p: xp.full(p.shape[0], 1.0e18), 1.0e18)
    mcc = MonteCarloCollisions(
        species_masses={"H-": M_P, "e": M_E},
        backgrounds=[_target("electrons", electrons, mass=M_E, temperature=23_000.0)],
        processes={
            "H-": [
                CollisionProcess(
                    "detachment",
                    "electrons",
                    constant(1.0e-18, threshold=0.75),
                    products={"electron": "e"},
                )
            ]
        },
        seed=4,
    )
    num = 20_000
    result = mcc.collide("H-", _line(num), beam(num, 0.1, M_P), _ones(num), dt=1e-8)
    removed = int(xp.count_nonzero(result.removed))
    assert removed > 0
    assert removed == result.diagnostics.counts["detachment:electrons"]
    created = sum(int(new.velocities.shape[0]) for new in result.created["e"])
    assert created == removed


def test_mutual_neutralization_removes_both_species_at_the_same_rate() -> None:
    """H+ + H-: two charge transfers without products, each on the other's density."""
    n_plus, n_minus, sigma = 2.0e18, 1.0e18, 1.0e-17
    plus_density = DynamicProfile(lambda p: xp.full(p.shape[0], n_plus), n_plus)
    minus_density = DynamicProfile(lambda p: xp.full(p.shape[0], n_minus), n_minus)
    mcc = MonteCarloCollisions(
        species_masses={"H+": M_P, "H-": M_P},
        backgrounds=[
            _target("H+ density", plus_density, mass=M_P, temperature=2_000.0),
            _target("H- density", minus_density, mass=M_P, temperature=2_000.0),
        ],
        processes={
            "H-": [CollisionProcess("charge_transfer", "H+ density", constant(sigma))],
            "H+": [CollisionProcess("charge_transfer", "H- density", constant(sigma))],
        },
        seed=5,
    )
    # markers in proportion to the densities, equal weights: the number of
    # physical particles removed per unit volume must be equal on both sides
    num_minus = 100_000
    num_plus = 2 * num_minus
    dt = 1e-7
    minus = mcc.collide(
        "H-", _line(num_minus), beam(num_minus, 0.5, M_P), _ones(num_minus), dt
    )
    plus = mcc.collide(
        "H+", _line(num_plus), beam(num_plus, 0.5, M_P), _ones(num_plus), dt
    )
    lost_minus = int(xp.count_nonzero(minus.removed))
    lost_plus = int(xp.count_nonzero(plus.removed))
    assert lost_minus > 1000 and not minus.created and not plus.created
    tolerance = 5.0 * math.sqrt(lost_minus + lost_plus)
    assert lost_plus == pytest.approx(lost_minus, abs=tolerance)
