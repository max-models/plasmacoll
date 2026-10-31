"""Event-driven collisions: the inverse mean free path and a collision for certain."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import (
    CollisionProcess,
    DensityProfile,
    MonteCarloCollisions,
    NeutralBackground,
)

from ._helpers import M_P, constant

SPEED = 1.0e6


def _operator(profile: DensityProfile | None = None) -> MonteCarloCollisions:
    """H+ on H2 with charge transfer (3e-20 m^2) and elastic scattering (1e-20 m^2)."""
    return MonteCarloCollisions(
        species_masses={"H+": M_P, "H": M_P},
        backgrounds=[NeutralBackground("H2", 1.0e19, 0.0, 2 * M_P, profile=profile)],
        processes={
            "H+": [
                CollisionProcess(
                    "charge_transfer",
                    "H2",
                    constant(3.0e-20),
                    products={"neutral": "H"},
                ),
                CollisionProcess("elastic", "H2", constant(1.0e-20)),
            ]
        },
        seed=4,
    )


def _beam(num: int) -> tuple:
    positions = xp.zeros((num, 1))
    velocities = xp.zeros((num, 3))
    velocities[:, 0] = SPEED
    return positions, velocities, xp.ones(num)


def test_the_attenuation_is_n_sigma_where_the_marker_is() -> None:
    profile = DensityProfile(axis=0, positions=[0.0, 1.0], factors=[0.0, 1.0])
    mcc = _operator(profile)
    positions = xp.asarray([[0.0], [0.5], [1.0]])
    velocities = xp.zeros((3, 3))
    velocities[:, 0] = SPEED
    np.testing.assert_allclose(
        xp.to_numpy(mcc.attenuation("H+", positions, velocities)),
        [0.0, 0.5e19 * 4e-20, 1e19 * 4e-20],
    )
    # at rest there is no free path to integrate; other species have none
    assert float(mcc.attenuation("H+", positions[:1], xp.zeros((1, 3)))[0]) == 0.0
    assert not xp.any(mcc.attenuation("H", positions, velocities))


def test_every_selected_marker_collides_in_proportion_to_sigma() -> None:
    mcc = _operator()
    num = 40_000
    positions, velocities, weights = _beam(num)
    selected = xp.arange(0, num, 2)
    result = mcc.collide_now("H+", positions, velocities, weights, selected)
    counts = result.diagnostics.counts
    assert counts["charge_transfer:H2"] + counts["elastic:H2"] == num // 2
    assert result.diagnostics.null_collisions == 0
    share = counts["charge_transfer:H2"] / (num // 2)
    assert share == pytest.approx(0.75, abs=4.0 * np.sqrt(0.75 * 0.25 / (num // 2)))
    removed = xp.to_numpy(result.removed)
    assert removed.sum() == counts["charge_transfer:H2"]
    assert not removed[1::2].any()
    np.testing.assert_array_equal(
        xp.to_numpy(result.velocities[1::2]), xp.to_numpy(velocities[1::2])
    )
    assert sum(len(b.weights) for b in result.created["H"]) == removed.sum()


def test_all_markers_collide_without_indices_and_in_place() -> None:
    mcc = _operator()
    positions, velocities, weights = _beam(100)
    result = mcc.collide_now("H+", positions, velocities, weights, in_place=True)
    assert result.velocities is velocities
    assert result.diagnostics.real_collisions == 100
    empty = mcc.collide_now(
        "H+", positions, velocities, weights, xp.zeros(0, dtype=int)
    )
    assert empty.diagnostics.candidates == 0
    nothing = mcc.collide_now("H", positions, velocities, weights)
    assert nothing.diagnostics.candidates == 0


def test_a_marker_with_nothing_to_hit_is_rejected() -> None:
    profile = DensityProfile(axis=0, positions=[0.0, 1.0], factors=[0.0, 1.0])
    mcc = _operator(profile)
    positions, velocities, weights = _beam(3)
    with pytest.raises(ValueError, match="no collision where it is"):
        mcc.collide_now("H+", positions, velocities, weights)
    with pytest.raises(ValueError, match="three velocity components"):
        mcc.collide_now("H+", positions + 1.0, velocities[:, :2], weights)
