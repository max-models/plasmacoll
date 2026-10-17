"""collide_species with a subset of incident species, per-species dt and strided views.

The cross sections are synthetic, not physical data.
"""

from __future__ import annotations

import numpy as np
import pytest

from plasmacoll import CollisionProcess, MonteCarloCollisions, NeutralBackground
from plasmacoll.markers import ParticleArrays
from plasmacoll.mcc import _max_squared_speed
from plasmacoll.tests.unit._helpers import HEAVY, M_E, M_P, beam, constant


def test_the_largest_speed_of_a_transposed_view_matches_a_copy() -> None:
    rng = np.random.default_rng(0)
    storage = rng.normal(size=(3, 1000))
    view = storage[:, :700].T  # component-major storage, marker-major view
    assert not view.flags.c_contiguous
    assert _max_squared_speed(view) == pytest.approx(
        _max_squared_speed(np.ascontiguousarray(view))
    )
    full = storage.T  # the whole storage: the transpose is contiguous
    assert not full.flags.c_contiguous
    assert _max_squared_speed(full) == pytest.approx(
        _max_squared_speed(np.ascontiguousarray(full))
    )
    assert _max_squared_speed(np.zeros((0, 3))) == 0.0


def _operator() -> MonteCarloCollisions:
    process = CollisionProcess("elastic", "gas", constant(1.0e-18))
    return MonteCarloCollisions(
        species_masses={"e": M_E, "p": M_P},
        backgrounds=[NeutralBackground("gas", 1.0e20, 0.0, HEAVY)],
        processes={"e": [process], "p": [process]},
        seed=3,
    )


def _markers(num, energy, mass):
    return ParticleArrays(
        positions=np.zeros((num, 1)), velocities=beam(num, energy, mass),
        weights=np.ones(num),
    )  # fmt: skip


def test_only_the_incident_species_collide() -> None:
    mcc = _operator()
    species = {"e": _markers(1000, 10.0, M_E), "p": _markers(1000, 10.0, M_P)}
    before = species["p"].velocities.copy()
    diagnostics = mcc.collide_species(species, dt=1e-8, incident=["e"])
    assert set(diagnostics) == {"e"}
    assert diagnostics["e"].real_collisions > 0
    assert np.array_equal(species["p"].velocities, before)


def test_each_species_can_have_its_own_time_step() -> None:
    mcc = _operator()
    species = {"e": _markers(20_000, 10.0, M_E), "p": _markers(20_000, 10.0, M_P)}
    diagnostics = mcc.collide_species(species, dt={"e": 1e-10, "p": 4e-8})
    assert (
        diagnostics["e"].collision_probability < diagnostics["p"].collision_probability
    )
