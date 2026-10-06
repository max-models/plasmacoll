"""The marker set protocol and ParticleArrays."""

from __future__ import annotations

import cunumpy as xp
import pytest

from pymcc import MarkerSet, ParticleArrays
from pymcc.constants import BOLTZMANN, ELEMENTARY_CHARGE

from ._helpers import M_HE


def test_particle_arrays_satisfy_the_protocol() -> None:
    assert isinstance(ParticleArrays.empty(), MarkerSet)


def test_empty_set() -> None:
    markers = ParticleArrays.empty(ndim=2)
    assert len(markers) == 0
    assert markers.positions.shape == (0, 2)
    assert markers.total_weight == 0.0
    assert markers.mean_energy_ev(M_HE) == 0.0
    assert markers.temperature(M_HE) == 0.0


@pytest.mark.parametrize(
    ("positions", "velocities", "weights", "message"),
    [
        (xp.zeros(3), xp.zeros((3, 3)), xp.ones(3), "positions"),
        (xp.zeros((3, 1)), xp.zeros((3, 2)), xp.ones(3), "velocities"),
        (xp.zeros((2, 1)), xp.zeros((3, 3)), xp.ones(3), "one row"),
        (xp.zeros((3, 1)), xp.zeros((3, 3)), xp.ones(2), "one row"),
    ],
)
def test_shapes_are_validated(positions, velocities, weights, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ParticleArrays(positions, velocities, weights)


def test_remove_and_add() -> None:
    markers = ParticleArrays(
        xp.arange(4.0)[:, None], xp.zeros((4, 3)), xp.arange(1.0, 5.0)
    )
    markers.remove(xp.asarray([True, False, True, False]))
    assert bool(xp.allclose(markers.positions[:, 0], [1.0, 3.0]))
    assert markers.total_weight == 6.0
    markers.add(xp.zeros((1, 1)), xp.ones((1, 3)), xp.full(1, 5.0))
    assert len(markers) == 3
    assert markers.total_weight == 11.0


def test_maxwellian_has_the_requested_moments() -> None:
    temperature, drift = 500.0, (300.0, -100.0, 0.0)
    markers = ParticleArrays.maxwellian(
        200_000, M_HE, temperature, weight=2.0, drift=drift, ndim=3, seed=3
    )
    assert markers.positions.shape == (200_000, 3)
    assert markers.total_weight == 400_000.0
    assert markers.temperature(M_HE) == pytest.approx(temperature, rel=0.01)
    mean = xp.mean(markers.velocities, axis=0)
    assert bool(xp.allclose(mean, xp.asarray(drift), atol=5.0))
    drift_energy = 0.5 * M_HE * sum(d**2 for d in drift)
    expected = (1.5 * BOLTZMANN * temperature + drift_energy) / ELEMENTARY_CHARGE
    assert markers.mean_energy_ev(M_HE) == pytest.approx(expected, rel=0.01)


def test_maxwellian_is_reproducible() -> None:
    first = ParticleArrays.maxwellian(10, M_HE, 300.0, seed=1)
    second = ParticleArrays.maxwellian(10, M_HE, 300.0, seed=1)
    assert bool(xp.array_equal(first.velocities, second.velocities))
    cold = ParticleArrays.maxwellian(10, M_HE, 0.0)
    assert bool(xp.all(cold.velocities == 0.0))


@pytest.mark.parametrize(
    ("num", "mass", "temperature", "message"),
    [(-1, 1.0, 1.0, "num"), (1, 0.0, 1.0, "mass"), (1, 1.0, -1.0, "temperature")],
)
def test_maxwellian_rejects_invalid_arguments(
    num: int, mass: float, temperature: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ParticleArrays.maxwellian(num, mass, temperature)
