"""Splitting and merging markers conserves what it must."""

from __future__ import annotations

import cunumpy as xp
import numpy as np
import pytest

from plasmacoll import ParticleArrays, merge_markers, split_markers

from ._helpers import M_HE


def _host(array) -> np.ndarray:
    return np.asarray(xp.to_numpy(array))


def _moments(markers: ParticleArrays) -> tuple[float, np.ndarray, float]:
    weights = _host(markers.weights)
    velocities = _host(markers.velocities)
    return (
        weights.sum(),
        (weights[:, None] * velocities).sum(axis=0),
        (weights * (velocities**2).sum(axis=1)).sum(),
    )


def _markers(num: int, ndim: int = 2, seed: int = 0) -> ParticleArrays:
    rng = np.random.default_rng(seed)
    return ParticleArrays(
        xp.asarray(rng.uniform(0.0, 1.0, (num, ndim))),
        xp.asarray(rng.normal(0.0, 1.0e3, (num, 3)) + [500.0, 0.0, 0.0]),
        xp.asarray(rng.uniform(0.5, 2.0, num)),
    )


@pytest.mark.parametrize("cell_size", [None, 0.25, (0.5, 0.25)])
def test_merging_conserves_weight_momentum_and_energy(cell_size) -> None:
    markers = _markers(5000)
    before = _moments(markers)
    removed = merge_markers(markers, 500, cell_size=cell_size, seed=1)
    after = _moments(markers)
    assert removed == 5000 - len(markers)
    assert len(markers) < 1000
    for old, new in zip(before, after, strict=True):
        np.testing.assert_allclose(new, old, rtol=1e-9, atol=1e-6)


def test_merged_markers_stay_in_their_cell() -> None:
    markers = _markers(4000, ndim=1)
    cells_before = set(np.floor(_host(markers.positions)[:, 0] / 0.25).astype(int))
    merge_markers(markers, 400, cell_size=0.25, rng=np.random.default_rng(2))
    cells_after = set(np.floor(_host(markers.positions)[:, 0] / 0.25).astype(int))
    assert cells_after <= cells_before


def test_merging_keeps_the_temperature_of_a_maxwellian() -> None:
    markers = ParticleArrays.maxwellian(20_000, M_HE, 1000.0, seed=3)
    merge_markers(markers, 2000, seed=4)
    assert markers.temperature(M_HE) == pytest.approx(1000.0, rel=0.03)


def test_merging_does_nothing_below_the_target_or_without_groups() -> None:
    markers = _markers(100)
    assert merge_markers(markers, 200) == 0
    assert len(markers) == 100
    # One marker per cell: no group reaches three markers.
    sparse = ParticleArrays(
        xp.asarray(np.arange(10.0)[:, None]), xp.ones((10, 3)), xp.ones(10)
    )
    assert merge_markers(sparse, 2, cell_size=1.0) == 0
    assert len(sparse) == 10


def test_merging_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="target"):
        merge_markers(_markers(10), 1)
    with pytest.raises(ValueError, match="cell_size"):
        merge_markers(_markers(10), 2, cell_size=0.0)


def test_splitting_divides_heavy_markers() -> None:
    markers = ParticleArrays(
        xp.zeros((3, 1)),
        xp.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
        xp.asarray([0.5, 2.5, 1.0]),
    )
    before = _moments(markers)
    added = split_markers(markers, 1.0)
    assert added == 2
    assert len(markers) == 5
    assert _host(markers.weights).max() <= 1.0
    for old, new in zip(before, _moments(markers), strict=True):
        np.testing.assert_allclose(new, old)
    assert split_markers(markers, 1.0) == 0
    with pytest.raises(ValueError, match="max_weight"):
        split_markers(markers, 0.0)
