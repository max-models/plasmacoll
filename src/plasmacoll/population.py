"""Control of the number of markers: splitting heavy markers and merging light ones.

Collisions give products the weight of the incident marker, so ionization
doubles the number of electron markers every ionization time, and attachment
or charge transfer drains a species. :func:`merge_markers` reduces a species to
a target number of markers and :func:`split_markers` divides markers heavier
than a limit. Both work on any :class:`~plasmacoll.markers.MarkerSet`.

Merging groups markers that are close in phase space (the same spatial cell,
the same velocity octant and neighbouring speeds) and replaces every group of
three or more by two markers of half the group's weight with velocities
``u +- s n``, where ``u`` is the group's mean velocity, ``s`` its velocity
spread and ``n`` a random direction (the non-relativistic form of Vranic et
al., Comput. Phys. Commun. 191, 65 (2015)). This conserves the weight, the
momentum and the kinetic energy of every group exactly; the merged markers sit
at the group's mean position.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import cunumpy as xp
import numpy as np

from plasmacoll.markers import MarkerSet

__all__ = ["merge_markers", "split_markers"]


def split_markers(markers: MarkerSet, max_weight: float) -> int:
    """Split every marker heavier than ``max_weight`` into equal lighter copies.

    A marker of weight ``w`` becomes ``ceil(w / max_weight)`` markers at the
    same position and velocity that share its weight.

    Returns:
        The number of markers added.
    """
    if max_weight <= 0.0:
        raise ValueError("max_weight must be > 0")
    weights = markers.weights
    heavy = weights > max_weight
    if not bool(xp.any(heavy)):
        return 0
    # The copies' row indices are built on the host: CuPy's repeat takes no
    # array of repeat counts.
    host_weights = np.asarray(xp.to_numpy(weights), dtype=float)
    rows = np.flatnonzero(host_weights > max_weight)
    counts = np.ceil(host_weights[rows] / max_weight).astype(np.int64)
    copies = xp.asarray(np.repeat(rows, counts))
    positions = markers.positions[copies]
    velocities = markers.velocities[copies]
    new_weights = xp.asarray(np.repeat(host_weights[rows] / counts, counts))
    markers.remove(heavy)
    markers.add(positions, velocities, new_weights)
    return int(counts.sum()) - int(rows.shape[0])


def merge_markers(
    markers: MarkerSet,
    target: int,
    cell_size: float | Sequence[float] | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> int:
    """Merge markers until about ``target`` remain, conserving weight, momentum and energy.

    Within each spatial cell (of size ``cell_size`` along every position
    component; one cell for all if None) and velocity octant, the markers are
    sorted by speed and grouped in runs of ``ceil(2 N / target)``; every group
    of three or more becomes two markers. Groups are formed per cell, so the
    result can exceed ``target`` when the cells hold few markers each.

    The grouping runs on the host (NumPy); merge every few steps, not every step.

    Args:
        markers: The markers of one species.
        target: The number of markers to aim for (>= 2).
        cell_size: The size of the spatial cells in the units of the positions.
        rng: The NumPy generator of the random directions.
        seed: Seed of the generator made when ``rng`` is None.

    Returns:
        The number of markers removed.
    """
    if target < 2:
        raise ValueError("target must be >= 2")
    num = int(markers.weights.shape[0])
    if num <= target:
        return 0
    if rng is None:
        rng = np.random.default_rng(seed)

    positions = np.asarray(xp.to_numpy(markers.positions), dtype=float)
    velocities = np.asarray(xp.to_numpy(markers.velocities), dtype=float)
    weights = np.asarray(xp.to_numpy(markers.weights), dtype=float)

    # Bin keys, most significant first: spatial cell per axis, velocity octant.
    keys = []
    if cell_size is not None:
        size = np.broadcast_to(np.asarray(cell_size, dtype=float), positions.shape[1:])
        if np.any(size <= 0.0):
            raise ValueError("cell_size must be > 0")
        cells = np.floor(positions / size).astype(np.int64)
        keys.extend(cells[:, axis] for axis in range(cells.shape[1]))
    keys.append((velocities > 0.0) @ np.array([1, 2, 4]))
    speed = np.sqrt(np.sum(velocities**2, axis=1))
    order = np.lexsort([speed, *reversed(keys)])
    sorted_keys = np.stack([key[order] for key in keys], axis=1)

    # Runs of `size` consecutive markers within each bin form the groups.
    group_size = max(3, math.ceil(2.0 * num / target))
    new_bin = np.ones(num, dtype=bool)
    new_bin[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    bin_start = np.maximum.accumulate(np.where(new_bin, np.arange(num), 0))
    rank_in_bin = np.arange(num) - bin_start
    new_group = new_bin | (rank_in_bin % group_size == 0)
    group = np.cumsum(new_group) - 1
    group_counts = np.bincount(group)
    merged = group_counts[group] >= 3  # in sorted order
    if not np.any(merged):
        return 0

    members = order[merged]
    member_group = group[merged]
    groups, member_group = np.unique(member_group, return_inverse=True)
    num_groups = int(groups.shape[0])
    w = weights[members]
    total = np.bincount(member_group, weights=w, minlength=num_groups)
    safe_total = np.where(total > 0.0, total, 1.0)

    def weighted_mean(values: np.ndarray) -> np.ndarray:
        return (
            np.stack(
                [
                    np.bincount(
                        member_group, weights=w * values[:, axis], minlength=num_groups
                    )
                    for axis in range(values.shape[1])
                ],
                axis=1,
            )
            / safe_total[:, None]
        )

    mean_position = weighted_mean(positions[members])
    mean_velocity = weighted_mean(velocities[members])
    mean_square = (
        np.bincount(
            member_group,
            weights=w * np.sum(velocities[members] ** 2, axis=1),
            minlength=num_groups,
        )
        / safe_total
    )
    spread = np.sqrt(np.clip(mean_square - np.sum(mean_velocity**2, axis=1), 0.0, None))
    direction = _random_directions(rng, num_groups)
    offset = spread[:, None] * direction

    remove = np.zeros(num, dtype=bool)
    remove[members] = True
    markers.remove(xp.asarray(remove))
    markers.add(
        xp.asarray(np.concatenate([mean_position, mean_position])),
        xp.asarray(np.concatenate([mean_velocity + offset, mean_velocity - offset])),
        xp.asarray(np.concatenate([0.5 * total, 0.5 * total])),
    )
    return int(members.shape[0]) - 2 * num_groups


def _random_directions(rng: Any, num: int) -> np.ndarray:
    """Return ``num`` host unit vectors distributed uniformly on the sphere."""
    cos_theta = 1.0 - 2.0 * rng.random(num)
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 0.0, None))
    phi = 2.0 * np.pi * rng.random(num)
    return np.stack(
        (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta), axis=1
    )
