---
title: Use in a particle-in-cell code
description: Plugging your own particle containers into the operator, MPI and GPUs.
sidebar:
  order: 4
---

In a particle-in-cell (PIC) cycle the collision step sits between pushes. plasmacoll never sees the
fields or the grid, only the markers of each species.

## The marker-set protocol

`collide_species` accepts any object with three array attributes and two methods (the
`plasmacoll.MarkerSet` protocol):

| Member                                  | Meaning                                                     |
| --------------------------------------- | ----------------------------------------------------------- |
| `positions`                             | shape `(N, ndim)`; read by density profiles, copied to products |
| `velocities`                            | shape `(N, 3)` in m/s; collided rows are updated in place   |
| `weights`                               | shape `(N,)`; physical particles per marker                 |
| `remove(mask)`                          | drop the markers flagged in a boolean mask of length `N`    |
| `add(positions, velocities, weights)`   | append new markers                                          |

An optional `alive` attribute (boolean, length `N`) restricts collisions to the flagged rows,
for containers that keep holes in preallocated storage. A thin adapter over an existing
container is enough:

```python
class SpeciesAdapter:
    def __init__(self, species):
        self._species = species

    @property
    def positions(self):
        return self._species.x[: self._species.n]

    @property
    def velocities(self):
        return self._species.v[: self._species.n]  # a view: updated in place

    @property
    def weights(self):
        return self._species.w[: self._species.n]

    @property
    def alive(self):
        return self._species.active[: self._species.n]

    def remove(self, mask):
        self._species.deactivate(mask)

    def add(self, positions, velocities, weights):
        self._species.append(positions, velocities, weights)
```

Velocities must have three components even in a 1D or 2D simulation (1D-3V, 2D-3V), because
scattering is three-dimensional.

For a single array set, `collide(species, positions, velocities, weights, dt, alive=None,
in_place=False)` returns an `MCCResult` and leaves the bookkeeping to the caller.

## Marker weights

Products inherit the weight of the incident marker, so charge is conserved marker by marker
whatever the weights. If your code uses one weight per species, give species that exchange
particles the same weight. Otherwise ionization grows and attachment drains the marker count;
call `plasmacoll.merge_markers(markers, target, cell_size=dx)` every few steps on species that
grew too large. It merges markers within one grid cell and velocity octant while conserving
weight, momentum and kinetic energy, and works on any `MarkerSet` through `remove` and `add`.
The grouping runs on the host, so on GPUs merge rarely rather than every step.
`split_markers(markers, max_weight)` divides markers that have become too heavy.

## Coupling to a neutral model

Pass `record_events=True` to `collide_species` and every species' `MCCDiagnostics.events` holds,
per background, the positions of its collisions and the momentum, energy and gas particles each
transferred. Deposit them with your shape functions to get the momentum, heating and particle
sources of a fluid or DSMC neutral model. See [the null-collision method](/plasmacoll/guides/null-collision-method/#coupling-to-the-gas).

## Charged-particle collisions

`ChargedCollisions.collide_species(species, dt, cell_volume=dx * area, cell_size=dx)` pairs the
markers within grid cells for Coulomb scattering and charged–charged reactions. See
[charged-particle collisions](/plasmacoll/guides/charged-collisions/).

## MPI

The operator holds no global state. With particle decomposition every rank builds the same
operator with `seed=seed, rank=comm.Get_rank()` and collides its own markers. The ranks draw
independent random streams, and the run is reproducible for a fixed seed and rank count. Sum
`MCCDiagnostics.counts` over the ranks for global collision counts.

## GPUs

plasmacoll uses [cunumpy](https://pypi.org/project/cunumpy/) for all array operations. With the CuPy
backend selected, pass CuPy arrays. The bound tables stay on the host (a handful of floats), and
everything per marker runs on the device. With a generator other than NumPy's, such as CuPy's,
the candidate selection draws one random number per marker, which runs in parallel on the device.
