"""Marker storage that :meth:`~plasmacoll.mcc.MonteCarloCollisions.collide_species` works on.

Any object with ``positions``, ``velocities`` and ``weights`` arrays and
``remove``/``add`` methods (the :class:`MarkerSet` protocol) can be collided,
so a particle-in-cell code can pass its own particle containers through a thin
adapter. :class:`ParticleArrays` is a simple implementation that keeps the
markers in three arrays and compacts them on removal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import cunumpy as xp

from plasmacoll._types import Array
from plasmacoll.constants import BOLTZMANN, ELEMENTARY_CHARGE

__all__ = ["MarkerSet", "ParticleArrays"]


@runtime_checkable
class MarkerSet(Protocol):
    """The markers of one species, as the collision operator needs them.

    ``positions`` has shape ``(N, ndim)`` (any ``ndim``, read only by density
    profiles and copied to products), ``velocities`` shape ``(N, 3)`` in m/s,
    ``weights`` shape ``(N,)``: the number of physical particles per marker.
    Collisions update rows of ``velocities`` in place.
    """

    @property
    def positions(self) -> Array:
        """Return the marker positions, shape ``(N, ndim)``."""
        ...

    @property
    def velocities(self) -> Array:
        """Return the marker velocities, shape ``(N, 3)``, updated in place."""
        ...

    @property
    def weights(self) -> Array:
        """Return the marker weights, shape ``(N,)``."""
        ...

    def remove(self, mask: Array) -> None:
        """Remove the markers flagged in the boolean ``mask`` of length ``N``."""
        ...

    def add(self, positions: Array, velocities: Array, weights: Array) -> None:
        """Append new markers."""
        ...


@dataclass
class ParticleArrays:
    """Markers of one species held in three arrays.

    Attributes:
        positions: Positions, shape ``(N, ndim)``.
        velocities: Velocities in m/s, shape ``(N, 3)``.
        weights: Physical particles per marker, shape ``(N,)``.
    """

    positions: Array
    velocities: Array
    weights: Array

    def __post_init__(self) -> None:
        """Validate the shapes and convert to float arrays of the active backend."""
        self.positions = xp.asarray(self.positions, dtype=float)
        self.velocities = xp.asarray(self.velocities, dtype=float)
        self.weights = xp.asarray(self.weights, dtype=float)
        if self.positions.ndim != 2:
            raise ValueError("positions must have shape (N, ndim)")
        if self.velocities.ndim != 2 or self.velocities.shape[1] != 3:
            raise ValueError("velocities must have shape (N, 3)")
        num = self.velocities.shape[0]
        if self.positions.shape[0] != num or self.weights.shape != (num,):
            raise ValueError(
                "positions, velocities and weights need one row per marker"
            )

    @classmethod
    def empty(cls, ndim: int = 1) -> ParticleArrays:
        """Return a set without markers, with ``ndim`` position components."""
        return cls(xp.zeros((0, ndim)), xp.zeros((0, 3)), xp.zeros(0))

    @classmethod
    def maxwellian(
        cls,
        num: int,
        mass: float,
        temperature: float,
        weight: float = 1.0,
        drift: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ndim: int = 1,
        rng: Any = None,
        seed: int | None = None,
    ) -> ParticleArrays:
        """Sample ``num`` markers from a drifting Maxwellian, all at the origin.

        Args:
            num: The number of markers.
            mass: Particle mass in kg.
            temperature: Temperature in K; see :func:`plasmacoll.constants.ev_to_kelvin`.
            weight: Physical particles per marker.
            drift: Mean velocity in m/s.
            ndim: Number of position components.
            rng: A NumPy or CuPy ``Generator``; one seeded with ``seed`` if None.
            seed: Seed of the generator made when ``rng`` is None.
        """
        if num < 0:
            raise ValueError("num must be >= 0")
        if mass <= 0.0:
            raise ValueError("mass must be > 0")
        if temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if rng is None:
            rng = xp.rng.get_rng(seed)
        thermal_speed = math.sqrt(BOLTZMANN * temperature / mass)
        velocities = rng.standard_normal((num, 3)) * thermal_speed + xp.asarray(
            drift, dtype=float
        )
        return cls(xp.zeros((num, ndim)), velocities, xp.full(num, float(weight)))

    def __len__(self) -> int:
        """Return the number of markers."""
        return int(self.weights.shape[0])

    def remove(self, mask: Array) -> None:
        """Remove the markers flagged in the boolean ``mask``."""
        keep = ~xp.asarray(mask, dtype=bool)
        self.positions = self.positions[keep]
        self.velocities = self.velocities[keep]
        self.weights = self.weights[keep]

    def add(self, positions: Array, velocities: Array, weights: Array) -> None:
        """Append new markers."""
        self.positions = xp.concatenate([self.positions, xp.asarray(positions)])
        self.velocities = xp.concatenate([self.velocities, xp.asarray(velocities)])
        self.weights = xp.concatenate([self.weights, xp.asarray(weights)])

    @property
    def total_weight(self) -> float:
        """Return the number of physical particles the markers represent."""
        return float(xp.sum(self.weights))

    def mean_energy_ev(self, mass: float) -> float:
        """Return the weighted mean kinetic energy in eV (0 without markers)."""
        total = self.total_weight
        if total <= 0.0:
            return 0.0
        energy = 0.5 * mass * xp.sum(self.weights * xp.sum(self.velocities**2, axis=1))
        return float(energy) / (total * ELEMENTARY_CHARGE)

    def temperature(self, mass: float) -> float:
        """Return the kinetic temperature in K about the mean velocity (0 if empty)."""
        total = self.total_weight
        if total <= 0.0:
            return 0.0
        weights = self.weights[:, None]
        mean = xp.sum(weights * self.velocities, axis=0) / total
        spread = xp.sum(weights * (self.velocities - mean) ** 2) / total
        return mass * float(spread) / (3.0 * BOLTZMANN)
